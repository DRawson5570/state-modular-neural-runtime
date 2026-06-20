import torch
import torch.nn as nn
import time
import os
import json
import re
import glob as globmod
from typing import Optional
from threading import Lock
from dataclasses import dataclass

from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig, DynamicCache

try:
    from transformers import BitsAndBytesConfig
    _BNB_AVAILABLE = True
except ImportError:
    _BNB_AVAILABLE = False

from compiled_context import ContextCompiler, ContextComposer, ContextGenerator


@dataclass
class EngineStats:
    turns: int = 0
    compiled_states: int = 0
    compiled_tokens: int = 0
    compiled_bytes: int = 0
    tool_calls: int = 0
    last_tps: float = 0.0
    last_tokens: int = 0
    gpu_vram_gb: float = 0.0
    mode: str = ""


class _CPUMLPWrapper(nn.Module):
    def __init__(self, mlp, gpu_device):
        super().__init__()
        self.mlp = mlp
        self.gpu = gpu_device

    def forward(self, x):
        return self.mlp(x.to("cpu")).to(self.gpu)


TOOLS = [
    {"type": "function", "function": {
        "name": "compile_text",
        "description": "Compile text into persistent memory so you can reason about it later. Use for any content the user provides (code, articles, data).",
        "parameters": {"type": "object", "properties": {
            "id": {"type": "string", "description": "Short identifier (e.g. 'main_py', 'article')"},
            "text": {"type": "string", "description": "The text content to compile"},
        }, "required": ["id", "text"]},
    }},
    {"type": "function", "function": {
        "name": "compile_file",
        "description": "Read a file from disk and compile it into persistent memory.",
        "parameters": {"type": "object", "properties": {
            "path": {"type": "string", "description": "File path to read and compile"},
        }, "required": ["path"]},
    }},
    {"type": "function", "function": {
        "name": "compile_directory",
        "description": "Compile all matching files in a directory into persistent memory.",
        "parameters": {"type": "object", "properties": {
            "path": {"type": "string", "description": "Directory path"},
            "pattern": {"type": "string", "description": "Glob pattern (default: '*.py')"},
        }, "required": ["path"]},
    }},
    {"type": "function", "function": {
        "name": "list_compiled",
        "description": "List all content currently compiled in persistent memory.",
        "parameters": {"type": "object", "properties": {}},
    }},
    {"type": "function", "function": {
        "name": "list_directory",
        "description": "List files and folders in a directory.",
        "parameters": {"type": "object", "properties": {
            "path": {"type": "string", "description": "Directory path"},
        }, "required": ["path"]},
    }},
    {"type": "function", "function": {
        "name": "search_files",
        "description": "Search for files matching a glob pattern.",
        "parameters": {"type": "object", "properties": {
            "pattern": {"type": "string", "description": "Glob pattern (e.g. '**/*.py')"},
            "path": {"type": "string", "description": "Root directory (default: '.')"},
        }, "required": ["pattern"]},
    }},
    {"type": "function", "function": {
        "name": "grep",
        "description": "Search file contents for a pattern. Returns matching lines with file and line number.",
        "parameters": {"type": "object", "properties": {
            "pattern": {"type": "string", "description": "Search pattern"},
            "path": {"type": "string", "description": "File or directory to search"},
        }, "required": ["pattern", "path"]},
    }},
    {"type": "function", "function": {
        "name": "write_file",
        "description": "Write content to a file and compile it into persistent memory.",
        "parameters": {"type": "object", "properties": {
            "path": {"type": "string", "description": "File path to write"},
            "content": {"type": "string", "description": "File content"},
        }, "required": ["path", "content"]},
    }},
]


class CompiledInference:

    def __init__(
        self,
        model_name: str,
        device: str = "cuda:0",
        hybrid: str = "auto",
        system_prompt: str = "",
        torch_dtype=torch.bfloat16,
    ):
        self._device = device
        self._lock = Lock()
        self.stats = EngineStats()

        self._config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
        self._d_model = self._config.hidden_size
        self._n_layers = self._config.num_hidden_layers
        self._tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

        self._mode = self._resolve_mode(hybrid, torch_dtype)
        self.stats.mode = self._mode

        if self._mode == "gpu":
            self._load_gpu(model_name, torch_dtype)
        else:
            self._load_hybrid(model_name, torch_dtype)

        self._compiler = ContextCompiler(self._model, self._tokenizer, device)
        self._composer = ContextComposer(
            device,
            inv_freq=self._compiler.inv_freq,
            attention_scaling=self._compiler.attention_scaling,
        )
        self._generator = ContextGenerator(self._model, self._tokenizer, device)
        self._system_prompt = system_prompt
        self._messages: list[dict] = []

        self.stats.gpu_vram_gb = torch.cuda.memory_allocated(device) / 1e9 if torch.cuda.is_available() else 0.0

    def _resolve_mode(self, hybrid, dtype):
        if hybrid == "gpu":
            return "gpu"
        if hybrid == "hybrid":
            return "hybrid"
        if not torch.cuda.is_available():
            return "hybrid"
        free_vram = torch.cuda.mem_get_info(self._device)[0]
        vocab = getattr(self._config, "vocab_size", 50000)
        n_params = (
            vocab * self._d_model
            + self._n_layers * (
                4 * self._d_model ** 2
                + 3 * self._d_model * getattr(self._config, "intermediate_size", 4 * self._d_model)
            )
        )
        if n_params * 0.55 < free_vram * 0.85:
            return "gpu"
        return "hybrid"

    def _load_gpu(self, model_name, dtype):
        kwargs = {"device_map": self._device, "trust_remote_code": True}
        if _BNB_AVAILABLE:
            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_compute_dtype=dtype
            )
        else:
            kwargs["dtype"] = dtype
        self._model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)
        self._model.eval()

    def _load_hybrid(self, model_name, dtype):
        self._model = AutoModelForCausalLM.from_pretrained(
            model_name, dtype=dtype, device_map="cpu"
        )
        inner = self._model.model if hasattr(self._model, "model") else self._model
        inner.embed_tokens.to(self._device)
        inner.norm.to(self._device)
        if hasattr(inner, "rotary_emb"):
            inner.rotary_emb.to(self._device)
        lm = getattr(self._model, "lm_head", None)
        if lm and hasattr(lm, "weight") and lm.weight.data_ptr() != inner.embed_tokens.weight.data_ptr():
            lm.to(self._device)
        for layer in inner.layers:
            layer.self_attn.to(self._device)
            layer.input_layernorm.to(self._device)
            if hasattr(layer, "post_attention_layernorm"):
                layer.post_attention_layernorm.to(self._device)
            layer.mlp = _CPUMLPWrapper(layer.mlp, self._device)
        self._model.eval()
        torch.cuda.empty_cache()

    def _execute_tool(self, name: str, args: dict) -> str:
        self.stats.tool_calls += 1
        try:
            if name == "compile_text":
                state = self._compiler.compile(args["id"], args["text"])
                self.stats.compiled_states = len(self._compiler.library)
                self.stats.compiled_tokens += state.n_tokens
                self.stats.compiled_bytes += state.size_bytes
                return f"Compiled '{args['id']}': {state.n_tokens} tokens ({state.size_bytes/1e6:.1f} MB)"

            elif name == "compile_file":
                path = args["path"]
                if not os.path.exists(path):
                    return f"Error: file not found: {path}"
                with open(path) as f:
                    text = f.read()
                cid = os.path.basename(path)
                state = self._compiler.compile(cid, text)
                self.stats.compiled_states = len(self._compiler.library)
                self.stats.compiled_tokens += state.n_tokens
                self.stats.compiled_bytes += state.size_bytes
                return f"Compiled '{cid}': {state.n_tokens} tokens, {len(text)} chars ({state.size_bytes/1e6:.1f} MB)"

            elif name == "compile_directory":
                path = args["path"]
                pattern = args.get("pattern", "*.py")
                files = sorted(globmod.glob(os.path.join(path, "**", pattern), recursive=True))
                if not files:
                    return f"No files matching {pattern} in {path}"
                results = []
                for fp in files:
                    try:
                        with open(fp) as f:
                            text = f.read()
                        cid = os.path.relpath(fp, path)
                        state = self._compiler.compile(cid, text)
                        results.append(f"  {cid}: {state.n_tokens} tokens")
                        self.stats.compiled_tokens += state.n_tokens
                        self.stats.compiled_bytes += state.size_bytes
                    except Exception as e:
                        results.append(f"  {fp}: ERROR {e}")
                self.stats.compiled_states = len(self._compiler.library)
                return f"Compiled {len(results)} files:\n" + "\n".join(results)

            elif name == "list_compiled":
                if not self._compiler.library:
                    return "No content compiled yet."
                lines = []
                for cid, state in self._compiler.library.items():
                    lines.append(f"  {cid}: {state.n_tokens} tokens ({state.size_bytes/1e6:.1f} MB)")
                total = sum(s.size_bytes for s in self._compiler.library.values())
                return f"{len(self._compiler.library)} items ({total/1e6:.1f} MB total):\n" + "\n".join(lines)

            elif name == "list_directory":
                path = args.get("path", ".")
                if not os.path.isdir(path):
                    return f"Error: not a directory: {path}"
                entries = sorted(os.listdir(path))
                items = []
                for e in entries[:50]:
                    fp = os.path.join(path, e)
                    if os.path.isdir(fp):
                        items.append(f"  {e}/")
                    else:
                        sz = os.path.getsize(fp)
                        items.append(f"  {e} ({sz} bytes)")
                return "\n".join(items) + (f"\n  ... and {len(entries)-50} more" if len(entries) > 50 else "")

            elif name == "search_files":
                pattern = args["pattern"]
                path = args.get("path", ".")
                matches = sorted(globmod.glob(os.path.join(path, pattern), recursive=True))[:30]
                if not matches:
                    return "No matches."
                return "\n".join(f"  {m}" for m in matches)

            elif name == "grep":
                pattern = args["pattern"]
                path = args["path"]
                results = []
                target_files = []
                if os.path.isfile(path):
                    target_files = [path]
                elif os.path.isdir(path):
                    target_files = sorted(globmod.glob(os.path.join(path, "**", "*"), recursive=True))
                for fp in target_files:
                    if not os.path.isfile(fp):
                        continue
                    try:
                        with open(fp) as f:
                            for i, line in enumerate(f, 1):
                                if pattern.lower() in line.lower():
                                    results.append(f"  {fp}:{i}: {line.rstrip()[:100]}")
                                    if len(results) >= 20:
                                        break
                    except (UnicodeDecodeError, IsADirectoryError):
                        continue
                    if len(results) >= 20:
                        break
                return "\n".join(results) if results else "No matches."

            elif name == "write_file":
                path = args["path"]
                content = args["content"]
                os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
                with open(path, "w") as f:
                    f.write(content)
                cid = os.path.basename(path)
                state = self._compiler.compile(cid, content)
                self.stats.compiled_states = len(self._compiler.library)
                return f"Written {len(content)} chars to {path} and compiled ({state.n_tokens} tokens)"

            else:
                return f"Unknown tool: {name}"
        except Exception as e:
            return f"Error: {type(e).__name__}: {e}"

    def _parse_tool_calls(self, text: str) -> list[dict]:
        calls = []
        for match in re.finditer(r'<tool_call>\s*(\{.*?\})\s*</tool_call>', text, re.DOTALL):
            try:
                call = json.loads(match.group(1))
                if "name" in call:
                    calls.append(call)
            except json.JSONDecodeError:
                continue
        if not calls:
            for match in re.finditer(r'✿FUNCTION✿\s*(\w+)\s*\n\s*✿ARGS✿\s*(\{.*?\})', text, re.DOTALL):
                try:
                    calls.append({"name": match.group(1), "arguments": json.loads(match.group(2))})
                except json.JSONDecodeError:
                    continue
        return calls

    def _build_messages(self, user_message: str) -> list[dict]:
        messages = []
        sys_content = self._system_prompt or "You are a helpful assistant."
        sys_content += "\n\nYou have tools available. Use them when you need to read files, compile content, or navigate a codebase."
        messages.append({"role": "system", "content": sys_content})
        for m in self._messages[-10:]:
            messages.append(m)
        messages.append({"role": "user", "content": user_message})
        return messages

    def _prefill(self, messages):
        try:
            prompt = self._tokenizer.apply_chat_template(
                messages, tools=TOOLS, tokenize=False, add_generation_prompt=True
            )
        except Exception:
            prompt = self._tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )

        compiled_states = list(self._compiler.library.values())
        if compiled_states:
            content_kv = self._composer.compose(*compiled_states)
            content_len = sum(s.n_tokens for s in compiled_states)
            input_ids = self._tokenizer(prompt, return_tensors="pt").input_ids.to(self._device)
            cache = DynamicCache()
            for k, v in content_kv:
                cache.update(k, v, len(cache.layers) if hasattr(cache, 'layers') else cache.__len__())
            pos = torch.arange(content_len, content_len + input_ids.shape[1], device=self._device).unsqueeze(0)
            with torch.no_grad():
                out = self._model(input_ids=input_ids, past_key_values=cache, position_ids=pos, use_cache=True)
            gen_pos = content_len + input_ids.shape[1]
        else:
            input_ids = self._tokenizer(prompt, return_tensors="pt").input_ids.to(self._device)
            with torch.no_grad():
                out = self._model(input_ids=input_ids, use_cache=True)
            gen_pos = input_ids.shape[1]

        eos_ids = set()
        if self._tokenizer.eos_token_id is not None:
            eos_ids.add(self._tokenizer.eos_token_id)
        for special in ["<|im_end|>", "<|endoftext|>"]:
            try:
                ids = self._tokenizer(special, add_special_tokens=False)["input_ids"]
                if ids:
                    eos_ids.update(ids)
            except Exception:
                pass

        return out.past_key_values, gen_pos, eos_ids, out.logits[0, -1, :]

    def _sample(self, logits, temperature, top_p):
        if temperature > 0.01:
            logits = logits / temperature
            if top_p < 1.0:
                sorted_logits, sorted_idx = logits.sort(descending=True)
                cumprob = sorted_logits.softmax(dim=-1).cumsum(dim=-1)
                mask = cumprob - sorted_logits.softmax(dim=-1) >= top_p
                sorted_logits[mask] = float("-inf")
                logits = torch.zeros_like(logits).scatter_(0, sorted_idx, sorted_logits)
            return torch.multinomial(logits.softmax(dim=-1), 1).item()
        return logits.argmax().item()

    def _generate_turn(self, messages, temperature=0.0, top_p=1.0, max_tokens=2048):
        cache, gen_pos, eos_ids, first_logits = self._prefill(messages)
        next_id = self._sample(first_logits, temperature, top_p)
        if next_id in eos_ids:
            return ""
        generated = [next_id]

        t0 = time.time()
        with torch.no_grad():
            for _ in range(max_tokens - 1):
                inp = torch.tensor([[next_id]], device=self._device)
                p = torch.tensor([[gen_pos]], device=self._device)
                out = self._model(input_ids=inp, past_key_values=cache, position_ids=p, use_cache=True)
                cache = out.past_key_values
                gen_pos += 1
                next_id = self._sample(out.logits[0, -1, :], temperature, top_p)
                if next_id in eos_ids:
                    break
                generated.append(next_id)

        elapsed = time.time() - t0
        self.stats.last_tokens = len(generated)
        self.stats.last_tps = len(generated) / elapsed if elapsed > 0 else 0
        return self._tokenizer.decode(generated, skip_special_tokens=True)

    def _stream_turn(self, messages, temperature=0.0, top_p=1.0, max_tokens=2048):
        cache, gen_pos, eos_ids, first_logits = self._prefill(messages)
        next_id = self._sample(first_logits, temperature, top_p)
        if next_id in eos_ids:
            return
        yield self._tokenizer.decode([next_id], skip_special_tokens=False)
        generated_count = 1

        t0 = time.time()
        with torch.no_grad():
            for _ in range(max_tokens - 1):
                inp = torch.tensor([[next_id]], device=self._device)
                p = torch.tensor([[gen_pos]], device=self._device)
                out = self._model(input_ids=inp, past_key_values=cache, position_ids=p, use_cache=True)
                cache = out.past_key_values
                gen_pos += 1
                next_id = self._sample(out.logits[0, -1, :], temperature, top_p)
                if next_id in eos_ids:
                    break
                generated_count += 1
                yield self._tokenizer.decode([next_id], skip_special_tokens=False)

        elapsed = time.time() - t0
        self.stats.last_tokens = generated_count
        self.stats.last_tps = generated_count / elapsed if elapsed > 0 else 0

    def chat(self, message: str, max_tokens: int = 2048, temperature: float = 0.0,
             top_p: float = 1.0, stream: bool = False, max_tool_rounds: int = 5):
        with self._lock:
            messages = self._build_messages(message)

            for _ in range(max_tool_rounds):
                response = self._generate_turn(messages, temperature, top_p, max_tokens)
                tool_calls = self._parse_tool_calls(response)
                if not tool_calls:
                    break
                messages.append({"role": "assistant", "content": response})
                for call in tool_calls:
                    name = call.get("name", "")
                    args = call.get("arguments", {})
                    if isinstance(args, str):
                        try:
                            args = json.loads(args)
                        except json.JSONDecodeError:
                            args = {}
                    result = self._execute_tool(name, args)
                    messages.append({"role": "tool", "content": result, "name": name})
            else:
                response = ""

            if stream:
                def gen():
                    chunks = []
                    for token in self._stream_turn(messages, temperature, top_p, max_tokens):
                        chunks.append(token)
                        yield token
                    full = "".join(chunks)
                    self._messages.append({"role": "user", "content": message})
                    self._messages.append({"role": "assistant", "content": full})
                    self.stats.turns += 1
                return gen()

            if not response:
                response = self._generate_turn(messages, temperature, top_p, max_tokens)
            self._messages.append({"role": "user", "content": message})
            self._messages.append({"role": "assistant", "content": response})
            self.stats.turns += 1
            return response

    def compile(self, content_id: str, text: str):
        try:
            state = self._compiler.compile(content_id, text)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            raise RuntimeError(f"GPU OOM compiling '{content_id}' ({len(text)} chars). Try shorter content or clear compiled states.")
        except ValueError as e:
            raise ValueError(f"Cannot compile '{content_id}': {e}")
        self.stats.compiled_states = len(self._compiler.library)
        self.stats.compiled_tokens += state.n_tokens
        self.stats.compiled_bytes += state.size_bytes
        return state

    def clear(self):
        self._compiler.reset()
        self._messages.clear()
        self.stats = EngineStats(mode=self._mode, gpu_vram_gb=self.stats.gpu_vram_gb)

    def save(self, path: str) -> int:
        n = self._compiler.save_to_disk(path)
        return n

    def load(self, path: str) -> int:
        n = self._compiler.load_from_disk(path)
        self.stats.compiled_states = len(self._compiler.library)
        self.stats.compiled_tokens = sum(s.n_tokens for s in self._compiler.library.values())
        self.stats.compiled_bytes = sum(s.size_bytes for s in self._compiler.library.values())
        return n

    @property
    def library(self):
        return self._compiler.library

    def __repr__(self):
        return (
            f"CompiledInference(mode={self._mode}, d={self._d_model}, "
            f"layers={self._n_layers}, compiled={self.stats.compiled_states} states, "
            f"{self.stats.compiled_tokens} tokens, {self.stats.compiled_bytes/1e6:.1f} MB)"
        )
