import torch
import time
import os
import json
from typing import Optional
from dataclasses import dataclass, field
from transformers import DynamicCache


def _rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def _extract_rope_params(model):
    inner = model.model if hasattr(model, "model") else model
    rotary_emb = getattr(inner, "rotary_emb", None)
    if rotary_emb is None and hasattr(inner, "layers") and len(inner.layers) > 0:
        rotary_emb = getattr(inner.layers[0].self_attn, "rotary_emb", None)

    if rotary_emb is not None and hasattr(rotary_emb, "inv_freq"):
        inv_freq = rotary_emb.inv_freq.float().cpu().clone()
        attention_scaling = float(getattr(rotary_emb, "attention_scaling", 1.0))
        return inv_freq, attention_scaling

    config = model.config
    head_dim = getattr(config, "head_dim", None) or config.hidden_size // config.num_attention_heads
    rope_params = getattr(config, "rope_parameters", {})
    rope_theta = rope_params.get("rope_theta", getattr(config, "rope_theta", 10000.0))
    inv_freq = 1.0 / (rope_theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))
    return inv_freq, 1.0


def _apply_rope(k, positions, inv_freq, inverse=False, attention_scaling=1.0):
    rotary_dim = inv_freq.shape[0] * 2
    head_dim = k.shape[-1]

    if rotary_dim < head_dim:
        k_rot = k[..., :rotary_dim]
        k_pass = k[..., rotary_dim:]
    else:
        k_rot = k
        k_pass = None

    freqs = torch.outer(positions.float(), inv_freq.float())
    emb = torch.cat((freqs, freqs), dim=-1)
    cos = (emb.cos() * attention_scaling).unsqueeze(0).unsqueeze(0)
    sin = (emb.sin() * attention_scaling).unsqueeze(0).unsqueeze(0)

    k_f = k_rot.float()
    if inverse:
        s2 = attention_scaling ** 2
        result = (k_f * cos - _rotate_half(k_f) * sin) / s2
    else:
        result = k_f * cos + _rotate_half(k_f) * sin

    result = result.to(k.dtype)
    if k_pass is not None:
        result = torch.cat([result, k_pass], dim=-1)
    return result


@dataclass
class CompiledState:
    content_id: str
    n_tokens: int
    position_start: int
    position_end: int
    kv_states: list = field(repr=False)
    compiled_at: float = field(default_factory=time.time)
    text_preview: str = ""
    derotated: bool = False

    @property
    def size_bytes(self):
        total = 0
        for k, v in self.kv_states:
            total += k.nbytes + v.nbytes
        return total


class ContextCompiler:
    def __init__(self, model, tokenizer, device="cuda:0"):
        self._model = model
        self._tok = tokenizer
        self._device = device
        self._position_counter = 0
        self._library: dict[str, CompiledState] = {}
        self._inv_freq, self._attention_scaling = _extract_rope_params(model)

    @property
    def inv_freq(self):
        return self._inv_freq

    @property
    def attention_scaling(self):
        return self._attention_scaling

    def compile(self, content_id: str, text: str, chunk_size: int = 4096) -> CompiledState:
        ids = self._tok.encode(text, add_special_tokens=False)
        if not ids:
            raise ValueError("Empty text after tokenization")

        all_keys = [[] for _ in range(self._n_layers)]
        all_vals = [[] for _ in range(self._n_layers)]
        pos_start = self._position_counter

        for chunk_start in range(0, len(ids), chunk_size):
            chunk_ids = ids[chunk_start:chunk_start + chunk_size]
            input_ids = torch.tensor([chunk_ids], device=self._device)
            position_ids = torch.arange(
                self._position_counter,
                self._position_counter + len(chunk_ids),
                device=self._device
            ).unsqueeze(0)

            with torch.no_grad():
                outputs = self._model(
                    input_ids=input_ids,
                    position_ids=position_ids,
                    use_cache=True,
                )

            kv = outputs.past_key_values
            for layer_idx in range(self._n_layers):
                k_chunk = kv.layers[layer_idx].keys.cpu()
                v_chunk = kv.layers[layer_idx].values.cpu()
                chunk_positions = torch.arange(
                    self._position_counter,
                    self._position_counter + len(chunk_ids),
                )
                k_chunk = _apply_rope(
                    k_chunk, chunk_positions, self._inv_freq,
                    inverse=True, attention_scaling=self._attention_scaling,
                )
                all_keys[layer_idx].append(k_chunk)
                all_vals[layer_idx].append(v_chunk)

            self._position_counter += len(chunk_ids)

        kv_states = []
        for layer_idx in range(self._n_layers):
            k = torch.cat(all_keys[layer_idx], dim=2) if len(all_keys[layer_idx]) > 1 else all_keys[layer_idx][0]
            v = torch.cat(all_vals[layer_idx], dim=2) if len(all_vals[layer_idx]) > 1 else all_vals[layer_idx][0]
            kv_states.append((k, v))

        state = CompiledState(
            content_id=content_id,
            n_tokens=len(ids),
            position_start=pos_start,
            position_end=self._position_counter,
            kv_states=kv_states,
            text_preview=text[:100],
            derotated=True,
        )
        self._library[content_id] = state
        return state

    def recompile(self, content_id: str, text: str) -> CompiledState:
        if content_id in self._library:
            del self._library[content_id]
        return self.compile(content_id, text)

    def compile_file(self, path: str) -> CompiledState:
        with open(path) as f:
            text = f.read()
        cid = os.path.basename(path) if os.path.sep in path or '/' in path else path
        return self.compile(cid, text)

    def remove(self, content_id: str):
        self._library.pop(content_id, None)

    @property
    def _n_layers(self):
        return self._model.config.num_hidden_layers

    @property
    def library(self):
        return self._library

    @property
    def position_counter(self):
        return self._position_counter

    def reset(self):
        self._library.clear()
        self._position_counter = 0

    def save_to_disk(self, path: str):
        os.makedirs(path, exist_ok=True)
        manifest = {}
        for cid, state in self._library.items():
            safe_name = cid.replace("/", "_").replace("\\", "_")
            state_file = f"{safe_name}.pt"
            torch.save({
                "kv_states": state.kv_states,
                "n_tokens": state.n_tokens,
                "position_start": state.position_start,
                "position_end": state.position_end,
                "text_preview": state.text_preview,
                "compiled_at": state.compiled_at,
                "derotated": state.derotated,
            }, os.path.join(path, state_file))
            manifest[cid] = {"file": state_file, "tokens": state.n_tokens,
                             "size_bytes": state.size_bytes}
        with open(os.path.join(path, "manifest.json"), "w") as f:
            json.dump({"position_counter": self._position_counter,
                        "states": manifest}, f, indent=2)
        return len(manifest)

    def load_from_disk(self, path: str) -> int:
        manifest_path = os.path.join(path, "manifest.json")
        if not os.path.exists(manifest_path):
            return 0
        with open(manifest_path) as f:
            data = json.load(f)
        self._position_counter = max(self._position_counter, data.get("position_counter", 0))
        loaded = 0
        for cid, info in data.get("states", {}).items():
            state_path = os.path.join(path, info["file"])
            if not os.path.exists(state_path):
                continue
            saved = torch.load(state_path, map_location="cpu", weights_only=False)
            state = CompiledState(
                content_id=cid,
                n_tokens=saved["n_tokens"],
                position_start=saved["position_start"],
                position_end=saved["position_end"],
                kv_states=saved["kv_states"],
                text_preview=saved.get("text_preview", ""),
                compiled_at=saved.get("compiled_at", 0.0),
                derotated=saved.get("derotated", False),
            )
            self._library[cid] = state
            loaded += 1
        return loaded


class ContextComposer:
    def __init__(self, device="cuda:0", inv_freq=None, attention_scaling=1.0):
        self._device = device
        self._inv_freq = inv_freq
        self._attention_scaling = attention_scaling

    def compose(self, *states: CompiledState) -> tuple:
        if not states:
            raise ValueError("No states to compose")

        total_bytes = sum(s.size_bytes for s in states)
        if torch.cuda.is_available():
            free_vram = torch.cuda.mem_get_info(self._device)[0]
            if total_bytes > free_vram * 0.8:
                total_tokens = sum(s.n_tokens for s in states)
                raise RuntimeError(
                    f"Composed KV states ({total_bytes/1e6:.0f} MB, {total_tokens} tokens) "
                    f"would exceed available GPU memory ({free_vram/1e6:.0f} MB free). "
                    f"Compile fewer files or use a GPU with more VRAM."
                )

        any_derotated = any(s.derotated for s in states)
        can_rerotate = self._inv_freq is not None and any_derotated

        if can_rerotate:
            return self._compose_derotated(states)

        if len(states) == 1:
            return self._to_gpu(states[0].kv_states)

        n_layers = len(states[0].kv_states)
        composed = []
        for layer_idx in range(n_layers):
            keys = [s.kv_states[layer_idx][0] for s in states]
            vals = [s.kv_states[layer_idx][1] for s in states]
            k = torch.cat(keys, dim=2).to(self._device)
            v = torch.cat(vals, dim=2).to(self._device)
            composed.append((k, v))
        return tuple(composed)

    def _compose_derotated(self, states):
        n_layers = len(states[0].kv_states)
        composed = []
        pos_cursor = 0

        per_state_positions = []
        for s in states:
            positions = torch.arange(pos_cursor, pos_cursor + s.n_tokens)
            per_state_positions.append(positions)
            pos_cursor += s.n_tokens

        for layer_idx in range(n_layers):
            layer_keys = []
            layer_vals = []
            for i, s in enumerate(states):
                k = s.kv_states[layer_idx][0]
                v = s.kv_states[layer_idx][1]
                positions = per_state_positions[i]

                if s.derotated:
                    k = _apply_rope(
                        k, positions, self._inv_freq,
                        inverse=False, attention_scaling=self._attention_scaling,
                    )
                else:
                    old_positions = torch.arange(s.position_start, s.position_end)
                    k = _apply_rope(
                        k, old_positions, self._inv_freq,
                        inverse=True, attention_scaling=self._attention_scaling,
                    )
                    k = _apply_rope(
                        k, positions, self._inv_freq,
                        inverse=False, attention_scaling=self._attention_scaling,
                    )

                layer_keys.append(k)
                layer_vals.append(v)

            k_cat = torch.cat(layer_keys, dim=2).to(self._device)
            v_cat = torch.cat(layer_vals, dim=2).to(self._device)
            composed.append((k_cat, v_cat))

        return tuple(composed)

    def _to_gpu(self, kv_states):
        return tuple((k.to(self._device), v.to(self._device)) for k, v in kv_states)


class ContextGenerator:
    def __init__(self, model, tokenizer, device="cuda:0"):
        self._model = model
        self._tok = tokenizer
        self._device = device

    def generate(self, composed_kv, max_new_tokens=256, temperature=0.0,
                 top_p=1.0, position_offset=0):
        cache = DynamicCache()
        for layer_idx, (k, v) in enumerate(composed_kv):
            cache.update(k, v, layer_idx)

        seq_len = composed_kv[0][0].shape[2]
        last_token_id = self._tok.bos_token_id or 0
        input_ids = torch.tensor([[last_token_id]], device=self._device)
        position_ids = torch.tensor([[seq_len + position_offset]], device=self._device)

        do_sample = temperature > 0.01
        eos_ids = set()
        if self._tok.eos_token_id is not None:
            eos_ids.add(self._tok.eos_token_id)
        try:
            eot = self._tok("<|endoftext|>", add_special_tokens=False)["input_ids"]
            if eot:
                eos_ids.add(eot[0])
        except Exception:
            pass
        try:
            im_end = self._tok("<|im_end|>", add_special_tokens=False)["input_ids"]
            if im_end:
                eos_ids.add(im_end[0])
        except Exception:
            pass

        generated = []
        t0 = time.time()

        with torch.no_grad():
            for _ in range(max_new_tokens):
                out = self._model(
                    input_ids=input_ids,
                    past_key_values=cache,
                    position_ids=position_ids,
                    use_cache=True,
                )
                cache = out.past_key_values
                logits = out.logits[0, -1, :]

                if do_sample:
                    logits = logits / temperature
                    if top_p < 1.0:
                        sorted_logits, sorted_idx = logits.sort(descending=True)
                        cumprob = sorted_logits.softmax(dim=-1).cumsum(dim=-1)
                        mask = cumprob - sorted_logits.softmax(dim=-1) >= top_p
                        sorted_logits[mask] = float("-inf")
                        logits = torch.zeros_like(logits).scatter_(0, sorted_idx, sorted_logits)
                    next_id = torch.multinomial(logits.softmax(dim=-1), 1).item()
                else:
                    next_id = logits.argmax().item()

                if next_id in eos_ids:
                    break

                generated.append(next_id)
                input_ids = torch.tensor([[next_id]], device=self._device)
                position_ids = position_ids + 1

        elapsed = time.time() - t0
        text = self._tok.decode(generated, skip_special_tokens=True)
        tps = len(generated) / elapsed if elapsed > 0 else 0

        return text, len(generated), tps

    def stream(self, composed_kv, max_new_tokens=256, temperature=0.0,
               top_p=1.0, position_offset=0):
        cache = DynamicCache()
        for layer_idx, (k, v) in enumerate(composed_kv):
            cache.update(k, v, layer_idx)

        seq_len = composed_kv[0][0].shape[2]
        last_token_id = self._tok.bos_token_id or 0
        input_ids = torch.tensor([[last_token_id]], device=self._device)
        position_ids = torch.tensor([[seq_len + position_offset]], device=self._device)

        eos_ids = set()
        if self._tok.eos_token_id is not None:
            eos_ids.add(self._tok.eos_token_id)
        for special in ["<|endoftext|>", "<|im_end|>"]:
            try:
                ids = self._tok(special, add_special_tokens=False)["input_ids"]
                if ids:
                    eos_ids.add(ids[0])
            except Exception:
                pass

        with torch.no_grad():
            for _ in range(max_new_tokens):
                out = self._model(
                    input_ids=input_ids,
                    past_key_values=cache,
                    position_ids=position_ids,
                    use_cache=True,
                )
                cache = out.past_key_values
                logits = out.logits[0, -1, :]

                if temperature > 0.01:
                    logits = logits / temperature
                    next_id = torch.multinomial(logits.softmax(dim=-1), 1).item()
                else:
                    next_id = logits.argmax().item()

                if next_id in eos_ids:
                    break

                yield self._tok.decode([next_id], skip_special_tokens=False)
                input_ids = torch.tensor([[next_id]], device=self._device)
                position_ids = position_ids + 1
