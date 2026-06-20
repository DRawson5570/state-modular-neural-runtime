#!/usr/bin/env python3
"""Compiled context on Qwen3.6-35B-A3B via llama-cpp-python.
Direct KV cache save/restore — the right way."""
import time, os, sys

os.environ["LLAMA_CPP_LIB_PATH"] = "/usr/local/lib/ollama"
LIB_DIR = "/usr/local/lib/ollama"
ld = os.environ.get("LD_LIBRARY_PATH", "")
if LIB_DIR not in ld:
    os.environ["LD_LIBRARY_PATH"] = LIB_DIR + ":" + LIB_DIR + "/cuda_v12:" + ld

import ctypes
_ggml = ctypes.CDLL(os.path.join(LIB_DIR, "libggml.so"), mode=ctypes.RTLD_GLOBAL)
_ggml.ggml_backend_load_all_from_path.argtypes = [ctypes.c_char_p]
_ggml.ggml_backend_load_all_from_path.restype = None
_ggml.ggml_backend_load_all_from_path(LIB_DIR.encode("utf-8"))
_ggml.ggml_backend_load_all_from_path((LIB_DIR + "/cuda_v12").encode("utf-8"))
print("GGML backends loaded (CPU + CUDA v12)")

from llama_cpp import Llama

GGUF = "/usr/share/ollama/.ollama/models/blobs/sha256-f5ee307a2982106a6eb82b62b2c00b575c9072145a759ae4660378acda8dcf2d"

print("=" * 60)
print("Qwen3.6-35B-A3B — Compiled Context (llama-cpp-python)")
print("=" * 60)

print("\nLoading model with tensor_split across 5 GPUs...")
t0 = time.time()
llm = Llama(
    model_path=GGUF,
    n_gpu_layers=-1,
    n_ctx=4096,
    n_threads=20,
    n_threads_batch=40,
    tensor_split=[1, 1, 1, 1, 1],
    verbose=False,
    use_mmap=True,
)
print("Loaded in %.1fs" % (time.time() - t0))

print("\n--- Phase 1: Basic generation ---")
resp = llm.create_chat_completion(
    messages=[{"role": "user", "content": "What is 2+2? One word answer."}],
    max_tokens=20,
    temperature=0,
)
text = resp["choices"][0]["message"]["content"]
print("Response: %s" % text.strip()[:200])

print("\n--- Phase 2: Compile content ---")
content = (
    "<|im_start|>system\n"
    "You answer questions about facts. Be concise.\n\n"
    "FACTS:\n"
    "The planet Zargthorp orbits a binary star system called Kepler-442.\n"
    "It has exactly three moons named Velnis, Krath, and Oppen.\n"
    "Zargthorp has a nitrogen-methane atmosphere and surface gravity of 1.3g.\n"
    "The largest moon Velnis has active cryovolcanoes.\n"
    "Krath is tidally locked.\n"
    "Oppen has a thin ring system made of ice particles.\n"
    "<|im_end|>\n"
)

tokens = llm.tokenize(content.encode("utf-8"), add_bos=True)
print("Compiling %d tokens..." % len(tokens))
t0 = time.time()
llm.eval(tokens)
compile_time = time.time() - t0
print("Compiled in %.2fs (%.0f tok/s prefill)" % (compile_time, len(tokens) / compile_time))

print("\n--- Phase 3: Save compiled state ---")
t0 = time.time()
compiled_state = llm.save_state()
save_time = time.time() - t0
state_size = len(compiled_state.llama_state)
print("State saved: %.1f MB in %.2fs" % (state_size / 1e6, save_time))

print("\n--- Phase 4: Generate from compiled state ---")
q1 = "<|im_start|>user\nWhat are the three moons of Zargthorp?<|im_end|>\n<|im_start|>assistant\n"
q1_tokens = llm.tokenize(q1.encode("utf-8"), add_bos=False)
llm.eval(q1_tokens)

generated = []
t0 = time.time()
for _ in range(100):
    token = llm.sample(temp=0.0)
    if token == llm.token_eos():
        break
    generated.append(token)
    llm.eval([token])
gen_time = time.time() - t0
text1 = llm.detokenize(generated).decode("utf-8", errors="replace")
tps1 = len(generated) / gen_time if gen_time > 0 else 0
print("Q1 (%d tok, %.1f tok/s): %s" % (len(generated), tps1, text1.strip()[:300]))

has_moons = all(m in text1.lower() for m in ["velnis", "krath", "oppen"])
print("All moons: %s" % has_moons)

print("\n--- Phase 5: Restore state + new query ---")
t0 = time.time()
llm.load_state(compiled_state)
restore_time = time.time() - t0
print("State restored in %.2fs" % restore_time)

q2 = "<|im_start|>user\nWhich moon has cryovolcanoes?<|im_end|>\n<|im_start|>assistant\n"
q2_tokens = llm.tokenize(q2.encode("utf-8"), add_bos=False)
llm.eval(q2_tokens)

generated2 = []
t0 = time.time()
for _ in range(50):
    token = llm.sample(temp=0.0)
    if token == llm.token_eos():
        break
    generated2.append(token)
    llm.eval([token])
gen_time2 = time.time() - t0
text2 = llm.detokenize(generated2).decode("utf-8", errors="replace")
tps2 = len(generated2) / gen_time2 if gen_time2 > 0 else 0
print("Q2 (%d tok, %.1f tok/s): %s" % (len(generated2), tps2, text2.strip()[:200]))

print("\n--- Phase 6: Restore + third query ---")
llm.load_state(compiled_state)
q3 = "<|im_start|>user\nWhat is Oppen's ring system made of?<|im_end|>\n<|im_start|>assistant\n"
q3_tokens = llm.tokenize(q3.encode("utf-8"), add_bos=False)
llm.eval(q3_tokens)

generated3 = []
t0 = time.time()
for _ in range(50):
    token = llm.sample(temp=0.0)
    if token == llm.token_eos():
        break
    generated3.append(token)
    llm.eval([token])
gen_time3 = time.time() - t0
text3 = llm.detokenize(generated3).decode("utf-8", errors="replace")
tps3 = len(generated3) / gen_time3 if gen_time3 > 0 else 0
print("Q3 (%d tok, %.1f tok/s): %s" % (len(generated3), tps3, text3.strip()[:200]))

print("\n--- Phase 7: Save to disk + reload ---")
disk_path = "/tmp/opencode/zargthorp_compiled.bin"
os.makedirs(os.path.dirname(disk_path), exist_ok=True)
t0 = time.time()
with open(disk_path, "wb") as f:
    f.write(compiled_state.llama_state)
disk_save = time.time() - t0
print("Saved to disk: %.1f MB in %.2fs" % (state_size / 1e6, disk_save))

llm.reset()
t0 = time.time()
with open(disk_path, "rb") as f:
    loaded_bytes = f.read()
from llama_cpp import LlamaState
loaded_state = LlamaState(
    input_ids=compiled_state.input_ids,
    scores=compiled_state.scores,
    n_tokens=compiled_state.n_tokens,
    llama_state=loaded_bytes,
    llama_state_size=compiled_state.llama_state_size,
    seed=getattr(compiled_state, "seed", 0),
)
llm.load_state(loaded_state)
disk_load = time.time() - t0
print("Loaded from disk in %.2fs" % disk_load)

q4 = "<|im_start|>user\nWhat star system does Zargthorp orbit?<|im_end|>\n<|im_start|>assistant\n"
q4_tokens = llm.tokenize(q4.encode("utf-8"), add_bos=False)
llm.eval(q4_tokens)

generated4 = []
t0 = time.time()
for _ in range(50):
    token = llm.sample(temp=0.0)
    if token == llm.token_eos():
        break
    generated4.append(token)
    llm.eval([token])
gen_time4 = time.time() - t0
text4 = llm.detokenize(generated4).decode("utf-8", errors="replace")
tps4 = len(generated4) / gen_time4 if gen_time4 > 0 else 0
print("Q4 (%d tok, %.1f tok/s): %s" % (len(generated4), tps4, text4.strip()[:200]))

print("\n" + "=" * 60)
print("RESULTS")
print("=" * 60)
print("Model: Qwen3.6-35B-A3B (36B params, 3B active, Q4_K_M)")
print("Backend: llama-cpp-python (native CUDA, tensor split 5 GPUs)")
print("Compiled state: %.1f MB" % (state_size / 1e6))
print("Compile speed: %.0f tok/s" % (len(tokens) / compile_time))
print("Generation: %.1f / %.1f / %.1f / %.1f tok/s" % (tps1, tps2, tps3, tps4))
print("State restore: %.2fs (from RAM) / %.2fs (from disk)" % (restore_time, disk_load))
print("Content compiled ONCE, queried 4x from saved state")

cryo = "velnis" in text2.lower() or "cryovolcan" in text2.lower()
ice = "ice" in text3.lower()
kepler = "kepler" in text4.lower() or "442" in text4.lower()
print("\nFact recall: moons=%s cryo=%s ice=%s kepler=%s" % (has_moons, cryo, ice, kepler))
if has_moons and cryo:
    print("\nCOMPILED CONTEXT WORKS ON 35B MoE. The right way.")
