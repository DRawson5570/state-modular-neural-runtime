# Architecture — Zero Context Window

> *Technical architecture of the compiled KV state inference system. 2026-06-20.*

---

## 1. The Core Insight

The transformer's attention mechanism queries key-value pairs in its KV cache.
It cannot distinguish between KV states built from live attention and KV states
loaded from pre-computed storage in RAM. KV states are KV states.

This means: instead of fitting content into a "context window," the system
COMPILES content into KV states offline, stores them in RAM, and loads them
before generation. The model wakes up already knowing everything. There was
no prompt. There was no reading. There is no context window.

---

## 2. Three Components

### 2.1 The Compiler (`ContextCompiler`)

Takes any text, runs a model forward pass, and saves the resulting KV states
(key/value tensors from every layer) to system RAM.

- Content is processed ONCE. Cost: O(content_length), one time.
- Chunked processing for content exceeding GPU memory (default: 4096 tokens).
- Position tracking via cumulative counter for correct RoPE rotations.
- RoPE de-rotation (§3) strips position information from K states after
  extraction, producing position-neutral keys for composable storage.

KV state shape per layer: `[batch=1, n_kv_heads, seq_len, head_dim]`.
Per-token storage: ~56 KB for a 7B model (28 layers x 4 KV heads x 128 dim
x 2 bytes x 2 K/V).

### 2.2 The Composer (`ContextComposer`)

Selects relevant compiled states from RAM, assigns sequential positions,
re-rotates K states with correct RoPE, and loads the composed KV tensor to GPU.

- Order-independent composition via RoPE de-rotation (§3).
- VRAM budget check before GPU transfer (refuses if >80% of free VRAM).
- Handles mixed states: old non-de-rotated states are de-rotated on the fly
  using their original position metadata, then re-rotated at new positions.

### 2.3 The Generator (`ContextGenerator`)

Generates from the composed state. The model has no prompt to process — all
content was pre-compiled. Generation starts immediately.

- Cost: O(output_length), independent of compiled content size.
- Supports greedy and sampled decoding (temperature, top_p).
- Streaming via Python generator (yields tokens as produced).
- Position starts at `total_compiled_tokens`, continuing the sequence.

---

## 3. RoPE De-rotation — Composable KV States

RoPE (Rotary Position Embedding) encodes position into K states during the
forward pass. Without de-rotation, K states are tied to the positions they
were compiled at — composition order affects attention patterns.

### 3.1 Pipeline

```
Compilation:
  Model forward(pos=p)  ->  K_rotated = R(p) @ K_raw
  De-rotate:            ->  K_neutral = R(-p) @ K_rotated = K_raw
  Store K_neutral on CPU (position-free)

Composition:
  Assign positions:     ->  state_0 at [0, n0), state_1 at [n0, n0+n1), ...
  Re-rotate:            ->  K_final = R(new_pos) @ K_neutral
  Concatenate, move to GPU
```

### 3.2 Mathematics

Forward rotation (matches HuggingFace `apply_rotary_pos_emb`):
```
k_rotated = k * cos(theta) + rotate_half(k) * sin(theta)
```

De-rotation (inverse — negate sin):
```
k_original = k_rotated * cos(theta) - rotate_half(k_rotated) * sin(theta)
```

Proof: substitute and use `rotate_half(rotate_half(x)) = -x`:
```
k * cos^2 + rh(k)*sin*cos - rh(k)*cos*sin + k*sin^2 = k*(cos^2 + sin^2) = k
```

### 3.3 What This Enables

- **Order-independent composition**: compose(A,B) and compose(B,A) produce
  identical generation output. Verified: all orderings of 3 segments produce
  exact-match text (42/42 tests passed).
- **Dynamic subset selection**: compile 200 documents, compose any subset.
- **Hot-swap**: add/remove content without recompiling.
- **No position gaps**: removed content doesn't leave holes.

### 3.4 Precision

Round-trip (rotate -> bf16 -> derotate -> rerotate -> bf16) introduces ~0.1
per-element quantization noise in bfloat16. Generation output is identical
across all composition orderings. The model operates well within this tolerance.

### 3.5 RoPE Parameter Extraction

`_extract_rope_params(model)` extracts `inv_freq` and `attention_scaling` from
the model's rotary embedding module. Falls back to computing from config
(rope_theta, head_dim). Handles Qwen2, LLaMA, Mistral, and any HF model using
the standard `rotate_half` convention.

---

## 4. The Production Engine (`CompiledInference`)

Wraps the three core components into a production-ready inference engine.

### 4.1 Model Loading

Two modes, auto-detected based on available VRAM:

| Mode | Where weights live | When to use |
|------|-------------------|-------------|
| **GPU** | All on GPU (4-bit quantized) | Model fits in VRAM |
| **Hybrid** | Attention on GPU, FFN on CPU | Model too large for VRAM |

Hybrid mode uses `_CPUMLPWrapper`: FFN layers stay on CPU, data transfers to
CPU for FFN compute, results return to GPU. Attention stays on GPU where
parallelism matters. PCIe bandwidth is the bottleneck (~0.5ms per layer).

### 4.2 Chat Pipeline

```
User message
    |
    v
Build messages (system + history + user)
    |
    v
Apply chat template (tokenize)
    |
    v
Compose compiled states -> GPU KV cache
    |
    v
Prefill prompt ON TOP of compiled KV
    |
    v
Generate (with tool call detection)
    |
    v
If tool calls: execute tools, append results, loop
    |
    v
Return response
```

### 4.3 Tool Calling

Seven built-in tools for autonomous operation:

| Tool | Purpose |
|------|---------|
| `compile_text` | Compile arbitrary text into memory |
| `compile_file` | Read and compile a file from disk |
| `compile_directory` | Compile all matching files in a directory |
| `list_compiled` | List all compiled content |
| `list_directory` | List files in a directory |
| `grep` | Search file contents |
| `write_file` | Write content to disk and compile it |

### 4.4 Disk Persistence

- `engine.save(path)`: saves all compiled states + manifest to disk.
- `engine.load(path)`: reloads states from disk. Instant — no reprocessing.
- Manifest tracks content IDs, token counts, sizes, and `derotated` flag.
- Backward compatible: loads old non-de-rotated states transparently.

---

## 5. Five Control Mechanisms

### 5.1 Compiled KV States (Knowledge)

Compile any text into KV states. The model attends to compiled content during
generation as if it just read it. Content compiled once, queried unlimited times.

Proven: Zargthorp facts, 767-line codebase (4 files), 1.58M-token
needle-in-haystack (33ms search, 30 tok/s, 21 MB KV cache).

### 5.2 System-Role Compilation (Behavior)

Compile a system message into KV states. The model follows behavioral
directives from the compiled state.

Proven: pirate voice responses, one-word answer mode, step-by-step CoT.
Must be compiled as SYSTEM role — assistant demonstrations don't work.

### 5.3 Multi-Layer Steering (Trained Knowledge Override)

Override deeply trained factual knowledge by injecting W_lm[target_token]
vectors at 9 layers simultaneously with RMS normalization.

Proven: "The capital of France is Lyon" at alpha=0.08 across 9 layers
(Qwen 1.5B fp16). Single-layer injection at any alpha cannot override
the Paris association. Nine-layer injection breaks it cleanly.

Note: fails on 4-bit quantized models (quantization noise scrambles the
injected vector across 14 layers). Use logit bias for quantized models.

### 5.4 Logit Bias (Output Token Control)

`LogitsProcessor` adds bias (default 25.0) to target token logits with
adaptive decay. Operates at the output level — quantization-agnostic.

Proven: verbatim "Velnis, Krath, and Oppen" on Qwen 7B 4-bit at 30 tok/s.

### 5.5 Thought Injection (Reasoning Control)

Compile arbitrary text into the KV cache mid-generation. The model
"remembers" having thought those words and follows through.

Proven: injecting "Use quickselect, not sorted()" caused the model to
implement O(n) quickselect with partition. Injecting "Track nesting depth"
caused correct parenthesis parser. The model cannot distinguish injected
thoughts from its own.

---

## 6. Self-Steering Loop

### 6.1 Architecture

```
Generate -> Test -> Pass? -> Accept
                |
                v (fail)
         Select strategy (ranked by success rate)
                |
                v
         Inject as thought (compiled into KV cache)
                |
                v
         Retry -> Test -> Pass? -> Accept + update stats
                    |
                    v (fail)
              Try next strategy (up to 5 retries)
```

### 6.2 Strategy Library

18 strategies across categories: general (step_by_step, trace_examples,
simplify, verify), string, math, list, recursion. Success rates accumulate
across sessions via disk persistence.

### 6.3 Results

| Metric | Value |
|---|---|
| Baseline (Qwen 7B) | 84.1% (138/164) |
| Self-steered | **90.9%** (149/164) |
| Fixed autonomously | 11 problems |
| MVP strategy | `step_by_step` — 100% fix rate (7/7) |

### 6.4 Three-Tier Retry (Most Effective)

1. **Error injection** (simplest): show error message -> retry. Catches "oops"
   mistakes. 4 fixes on first 50 problems (98% pass rate).
2. **Strategy library** (targeted): classify error -> inject approach -> retry.
3. **Manual thought** (maximum): hand-craft specific algorithm hint -> retry.

### 6.5 What Doesn't Work

**Self-reflection**: injecting error + "What am I doing wrong? Why? How to
fix?" produces 0 fixes on both 1.5B AND 7B models. The model can't
self-diagnose. It needs DATA (the error message) or WISDOM (the strategy),
not QUESTIONS. The reflection questions are noise that dilutes the signal.

---

## 7. KV Cache Economics

### 7.1 Per-Token Cost

For Qwen 7B (28 layers, 4 KV heads, 128 head_dim):
```
Per token: 28 layers x 4 heads x 128 dim x 2 bytes x 2 (K+V) = 57,344 bytes ~ 56 KB
```

### 7.2 Compiled Context Sizes

| Content | Tokens | KV Size in RAM |
|---|---|---|
| One sentence (facts) | 69 | 4 MB |
| One source file (500 lines) | ~2,000 | 112 MB |
| 20-file project | ~40,000 | 2.2 GB |
| 1.58M token corpus | 1,580,000 | 86 GB |
| 4M token corpus | 4,000,000 | 224 GB |

### 7.3 Enforced Small Context Window

Since compiled context guarantees <200 tokens of live context, the model's
context window can be capped at 512 tokens:

| Context window | KV cache allocation (7B) |
|---|---|
| 128K (default) | 14.3 GB |
| 4K | 229 MB |
| **512 (compiled)** | **28 MB** |

A 27B model at 4-bit: 14 GB weights + 28 MB KV = **14 GB**. Fits on
a single 16 GB GPU. With the default 128K context, it needs 28+ GB.

---

## 8. Compute Breakdown

With compiled context, attention over the live KV cache is effectively free
(~50 tokens). But per-token throughput doesn't improve because attention was
never the per-token bottleneck:

| Component | Per-token cost (7B) | % of total |
|---|---|---|
| Q/K/V/O projections | 817M multiply-adds | 12.5% |
| Attention (QK^T, softmax, V) | ~0 (50 tokens) | ~0% |
| FFN (gate + up + down) | 5,700M multiply-adds | **87.5%** |

The compiled system's win is **memory** (634 GB -> 21 MB KV cache) and
**time-to-first-token** (minutes of prefill -> milliseconds). Per-token
decode speed is unchanged because FFN (87.5%) dominates.

---

## 9. llama.cpp Backend for Hybrid Attention Models

### 9.1 The Problem

Models with hybrid attention architectures (standard attention + linear
attention / Gated DeltaNet) cannot run through HuggingFace's torch fallback
on older GPUs. The native kernels (`flash-linear-attention`, `causal-conv1d`)
require Volta+ (compute 7.0+). The torch fallback produces garbage output.

### 9.2 The Solution

Use llama.cpp as the inference backend via `llama-cpp-python`. llama.cpp
implements hybrid attention in optimized C++/CUDA kernels that work on any
GPU architecture (including Maxwell, compute 5.2).

**Key technique**: Install a minimal `llama-cpp-python` (no compilation),
then point it at a pre-built `libllama.so` via `LLAMA_CPP_LIB_PATH` env var.
Load GGML backends via ctypes before importing:

```python
import ctypes, os
os.environ["LLAMA_CPP_LIB_PATH"] = "/usr/local/lib/ollama"
LIB_DIR = "/usr/local/lib/ollama"

_ggml = ctypes.CDLL(os.path.join(LIB_DIR, "libggml.so"), mode=ctypes.RTLD_GLOBAL)
_ggml.ggml_backend_load_all_from_path.argtypes = [ctypes.c_char_p]
_ggml.ggml_backend_load_all_from_path.restype = None
_ggml.ggml_backend_load_all_from_path(LIB_DIR.encode("utf-8"))
_ggml.ggml_backend_load_all_from_path((LIB_DIR + "/cuda_v12").encode("utf-8"))

from llama_cpp import Llama
```

### 9.3 KV Cache State Management

`llama-cpp-python` exposes `save_state()` / `load_state()` for full context
state serialization — this IS compiled context:

```python
llm.eval(tokens)                    # compile content
state = llm.save_state()            # save to RAM (69 MB for 123 tokens)
llm.load_state(state)               # restore (0.3s)
with open(path, "wb") as f:         # save to disk
    f.write(state.llama_state)
```

### 9.4 Proven Results (Qwen3.6-35B-A3B)

| Metric | Value |
|---|---|
| Model | 36B params, 3B active (MoE, 256 experts, 8 active) |
| Hardware | 5x Tesla M40 24GB (Maxwell, compute 5.2) |
| Generation | **33.3 tok/s** |
| Prefill | 115 tok/s |
| Compiled state | 69.2 MB, save/restore 0.3s |
| Disk persistence | Works (0.31s load from disk) |
| Fact recall | 4/4 across 4 queries from saved state |

---

## 10. Future Directions

### Burn-In (KV States -> Permanent Weights)

Compile content into KV states. Auto-generate Q&A pairs (teacher = model +
compiled states, student = model without). Train LoRA on the pairs. Merge
into base weights. The model permanently knows the content.

### Pre-Compiled State Fine-Tuning (PCS-FT)

Compile context ONCE (no_grad), train ONLY on target tokens. 65x activation
memory reduction. 100K+ context training on consumer GPUs.

### CoT Control Patterns

- **Pattern A**: model emits `[LOAD_STATE:file]` mid-thought -> system loads KV
- **Pattern B**: verbose thinking, then strip thought tokens from KV
- **Pattern E**: O(1) backtracking via tensor slice on KV cache
- **Pattern G**: model calls tool that modifies its OWN KV cache (self-agency)

### Compiled Dispositions

Compile DESIRE, not just knowledge. Internal orientations (curiosity,
creativity, persistence) as compiled KV states.

### Continuous Cognition

Non-stop thinking loop with random impulse injection from a compiled
thought library. The model thinks between prompts.
