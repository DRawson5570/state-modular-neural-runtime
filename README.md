# State-Modular Neural Runtime

The model knows what it needs to, when it needs to.

Content lives as text in system RAM — unlimited. When a question arrives,
the relevant content is found (milliseconds), compiled into the model's
KV cache (the model processes it through its own forward pass), and the
model genuinely reasons about it at full speed. When the topic changes,
different content is compiled. The model now knows that instead.

The system has no context window. The model has working memory. Just like
you have working memory but access to every book ever written. Nobody says
you have a "7-item context window." They say you can think about anything.

## What This Is

A **zero-prefill architecture** that bypasses three costs of standard inference:

1. **Bypassed: Dynamic prefill** — The O(N²) forward pass that generates
   initial K/V matrices is executed once, the activations are extracted,
   RoPE coordinates are stripped (de-rotation), and the result is written
   to RAM. Never recomputed.

2. **Bypassed: Absolute RoPE positioning** — De-rotated keys can be
   recomposed in any order and re-sequentialized on the fly without
   another forward pass. 42/42 composition tests pass.

3. **Bypassed: Active KV cache growth** — A sliding window enforces
   constant active cache size during generation. New tokens don't grow
   the cache quadratically.

## What This Is NOT

The attention step is still physically active. During decoding, the query
vector of each generated token calculates dot-product similarity against
every key in the composed `past_key_values` tensor. The context window is
populated via fast RAM transfer instead of dynamic forward pass, but it is
still present and queried by the attention heads.

**The model's active knowledge is bounded by GPU VRAM.** At ~56 KB/token
for a 7B model, compiled state scales linearly. This system shifted the
memory bottleneck from expensive GPU compute to passive RAM/disk storage.
It did not eliminate the storage footprint.

**Independently compiled states lack cross-attention.** File A's KV states
don't reflect awareness of File B. The model bridges them through its
active attention during decoding, not through pre-computed cross-file
relationships.

## Proven Results

| What | Result | Hardware |
|---|---|---|
| Compiled state recall | 4/4 facts across 4 queries from saved state | RTX 3080, 5x M40 |
| Prefill elimination | 0ms from saved state (vs minutes for 1M+ tokens) | RTX 3080 |
| RoPE de-rotation | Order-independent composition, 42/42 tests | RTX 3080 |
| 35B MoE generation | 33.3 tok/s with state save/restore | 5x M40 24GB |
| Retrieval + compile | 100K corpus, 5ms CPU search, 32 tok/s generation | 5x M40 24GB |
| HumanEval (thought injection) | 97.0% (7B model, zero training) | RTX 3080 |
| HumanEval (self-steering) | 90.9% autonomous | RTX 3080 |

### The 1.58M Token Test

This was retrieval + compilation, not the model holding 1.58M tokens in
state. A CPU text search found the needle in 33ms. The relevant chunk was
compiled into 21 MB of active KV. The model answered from that chunk at
30 tok/s. The system accessed 1.58M tokens. The model knew ~200 tokens.

## Quick Start

```bash
pip install -r requirements.txt

python compiled_chat.py --model Qwen/Qwen2.5-7B-Instruct

# Commands:
#   /compile_file src/main.py
#   /compiled
#   "Explain the main function"
```

### API Server

```bash
python compiled_server.py --model Qwen/Qwen2.5-7B-Instruct --port 8000

curl -X POST http://localhost:8000/v1/compile \
  -d '{"id": "facts", "text": "The planet Zargthorp has 3 moons."}'

curl http://localhost:8000/v1/chat/completions \
  -d '{"messages": [{"role": "user", "content": "What are the moons of Zargthorp?"}]}'
```

### Programmatic

```python
from compiled_engine import CompiledInference

engine = CompiledInference("Qwen/Qwen2.5-7B-Instruct")
engine.compile("codebase", open("src/main.py").read())
response = engine.chat("What does the main function do?")
engine.save("./compiled_states/")
engine.load("./compiled_states/")
```

## Architecture

1. **Compiler** (`ContextCompiler`) — Forward pass on content, saves KV
   states to CPU RAM. RoPE de-rotation stores position-neutral keys.

2. **Composer** (`ContextComposer`) — Selects compiled states, assigns
   sequential positions, re-rotates keys, loads to GPU.

3. **Generator** (`ContextGenerator`) — Generates from composed state.
   Zero prefill latency.

### Control Mechanisms

| # | Mechanism | What it does |
|---|---|---|
| 1 | Compiled KV states | Pre-compute content, reload instantly |
| 2 | System-role compilation | Compile behavioral directives into KV |
| 3 | Multi-layer steering | Override trained facts via W_lm at 9 layers |
| 4 | Logit bias | Verbatim reproduction on quantized models |
| 5 | Thought injection | Inject reasoning hints mid-generation |

### Self-Steering Loop

Generate → test → on failure, select strategy from ranked library →
inject as thought → retry. 18 strategies, success rates tracked across
sessions.

## Files

| File | Purpose |
|------|---------|
| `compiled_context.py` | Compiler, Composer, Generator, RoPE de-rotation |
| `compiled_engine.py` | Production engine with tools, streaming, persistence |
| `compiled_server.py` | OpenAI-compatible API server |
| `compiled_chat.py` | Interactive CLI |
| `self_steering.py` | Autonomous strategy selection and retry |
| `test_rope_derotation.py` | RoPE de-rotation test suite (42 tests) |
| `test_35b_llamacpp.py` | 35B MoE via llama.cpp (33 tok/s) |
| `test_100k_pathb.py` | 100K corpus retrieval + compilation |
| `bench_humaneval*.py` | HumanEval benchmark scripts |
| `docs/ARCHITECTURE.md` | Technical architecture |

## Requirements

- Python 3.10+
- PyTorch 2.0+
- Any HuggingFace causal language model
- For hybrid attention models: `llama-cpp-python` + GGUF
- GPU with VRAM for model weights + active compiled states
- System RAM for state storage (~56 KB/token for 7B)
