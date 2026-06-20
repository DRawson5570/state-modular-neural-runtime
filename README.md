# Zero Context Window

> *"Do not try to fit the prompt into the context window. That's impossible.
> Instead, only try to realize the truth... there is no context window."*

**Paper:** [Zero Token Prefill](Zero_Token_Prefill.pdf)

## The Core Idea

The transformer cannot distinguish between KV states built from live attention
and KV states loaded from pre-computed storage. Content can be compiled via a
forward pass into persistent KV states, saved to RAM or disk, and restored
instantly. The model wakes up already knowing the compiled content. Zero
prefill latency. Content processed once, queried forever.

## Two Tiers of Knowledge

This system provides two distinct capabilities. They are both real, but they
are different things and should not be conflated.

### Tier 1: Compiled Context (The Model Genuinely Knows)

Content is processed through the model's forward pass. The resulting KV states
(or recurrent state for hybrid models) are saved. When restored, the model
has the content in its active attention — it genuinely knows it, the same way
it knows a prompt it just read. No search, no retrieval. The model knows.

**Bounded by GPU VRAM.** At ~56 KB per token for a 7B model, a 10 GB GPU
holds ~70K tokens of compiled content after model weights. A 35B MoE model
compiled 123 tokens into 69 MB of state. This is the real "I know Kung Fu"
tier — limited by memory, but genuine knowledge.

| Model | Compiled Content | State Size | Speed | Hardware |
|---|---|---|---|---|
| Qwen 7B 4-bit | 69 tokens (Zargthorp facts) | 4 MB | 30 tok/s | RTX 3080 |
| Qwen3.6-35B-A3B | 123 tokens (Zargthorp facts) | 69 MB | 33 tok/s | 5x M40 24GB |

Every fact recalled perfectly. Content compiled once, queried 4 times from
saved state. State saved to disk, reloaded in 0.3s. No reprocessing.

### Tier 2: Indexed Corpus (The System Can Access)

For content beyond what fits in active compiled state, the corpus is stored as
text in RAM or on disk. A CPU-based search (text match, semantic index, or
embedding search) locates relevant content in milliseconds. Only the relevant
chunk is compiled into the model's active state. The model then knows that
chunk genuinely (Tier 1), while the rest of the corpus remains as searchable
text.

**This is retrieval-augmented compilation, not "the model knows everything."**
The model knows what's compiled into its state. The system can access anything
in the indexed corpus. These are different capabilities.

| Corpus Size | Search Time (CPU) | Compile Time (GPU) | Generation | Active KV |
|---|---|---|---|---|
| 100K tokens (625 KB) | 5 ms | 1,038 ms (~110 tokens) | 32 tok/s | 69 MB constant |

5/5 needles found in a 100K-token haystack. But the model didn't "know" 100K
tokens — it was told the relevant paragraph each time. The system accessed
100K tokens. The model knew ~110 tokens.

### What the 1.58M Token Test Actually Proved

The earlier 1.58M-token needle-in-haystack test demonstrated **Tier 2** — system-level
access, not model knowledge. A CPU text search found the needle in 33ms. The
relevant chunk was compiled into 21 MB of KV states. The model answered from
that chunk at 30 tok/s. The "30,000x KV reduction" compared a traditional
system where the model attends to ALL 1.58M tokens (634 GB KV) with our
system where the model attends to only the retrieved chunk (21 MB KV). The
model never held 1.58M tokens in its state.

**What it proved:** the system can access arbitrarily large corpora with
constant active KV cache and constant generation speed. The search scales
linearly on CPU. The compilation and generation are constant. This is real
and useful — but it is retrieval + compilation, not omniscient knowledge.

### HumanEval (7B model, zero training)

| Method | Score |
|---|---|
| Baseline | 79.9% |
| + Thought injection (reasoning library) | **97.0%** |
| + Self-steering loop (autonomous) | **90.9%** |

## Quick Start

```bash
pip install -r requirements.txt

# Interactive chat with compiled context
python compiled_chat.py --model Qwen/Qwen2.5-7B-Instruct

# Inside the chat:
#   /compile_file src/main.py      — compile a file into the model's brain
#   /compile_file src/utils.py     — compile another
#   /compiled                      — see what's compiled
#   "Explain the main function"    — ask questions about compiled content
```

### OpenAI-Compatible API Server

```bash
python compiled_server.py --model Qwen/Qwen2.5-7B-Instruct --port 8000

# Compile content
curl -X POST http://localhost:8000/v1/compile \
  -H "Content-Type: application/json" \
  -d '{"id": "facts", "text": "The planet Zargthorp has 3 moons."}'

# Chat (standard OpenAI format)
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"messages": [{"role": "user", "content": "What are the moons of Zargthorp?"}]}'

# Streaming
curl http://localhost:8000/v1/chat/completions \
  -d '{"messages": [{"role": "user", "content": "Hello"}], "stream": true}'
```

### Programmatic Usage

```python
from compiled_engine import CompiledInference

engine = CompiledInference("Qwen/Qwen2.5-7B-Instruct")

# Compile content — model processes it once, stores KV states in RAM
engine.compile("codebase", open("src/main.py").read())
engine.compile("docs", open("README.md").read())

# Chat — model attends to all compiled content
response = engine.chat("What does the main function do?")
print(response)

# Save/load compiled states to disk
engine.save("./compiled_states/")
engine.load("./compiled_states/")  # instant reload, no reprocessing
```

## Architecture

Three components:

1. **Compiler** (`ContextCompiler`) — forward pass on content, saves KV states to CPU RAM.
   RoPE de-rotation stores position-neutral keys for order-independent composition.

2. **Composer** (`ContextComposer`) — selects compiled states, assigns sequential positions,
   re-rotates keys, loads to GPU. Compose any subset in any order.

3. **Generator** (`ContextGenerator`) — generates from composed state. The model has no
   prompt to process. Generation starts immediately.

### Five Control Mechanisms

| # | Mechanism | Target | What it does |
|---|---|---|---|
| 1 | Compiled KV states | Knowledge | Compile text/code/docs into the model's active state |
| 2 | System-role compilation | Behavior | Compile behavioral directives (pirate voice, terse, CoT) |
| 3 | Multi-layer steering | Trained knowledge | Override deeply trained facts (Paris -> Lyon) |
| 4 | Logit bias | Output tokens | Verbatim reproduction on quantized models |
| 5 | Thought injection | Reasoning | Inject algorithm hints mid-generation |

### Self-Steering Loop

Autonomous retry pipeline: generate -> test -> on failure, select strategy
from ranked library -> inject as thought -> retry. 18 strategies, success
rates tracked across sessions.

```python
from self_steering import SelfSteeringLoop

loop = SelfSteeringLoop(model, tokenizer)
result = loop.generate_with_retry(prompt, validator=run_tests)
```

## Files

| File | Purpose |
|------|---------|
| `compiled_context.py` | Core: Compiler, Composer, Generator, RoPE de-rotation |
| `compiled_engine.py` | Production engine with tools, streaming, persistence |
| `compiled_server.py` | OpenAI-compatible API server |
| `compiled_chat.py` | Interactive CLI |
| `self_steering.py` | Autonomous strategy selection and retry |
| `test_rope_derotation.py` | RoPE de-rotation test suite (42 tests) |
| `test_35b_llamacpp.py` | 35B MoE compiled context via llama.cpp (33 tok/s) |
| `test_100k_pathb.py` | 100K corpus retrieval + compilation (Path B) |
| `bench_humaneval*.py` | HumanEval benchmark scripts |
| `docs/SPEC_COMPILED_CONTEXT.md` | Full specification (20 sections) |
| `docs/ARCHITECTURE.md` | Technical architecture |

## Requirements

- Python 3.10+
- PyTorch 2.0+ (for HuggingFace backend)
- Any HuggingFace causal language model (Qwen, LLaMA, Mistral, etc.)
- For hybrid attention models (Qwen3.5/3.6): `llama-cpp-python` + GGUF model
- GPU with enough VRAM for the model weights (KV cache is negligible)
- System RAM for compiled states (~56 KB per token for 7B)

## What This Is and What It Isn't

**What it is:** A system that eliminates prefill latency by pre-compiling
content into reusable KV states. Combined with CPU-based corpus indexing,
it provides instant access to arbitrarily large knowledge bases at constant
generation speed. The compiled states are genuine model knowledge — the
model attends to them identically to live input.

**What it isn't:** A way to make a model "know" unlimited content
simultaneously. The model's active knowledge is bounded by what fits in
its compiled state (GPU VRAM for pure transformers, recurrent state
capacity for hybrid models). Content beyond that boundary requires
retrieval before compilation.
