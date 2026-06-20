# Zero Context Window

> *"Do not try to fit the prompt into the context window. That's impossible.
> Instead, only try to realize the truth... there is no context window."*

The transformer cannot distinguish between KV states built from live attention
and KV states loaded from pre-computed storage. Content is compiled via a
forward pass into persistent KV states. The model wakes up already knowing
everything. There is no prompt. There is no context window.

## What This Does

Compile any content (code, documents, conversation, system prompts) into
persistent KV states stored in system RAM. Compose any subset in any order
at query time. Generate with the model "already knowing" everything it was
compiled on. Zero prefill. Constant KV cache. Unlimited compiled context.

### Real Numbers (Qwen 7B 4-bit, RTX 3080)

| Context | KV Cache | Traditional KV | Reduction | Speed |
|---|---|---|---|---|
| 132K tokens | 21 MB | 53 GB | 2,500x | 31 tok/s |
| 708K tokens | 21 MB | 284 GB | 13,500x | 31 tok/s |
| **1.58M tokens** | **21 MB** | **634 GB** | **30,000x** | **30 tok/s** |

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
| 1 | Compiled KV states | Knowledge | Compile text/code/docs into the model's brain |
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
| `bench_humaneval*.py` | HumanEval benchmark scripts |
| `docs/SPEC_COMPILED_CONTEXT.md` | Full specification (20 sections) |
| `docs/ARCHITECTURE.md` | Technical architecture |

## Requirements

- Python 3.10+
- PyTorch 2.0+
- Any HuggingFace causal language model (Qwen, LLaMA, Mistral, etc.)
- GPU with enough VRAM for the model weights (KV cache is negligible)
- System RAM for compiled states (~56 KB per token for 7B)

## Key Insight

The GPU becomes an attention ASIC. Compiled states live in system RAM (cheap,
abundant). The live KV cache stays tiny (~21 MB constant). The context window
is an illusion created by the assumption that KV states must be built from
live attention. They don't. Compile once, query forever.
