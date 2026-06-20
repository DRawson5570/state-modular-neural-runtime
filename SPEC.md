# SPEC: State-Modular Neural Runtime

> Definitive architectural specification. Read this FIRST before any work.
> If you are an agent and you find yourself building something not described
> here, STOP and re-read this document. You have drifted.

---

## 1. The Core Principle

**The model knows what it needs to, when it needs to.**

The system has no context window. The model has working memory. Content
lives as text in system RAM — unlimited. When the model needs to reason
about specific content, it is compiled into the model's processing
pipeline through one of two paths. When the topic changes, different
content is compiled. The model now knows that instead.

This is how humans work. You have working memory (~7 items) but access
to every book ever written. Nobody says you have a "7-item context window."
They say you can think about anything.

---

## 2. The Two Paths

### Path A — Compiled KV States (Comprehension via Attention)

Content is processed through the model's full forward pass. The resulting
KV states (key/value tensors from every layer) are saved to system RAM or
disk. When restored, the model attends to them — genuine comprehension.
The model processed the content through its own neural network. The KV
states are its own internal representations.

**Mechanism:**
1. Tokenize content
2. Forward pass through all layers → KV states produced
3. RoPE de-rotation strips position information → position-neutral storage
4. Save to RAM (instant) or disk (persistent)
5. On query: re-rotate with fresh positions → load to GPU → generate

**Properties:**
- Genuine comprehension: the model processed the content itself
- Zero prefill on reload: no reprocessing, instant restoration
- Order-independent composition: de-rotated states compose in any order
- Bounded by n_ctx (context window) and GPU VRAM
- O(W) decode cost where W = active window size

**Proven results:**
- Qwen 7B 4-bit: 32.6 tok/s, all facts recalled from compiled state
- Qwen3.6-35B-A3B: 33.3 tok/s (8K), 26.7 tok/s (40K context)
- RoPE de-rotation: 42/42 order-independence tests (Qwen 0.5B)
- State save/restore: 0.3s from RAM, 0.31s from disk
- Disk persistence: save to file, reload across sessions

**Key files:**
- `compiled_context.py` — ContextCompiler, ContextComposer, ContextGenerator
- `compiled_engine.py` — Production engine with tools and streaming
- `compiled_server.py` — OpenAI-compatible API server (HuggingFace backend)

### Path B — Steerer + FFN (Reasoning via Residual Stream)

Content is compiled into structural features on CPU. The steerer injects
these features directly into the residual stream at early/mid layers.
The subsequent FFN layers — which act as key-value memory banks — READ
these features, trigger associations, and produce genuine reasoning.

**This is NOT pattern matching.** The injected features flow through
all subsequent transformer layers. Each FFN layer reads the residual
stream (which now contains the injected features), performs its key-value
lookup computation, and writes the results back. The model reasons about
the content through its own FFN computation — in latent space, without
the raw content ever appearing in the attention window.

**Mechanism:**
1. Parse content on CPU (AST, n-grams, PPMI, topic vectors)
2. Compute per-token features (21 channels, bounded [0,1])
3. Steerer MLP (22K params) projects features → d_model offset
4. RMS-normalized injection at 9 layers (2 early, 5 mid, 2 late)
5. FFN layers process the modified residual stream
6. Late attention handles local syntax routing

**Properties:**
- O(1) per token — no KV cache growth
- Constant ~21 MB active cache regardless of content size
- Attention layers never see the compiled content
- The model reasons through FFN, not attention
- Scales to unlimited content (bounded by superposition capacity)

**Two sub-mechanisms:**
1. **Steerer injection** — structural features into residual stream
   (reasoning about content structure, topics, patterns)
2. **Cartridge injection** — logit bias or W_lm vectors for specific
   fact recall (verbatim token reproduction)

**Proven results:**
- 820M steered backbone: PPL 36.9 (below oracle ceiling of 69.7)
- O(T×W) sliding window + steerer matches full O(T²) attention
- Cartridge injection: verbatim fact reproduction at alpha=400
- Code channel: AST features for code structure awareness

**Key files:**
- `conversation_channel.py` — FactInjector, ConversationRetriever
- `code_channel.py` — CodeChannelComputer (AST features)
- `superposition_steerer_v3.py` — SuperpositionSteererV3 (in compiled_priors)

### The Bridge — Triggers

Path B tells the model WHAT EXISTS. Path A lets it READ the details.
Triggers connect them.

During generation, the model can emit trigger spans. The system
intercepts these, processes the request (reads a file, searches context),
compiles the result into the model's KV cache (Path A), and generation
continues. The model navigates with awareness (Path B) and reads with
comprehension (Path A).

```
Path B (steerer features) → model knows "project has auth.py, database.py"
User asks about authentication
Model emits → [READ:src/auth.py]
System reads file → compiles into KV cache (Path A)
Model reasons → "The authenticate function validates JWTs..."
```

**Available triggers:**
| Trigger | Action | Path |
|---|---|---|
| `[READ:path]` | Read file, compile into KV cache | A |
| `[SEARCH:query]` | Search accumulated context | A |
| `[FILE:path]` | Structural summary (AST) | B |
| `[SYMBOL:name]` | Find definition across repo | B |
| `[CALC:expr]` | Math evaluation | — |
| `[QUERY:entity]` | Compiled PPMI lookup | B |
| `[UPDATE:path]` | Background reprocessing | B |

**Proven results:**
- Qwen 7B: emitted [READ:], system compiled file, model found SECRET=42
- Qwen3.6-35B: emitted [READ:], analyzed both functions, found SECRET=42
- Reasoning loop: trigger detection, result injection, generation continues

**Key files:**
- `code_tools.py` — Trigger handlers (FILE, SYMBOL, DIAG, etc.)
- `server_35b.py` — Trigger-aware OpenAI API server (llama.cpp backend)

---

## 3. Scaling Taxonomy

| Context Scale | Memory | Recommended Mechanism | Latency |
|---|---|---|---|
| **Small** (< 10K tokens) | < 560 MB | Path B: Pure steerer injection | ~0 ms |
| **Medium-Large** (10K – 500K) | 560 MB – 28 GB | Path A: KV cache snapshot loading | < 1.5s |
| **Ultra-Large** (> 500K) | > 28 GB | Hybrid: CPU index + selective KV loading | < 200ms per segment |

**Small contexts (< 10K):** The steerer compresses the content into
residual stream features. The FFN reasons about it. No KV cache needed
for the content. Attention handles only local tokens.

**Medium-large contexts (10K – 500K):** Content is compiled into KV
states and loaded from RAM. At 200K tokens on Qwen 7B, the state is
~11.2 GB — loads in ~0.93s over PCIe Gen 3. The model attends to the
full compiled state with exact, lossless access.

**Ultra-large contexts (> 500K):** A CPU semantic index locates relevant
segments. Only the relevant KV segments are loaded to GPU. The model
reasons about the loaded segments while the rest stays in RAM.

---

## 4. What We Solved

### The O(N²) Barrier — Eliminated

The quadratic complexity term is mathematically absent from the online
generation loop:

- **Prefill:** O(Q²) where Q is the short query length — effectively
  O(1) relative to compiled history. Content compiled once, never
  reprocessed.
- **Decoding:** O(W) where W is the sliding window — constant
  regardless of compiled history size. The steerer carries global context.

Standard transformers are bound by O(N²) compute complexity. This
architecture eliminated that bottleneck entirely.

### Remaining Barriers (Physics, Not Algorithms)

**Barrier 1: Superposition Capacity (Path B)**

The residual stream is d_model dimensions (e.g., 3584 for Qwen 7B).
Via superposition, more than d_model features can be packed into
almost-orthogonal directions — but interference noise grows. At extreme
scale (millions of tokens in d_model dimensions), features overlap,
causing cognitive decay. This is an information-theoretic limit of vector
space density, not an algorithmic limit.

**Barrier 2: Memory Bandwidth (Path A)**

Loading large compiled KV caches requires streaming key tensors through
the GPU memory bus for every generated token. This scales linearly:
O(N_total × d). At millions of tokens, the memory bus saturates. This
is a hardware physics limit, not an algorithmic limit.

**The victory is permanent.** The constraints are hardware physics that
improve with each hardware generation. The algorithmic victory does not
degrade.

---

## 5. What Is NOT Proven

Be explicit about what has not been demonstrated:

1. **FFN-level knowledge injection for content comprehension** — The
   concept compilation pipeline (pseudo-inverse + SVD weight deltas)
   transfers structural knowledge but not specific factual details at
   current rank. Mechanism 7 (compiled FFN rules) proves 50K
   trigger→response rules at 100% accuracy on synthetic benchmarks
   but has not been tested for natural language comprehension.

2. **Steerer on pretrained Qwen models at runtime** — The steerer was
   trained alongside the 820M DeepSeek backbone. It has not been
   demonstrated as a runtime injection mechanism on pretrained Qwen 7B
   or 35B models. The steerer requires training to learn how to
   interpret compiled features.

3. **Path B at > 100K tokens** — The superposition capacity limit has
   not been empirically measured. We do not know when cognitive decay
   begins for the steerer.

4. **Cross-file attention from compiled states** — Independently
   compiled files do not have cross-attention. The model bridges them
   through active attention during decoding, not through pre-computed
   cross-file relationships.

---

## 6. The Two Consumption Patterns

"Compiled Context" has two consumption patterns. Confusing them
causes architectural drift.

**Pattern 1: Dense KV Cache Loading (Uses Attention)**
- Serialize and restore raw K/V activations into `past_key_values`
- The model performs full attention over loaded states
- Subject to PCIe transfer speed and attention compute scaling
- This is Path A

**Pattern 2: Steered Backplanes + Cartridges (Bypasses Attention)**
- Encode structure into steerer features
- Inject facts via logit bias
- Results in constant 21 MB KV cache and O(1) execution
- This is Path B

Both are valid. They serve different purposes at different scales.
Do not confuse one for the other. Do not claim one delivers what
only the other provides.

---

## 7. File Inventory

### Path A (Comprehension)
| File | What it does |
|---|---|
| `compiled_context.py` | ContextCompiler, ContextComposer, ContextGenerator, RoPE de-rotation |
| `compiled_engine.py` | Production engine: compile, chat, stream, tools, save/load |
| `compiled_server.py` | OpenAI API server (HuggingFace backend, 7B) |
| `compiled_chat.py` | Interactive CLI |

### Path B (Reasoning)
| File | What it does |
|---|---|
| `conversation_channel.py` | FactInjector (W_lm cartridge), ConversationRetriever |
| `code_channel.py` | CodeChannelComputer (AST features for code awareness) |

### Bridge (Triggers)
| File | What it does |
|---|---|
| `code_tools.py` | Trigger handlers: [FILE:], [SYMBOL:], [DIAG:], etc. |
| `server_35b.py` | Trigger-aware OpenAI API server (llama.cpp, 35B MoE) |

### Self-Improvement
| File | What it does |
|---|---|
| `self_steering.py` | Autonomous retry with strategy library |
| `bench_humaneval_reasoning.py` | 3-pass reasoning library (97%) |
| `bench_selfsteer_full.py` | Autonomous self-steering (90.9%) |

### Tests
| File | What it does |
|---|---|
| `test_rope_derotation.py` | RoPE de-rotation suite (42 tests) |
| `test_35b_llamacpp.py` | 35B compiled context proof |
| `test_100k_pathb.py` | 100K retrieval pipeline |
| `test_full_7b.py` | Complete architecture test (7B, RTX 3080) |

---

## 8. Hardware Configurations

### Local (RTX 3080, 10 GB VRAM)
- Qwen 7B 4-bit: Path A + Bridge (Path B limited — W_lm breaks on 4-bit)
- Qwen 1.5B float16: Path A + Path B (full architecture, both paths)
- Development and testing

### pe2 (5× Tesla M40 24GB, 503 GB RAM)
- Qwen3.6-35B-A3B via llama-cpp-python: Path A + Bridge
- 33 tok/s at 8K context, 26.7 tok/s at 40K
- NVIDIA driver 580 (upgraded from 535 for Ollama compatibility)
- Production deployment

### pe3 (2× Tesla M40 12GB, 204 GB RAM)
- Training and evaluation
- NVIDIA driver 580

---

## 9. Anti-Drift Rules

If you are an agent working on this system:

1. **Read this spec FIRST.** Before writing any code, before proposing
   any architecture, before running any experiment.

2. **There are TWO paths.** Path A (attention) and Path B (FFN). They
   are complementary. Do not build one and call it the other.

3. **Path A is comprehension.** The model attends to compiled content.
   Bounded by n_ctx and VRAM. Do not claim it eliminates the context
   window — it manages it efficiently.

4. **Path B is reasoning.** The steerer injects into the residual
   stream. The FFN reasons about it. This bypasses attention. This IS
   the zero-context-window mechanism — but it requires compiled features,
   not raw content.

5. **The bridge connects them.** The model uses Path B awareness to
   decide what to read via Path A. Triggers are the mechanism. Do not
   build a system without triggers.

6. **Do not conflate the two consumption patterns.** Dense KV loading
   (Pattern 1) uses attention. Steered injection (Pattern 2) bypasses
   attention. They have different scaling properties. Do not claim
   Pattern 1 results for Pattern 2 or vice versa.

7. **Do not claim "the model knows unlimited content."** Path A is
   bounded by VRAM. Path B is bounded by superposition capacity. The
   SYSTEM has no context window. The MODEL has working memory.

8. **The O(N²) barrier is eliminated.** This is the real achievement.
   The remaining limits are hardware physics (memory bandwidth,
   superposition capacity), not algorithms. Do not overstate by claiming
   physics barriers are also eliminated.

9. **Test before claiming.** Every claim in this document is tagged as
   proven or not proven. Do not add unproven claims without tagging them.

10. **When in doubt, re-read Section 1.** "The model knows what it
    needs to, when it needs to." That is the entire system in one
    sentence. If your work doesn't serve that sentence, you have drifted.
