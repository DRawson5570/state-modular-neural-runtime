import torch
import sys
import time

from compiled_context import (
    _rotate_half, _apply_rope, _extract_rope_params,
    ContextCompiler, ContextComposer, ContextGenerator, CompiledState,
)

PASS = 0
FAIL = 0

def check(name, condition, detail=""):
    global PASS, FAIL
    if condition:
        PASS += 1
        print(f"  PASS: {name}")
    else:
        FAIL += 1
        print(f"  FAIL: {name} {detail}")


def test_rotate_half():
    print("\n=== Test: _rotate_half ===")
    x = torch.randn(1, 4, 10, 128)
    rh = _rotate_half(x)
    check("output shape preserved", rh.shape == x.shape)
    rhrh = _rotate_half(rh)
    check("rotate_half(rotate_half(x)) == -x",
          torch.allclose(rhrh, -x, atol=1e-7),
          f"max diff: {(rhrh + x).abs().max().item():.2e}")


def test_apply_rope_roundtrip():
    print("\n=== Test: _apply_rope round-trip ===")
    head_dim = 128
    inv_freq = 1.0 / (10000.0 ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))
    k = torch.randn(1, 4, 20, head_dim)
    positions = torch.arange(0, 20)

    k_rotated = _apply_rope(k, positions, inv_freq, inverse=False)
    k_back = _apply_rope(k_rotated, positions, inv_freq, inverse=True)
    check("derotate(rotate(K)) ≈ K",
          torch.allclose(k_back, k, atol=1e-5),
          f"max diff: {(k_back - k).abs().max().item():.2e}")

    k_rotated2 = _apply_rope(k_back, positions, inv_freq, inverse=False)
    check("rotate(derotate(K_rot)) ≈ K_rot",
          torch.allclose(k_rotated2, k_rotated, atol=1e-5),
          f"max diff: {(k_rotated2 - k_rotated).abs().max().item():.2e}")


def test_apply_rope_matches_hf():
    print("\n=== Test: _apply_rope matches HuggingFace ===")
    from transformers.models.qwen2.modeling_qwen2 import rotate_half, apply_rotary_pos_emb

    head_dim = 128
    base = 1000000.0
    inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))
    positions = torch.arange(0, 15).unsqueeze(0)

    inv_freq_exp = inv_freq[None, :, None].float().expand(1, -1, 1)
    pos_exp = positions[:, None, :].float()
    freqs = (inv_freq_exp @ pos_exp).transpose(1, 2)
    emb = torch.cat((freqs, freqs), dim=-1)
    cos_hf = emb.cos()
    sin_hf = emb.sin()

    q = torch.randn(1, 28, 15, head_dim)
    k = torch.randn(1, 4, 15, head_dim)
    _, k_hf = apply_rotary_pos_emb(q, k, cos_hf, sin_hf, unsqueeze_dim=1)

    k_ours = _apply_rope(k, positions.squeeze(0), inv_freq, inverse=False)
    check("forward rotation matches HF",
          torch.allclose(k_ours, k_hf, atol=1e-5),
          f"max diff: {(k_ours - k_hf).abs().max().item():.2e}")

    k_derot = _apply_rope(k_hf, positions.squeeze(0), inv_freq, inverse=True)
    check("de-rotation recovers original K",
          torch.allclose(k_derot, k, atol=1e-5),
          f"max diff: {(k_derot - k).abs().max().item():.2e}")


def test_apply_rope_different_positions():
    print("\n=== Test: _apply_rope position reassignment ===")
    head_dim = 128
    inv_freq = 1.0 / (10000.0 ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))
    k = torch.randn(1, 4, 10, head_dim)
    pos_a = torch.arange(0, 10)
    pos_b = torch.arange(100, 110)

    k_at_a = _apply_rope(k, pos_a, inv_freq, inverse=False)
    k_at_b = _apply_rope(k, pos_b, inv_freq, inverse=False)
    check("different positions produce different K",
          not torch.allclose(k_at_a, k_at_b, atol=1e-3))

    k_derot = _apply_rope(k_at_a, pos_a, inv_freq, inverse=True)
    k_rereot = _apply_rope(k_derot, pos_b, inv_freq, inverse=False)
    check("derotate(pos_a) then rotate(pos_b) ≈ direct rotate(pos_b)",
          torch.allclose(k_rereot, k_at_b, atol=1e-5),
          f"max diff: {(k_rereot - k_at_b).abs().max().item():.2e}")


def test_apply_rope_attention_scaling():
    print("\n=== Test: _apply_rope with attention_scaling ===")
    head_dim = 128
    inv_freq = 1.0 / (10000.0 ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))
    k = torch.randn(1, 4, 10, head_dim)
    positions = torch.arange(0, 10)

    k_rot = _apply_rope(k, positions, inv_freq, inverse=False, attention_scaling=0.7)
    k_back = _apply_rope(k_rot, positions, inv_freq, inverse=True, attention_scaling=0.7)
    check("round-trip with attention_scaling=0.7",
          torch.allclose(k_back, k, atol=1e-4),
          f"max diff: {(k_back - k).abs().max().item():.2e}")


def test_apply_rope_partial_rotation():
    print("\n=== Test: _apply_rope partial rotation ===")
    head_dim = 128
    rotary_dim = 96
    inv_freq = 1.0 / (10000.0 ** (torch.arange(0, rotary_dim, 2, dtype=torch.float32) / rotary_dim))
    k = torch.randn(1, 4, 10, head_dim)
    positions = torch.arange(0, 10)

    k_rot = _apply_rope(k, positions, inv_freq, inverse=False)
    check("passthrough dims unchanged",
          torch.allclose(k_rot[..., rotary_dim:], k[..., rotary_dim:].float(), atol=1e-7))
    check("rotary dims changed",
          not torch.allclose(k_rot[..., :rotary_dim], k[..., :rotary_dim].float(), atol=1e-3))

    k_back = _apply_rope(k_rot, positions, inv_freq, inverse=True)
    check("round-trip preserves all dims",
          torch.allclose(k_back, k, atol=1e-5),
          f"max diff: {(k_back - k).abs().max().item():.2e}")


def test_apply_rope_empty():
    print("\n=== Test: _apply_rope empty tensor ===")
    head_dim = 128
    inv_freq = 1.0 / (10000.0 ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))
    k = torch.randn(1, 4, 0, head_dim)
    positions = torch.arange(0, 0)
    k_rot = _apply_rope(k, positions, inv_freq, inverse=False)
    check("empty tensor produces empty output", k_rot.shape == (1, 4, 0, head_dim))


def test_apply_rope_dtype_preservation():
    print("\n=== Test: _apply_rope dtype preservation ===")
    head_dim = 128
    inv_freq = 1.0 / (10000.0 ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))
    positions = torch.arange(0, 10)

    for dtype in [torch.float32, torch.float16, torch.bfloat16]:
        k = torch.randn(1, 4, 10, head_dim, dtype=dtype)
        k_rot = _apply_rope(k, positions, inv_freq, inverse=False)
        check(f"dtype {dtype} preserved", k_rot.dtype == dtype)


def test_model_compile_derotation():
    print("\n=== Test: Model compile + de-rotation ===")
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_name = "Qwen/Qwen2.5-0.5B-Instruct"
    print(f"  Loading {model_name}...")
    tok = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.bfloat16, device_map="cuda:0",
        trust_remote_code=True,
    )
    model.eval()

    compiler = ContextCompiler(model, tok, device="cuda:0")
    composer = ContextComposer("cuda:0", inv_freq=compiler.inv_freq,
                               attention_scaling=compiler.attention_scaling)
    generator = ContextGenerator(model, tok, device="cuda:0")

    check("inv_freq extracted", compiler.inv_freq is not None)
    check("inv_freq shape", compiler.inv_freq.shape[0] > 0,
          f"shape: {compiler.inv_freq.shape}")

    text_a = "The planet Zargthorp has exactly three moons named Velnis, Krath, and Oppen."
    text_b = "The capital of France is Paris. The Eiffel Tower is located in Paris."
    text_c = "Python is a programming language created by Guido van Rossum in 1991."

    print("  Compiling three segments...")
    state_a = compiler.compile("fact_a", text_a)
    state_b = compiler.compile("fact_b", text_b)
    state_c = compiler.compile("fact_c", text_c)

    check("state_a derotated", state_a.derotated)
    check("state_b derotated", state_b.derotated)
    check("state_c derotated", state_c.derotated)

    print("  Testing compose(A, B, C)...")
    kv_abc = composer.compose(state_a, state_b, state_c)
    total_tokens = state_a.n_tokens + state_b.n_tokens + state_c.n_tokens
    check("composed seq_len correct",
          kv_abc[0][0].shape[2] == total_tokens,
          f"expected {total_tokens}, got {kv_abc[0][0].shape[2]}")

    out_abc, n_abc, tps = generator.generate(kv_abc, max_new_tokens=50)
    print(f"  ABC output ({tps:.0f} tok/s): {out_abc[:100]}")
    check("ABC generates text", len(out_abc) > 0)

    return model, tok, compiler, composer, generator, state_a, state_b, state_c


def test_order_independence(model, tok, compiler, composer, generator, state_a, state_b, state_c):
    print("\n=== Test: Order independence ===")
    kv_abc = composer.compose(state_a, state_b, state_c)
    kv_bca = composer.compose(state_b, state_c, state_a)
    kv_cab = composer.compose(state_c, state_a, state_b)

    total = state_a.n_tokens + state_b.n_tokens + state_c.n_tokens
    check("ABC seq_len", kv_abc[0][0].shape[2] == total)
    check("BCA seq_len", kv_bca[0][0].shape[2] == total)
    check("CAB seq_len", kv_cab[0][0].shape[2] == total)

    question = "What are the moons of Zargthorp?"
    from transformers import DynamicCache

    def generate_with_question(kv, q):
        msgs = [
            {"role": "system", "content": "Answer concisely."},
            {"role": "user", "content": q},
        ]
        prompt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        input_ids = tok(prompt, return_tensors="pt").input_ids.to("cuda:0")

        cache = DynamicCache()
        for layer_idx, (k, v) in enumerate(kv):
            cache.update(k, v, layer_idx)

        content_len = kv[0][0].shape[2]
        pos = torch.arange(content_len, content_len + input_ids.shape[1], device="cuda:0").unsqueeze(0)

        with torch.no_grad():
            out = model(input_ids=input_ids, past_key_values=cache, position_ids=pos, use_cache=True)

        generated = []
        gen_pos = content_len + input_ids.shape[1]
        cache = out.past_key_values
        next_id = out.logits[0, -1, :].argmax().item()
        for _ in range(50):
            if next_id == tok.eos_token_id:
                break
            try:
                im_end = tok("<|im_end|>", add_special_tokens=False)["input_ids"]
                if im_end and next_id == im_end[0]:
                    break
            except Exception:
                pass
            generated.append(next_id)
            inp = torch.tensor([[next_id]], device="cuda:0")
            p = torch.tensor([[gen_pos]], device="cuda:0")
            with torch.no_grad():
                out = model(input_ids=inp, past_key_values=cache, position_ids=p, use_cache=True)
            cache = out.past_key_values
            gen_pos += 1
            next_id = out.logits[0, -1, :].argmax().item()
        return tok.decode(generated, skip_special_tokens=True)

    out_abc = generate_with_question(kv_abc, question)
    out_bca = generate_with_question(kv_bca, question)
    out_cab = generate_with_question(kv_cab, question)

    print(f"  ABC: {out_abc[:120]}")
    print(f"  BCA: {out_bca[:120]}")
    print(f"  CAB: {out_cab[:120]}")

    has_moons_abc = any(m in out_abc.lower() for m in ["velnis", "krath", "oppen"])
    has_moons_bca = any(m in out_bca.lower() for m in ["velnis", "krath", "oppen"])
    has_moons_cab = any(m in out_cab.lower() for m in ["velnis", "krath", "oppen"])

    check("ABC mentions moons", has_moons_abc, f"output: {out_abc[:80]}")
    check("BCA mentions moons", has_moons_bca, f"output: {out_bca[:80]}")
    check("CAB mentions moons", has_moons_cab, f"output: {out_cab[:80]}")

    check("ABC == BCA (exact match)", out_abc == out_bca,
          f"\n    ABC: {out_abc[:80]}\n    BCA: {out_bca[:80]}")
    check("ABC == CAB (exact match)", out_abc == out_cab,
          f"\n    ABC: {out_abc[:80]}\n    CAB: {out_cab[:80]}")


def test_kv_state_equivalence(model, tok, compiler, composer, generator, state_a, state_b, state_c):
    print("\n=== Test: KV state position equivalence ===")
    kv_ab = composer.compose(state_a, state_b)
    kv_ba = composer.compose(state_b, state_a)

    n_a = state_a.n_tokens
    n_b = state_b.n_tokens

    k_ab_a = kv_ab[0][0][:, :, :n_a, :]
    k_ba_a = kv_ba[0][0][:, :, n_b:, :]
    v_ab_a = kv_ab[0][1][:, :, :n_a, :]
    v_ba_a = kv_ba[0][1][:, :, n_b:, :]

    check("V states of A identical regardless of order",
          torch.allclose(v_ab_a, v_ba_a, atol=1e-5),
          f"max diff: {(v_ab_a - v_ba_a).abs().max().item():.2e}")

    k_derot_ab = _apply_rope(
        k_ab_a.cpu(), torch.arange(0, n_a),
        compiler.inv_freq, inverse=True, attention_scaling=compiler.attention_scaling,
    )
    k_derot_ba = _apply_rope(
        k_ba_a.cpu(), torch.arange(n_b, n_b + n_a),
        compiler.inv_freq, inverse=True, attention_scaling=compiler.attention_scaling,
    )
    check("K states of A equivalent after de-rotation (bf16 tolerance)",
          torch.allclose(k_derot_ab, k_derot_ba, atol=0.2),
          f"max diff: {(k_derot_ab - k_derot_ba).abs().max().item():.2e}")


def test_save_load_derotated(model, tok, compiler, composer, generator, state_a, state_b, state_c):
    print("\n=== Test: Save/Load with derotated flag ===")
    import tempfile, shutil
    tmpdir = tempfile.mkdtemp(prefix="derot_test_")
    try:
        n = compiler.save_to_disk(tmpdir)
        check("saved states", n == 3, f"expected 3, got {n}")

        compiler2 = ContextCompiler(model, tok, device="cuda:0")
        loaded = compiler2.load_from_disk(tmpdir)
        check("loaded states", loaded == 3, f"expected 3, got {loaded}")

        for cid, state in compiler2.library.items():
            check(f"  {cid} derotated flag preserved", state.derotated)

        loaded_a = compiler2.library.get("fact_a")
        if loaded_a:
            orig_k = state_a.kv_states[0][0]
            load_k = loaded_a.kv_states[0][0]
            check("loaded K states match original",
                  torch.allclose(orig_k, load_k, atol=1e-6),
                  f"max diff: {(orig_k - load_k).abs().max().item():.2e}")

        kv = composer.compose(*compiler2.library.values())
        out, n_tok, tps = generator.generate(kv, max_new_tokens=30)
        check("loaded states generate text", len(out) > 0, f"output: {out[:80]}")
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def test_single_state_derotated(model, tok, compiler, composer, generator, state_a, *_):
    print("\n=== Test: Single derotated state ===")
    kv = composer.compose(state_a)
    check("single state seq_len", kv[0][0].shape[2] == state_a.n_tokens)
    out, n_tok, tps = generator.generate(kv, max_new_tokens=30)
    check("single state generates text", len(out) > 0, f"output: {out[:80]}")


if __name__ == "__main__":
    print("=" * 60)
    print("RoPE De-rotation Test Suite")
    print("=" * 60)

    test_rotate_half()
    test_apply_rope_roundtrip()
    test_apply_rope_matches_hf()
    test_apply_rope_different_positions()
    test_apply_rope_attention_scaling()
    test_apply_rope_partial_rotation()
    test_apply_rope_empty()
    test_apply_rope_dtype_preservation()

    print("\n--- Model-based tests (requires GPU + Qwen 0.5B) ---")
    try:
        result = test_model_compile_derotation()
        model, tok, compiler, composer, generator, sa, sb, sc = result
        test_order_independence(model, tok, compiler, composer, generator, sa, sb, sc)
        test_kv_state_equivalence(model, tok, compiler, composer, generator, sa, sb, sc)
        test_save_load_derotated(model, tok, compiler, composer, generator, sa, sb, sc)
        test_single_state_derotated(model, tok, compiler, composer, generator, sa, sb, sc)
    except Exception as e:
        print(f"  SKIP model tests: {e}")
        import traceback
        traceback.print_exc()

    print("\n" + "=" * 60)
    print(f"Results: {PASS} passed, {FAIL} failed")
    print("=" * 60)
    sys.exit(1 if FAIL > 0 else 0)
