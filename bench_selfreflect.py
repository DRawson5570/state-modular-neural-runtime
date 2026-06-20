#!/usr/bin/env python3
"""Self-reflection experiment: inject the ERROR + REFLECTION QUESTIONS
and let the model generate its OWN fix strategy.

Instead of: system picks strategy → injects thought
This does:  system injects error + "What am I doing wrong?" → model reflects → fixes itself
"""
import sys, os, re, subprocess, tempfile, time, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from datasets import load_dataset

INST = (
    "Complete the following Python function. "
    "Return ONLY the complete function inside a ```python code block.\n\n"
    "```python\n{prompt}\n```"
)

REFLECTION_TEMPLATE = (
    "Wait. My first attempt failed with this error:\n"
    "```\n{error}\n```\n\n"
    "Let me reflect:\n"
    "1. What am I doing wrong?\n"
    "2. Why is this happening?\n"
    "3. How should I fix it?\n\n"
    "Let me think step by step and write a corrected version.\n\n"
)


def extract_code(text):
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    m = re.search(r"```(?:python|py)?\s*\n(.*?)```", text, re.DOTALL)
    return m.group(1) if m else text


def run_test(prog, test_code, entry):
    full = f"{prog}\n\n{test_code}\n\ncheck({entry})\n"
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False, encoding="utf-8") as f:
        f.write(full)
        tmp = f.name
    try:
        r = subprocess.run(["python3", tmp], capture_output=True, text=True, timeout=10)
        return r.returncode == 0, r.stderr[:200] if r.returncode != 0 else ""
    except subprocess.TimeoutExpired:
        return False, "TIMEOUT: possible infinite loop"
    except Exception as e:
        return False, str(e)[:100]
    finally:
        try:
            os.unlink(tmp)
        except:
            pass


def gen_with_thought(model, tok, device, prompt, thought=None, max_new=512):
    ids = tok(prompt, return_tensors="pt").input_ids.to(device)
    with torch.no_grad():
        out = model(ids, use_cache=True)
    cache = out.past_key_values
    gp = ids.shape[1]
    if thought:
        t_ids = tok(thought, add_special_tokens=False, return_tensors="pt").input_ids.to(device)
        t_pos = torch.arange(gp, gp + t_ids.shape[1], device=device).unsqueeze(0)
        with torch.no_grad():
            out = model(t_ids, past_key_values=cache, position_ids=t_pos, use_cache=True)
        cache = out.past_key_values
        gp += t_ids.shape[1]
    eos = set()
    if tok.eos_token_id is not None:
        eos.add(tok.eos_token_id)
    try:
        eos.update(tok("<|im_end|>", add_special_tokens=False)["input_ids"])
    except Exception:
        pass
    generated = list(tok.encode(thought, add_special_tokens=False)) if thought else []
    nid = out.logits[0, -1, :].argmax().item()
    if nid in eos:
        return tok.decode(generated, skip_special_tokens=True)
    generated.append(nid)
    with torch.no_grad():
        for _ in range(max_new):
            o = model(
                torch.tensor([[nid]], device=device),
                past_key_values=cache,
                position_ids=torch.tensor([[gp]], device=device),
                use_cache=True,
            )
            cache = o.past_key_values
            gp += 1
            nid = o.logits[0, -1, :].argmax().item()
            if nid in eos:
                break
            generated.append(nid)
    return tok.decode(generated, skip_special_tokens=True)


def main():
    print("=" * 60)
    print("  SELF-REFLECTION EXPERIMENT")
    print("  Inject error + 'What am I doing wrong?' → model fixes itself")
    print("=" * 60)

    tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct")
    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2.5-7B-Instruct",
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16
        ),
        device_map="cuda:0",
    ).eval()
    device = "cuda:0"

    he = load_dataset("openai/openai_humaneval", split="test")
    problems = list(he)

    baseline_pass = 0
    reflect_pass = 0
    fixed_by_reflect = 0
    results = []
    t0 = time.time()

    for i, ex in enumerate(problems):
        prompt = tok.apply_chat_template(
            [{"role": "user", "content": INST.format(prompt=ex["prompt"])}],
            tokenize=False,
            add_generation_prompt=True,
        )

        resp = gen_with_thought(model, tok, device, prompt)
        code = extract_code(resp)
        prog = code if f"def {ex['entry_point']}" in code else f"{ex['prompt']}{code}"
        ok_base, err = run_test(prog, ex["test"], ex["entry_point"])

        if ok_base:
            baseline_pass += 1
            reflect_pass += 1
            results.append({"id": ex["task_id"], "baseline": True, "reflect": True})
        else:
            reflection = REFLECTION_TEMPLATE.format(error=err[:150])
            resp2 = gen_with_thought(model, tok, device, prompt, thought=reflection)
            code2 = extract_code(resp2)
            prog2 = code2 if f"def {ex['entry_point']}" in code2 else f"{ex['prompt']}{code2}"
            ok_ref, err2 = run_test(prog2, ex["test"], ex["entry_point"])

            if ok_ref:
                reflect_pass += 1
                fixed_by_reflect += 1
                results.append({"id": ex["task_id"], "baseline": False, "reflect": True})
            else:
                results.append({"id": ex["task_id"], "baseline": False, "reflect": False,
                                "error": err2[:80]})

        if (i + 1) % 10 == 0 or (i + 1) == len(problems):
            elapsed = time.time() - t0
            print(
                f"[{i+1:3d}/164] baseline={baseline_pass} reflect={reflect_pass} "
                f"fixed={fixed_by_reflect} | {elapsed:.0f}s",
                flush=True,
            )

    elapsed = time.time() - t0
    print(f"\n{'=' * 60}")
    print(f"  SELF-REFLECTION RESULTS")
    print(f"{'=' * 60}")
    print(f"  Baseline:            {baseline_pass}/164 = {baseline_pass/164:.1%}")
    print(f"  With reflection:     {reflect_pass}/164 = {reflect_pass/164:.1%}")
    print(f"  Fixed by reflection: {fixed_by_reflect}")
    print(f"  Time:                {elapsed:.0f}s")
    print(f"{'=' * 60}")

    print(f"\nFixed by self-reflection:")
    for r in results:
        if not r["baseline"] and r["reflect"]:
            print(f"  {r['id']}")

    with open("selfrefect_results.json", "w") as f:
        json.dump({"baseline": baseline_pass, "reflect": reflect_pass,
                    "fixed": fixed_by_reflect, "results": results}, f, indent=2)


if __name__ == "__main__":
    main()
