#!/usr/bin/env python3
"""Full 164-problem HumanEval with self-steering loop. No hand-crafted thoughts."""
import sys, os, re, subprocess, tempfile, time, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from datasets import load_dataset
from self_steering import SelfSteeringLoop

INST = (
    "Complete the following Python function. "
    "Return ONLY the complete function inside a ```python code block.\n\n"
    "```python\n{prompt}\n```"
)

def extract_code(text):
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    m = re.search(r"```(?:python|py)?\s*\n(.*?)```", text, re.DOTALL)
    return m.group(1) if m else text

def make_validator(ex):
    def validate(output):
        code = extract_code(output)
        prog = code if f"def {ex['entry_point']}" in code else f"{ex['prompt']}{code}"
        full = f"{prog}\n\n{ex['test']}\n\ncheck({ex['entry_point']})\n"
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(full); tmp = f.name
        try:
            r = subprocess.run(["python3", tmp], capture_output=True, text=True, timeout=10)
            return r.returncode == 0, r.stderr[:100] if r.returncode != 0 else ""
        except subprocess.TimeoutExpired:
            return False, "TIMEOUT"
        except:
            return False, "ERROR"
        finally:
            try: os.unlink(tmp)
            except: pass
    return validate

def main():
    print("=" * 60)
    print("  SELF-STEERING LOOP — Full 164 HumanEval")
    print("  Autonomous strategy selection + retry")
    print("=" * 60)

    tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct")
    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2.5-7B-Instruct",
        quantization_config=BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16),
        device_map="cuda:0",
    ).eval()

    loop = SelfSteeringLoop(model, tok, max_retries=3)
    he = load_dataset("openai/openai_humaneval", split="test")
    problems = list(he)

    baseline_pass = 0
    steered_pass = 0
    fixed = 0
    results = []
    t0 = time.time()

    for i, ex in enumerate(problems):
        prompt = tok.apply_chat_template(
            [{"role": "user", "content": INST.format(prompt=ex["prompt"])}],
            tokenize=False, add_generation_prompt=True,
        )
        result = loop.generate_with_retry(prompt, validator=make_validator(ex))

        if result.attempts == 1 and result.passed:
            baseline_pass += 1
        steered_pass += int(result.passed)
        if result.attempts > 1 and result.passed:
            fixed += 1

        results.append({
            "id": ex["task_id"],
            "passed": result.passed,
            "attempts": result.attempts,
            "strategy": result.strategy_used,
        })

        if (i + 1) % 10 == 0 or (i + 1) == len(problems):
            elapsed = time.time() - t0
            print(
                f"[{i+1:3d}/164] baseline={baseline_pass} steered={steered_pass} "
                f"fixed={fixed} | {elapsed:.0f}s",
                flush=True,
            )

    elapsed = time.time() - t0
    print(f"\n{'=' * 60}")
    print(f"  SELF-STEERING RESULTS")
    print(f"{'=' * 60}")
    print(f"  Baseline (attempt 1):  {baseline_pass}/164 = {baseline_pass/164:.1%}")
    print(f"  Self-steered:          {steered_pass}/164 = {steered_pass/164:.1%}")
    print(f"  Fixed by steering:     {fixed}")
    print(f"  Improvement:           +{steered_pass - baseline_pass} ({(steered_pass-baseline_pass)/164*100:.1f}%)")
    print(f"  Time:                  {elapsed:.0f}s")
    print(f"  Strategy stats:        {loop.stats}")
    print(f"{'=' * 60}")

    loop.save_strategies("strategies_full_run")

    with open("selfsteer_full_results.json", "w") as f:
        json.dump({
            "baseline": baseline_pass, "steered": steered_pass,
            "fixed": fixed, "total": 164,
            "time_s": elapsed, "stats": loop.stats, "results": results,
        }, f, indent=2)
    print("Results saved to selfsteer_full_results.json")

if __name__ == "__main__":
    main()
