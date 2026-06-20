#!/usr/bin/env python3
"""HumanEval with self-verification via compiled error feedback."""
import sys, os, re, subprocess, tempfile, time, argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from datasets import load_dataset
from compiled_engine import CompiledInference

INSTRUCTION = (
    "Complete the following Python function. "
    "Return ONLY the complete function (including the signature) inside a single "
    "```python code block. Do not include explanations, examples, or test code.\n\n"
    "```python\n{prompt}\n```"
)

FIX_INSTRUCTION = (
    "Your previous implementation of `{entry}` failed with this error:\n"
    "```\n{error}\n```\n\n"
    "Your previous code was:\n```python\n{code}\n```\n\n"
    "Fix the implementation. Return ONLY the corrected function inside a "
    "```python code block."
)


def extract_code(text):
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    m = re.search(r"```(?:python|py)?\s*\n(.*?)```", text, re.DOTALL)
    return m.group(1) if m else text


def build_program(prompt, code, entry_point):
    code = code.strip("\n")
    if f"def {entry_point}" in code:
        return code
    return f"{prompt}{code}"


def run_test(program, test_code, entry_point, timeout=10.0):
    full = f"{program}\n\n{test_code}\n\ncheck({entry_point})\n"
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False, encoding="utf-8") as f:
        f.write(full)
        tmp = f.name
    try:
        r = subprocess.run(["python3", tmp], capture_output=True, text=True, timeout=timeout)
        return r.returncode == 0, r.stderr[:300] if r.returncode != 0 else ""
    except subprocess.TimeoutExpired:
        return False, "TIMEOUT: code took too long to execute (possible infinite loop)"
    except Exception as e:
        return False, str(e)[:200]
    finally:
        try:
            os.unlink(tmp)
        except OSError:
            pass


def extract_doctests(prompt):
    lines = prompt.split("\n")
    tests = []
    for line in lines:
        stripped = line.strip()
        if stripped.startswith(">>>"):
            tests.append(stripped[4:])
    return tests


def quick_test(program, entry_point):
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False, encoding="utf-8") as f:
        f.write(program)
        tmp = f.name
    try:
        r = subprocess.run(["python3", "-c", f"exec(open('{tmp}').read()); print('OK')"],
                          capture_output=True, text=True, timeout=5)
        return r.returncode == 0, r.stderr[:200]
    except:
        return False, "quick test failed"
    finally:
        try:
            os.unlink(tmp)
        except:
            pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--n", type=int, default=0)
    ap.add_argument("--retries", type=int, default=2, help="Max retry attempts on failure")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    engine = CompiledInference(args.model)
    print(f"Engine: {engine}")
    print(f"Max retries per problem: {args.retries}")

    he = load_dataset("openai/openai_humaneval", split="test")
    problems = list(he)
    if args.n > 0:
        problems = problems[:args.n]

    passes = 0
    fixed = 0
    fails = []
    t0 = time.time()

    for i, ex in enumerate(problems):
        prompt = ex["prompt"]
        entry = ex["entry_point"]
        question = INSTRUCTION.format(prompt=prompt)

        response = engine.chat(question, max_tokens=512, temperature=0.0)
        code = extract_code(response)
        program = build_program(prompt, code, entry)
        ok, err = run_test(program, ex["test"], entry)

        if ok:
            passes += 1
        else:
            for attempt in range(args.retries):
                fix_prompt = FIX_INSTRUCTION.format(
                    entry=entry,
                    error=err[:200],
                    code=code[:400],
                )
                engine.compile(f"error_{i}_{attempt}", f"Error feedback for {entry}:\n{err[:200]}\n\nFailed code:\n{code[:400]}")

                response2 = engine.chat(fix_prompt, max_tokens=512, temperature=0.0)
                code2 = extract_code(response2)
                program2 = build_program(prompt, code2, entry)
                ok2, err2 = run_test(program2, ex["test"], entry)

                engine._compiler.remove(f"error_{i}_{attempt}")

                if ok2:
                    passes += 1
                    fixed += 1
                    if args.verbose:
                        print(f"  FIXED {ex['task_id']} on retry {attempt+1}")
                    break
                else:
                    code = code2
                    err = err2

            if not ok2:
                fails.append(ex["task_id"])
                if args.verbose:
                    print(f"  FAIL {ex['task_id']}: {err[:60]}")

        if (i + 1) % 10 == 0 or (i + 1) == len(problems):
            elapsed = time.time() - t0
            print(f"[{i+1}/{len(problems)}] pass={passes} fixed={fixed} "
                  f"({passes/(i+1):.1%}) | {elapsed:.0f}s")

    rate = passes / len(problems)
    print(f"\n{'='*60}")
    print(f"  {args.model} — SELF-VERIFICATION (max {args.retries} retries)")
    print(f"  HumanEval pass@1: {passes}/{len(problems)} = {rate:.1%}")
    print(f"  Fixed by retry: {fixed}")
    print(f"  Time: {time.time()-t0:.0f}s")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
