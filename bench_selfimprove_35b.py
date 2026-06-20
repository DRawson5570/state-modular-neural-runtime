#!/usr/bin/env python3
"""Self-Improving HumanEval — the 35B model analyzes its own failures,
generates strategies, compiles them as context, and retries.

If Pass 2 > Pass 1, the model improved itself. Not through training.
Through compiled experience.

Usage:
    python3 bench_selfimprove_35b.py [--server http://pe2:8000] [--max-problems 164]
"""
import requests, json, re, subprocess, tempfile, time, os, sys, argparse
from datasets import load_dataset

INST = (
    "Complete the following Python function. "
    "Return ONLY the complete function inside a ```python code block.\n\n"
    "```python\n{prompt}\n```"
)

ANALYSIS_PROMPT = (
    "You wrote code for a programming problem, but it failed.\n\n"
    "**Problem:**\n```python\n{prompt}\n```\n\n"
    "**Your code:**\n```python\n{code}\n```\n\n"
    "**Error:**\n```\n{error}\n```\n\n"
    "Analyze WHY your code failed. Then write a brief strategy (2-3 sentences) "
    "that you would tell yourself BEFORE attempting this problem, to avoid "
    "making the same mistake. Focus on the specific reasoning approach, not "
    "implementation details. Start your strategy with 'STRATEGY:'"
)


def chat(server, prompt, temperature=0):
    r = requests.post(
        server + "/v1/chat/completions",
        json={
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 2048,
            "temperature": temperature,
        },
        timeout=120,
    )
    data = r.json()
    text = data["choices"][0]["message"]["content"]
    text = text.replace("<|im_end|>", "").strip()
    return text


def compile_state(server, content_id, text):
    requests.delete(server + "/v1/compiled/" + content_id, timeout=10)
    r = requests.post(
        server + "/v1/compile",
        json={"id": content_id, "text": text},
        timeout=60,
    )
    return r.json()


def clear_compiled(server):
    r = requests.get(server + "/v1/compiled", timeout=10)
    data = r.json()
    for cid in data.get("compiled_states", {}):
        requests.delete(server + "/v1/compiled/" + cid, timeout=10)


def extract_code(text):
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    m = re.search(r"```(?:python|py)?\s*\n(.*?)```", text, re.DOTALL)
    return m.group(1).strip() if m else text.strip()


def test_code(code, test_code_str, timeout_sec=10):
    full = code + "\n\n" + test_code_str
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write(full)
        f.flush()
        try:
            result = subprocess.run(
                [sys.executable, f.name],
                capture_output=True, text=True, timeout=timeout_sec,
            )
            return result.returncode == 0, result.stderr[:500] if result.returncode != 0 else ""
        except subprocess.TimeoutExpired:
            return False, "TIMEOUT"
        except Exception as e:
            return False, str(e)
        finally:
            os.unlink(f.name)


def extract_strategy(text):
    if "STRATEGY:" in text:
        return text.split("STRATEGY:", 1)[1].strip()
    lines = text.strip().split("\n")
    return lines[-1].strip() if lines else text.strip()


def run_phase1_baseline(server, problems, max_problems):
    print("\n" + "=" * 60)
    print("PHASE 1: BASELINE (no strategies)")
    print("=" * 60)

    results = []
    passed = 0

    for i, problem in enumerate(problems[:max_problems]):
        task_id = problem["task_id"]
        prompt = problem["prompt"]
        tests = problem["test"]
        entry = problem.get("entry_point", "")

        clear_compiled(server)
        text = chat(server, INST.format(prompt=prompt))
        code = extract_code(text)
        ok, error = test_code(code, tests + "\n" + "check(%s)" % entry)

        if ok:
            passed += 1

        results.append({
            "task_id": task_id,
            "passed": ok,
            "code": code,
            "error": error,
            "raw_response": text[:500],
        })

        status = "PASS" if ok else "FAIL"
        print("[%d/%d] %s %s (running: %.1f%%)" % (
            i + 1, max_problems, status, task_id,
            100 * passed / (i + 1),
        ))

    print("\nBASELINE: %d/%d = %.1f%%" % (passed, max_problems, 100 * passed / max_problems))
    return results


def run_phase2_selfimprove(server, problems, phase1_results, max_problems):
    print("\n" + "=" * 60)
    print("PHASE 2: SELF-ANALYSIS + RETRY")
    print("=" * 60)

    failures = [r for r in phase1_results if not r["passed"]]
    print("Analyzing %d failures..." % len(failures))

    strategy_library = []
    fixed = 0

    for i, result in enumerate(failures):
        task_id = result["task_id"]
        problem = next(p for p in problems if p["task_id"] == task_id)
        prompt = problem["prompt"]
        tests = problem["test"]
        entry = problem.get("entry_point", "")

        clear_compiled(server)
        analysis = chat(server, ANALYSIS_PROMPT.format(
            prompt=prompt,
            code=result["code"],
            error=result["error"][:300],
        ))

        strategy = extract_strategy(analysis)

        compile_state(server, "strategy", strategy)
        text = chat(server, INST.format(prompt=prompt))
        code = extract_code(text)
        ok, error = test_code(code, tests + "\n" + "check(%s)" % entry)

        if ok:
            fixed += 1
            strategy_library.append({
                "strategy": strategy,
                "source_problem": task_id,
                "analysis": analysis[:300],
            })

        status = "FIXED" if ok else "still_fail"
        print("[%d/%d] %s %s — strategy: %s" % (
            i + 1, len(failures), status, task_id, strategy[:80],
        ))

        result["phase2_passed"] = ok
        result["phase2_code"] = code
        result["phase2_error"] = error
        result["strategy"] = strategy
        result["analysis"] = analysis[:500]

    print("\nSELF-IMPROVEMENT: fixed %d/%d failures" % (fixed, len(failures)))
    print("Strategy library: %d successful strategies" % len(strategy_library))
    return strategy_library


def run_phase3_library(server, problems, phase1_results, strategy_library, max_problems):
    print("\n" + "=" * 60)
    print("PHASE 3: LIBRARY REUSE (cross-problem transfer)")
    print("=" * 60)

    remaining = [r for r in phase1_results if not r["passed"] and not r.get("phase2_passed")]
    if not remaining:
        print("No remaining failures!")
        return

    if not strategy_library:
        print("No strategies in library!")
        return

    combined_strategies = "\n\n".join(
        "Strategy from %s: %s" % (s["source_problem"], s["strategy"])
        for s in strategy_library[:10]
    )

    fixed = 0
    for i, result in enumerate(remaining):
        task_id = result["task_id"]
        problem = next(p for p in problems if p["task_id"] == task_id)
        prompt = problem["prompt"]
        tests = problem["test"]
        entry = problem.get("entry_point", "")

        clear_compiled(server)
        compile_state(server, "library",
            "Before solving, review these proven strategies:\n\n" + combined_strategies)

        text = chat(server, INST.format(prompt=prompt))
        code = extract_code(text)
        ok, error = test_code(code, tests + "\n" + "check(%s)" % entry)

        if ok:
            fixed += 1

        status = "FIXED" if ok else "still_fail"
        print("[%d/%d] %s %s" % (i + 1, len(remaining), status, task_id))

        result["phase3_passed"] = ok

    print("\nLIBRARY TRANSFER: fixed %d/%d remaining failures" % (fixed, len(remaining)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--server", default="http://pe2:8000")
    parser.add_argument("--max-problems", type=int, default=164)
    parser.add_argument("--output", default="selfimprove_35b_results.json")
    args = parser.parse_args()

    print("Self-Improving HumanEval — Qwen3.6-35B-A3B")
    print("Server: %s" % args.server)

    r = requests.get(args.server + "/v1/models", timeout=10)
    print("Model: %s" % r.json()["data"][0]["id"])

    ds = load_dataset("openai/openai_humaneval", split="test")
    problems = list(ds)
    print("Loaded %d problems" % len(problems))

    t0 = time.time()

    phase1 = run_phase1_baseline(args.server, problems, args.max_problems)
    baseline_pass = sum(1 for r in phase1 if r["passed"])

    strategy_lib = run_phase2_selfimprove(args.server, problems, phase1, args.max_problems)

    run_phase3_library(args.server, problems, phase1, strategy_lib, args.max_problems)

    elapsed = time.time() - t0

    p1_pass = sum(1 for r in phase1 if r["passed"])
    p2_pass = p1_pass + sum(1 for r in phase1 if r.get("phase2_passed"))
    p3_pass = p2_pass + sum(1 for r in phase1 if r.get("phase3_passed"))
    total = min(args.max_problems, len(problems))

    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print("Model: Qwen3.6-35B-A3B (36B params, 3B active)")
    print("Time: %.0f minutes" % (elapsed / 60))
    print()
    print("Phase 1 (baseline):           %d/%d = %.1f%%" % (p1_pass, total, 100 * p1_pass / total))
    print("Phase 2 (self-analysis+retry): %d/%d = %.1f%% (+%d)" % (p2_pass, total, 100 * p2_pass / total, p2_pass - p1_pass))
    print("Phase 3 (library transfer):    %d/%d = %.1f%% (+%d)" % (p3_pass, total, 100 * p3_pass / total, p3_pass - p2_pass))
    print()
    print("Self-generated strategies: %d" % len(strategy_lib))

    if p2_pass > p1_pass:
        print("\nTHE MODEL IMPROVED ITSELF. +%d problems fixed through self-analysis." % (p2_pass - p1_pass))
    if p3_pass > p2_pass:
        print("STRATEGY TRANSFER WORKS. +%d problems fixed by reusing strategies from other problems." % (p3_pass - p2_pass))

    output = {
        "model": "Qwen3.6-35B-A3B",
        "server": args.server,
        "total_problems": total,
        "phase1_passed": p1_pass,
        "phase2_passed": p2_pass,
        "phase3_passed": p3_pass,
        "strategy_library": strategy_lib,
        "elapsed_seconds": elapsed,
        "results": phase1,
    }
    with open(args.output, "w") as f:
        json.dump(output, f, indent=2)
    print("\nResults saved to %s" % args.output)


if __name__ == "__main__":
    main()
