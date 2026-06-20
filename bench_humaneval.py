#!/usr/bin/env python3
"""HumanEval benchmark with No Spoon compiled context."""
import sys, os, re, subprocess, tempfile, time, argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
from datasets import load_dataset
from compiled_engine import CompiledInference

INSTRUCTION = (
    "Complete the following Python function. "
    "Return ONLY the complete function (including the signature) inside a single "
    "```python code block. Do not include explanations, examples, or test code.\n\n"
    "```python\n{prompt}\n```"
)

REFERENCE_LIBRARY = '''
# Python Algorithm Reference Library — Common Patterns

## String Operations
- Use ''.join() for efficient string building
- str.split(), str.strip(), str.replace() for manipulation
- Use collections.Counter for character frequency
- ord() and chr() for ASCII operations
- Regular expressions via re module for pattern matching

## List/Array Patterns
- List comprehensions: [x for x in items if condition]
- Two-pointer technique for sorted arrays
- Sliding window for subarray problems
- enumerate() for index+value iteration
- zip() for parallel iteration
- sorted() with key= for custom sorting
- collections.defaultdict for grouping

## Math Utilities
- math.gcd, math.lcm for number theory
- Integer division: a // b, modulo: a % b
- Check prime: test divisors up to sqrt(n)
- Fibonacci: iterative with two variables
- Factorial: math.factorial or iterative
- abs() for absolute value, round() for rounding
- sum(), min(), max() for aggregation
- Binary: bin(), hex(), oct() for base conversion

## Common Algorithm Patterns
- Binary search: lo, hi = 0, len(arr)-1; while lo <= hi
- Recursion with memoization: functools.lru_cache
- Stack-based: matching brackets, expression evaluation
- Dictionary lookup: O(1) membership testing
- Set operations: intersection, union, difference
- Sorting: sorted() returns new list, .sort() in-place

## Functional Python
- map(fn, iterable), filter(fn, iterable)
- any(), all() for boolean aggregation
- lambda functions for inline operations
- itertools: permutations, combinations, product, chain
- functools: reduce, partial

## Type Checking and Conversion
- isinstance(x, (int, float)) for type checks
- int(), float(), str(), list(), tuple() for conversion
- bool(): 0, None, empty containers are False
'''


def extract_code(text):
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    m = re.search(r"```(?:python|py)?\s*\n(.*?)```", text, re.DOTALL)
    if m:
        return m.group(1)
    return text


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
        return r.returncode == 0
    except (subprocess.TimeoutExpired, Exception):
        return False
    finally:
        try:
            os.unlink(tmp)
        except OSError:
            pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--n", type=int, default=0, help="0 = all 164")
    ap.add_argument("--compiled", action="store_true", help="Use compiled reference library")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    engine = CompiledInference(args.model)
    print(f"Engine: {engine}")

    if args.compiled:
        engine.compile("python_reference", REFERENCE_LIBRARY)
        print(f"Compiled reference library: {engine.stats.compiled_tokens} tokens")

    he = load_dataset("openai/openai_humaneval", split="test")
    problems = list(he)
    if args.n > 0:
        problems = problems[:args.n]

    passes = 0
    fails = []
    t0 = time.time()

    for i, ex in enumerate(problems):
        prompt = ex["prompt"]
        entry = ex["entry_point"]
        question = INSTRUCTION.format(prompt=prompt)

        response = engine.chat(question, max_tokens=512, temperature=0.0)
        code = extract_code(response)
        program = build_program(prompt, code, entry)
        ok = run_test(program, ex["test"], entry)
        passes += int(ok)
        if not ok:
            fails.append(ex["task_id"])
        if args.verbose and not ok:
            print(f"\n--- FAIL {ex['task_id']} ---\n{response[:300]}\n")
        if (i + 1) % 10 == 0 or (i + 1) == len(problems):
            elapsed = time.time() - t0
            print(f"[{i+1}/{len(problems)}] pass={passes} ({passes/(i+1):.1%}) "
                  f"| {elapsed:.0f}s | {engine.stats.last_tps:.0f} tok/s")

    rate = passes / len(problems)
    mode = "COMPILED" if args.compiled else "BASELINE"
    print(f"\n{'='*50}")
    print(f"  {args.model} — {mode}")
    print(f"  HumanEval pass@1: {passes}/{len(problems)} = {rate:.1%}")
    print(f"  Time: {time.time()-t0:.0f}s")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()
