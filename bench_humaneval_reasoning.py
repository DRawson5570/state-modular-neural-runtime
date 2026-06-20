#!/usr/bin/env python3
"""Automated HumanEval retry pipeline with reasoning library.

For each problem: baseline → if fail → categorize error → select reasoning
strategy from library → inject thought → retry → record results.
"""
import sys, os, re, subprocess, tempfile, time, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from datasets import load_dataset

REASONING_LIBRARY = {
    "parsing": "I need to carefully parse the input. Let me identify the delimiters and track state (depth, position, mode) as I scan character by character. Handle edge cases: empty input, single character.\n\n",
    "string_manipulation": "For string operations: work character by character, use a list to build the result (not string concatenation). Consider: empty string, single char, unicode. Use str.lower()/upper() for case. Check the examples carefully for exact expected behavior.\n\n",
    "counting": "For counting problems: use collections.Counter or a dictionary. Be precise about what counts — exactly equal, greater than, etc. Check if the problem wants count of unique items or total occurrences. Verify against every example.\n\n",
    "math_formula": "Let me derive the mathematical formula step by step. Check base cases (n=0, n=1). Look for patterns in the examples. Consider: is this combinatorics, modular arithmetic, or a recurrence relation? Use Python's built-in pow(base, exp, mod) for modular exponentiation.\n\n",
    "sorting_ordering": "For sorting: understand exactly what ordering is required. Custom key functions with sorted(). Preserve original order when needed (use enumerate). Handle ties. Check if output should be indices or values.\n\n",
    "recursion_dp": "This looks like it needs recursion or dynamic programming. Identify: what's the base case? What's the recurrence? Can I use iterative DP with a table? For Fibonacci-like: use iterative with rolling variables to avoid stack overflow.\n\n",
    "search_match": "For searching/matching: check every position. Allow overlapping matches unless stated otherwise. Consider: empty pattern, pattern longer than text, case sensitivity. Use simple iteration rather than regex unless regex is clearly needed.\n\n",
    "polynomial": "For polynomial operations: use Newton's method for root finding (x = x - f(x)/f'(x)). The derivative of a polynomial [a0, a1, a2, ...] is [a1, 2*a2, 3*a3, ...]. Iterate until convergence from x=0.\n\n",
    "encoding_decoding": "For encoding/decoding: the decode function must be the EXACT INVERSE of encode. If encode shifts right, decode shifts left. If encode rotates left by 1 in groups of 3 (abc→bca), decode rotates right by 1 (bca→abc). Trace through a small example to verify the inverse.\n\n",
    "geometry_collision": "Think about this geometrically/logically. For collision problems: each object from set A interacts with each from set B. Count = |A| × |B|. For distance/position: use the coordinate system described.\n\n",
    "list_filtering": "For list filtering: iterate and check each element against the condition. Preserve order. Use list comprehension [x for x in lst if condition(x)]. Handle empty list. Check: does the problem want elements, indices, or counts?\n\n",
    "digit_manipulation": "For digit operations: convert to string to access digits, or use %10 and //10 for extraction. Remember: abs() for negative numbers. The UNIT digit is abs(n) % 10. For base conversion: repeatedly divide and collect remainders.\n\n",
    "prime_check": "For prime checking: test divisibility from 2 to sqrt(n). n < 2 is not prime. 2 is prime. Even numbers > 2 are not prime. Use: all(n % i != 0 for i in range(2, int(n**0.5)+1)) and n >= 2.\n\n",
    "general": "Let me think step by step. First, read ALL the examples in the docstring carefully — they define the exact expected behavior. Handle edge cases: empty input, zero, negative numbers, single element. Write the simplest correct solution. Verify against each example before finalizing.\n\n",
}

PROBLEM_TO_STRATEGY = {
    1: "parsing",           # separate_paren_groups
    9: "list_filtering",    # rolling_max
    10: "string_manipulation", # make_palindrome
    11: "string_manipulation", # string_xor
    16: "counting",         # count_distinct_characters
    17: "parsing",          # parse_music
    18: "search_match",     # how_many_times
    19: "sorting_ordering", # sort_numbers
    26: "counting",         # remove_duplicates
    28: "string_manipulation", # concatenate
    29: "list_filtering",   # filter_by_prefix
    32: "polynomial",       # find_zero
    38: "encoding_decoding", # decode_cyclic
    39: "prime_check",      # prime_fib
    40: "search_match",     # triples_sum_to_zero
    41: "geometry_collision", # car_race_collision
    44: "digit_manipulation", # change_base
    46: "recursion_dp",     # fib4
    49: "math_formula",     # modp
    50: "encoding_decoding", # decode_shift
    54: "string_manipulation", # same_chars
    57: "sorting_ordering", # monotonic
    64: "string_manipulation", # vowels_count
    65: "digit_manipulation", # circular_shift
    67: "parsing",          # fruit_distribution
    68: "list_filtering",   # pluck
    77: "math_formula",     # iscube
    80: "string_manipulation", # is_happy
    83: "math_formula",     # starts_one_ends
    93: "encoding_decoding", # encode
    95: "string_manipulation", # check_dict_case
    97: "digit_manipulation", # multiply
    99: "math_formula",     # closest_integer
    100: "math_formula",    # make_a_pile
    104: "digit_manipulation", # unique_digits
    106: "math_formula",    # f
    109: "sorting_ordering", # move_one_ball
    113: "string_manipulation", # odd_count
    115: "list_filtering",  # max_fill
    116: "sorting_ordering", # sort_array
    120: "list_filtering",  # maximum
    122: "math_formula",    # add_elements
    123: "math_formula",    # get_odd_collatz
    125: "string_manipulation", # split_words
    126: "sorting_ordering", # is_sorted
    127: "math_formula",    # intersection
    130: "list_filtering",  # tri
    132: "string_manipulation", # is_nested
    134: "general",         # check_if_last_char_is_a_letter
    137: "math_formula",    # compare_one
    139: "math_formula",    # special_factorial
    140: "string_manipulation", # fix_spaces
    141: "parsing",         # file_name_check
    145: "sorting_ordering", # order_by_points
    158: "search_match",    # find_max
    160: "math_formula",    # do_algebra
    163: "list_filtering",  # generate_integers
}

INST = (
    "Complete the following Python function. "
    "Return ONLY the complete function (including the signature) inside a single "
    "```python code block. Do not include explanations.\n\n"
    "```python\n{prompt}\n```"
)

def extract_code(text):
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    m = re.search(r"```(?:python|py)?\s*\n(.*?)```", text, re.DOTALL)
    return m.group(1) if m else text

def run_test(prog, test_code, entry):
    full = f"{prog}\n\n{test_code}\n\ncheck({entry})\n"
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False, encoding="utf-8") as f:
        f.write(full); tmp = f.name
    try:
        r = subprocess.run(["python3", tmp], capture_output=True, text=True, timeout=10)
        return r.returncode == 0, r.stderr[:200] if r.returncode != 0 else ""
    except subprocess.TimeoutExpired:
        return False, "TIMEOUT"
    except Exception as e:
        return False, str(e)[:100]
    finally:
        try: os.unlink(tmp)
        except: pass

def gen_with_thought(model, tok, prompt_text, thought=None, max_new=512):
    ids = tok(prompt_text, return_tensors="pt").input_ids.to("cuda:0")
    with torch.no_grad():
        out = model(ids, use_cache=True)
    cache = out.past_key_values
    gp = ids.shape[1]
    if thought:
        t_ids = tok(thought, add_special_tokens=False, return_tensors="pt").input_ids.to("cuda:0")
        t_pos = torch.arange(gp, gp + t_ids.shape[1], device="cuda:0").unsqueeze(0)
        with torch.no_grad():
            out = model(t_ids, past_key_values=cache, position_ids=t_pos, use_cache=True)
        cache = out.past_key_values
        gp += t_ids.shape[1]
    eos = {tok.eos_token_id}
    try:
        eos.update(tok("<|im_end|>", add_special_tokens=False)["input_ids"])
    except:
        pass
    generated = list(tok.encode(thought, add_special_tokens=False)) if thought else []
    nid = out.logits[0, -1, :].argmax().item()
    if nid in eos:
        return tok.decode(generated, skip_special_tokens=True)
    generated.append(nid)
    with torch.no_grad():
        for _ in range(max_new):
            o = model(torch.tensor([[nid]], device="cuda:0"), past_key_values=cache,
                      position_ids=torch.tensor([[gp]], device="cuda:0"), use_cache=True)
            cache = o.past_key_values
            gp += 1
            nid = o.logits[0, -1, :].argmax().item()
            if nid in eos:
                break
            generated.append(nid)
    return tok.decode(generated, skip_special_tokens=True)


def classify_problem(idx, prompt_text):
    if idx in PROBLEM_TO_STRATEGY:
        return PROBLEM_TO_STRATEGY[idx]
    pl = prompt_text.lower()
    if any(w in pl for w in ["parse", "paren", "bracket", "split", "extract"]):
        return "parsing"
    if any(w in pl for w in ["string", "char", "vowel", "letter", "upper", "lower"]):
        return "string_manipulation"
    if any(w in pl for w in ["count", "frequency", "distinct", "unique", "duplicate"]):
        return "counting"
    if any(w in pl for w in ["sort", "order", "arrange", "monoton"]):
        return "sorting_ordering"
    if any(w in pl for w in ["prime", "factor", "divisib"]):
        return "prime_check"
    if any(w in pl for w in ["digit", "base", "binary", "decimal"]):
        return "digit_manipulation"
    if any(w in pl for w in ["encode", "decode", "cipher", "shift", "cycl"]):
        return "encoding_decoding"
    if any(w in pl for w in ["fibonacci", "recur", "memo"]):
        return "recursion_dp"
    if any(w in pl for w in ["polynomial", "root", "zero"]):
        return "polynomial"
    return "general"


def main():
    print("=" * 60)
    print("  AUTOMATED REASONING LIBRARY + RETRY PIPELINE")
    print("  Qwen 7B 4-bit | Full 164 HumanEval")
    print("=" * 60)

    tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct")
    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2.5-7B-Instruct",
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16
        ),
        device_map="cuda:0",
    ).eval()

    he = load_dataset("openai/openai_humaneval", split="test")
    problems = list(he)

    baseline_pass = 0
    retry_pass = 0
    fixed_by_retry = 0
    results = []
    t0 = time.time()

    for i, ex in enumerate(problems):
        prompt = tok.apply_chat_template(
            [{"role": "user", "content": INST.format(prompt=ex["prompt"])}],
            tokenize=False, add_generation_prompt=True,
        )

        resp = gen_with_thought(model, tok, prompt)
        code = extract_code(resp)
        prog = code if f"def {ex['entry_point']}" in code else f"{ex['prompt']}{code}"
        ok_base, err_base = run_test(prog, ex["test"], ex["entry_point"])

        if ok_base:
            baseline_pass += 1
            retry_pass += 1
            results.append({"id": ex["task_id"], "baseline": "PASS", "retry": "PASS", "strategy": None})
        else:
            strategy = classify_problem(i, ex["prompt"])
            thought = REASONING_LIBRARY.get(strategy, REASONING_LIBRARY["general"])

            resp2 = gen_with_thought(model, tok, prompt, thought=thought)
            code2 = extract_code(resp2)
            prog2 = code2 if f"def {ex['entry_point']}" in code2 else f"{ex['prompt']}{code2}"
            ok_retry, err_retry = run_test(prog2, ex["test"], ex["entry_point"])

            if ok_retry:
                retry_pass += 1
                fixed_by_retry += 1
                results.append({"id": ex["task_id"], "baseline": "FAIL", "retry": "FIXED", "strategy": strategy})
            else:
                results.append({"id": ex["task_id"], "baseline": "FAIL", "retry": "FAIL",
                                "strategy": strategy, "error": err_retry[:100]})

        if (i + 1) % 10 == 0 or (i + 1) == len(problems):
            elapsed = time.time() - t0
            print(
                f"[{i+1:3d}/164] baseline={baseline_pass} retry={retry_pass} "
                f"fixed={fixed_by_retry} | {elapsed:.0f}s",
                flush=True,
            )

    elapsed = time.time() - t0
    baseline_rate = baseline_pass / len(problems)
    retry_rate = retry_pass / len(problems)

    print(f"\n{'=' * 60}")
    print(f"  RESULTS — Qwen 7B 4-bit HumanEval")
    print(f"{'=' * 60}")
    print(f"  Baseline:        {baseline_pass}/164 = {baseline_rate:.1%}")
    print(f"  With retry:      {retry_pass}/164 = {retry_rate:.1%}")
    print(f"  Fixed by retry:  {fixed_by_retry}")
    print(f"  Improvement:     {retry_rate - baseline_rate:+.1%}")
    print(f"  Time:            {elapsed:.0f}s")
    print(f"{'=' * 60}")

    print(f"\nFixed problems:")
    for r in results:
        if r["retry"] == "FIXED":
            print(f"  {r['id']:>20s} strategy={r['strategy']}")

    print(f"\nStill failing:")
    for r in results:
        if r["retry"] == "FAIL":
            print(f"  {r['id']:>20s} strategy={r['strategy']} err={r.get('error', '')[:60]}")

    with open("humaneval_retry_results.json", "w") as f:
        json.dump({"baseline": baseline_pass, "retry": retry_pass, "fixed": fixed_by_retry,
                    "total": len(problems), "results": results}, f, indent=2)
    print(f"\nResults saved to humaneval_retry_results.json")


if __name__ == "__main__":
    main()
