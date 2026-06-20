"""Self-Steering Cognitive Loop — Pattern G + Pattern D automated.

The model generates, evaluates, and self-corrects via compiled thought
injection. Reasoning strategies are selected from a library and injected
mid-generation. Successful strategies are saved for future sessions.

Usage:
    loop = SelfSteeringLoop(model, tokenizer)
    loop.load_strategies("strategies/")
    result = loop.generate_with_retry(prompt, validator=test_fn)
"""
import torch
import json
import os
import time
from typing import Optional, Callable
from dataclasses import dataclass, field


@dataclass
class Strategy:
    name: str
    category: str
    thought: str
    successes: int = 0
    attempts: int = 0

    @property
    def success_rate(self):
        return self.successes / max(self.attempts, 1)


@dataclass
class SteeringResult:
    output: str
    passed: bool
    strategy_used: Optional[str] = None
    attempts: int = 1
    time_s: float = 0.0


class SelfSteeringLoop:

    def __init__(self, model, tokenizer, device="cuda:0", max_retries=5):
        self._model = model
        self._tok = tokenizer
        self._device = device
        self._max_retries = max_retries
        self._strategies: dict[str, Strategy] = {}
        self._history: list[dict] = []
        self._load_default_strategies()

    def _load_default_strategies(self):
        defaults = {
            "step_by_step": Strategy("step_by_step", "general",
                "Let me think step by step. First, what are the edge cases? Empty input, zero, negative, single element. Now the simplest correct approach.\n\n"),
            "trace_examples": Strategy("trace_examples", "general",
                "Let me trace through each example in the docstring carefully to understand the exact expected behavior before writing code.\n\n"),
            "simplify": Strategy("simplify", "general",
                "Wait, I'm overcomplicating this. Let me step back and use the simplest possible approach. Python has built-in functions for most of this.\n\n"),
            "verify": Strategy("verify", "general",
                "Let me verify my solution against ALL the examples before finalizing. Trace through each one mentally.\n\n"),
            "parsing": Strategy("parsing", "string",
                "I need to track state as I scan character by character. Use a counter or stack for depth/nesting. Handle edge cases: empty input, single character.\n\n"),
            "encoding": Strategy("encoding", "string",
                "The decode function must be the EXACT INVERSE of encode. If encode shifts +N, decode shifts -N. If encode rotates left, decode rotates right. Trace through a small example to verify the round-trip.\n\n"),
            "string_ops": Strategy("string_ops", "string",
                "For string operations: work character by character, build result as a list. Consider case sensitivity. Use str.lower()/upper(). Check empty string, single char.\n\n"),
            "math_formula": Strategy("math_formula", "math",
                "Let me derive the mathematical formula step by step. Check base cases (n=0, n=1). Look for patterns in the examples. Use pow(base, exp, mod) for modular arithmetic.\n\n"),
            "counting": Strategy("counting", "math",
                "For counting: use collections.Counter or a dict. Be precise about what to count. Keep only elements matching the criteria. Preserve order when needed.\n\n"),
            "digits": Strategy("digits", "math",
                "For digit operations: convert to string for digit access, or use %10 and //10. Remember abs() for negatives. Unit digit = abs(n)%10. For base conversion: repeatedly divide and collect remainders.\n\n"),
            "prime_check": Strategy("prime_check", "math",
                "For prime checking: test divisibility from 2 to sqrt(n). n < 2 is not prime. 2 is prime. all(n % i != 0 for i in range(2, int(n**0.5)+1)) and n >= 2.\n\n"),
            "sorting": Strategy("sorting", "list",
                "For custom sorting: sorted() with a key function. Python's sort is stable. Handle: empty list, single element, negative numbers. For digit sum sorting: define a key function.\n\n"),
            "list_filtering": Strategy("list_filtering", "list",
                "For list filtering: [x for x in lst if condition(x)]. Preserve order. Handle empty list. Check: does the problem want elements, indices, or counts?\n\n"),
            "polynomial": Strategy("polynomial", "math",
                "For polynomial root finding: use Newton's method. x_new = x - poly(xs,x) / derivative(xs,x). The derivative of [a0,a1,a2,...] is [a1, 2*a2, 3*a3,...]. Iterate from x=0 until convergence.\n\n"),
            "geometry": Strategy("geometry", "math",
                "Think logically about the geometry. For collision problems: each object from set A interacts with each from set B. Count = |A| * |B|. For n cars from each direction: n*n collisions.\n\n"),
            "recursion_dp": Strategy("recursion_dp", "math",
                "This needs recursion or DP. Identify: what's the base case? What's the recurrence? Use iterative DP with rolling variables to avoid stack overflow. For fib-like: use a loop with N variables.\n\n"),
            "search_match": Strategy("search_match", "string",
                "For searching/matching: check every starting position. Allow overlapping matches. For substring rotations: generate all rotations, check each as substring.\n\n"),
        }
        self._strategies.update(defaults)

    def add_strategy(self, name: str, category: str, thought: str):
        self._strategies[name] = Strategy(name, category, thought)

    def _gen_with_thought(self, prompt, thought=None, max_new=512):
        ids = self._tok(prompt, return_tensors="pt").input_ids.to(self._device)
        with torch.no_grad():
            out = self._model(ids, use_cache=True)
        cache = out.past_key_values
        gp = ids.shape[1]

        if thought:
            t_ids = self._tok(thought, add_special_tokens=False, return_tensors="pt").input_ids.to(self._device)
            t_pos = torch.arange(gp, gp + t_ids.shape[1], device=self._device).unsqueeze(0)
            with torch.no_grad():
                out = self._model(t_ids, past_key_values=cache, position_ids=t_pos, use_cache=True)
            cache = out.past_key_values
            gp += t_ids.shape[1]

        eos = set()
        if self._tok.eos_token_id is not None:
            eos.add(self._tok.eos_token_id)
        try:
            eos.update(self._tok("<|im_end|>", add_special_tokens=False)["input_ids"])
        except Exception:
            pass

        generated = list(self._tok.encode(thought, add_special_tokens=False)) if thought else []
        nid = out.logits[0, -1, :].argmax().item()
        if nid in eos:
            return self._tok.decode(generated, skip_special_tokens=True)
        generated.append(nid)

        with torch.no_grad():
            for _ in range(max_new):
                o = self._model(
                    torch.tensor([[nid]], device=self._device),
                    past_key_values=cache,
                    position_ids=torch.tensor([[gp]], device=self._device),
                    use_cache=True,
                )
                cache = o.past_key_values
                gp += 1
                nid = o.logits[0, -1, :].argmax().item()
                if nid in eos:
                    break
                generated.append(nid)

        return self._tok.decode(generated, skip_special_tokens=True)

    def _select_strategy(self, attempt: int, error: str = "", category: str = "") -> Optional[Strategy]:
        candidates = list(self._strategies.values())
        if category:
            cat_matches = [s for s in candidates if s.category == category]
            if cat_matches:
                candidates = cat_matches

        candidates.sort(key=lambda s: s.success_rate, reverse=True)

        if attempt < len(candidates):
            return candidates[attempt]
        return None

    def generate_with_retry(
        self,
        prompt: str,
        validator: Optional[Callable[[str], tuple[bool, str]]] = None,
        category: str = "",
        max_new: int = 512,
    ) -> SteeringResult:
        t0 = time.time()

        output = self._gen_with_thought(prompt, max_new=max_new)
        if validator:
            passed, error = validator(output)
        else:
            passed, error = True, ""

        if passed:
            return SteeringResult(output=output, passed=True, time_s=time.time() - t0)

        for attempt in range(self._max_retries):
            strategy = self._select_strategy(attempt, error, category)
            if not strategy:
                break

            strategy.attempts += 1
            output = self._gen_with_thought(prompt, thought=strategy.thought, max_new=max_new)

            if validator:
                passed, error = validator(output)
            else:
                passed, error = True, ""

            if passed:
                strategy.successes += 1
                self._history.append({
                    "strategy": strategy.name,
                    "category": category,
                    "success": True,
                    "attempt": attempt + 1,
                })
                return SteeringResult(
                    output=output, passed=True,
                    strategy_used=strategy.name,
                    attempts=attempt + 2,
                    time_s=time.time() - t0,
                )

        return SteeringResult(
            output=output, passed=False,
            strategy_used=strategy.name if strategy else None,
            attempts=self._max_retries + 1,
            time_s=time.time() - t0,
        )

    def save_strategies(self, path: str):
        os.makedirs(path, exist_ok=True)
        data = {}
        for name, s in self._strategies.items():
            data[name] = {
                "category": s.category,
                "thought": s.thought,
                "successes": s.successes,
                "attempts": s.attempts,
            }
        with open(os.path.join(path, "strategies.json"), "w") as f:
            json.dump(data, f, indent=2)
        with open(os.path.join(path, "history.json"), "w") as f:
            json.dump(self._history, f, indent=2)

    def load_strategies(self, path: str):
        strat_path = os.path.join(path, "strategies.json")
        if os.path.exists(strat_path):
            with open(strat_path) as f:
                data = json.load(f)
            for name, info in data.items():
                self._strategies[name] = Strategy(
                    name=name,
                    category=info["category"],
                    thought=info["thought"],
                    successes=info.get("successes", 0),
                    attempts=info.get("attempts", 0),
                )
        hist_path = os.path.join(path, "history.json")
        if os.path.exists(hist_path):
            with open(hist_path) as f:
                self._history = json.load(f)

    @property
    def stats(self):
        total = sum(s.attempts for s in self._strategies.values())
        successes = sum(s.successes for s in self._strategies.values())
        return {
            "strategies": len(self._strategies),
            "total_attempts": total,
            "total_successes": successes,
            "success_rate": successes / max(total, 1),
            "history_length": len(self._history),
        }
