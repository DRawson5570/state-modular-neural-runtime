#!/usr/bin/env python3
import sys, os, argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from compiled_engine import CompiledInference

def main():
    parser = argparse.ArgumentParser(description="Compiled Inference Chat — No Spoon")
    parser.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--hybrid", default="auto", choices=["auto", "gpu", "hybrid"])
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--system", type=str, default="")
    args = parser.parse_args()

    print(f"Loading {args.model}...")
    engine = CompiledInference(args.model, hybrid=args.hybrid, system_prompt=args.system)
    print(f"Ready: {engine}")
    print()
    print("Commands:")
    print("  /compile <id> <text>   — compile text into persistent memory")
    print("  /compile_file <path>   — compile a file")
    print("  /compiled              — list compiled states")
    print("  /stats                 — show engine stats")
    print("  /clear                 — clear everything")
    print("  /quit                  — exit")
    print()

    while True:
        try:
            user = input("\033[1;36mYou:\033[0m ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not user:
            continue

        if user.startswith("/quit"):
            break
        elif user.startswith("/compile_file "):
            path = user[14:].strip()
            if not os.path.exists(path):
                print(f"\033[1;31mFile not found: {path}\033[0m")
                continue
            with open(path) as f:
                text = f.read()
            cid = os.path.basename(path)
            state = engine.compile(cid, text)
            print(f"\033[1;33m[Compiled '{cid}': {state.n_tokens} tokens, {state.size_bytes/1e6:.1f} MB]\033[0m")
            continue
        elif user.startswith("/compile "):
            parts = user[9:].strip().split(" ", 1)
            if len(parts) < 2:
                print("\033[1;31mUsage: /compile <id> <text>\033[0m")
                continue
            state = engine.compile(parts[0], parts[1])
            print(f"\033[1;33m[Compiled '{parts[0]}': {state.n_tokens} tokens]\033[0m")
            continue
        elif user == "/compiled":
            if not engine.library:
                print("  (nothing compiled)")
            for cid, state in engine.library.items():
                print(f"  {cid}: {state.n_tokens} tokens ({state.size_bytes/1e6:.1f} MB)")
            continue
        elif user == "/stats":
            s = engine.stats
            print(f"  Turns: {s.turns} | Compiled: {s.compiled_states} states, "
                  f"{s.compiled_tokens} tokens ({s.compiled_bytes/1e6:.1f} MB) | "
                  f"Tools: {s.tool_calls} | Last: {s.last_tps:.1f} tok/s | Mode: {s.mode}")
            continue
        elif user == "/clear":
            engine.clear()
            print("\033[1;33m[Cleared all compiled state and history]\033[0m")
            continue

        print("\033[1;32mAssistant:\033[0m ", end="", flush=True)
        for token in engine.chat(user, max_tokens=args.max_tokens, stream=True,
                                  temperature=args.temperature):
            print(token, end="", flush=True)
        print()
        s = engine.stats
        print(f"\033[90m[{s.last_tokens} tok, {s.last_tps:.1f} tok/s, "
              f"{s.compiled_states} compiled, {s.turns} turns]\033[0m")
        print()

if __name__ == "__main__":
    main()
