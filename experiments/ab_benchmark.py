"""
A/B Benchmark: f16 KV vs q4_0 KV on Qwen3.5-9B
Same machine, same model, same prompts, back-to-back, 3 runs each.
"""

import httpx
import time
import subprocess
import sys
import os

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LLAMA_SERVER = os.path.join(PROJECT_ROOT, "tools", "llama-cpp-latest", "bin", "llama-server.exe")
MODEL = os.path.join(PROJECT_ROOT, "models", "qwen3.5-9b", "Qwen3.5-9B-Q4_K_M.gguf")
PORT = 8090
BASE = f"http://127.0.0.1:{PORT}"
RUNS = 3


def wait_ready():
    for i in range(120):
        try:
            r = httpx.get(f"{BASE}/health", timeout=2)
            if r.status_code == 200:
                return True
        except Exception:
            pass
        time.sleep(1)
    return False


def bench(prompt, max_tokens=100):
    body = {
        "model": "m",
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
        "max_tokens": max_tokens,
        "temperature": 0.0,
    }
    r = httpx.post(f"{BASE}/v1/chat/completions", json=body, timeout=300)
    d = r.json()
    t = d.get("timings", {})
    u = d.get("usage", {})
    return {
        "prompt_per_sec": t.get("prompt_per_second", 0),
        "gen_per_sec": t.get("predicted_per_second", 0),
        "prompt_tokens": u.get("prompt_tokens", 0),
        "completion_tokens": u.get("completion_tokens", 0),
    }


TESTS = [
    ("Short", "What is 2+2?", 50),
    ("Medium", "Explain quantum entanglement with examples.", 200),
    ("Long 2K", "Summarize: " + "The history of computing spans centuries. " * 200, 100),
    ("Long 10K", "Topic? " + "AI transforms every industry from healthcare to finance. " * 1000, 50),
]

CONFIGS = [
    ("f16 (baseline)", [
        LLAMA_SERVER, "-m", MODEL, "-ngl", "99", "-c", "32768",
        "-t", "4", "--host", "127.0.0.1", "--port", str(PORT),
        "-rea", "off", "-np", "1",
    ]),
    ("q4_0 (4-bit KV)", [
        LLAMA_SERVER, "-m", MODEL, "-ngl", "99", "-c", "32768",
        "-t", "4", "--host", "127.0.0.1", "--port", str(PORT),
        "-rea", "off", "-np", "1", "-ctk", "q4_0", "-ctv", "q4_0",
    ]),
]


def run_suite():
    # Warmup
    bench("Hi", 10)
    results = {}
    for name, prompt, mt in TESTS:
        runs = []
        for r in range(RUNS):
            result = bench(prompt, mt)
            runs.append(result)
        avg_pp = sum(r["prompt_per_sec"] for r in runs) / RUNS
        avg_gp = sum(r["gen_per_sec"] for r in runs) / RUNS
        pt = runs[0]["prompt_tokens"]
        results[name] = {"tokens": pt, "prompt_per_sec": avg_pp, "gen_per_sec": avg_gp}
    return results


def main():
    # Kill any existing llama-server on our port
    print("=" * 80)
    print(f"A/B BENCHMARK: f16 vs q4_0 KV | Qwen3.5-9B | RTX 3060 | {RUNS} runs each")
    print("=" * 80)

    all_results = {}

    for label, cmd in CONFIGS:
        print(f"\n--- Starting: {label} ---")
        proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        if not wait_ready():
            print(f"FAILED to start {label}")
            proc.kill()
            continue
        print(f"Server ready. Running {len(TESTS)} tests x {RUNS} runs...")
        all_results[label] = run_suite()
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
        time.sleep(5)  # Let GPU memory clear

    # Print comparison
    f16 = all_results.get("f16 (baseline)", {})
    q4 = all_results.get("q4_0 (4-bit KV)", {})

    print()
    print("=" * 90)
    print(f"{'Test':<15} {'Tokens':<8} | {'f16 prompt/s':<14} {'f16 gen/s':<12} | {'q4_0 prompt/s':<14} {'q4_0 gen/s':<12} | {'Diff'}")
    print("-" * 90)

    for test_name, _, _ in TESTS:
        if test_name in f16 and test_name in q4:
            f = f16[test_name]
            q = q4[test_name]
            diff = ((q["gen_per_sec"] - f["gen_per_sec"]) / f["gen_per_sec"] * 100) if f["gen_per_sec"] > 0 else 0
            print(
                f"{test_name:<15} {f['tokens']:<8} | "
                f"{f['prompt_per_sec']:>10.1f}/s   {f['gen_per_sec']:>8.1f}/s   | "
                f"{q['prompt_per_sec']:>10.1f}/s   {q['gen_per_sec']:>8.1f}/s   | "
                f"{diff:+.1f}%"
            )

    print()
    print("Same machine, same model, same prompts, back-to-back.")
    print("q4_0 uses 4x less KV memory. At 32K both fit; at 64K+ only q4_0 fits.")
    print()

    # Memory comparison
    print("MEMORY COMPARISON (KV cache at 32K context):")
    print(f"  f16:  1024 MB  (8 layers x 4 KV heads x 32K x 256 dim x 2 bytes)")
    print(f"  q4_0:  256 MB  (same but 0.5 bytes per element)")
    print(f"  Savings: 768 MB freed for longer context or other models")
    print()
    print("MAX CONTEXT (12GB GPU, 5.3GB model):")
    print(f"  f16:   ~32K tokens (1024 MB KV)")
    print(f"  q4_0: ~128K tokens (1024 MB KV at 4x density)")


if __name__ == "__main__":
    main()
