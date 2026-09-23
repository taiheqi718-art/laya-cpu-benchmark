"""
Repeatability and recovery: how much does one number move, and what moves it?

Across this session the SAME config (261-token state, 10 questions, 24 threads, steady
state) has measured 1675, 1686, 1696, 2150, 2415, 2654 and 3175 ms -- a 1.9x spread. Any
single figure in the benchmark is worthless until that spread is characterised.

Two candidate causes were visible in earlier runs:
  - recovery from prior load is slow (a 60 s idle did not restore burst performance)
  - something varies per process launch (hybrid P-core / E-core placement is the suspect)

This script isolates the first: identical measurement, identical process, with the idle
period before each round as the only variable. If idle length explains the spread, the
rounds order themselves by idle time. If it does not, the residual is per-launch or
scheduler noise and gets reported as the benchmark's error bar.
"""
from __future__ import annotations

import json
import os
import statistics
import sys
import time
from datetime import datetime, timezone

import torch

import laya
from bench import build_questions, build_state, time_steady

CTX = 256
NQ = 10
STEADY_S = 45
IDLES = [15, 30, 60, 120, 240, 480]
LOAD_S = 60  # heavy load applied before each idle, so every round starts from the same place


def main() -> None:
    print("loading ...", flush=True)
    agent = laya.load("convaiinnovations/laya", subfolder="multilingual", device="cpu")
    state = build_state(agent.tok, CTX, "zh")
    questions = build_questions(NQ)
    threads = torch.get_num_threads()
    print(f"threads={threads}\n", flush=True)

    def soak(seconds: float) -> None:
        """Drive the package to a known hot state so each round starts identically."""
        t0 = time.perf_counter()
        while (time.perf_counter() - t0) < seconds:
            agent.predict(state, questions)

    rows = []
    for idle in IDLES:
        print(f"round idle={idle}s: soaking {LOAD_S}s ...", flush=True)
        soak(LOAD_S)
        print(f"  idling {idle}s ...", flush=True)
        time.sleep(idle)
        r = time_steady(agent, state, questions, "cpu", STEADY_S)
        rows.append({"idle_s": idle, "threads": threads, **r})
        print(f"  idle={idle:<4}s -> steady p50 {r['p50_ms']:>8.2f} ms   "
              f"(burst-half {r['burst_p50_ms']:.0f} ms, x{r['turbo_ratio']:.2f}, "
              f"{r['reps']} reps)", flush=True)

    vals = [r["p50_ms"] for r in rows]
    bursts = [r["burst_p50_ms"] for r in rows]
    summary = {
        "steady_min": min(vals), "steady_max": max(vals),
        "steady_median": round(statistics.median(vals), 2),
        "steady_spread_x": round(max(vals) / min(vals), 3),
        "burst_min": min(bursts), "burst_max": max(bursts),
        "burst_spread_x": round(max(bursts) / min(bursts), 3),
        "idle_correlates": None,
    }
    # Does idle length actually order the results?
    ordered = [r["p50_ms"] for r in sorted(rows, key=lambda r: r["idle_s"])]
    summary["idle_correlates"] = ordered == sorted(ordered, reverse=True)

    print("\n=== summary ===")
    for k, v in summary.items():
        print(f"  {k:<18} {v}")
    print("\n  If steady_spread_x is small, idle length is not the driver and the")
    print("  remaining variance is this benchmark's honest error bar.")

    out = {
        "meta": {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "ctx": CTX, "n_questions": NQ, "steady_s": STEADY_S,
            "soak_s": LOAD_S, "idles": IDLES, "threads": threads,
            "torch": torch.__version__, "device": "cpu", "dtype": "torch.float32",
        },
        "summary": summary,
        "rows": rows,
    }
    out_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "repeatability_cpu_zh.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"\nwrote {path}")


if __name__ == "__main__":
    sys.exit(main())
