"""
Turbo-decay probe: is a short benchmark measuring the machine, or measuring turbo boost?

The same config (ctx~256, n=10, 24 threads) measured 1696 ms inside a short grid sweep,
2415 ms in steady state, and 3314 ms after 15 minutes of load. Thermal drift within the
steady-state run was -0.9%, so throttling does not explain the spread.

This logs EVERY call from the very first one after a long idle, with no warmup discarded,
so the decay curve itself is visible. If turbo is the cause, early calls are fast and the
curve settles onto a plateau within the first tens of seconds.
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
from bench import build_questions, build_state  # same inputs as the main benchmark

PRECOOL_S = 90
RUN_S = 240
CTX = 256
NQ = 10


def main() -> None:
    print(f"loading ... (precool {PRECOOL_S}s after load)", flush=True)
    agent = laya.load("convaiinnovations/laya", subfolder="multilingual", device="cpu")
    state = build_state(agent.tok, CTX, "zh")
    questions = build_questions(NQ)

    print(f"idling {PRECOOL_S}s so the package power budget fully recovers ...", flush=True)
    time.sleep(PRECOOL_S)

    print(f"running {RUN_S}s, logging every call from the first:", flush=True)
    samples = []
    t_start = time.perf_counter()
    i = 0
    while (time.perf_counter() - t_start) < RUN_S:
        t0 = time.perf_counter()
        agent.predict(state, questions)
        ms = (time.perf_counter() - t0) * 1000.0
        elapsed = time.perf_counter() - t_start
        samples.append({"iter": i, "elapsed_s": round(elapsed, 2), "ms": round(ms, 2)})
        if i < 12 or i % 10 == 0:
            print(f"  iter {i:<4} t={elapsed:>6.1f}s  {ms:>9.2f} ms", flush=True)
        i += 1

    def window(lo, hi):
        vals = [s["ms"] for s in samples if lo <= s["elapsed_s"] < hi]
        return round(statistics.median(vals), 1) if vals else None

    summary = {
        "first_call_ms": samples[0]["ms"],
        "p50_0_15s": window(0, 15),
        "p50_15_30s": window(15, 30),
        "p50_30_60s": window(30, 60),
        "p50_60_120s": window(60, 120),
        "p50_120_240s": window(120, 240),
        "n_calls": len(samples),
    }
    print("\n=== decay profile (median ms by elapsed window) ===")
    for k, v in summary.items():
        print(f"  {k:<16} {v}")

    plateau = summary["p50_120_240s"]
    early = summary["p50_0_15s"]
    if plateau and early:
        print(f"\n  early burst vs plateau: {early} ms -> {plateau} ms "
              f"({(plateau / early - 1) * 100:+.1f}%)")

    out = {
        "meta": {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "precool_s": PRECOOL_S, "run_s": RUN_S, "ctx": CTX, "n_questions": NQ,
            "threads": torch.get_num_threads(), "torch": torch.__version__,
            "device": "cpu", "dtype": "torch.float32",
            "note": "no warmup discarded; every call logged from first",
        },
        "summary": summary,
        "samples": samples,
    }
    out_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "turbo_decay_cpu_zh.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"\nwrote {path}")


if __name__ == "__main__":
    sys.exit(main())
