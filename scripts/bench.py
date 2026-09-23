"""
Laya latency benchmark: what "33 ms" actually costs on hardware you own.

Upstream publishes one GPU table (Tesla T4) and one buried CPU figure
(research/results/cpu_51_language_sweep.json: 4 threads, 1392.5 ms/case) with no
stated context length, question count, warmup or repetition policy. This script
measures the whole surface with one method so the numbers are comparable.

Method
  - Only `agent.predict()` is timed; model load and tokenizer construction are excluded.
  - CUDA runs call torch.cuda.synchronize() before and after each timed call.
  - Warmup repetitions are discarded, then repetitions run until either MAX_REPS or
    TIME_BUDGET_S, whichever comes first (slow configs would otherwise dominate runtime).
  - Reported statistic is the median (p50); p95 and min are recorded too.
  - Actual sequence length is read back from result["usage"]["input_tokens"], so the
    reported token counts are measured rather than intended.
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import statistics
import sys
import time
from datetime import datetime, timezone
from typing import Any, Dict, List

import torch

import laya

WARMUP = 3
MIN_REPS = 3
MAX_REPS = 15
TIME_BUDGET_S = 20.0

# ---------------------------------------------------------------- inputs

# A realistic support-ticket paragraph, used as a repeatable unit to hit target
# context lengths. Chinese is deliberate: upstream publishes no zh latency data,
# and CJK tokenization differs enough from English to matter.
UNIT_ZH = (
    "客户在九月十二日提交了工单，说明自己的企业账户在同一个结算周期内被重复扣款两次，"
    "订单号分别是 4411 和 4412，金额各为 1280 元。对方已经提供了银行流水截图作为凭证，"
    "并且明确要求在二十四小时内完成退款，否则将向监管部门投诉。客服初步核查后发现，"
    "该账户在本月确实存在两笔相同金额的扣款记录，但其中一笔的支付网关返回码异常。"
)
UNIT_EN = (
    "The customer filed a ticket on September 12 stating that their enterprise account "
    "was charged twice within the same billing cycle, on orders 4411 and 4412, for 1280 "
    "CNY each. They attached bank statements as evidence and demanded a refund within "
    "twenty-four hours, failing which they intend to escalate to the regulator. A first "
    "pass by support confirmed two identical charges this month, one of which carried an "
    "anomalous gateway response code."
)


def build_state(tok, target_tokens: int, lang: str) -> Dict[str, str]:
    """Repeat/trim the unit paragraph until it tokenizes to ~target_tokens."""
    unit = UNIT_ZH if lang == "zh" else UNIT_EN
    n_unit = len(tok(unit, add_special_tokens=False)["input_ids"])
    reps = max(1, round(target_tokens / n_unit))
    text = unit * reps
    ids = tok(text, add_special_tokens=False)["input_ids"]
    if len(ids) > target_tokens:
        text = tok.decode(ids[:target_tokens])
    return {"body": text}


def build_questions(n: int) -> Dict[str, Dict[str, Any]]:
    """n questions cycling through all three primitives, matching upstream's mix."""
    out: Dict[str, Dict[str, Any]] = {}
    for i in range(n):
        kind = i % 3
        if kind == 0:
            out[f"q{i}_noul"] = {
                "type": "noul",
                "instructions": "Does the sender ask for money back?",
            }
        elif kind == 1:
            out[f"q{i}_choice"] = {
                "type": "choice",
                "instructions": "Which team should handle this?",
                "criteria": {
                    "billing": "invoices, payments, refunds",
                    "technical": "bugs and outages",
                    "sales": "pricing and plans",
                },
            }
        else:
            out[f"q{i}_score"] = {
                "type": "score",
                "instructions": "How urgent is this?",
                "criteria": ["not urgent", "normal", "urgent", "critical"],
            }
    return out


# ---------------------------------------------------------------- timing

def time_steady(agent, state, questions, device: str, steady_s: float) -> Dict[str, Any]:
    """Measure a config's SUSTAINED cost, not its turbo-boost cost.

    Measured on this machine (see results/turbo_decay_cpu_zh.json): an idle laptop runs
    the same config at 1135 ms for the first 15 s and settles at 2654 ms after ~90 s, a
    2.34x spread. A benchmark that warms up and takes ten samples finishes inside the
    turbo window and reports a number production will never see.

    So: run the config back to back for at least `steady_s` seconds and at least 3 calls,
    then report the median of the calls in the LAST HALF of that window, by which point
    the package power budget has settled.
    """
    cuda = device.startswith("cuda")

    def one() -> tuple[float, Any]:
        if cuda:
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        res = agent.predict(state, questions)
        if cuda:
            torch.cuda.synchronize()
        return (time.perf_counter() - t0) * 1000.0, res

    marks: List[tuple[float, float]] = []  # (elapsed_at_end, ms)
    last = None
    t_start = time.perf_counter()
    while True:
        ms, last = one()
        marks.append((time.perf_counter() - t_start, ms))
        if len(marks) >= 3 and (time.perf_counter() - t_start) >= steady_s:
            break
        if len(marks) >= 400:
            break

    total = marks[-1][0]
    tail = [ms for (t, ms) in marks if t >= total / 2] or [marks[-1][1]]
    tail_sorted = sorted(tail)
    head = [ms for (t, ms) in marks if t < total / 2] or tail
    return {
        "p50_ms": round(statistics.median(tail_sorted), 2),
        "p95_ms": round(tail_sorted[min(len(tail_sorted) - 1, int(0.95 * len(tail_sorted)))], 2),
        "min_ms": round(min(tail_sorted), 2),
        "mean_ms": round(statistics.fmean(tail_sorted), 2),
        "reps": len(marks),
        "reps_in_tail": len(tail),
        "window_s": round(total, 1),
        "burst_p50_ms": round(statistics.median(sorted(head)), 2),
        "turbo_ratio": round(statistics.median(tail_sorted) / statistics.median(sorted(head)), 3),
        "input_tokens": last["usage"]["input_tokens"],
    }


def time_config(agent, state, questions, device: str) -> Dict[str, Any]:
    cuda = device.startswith("cuda")

    def one_call() -> float:
        if cuda:
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        res = agent.predict(state, questions)
        if cuda:
            torch.cuda.synchronize()
        return (time.perf_counter() - t0) * 1000.0, res

    # One probe call tells us how expensive this config is, so warmup and repetition
    # counts can adapt. A fixed policy would spend minutes on the 50q x 900tok cell
    # and milliseconds on the 1q x 32tok cell.
    probe_ms, last = one_call()
    if probe_ms < 1000:
        warmup, min_reps = WARMUP, MIN_REPS
    elif probe_ms < 8000:
        warmup, min_reps = 1, 3
    else:
        warmup, min_reps = 0, 2

    for _ in range(warmup):
        _, last = one_call()

    samples: List[float] = []
    t_start = time.perf_counter()
    while len(samples) < MAX_REPS:
        ms, last = one_call()
        samples.append(ms)
        if len(samples) >= min_reps and (time.perf_counter() - t_start) > TIME_BUDGET_S:
            break

    samples.sort()
    return {
        "p50_ms": round(statistics.median(samples), 2),
        "p95_ms": round(samples[min(len(samples) - 1, int(0.95 * len(samples)))], 2),
        "min_ms": round(samples[0], 2),
        "mean_ms": round(statistics.fmean(samples), 2),
        "reps": len(samples),
        "input_tokens": last["usage"]["input_tokens"],
    }


def rss_mb() -> float:
    try:
        import psutil

        return round(psutil.Process().memory_info().rss / 1024 / 1024, 1)
    except Exception:
        return -1.0


# ---------------------------------------------------------------- sweeps

def sweep_grid(agent, tok, device, lang, q_counts, ctx_lens, cooldown: float = 0.0,
               steady_s: float = 0.0) -> List[Dict[str, Any]]:
    rows = []
    for ctx in ctx_lens:
        state = build_state(tok, ctx, lang)
        for n in q_counts:
            questions = build_questions(n)
            if cooldown:
                time.sleep(cooldown)
            if steady_s:
                r = time_steady(agent, state, questions, device, steady_s)
            else:
                r = time_config(agent, state, questions, device)
            row = {
                "sweep": "grid_steady" if steady_s else "grid",
                "device": device,
                "lang": lang,
                "target_ctx": ctx,
                "n_questions": n,
                "threads": torch.get_num_threads(),
                **r,
                "ms_per_question": round(r["p50_ms"] / n, 2),
                "rss_mb": rss_mb(),
            }
            rows.append(row)
            extra = ""
            if steady_s:
                extra = f", burst {row['burst_p50_ms']:.0f} ms, x{row['turbo_ratio']:.2f}"
            print(
                f"  ctx~{ctx:<5} n={n:<3} -> p50 {row['p50_ms']:>9.2f} ms   "
                f"({row['ms_per_question']:>7.2f} ms/q, {row['input_tokens']:>5} tok, "
                f"{row['reps']} reps{extra})",
                flush=True,
            )
    return rows


def sweep_threads(agent, tok, device, lang, ctx, n, thread_list,
                  cooldown: float, control_threads: int,
                  steady_s: float = 45.0) -> List[Dict[str, Any]]:
    """Thread sweep with thermal-drift control.

    A laptop that has already run the grid sweep is hot, and on this machine that alone
    moved an identical config from 1696 ms to 3314 ms. So: cool down between configs,
    visit thread counts in random order (decorrelates drift from the variable under test),
    and re-measure one fixed control config throughout so residual drift is visible in the
    data rather than silently folded into the result.
    """
    import random

    rows = []
    state = build_state(tok, ctx, lang)
    questions = build_questions(n)

    def control(slot: str) -> float:
        torch.set_num_threads(control_threads)
        time.sleep(cooldown)
        r = time_steady(agent, state, questions, device, steady_s)
        rows.append({
            "sweep": "thread_control", "device": device, "lang": lang,
            "target_ctx": ctx, "n_questions": n, "threads": control_threads,
            "slot": slot, **r, "rss_mb": rss_mb(),
        })
        print(f"  [control {slot:<6} t={control_threads}] p50 {r['p50_ms']:>9.2f} ms", flush=True)
        return r["p50_ms"]

    base = control("start")

    order = list(thread_list)
    random.Random(0).shuffle(order)
    for idx, t in enumerate(order):
        time.sleep(cooldown)
        torch.set_num_threads(t)
        r = time_steady(agent, state, questions, device, steady_s)
        rows.append({
            "sweep": "threads", "device": device, "lang": lang,
            "target_ctx": ctx, "n_questions": n, "threads": t,
            "visit_order": idx, **r,
            "ms_per_question": round(r["p50_ms"] / n, 2),
            "rss_mb": rss_mb(),
        })
        print(f"  threads={t:<3} -> p50 {r['p50_ms']:>9.2f} ms   ({r['reps']} reps)", flush=True)
        if idx == len(order) // 2:
            control("mid")

    end = control("end")
    drift = (end / base - 1) * 100 if base else 0.0
    print(f"  control drift over sweep: {drift:+.1f}%  "
          f"({base:.0f} ms -> {end:.0f} ms)", flush=True)
    return rows


def sweep_sustained(agent, tok, device, lang, ctx, n, seconds) -> List[Dict[str, Any]]:
    """Laptops throttle. Nobody publishes this; run one config until it hurts."""
    state = build_state(tok, ctx, lang)
    questions = build_questions(n)
    for _ in range(WARMUP):
        agent.predict(state, questions)

    rows, t_start = [], time.perf_counter()
    i = 0
    while (time.perf_counter() - t_start) < seconds:
        if device.startswith("cuda"):
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        agent.predict(state, questions)
        if device.startswith("cuda"):
            torch.cuda.synchronize()
        ms = (time.perf_counter() - t0) * 1000.0
        rows.append({
            "sweep": "sustained",
            "device": device,
            "lang": lang,
            "target_ctx": ctx,
            "n_questions": n,
            "threads": torch.get_num_threads(),
            "iter": i,
            "elapsed_s": round(time.perf_counter() - t_start, 1),
            "ms": round(ms, 2),
        })
        i += 1
    first = statistics.median([r["ms"] for r in rows[: max(1, len(rows) // 10)]])
    last = statistics.median([r["ms"] for r in rows[-max(1, len(rows) // 10):]])
    print(f"  sustained {seconds}s: {len(rows)} iters, "
          f"first-decile p50 {first:.1f} ms -> last-decile p50 {last:.1f} ms "
          f"({(last / first - 1) * 100:+.1f}%)", flush=True)
    return rows


# ---------------------------------------------------------------- main

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--lang", default="zh", choices=["zh", "en"])
    ap.add_argument("--subfolder", default="multilingual")
    ap.add_argument("--model", default="convaiinnovations/laya")
    ap.add_argument("--tag", default="")
    ap.add_argument("--sustained-s", type=int, default=180)
    ap.add_argument("--skip-sustained", action="store_true")
    ap.add_argument("--only", choices=["grid", "threads", "sustained"], default=None,
                    help="run a single sweep instead of all three")
    ap.add_argument("--cooldown", type=float, default=20.0,
                    help="idle seconds between configs, to bleed off thermal drift")
    ap.add_argument("--precool", type=float, default=0.0,
                    help="idle seconds before the first measurement")
    ap.add_argument("--steady-s", type=float, default=0.0,
                    help="if set, measure each grid cell in steady state: run it back to "
                         "back for this many seconds and report the median of the last "
                         "half. 0 keeps the short-burst (turbo) method.")
    args = ap.parse_args()

    print(f"loading {args.model} [{args.subfolder}] on {args.device} ...", flush=True)
    t0 = time.perf_counter()
    agent = laya.load(args.model, subfolder=args.subfolder or None, device=args.device)
    load_s = round(time.perf_counter() - t0, 2)
    tok = agent.tok
    device = str(agent.device)
    print(f"loaded in {load_s}s  device={device} dtype={agent.dtype}\n", flush=True)

    n_params = sum(p.numel() for p in agent.model.parameters())
    n_embed = sum(p.numel() for n, p in agent.model.named_parameters() if "embed" in n.lower())

    meta = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "model": args.model,
        "subfolder": args.subfolder,
        "device": device,
        "dtype": str(agent.dtype),
        "lang": args.lang,
        "load_s": load_s,
        "max_len": agent.cfg.get("max_len"),
        "head_max_len": agent.cfg.get("head_max_len"),
        "encoder": agent.cfg.get("encoder"),
        "params_total_m": round(n_params / 1e6, 1),
        "params_embedding_m": round(n_embed / 1e6, 1),
        "params_non_embedding_m": round((n_params - n_embed) / 1e6, 1),
        "torch": torch.__version__,
        "laya": getattr(laya, "__version__", "unknown"),
        "python": platform.python_version(),
        "os": f"{platform.system()} {platform.release()}",
        "cpu": platform.processor(),
        "cpu_count_logical": os.cpu_count(),
        "torch_threads_default": torch.get_num_threads(),
        "warmup": WARMUP,
        "min_reps": MIN_REPS,
        "max_reps": MAX_REPS,
        "time_budget_s": TIME_BUDGET_S,
    }
    if device.startswith("cuda"):
        meta["gpu"] = torch.cuda.get_device_name(0)
        meta["gpu_capability"] = "sm_%d%d" % torch.cuda.get_device_capability(0)
        meta["cuda"] = torch.version.cuda

    meta["cooldown_s"] = args.cooldown
    meta["precool_s"] = args.precool
    meta["steady_s"] = args.steady_s
    meta["method"] = "steady_state" if args.steady_s else "short_burst"
    rows: List[Dict[str, Any]] = []
    only = args.only

    if args.precool:
        print(f"pre-cooling {args.precool:.0f}s ...", flush=True)
        time.sleep(args.precool)

    if only in (None, "grid"):
        print("[1/3] grid sweep: context length x question count", flush=True)
        rows += sweep_grid(
            agent, tok, device, args.lang,
            q_counts=[1, 5, 10, 25, 50],
            ctx_lens=[32, 128, 256, 512, 900],
            cooldown=args.cooldown,
            steady_s=args.steady_s,
        )

    if only in (None, "threads"):
        if not device.startswith("cuda"):
            print("\n[2/3] thread sweep (ctx~256, n=10)", flush=True)
            cpu_n = os.cpu_count() or 8
            cands = sorted({1, 2, 4, 8, 16, 24, cpu_n})
            rows += sweep_threads(
                agent, tok, device, args.lang, 256, 10,
                [t for t in cands if t <= cpu_n],
                cooldown=args.cooldown,
                control_threads=meta["torch_threads_default"],
            )
            torch.set_num_threads(meta["torch_threads_default"])
        else:
            print("\n[2/3] thread sweep skipped on cuda", flush=True)

    if only in (None, "sustained") and not args.skip_sustained:
        print(f"\n[3/3] sustained run ({args.sustained_s}s, ctx~256, n=10)", flush=True)
        rows += sweep_sustained(agent, tok, device, args.lang, 256, 10, args.sustained_s)
    elif only is None:
        print("\n[3/3] sustained run skipped", flush=True)

    tag = args.tag or f"{device.replace(':', '')}_{args.lang}"
    out_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"bench_{tag}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"meta": meta, "rows": rows}, f, indent=2, ensure_ascii=False)
    print(f"\nwrote {out_path}  ({len(rows)} rows)")


if __name__ == "__main__":
    sys.exit(main())
