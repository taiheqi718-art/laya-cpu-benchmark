"""
Does Chinese cost more than English on the same content?

Latency is driven by token count, so the practical question for a zh deployment is how
many tokens the same meaning costs under mmBERT's 256k multilingual vocab. This measures
that, then spot-checks that latency really does depend only on token count and not on the
language itself.
"""
from __future__ import annotations

import json
import os
import sys
import time
from datetime import datetime, timezone

import torch

import laya
from bench import UNIT_EN, UNIT_ZH, build_questions, build_state, time_steady

STEADY_S = 45


def main() -> None:
    agent = laya.load("convaiinnovations/laya", subfolder="multilingual", device="cpu")
    tok = agent.tok

    print("=== tokenization of semantically equivalent paragraphs ===")
    stats = {}
    for name, text in (("zh", UNIT_ZH), ("en", UNIT_EN)):
        ids = tok(text, add_special_tokens=False)["input_ids"]
        chars = len(text)
        stats[name] = {
            "chars": chars,
            "tokens": len(ids),
            "chars_per_token": round(chars / len(ids), 2),
        }
        print(f"  {name}: {chars:>4} chars -> {len(ids):>4} tokens "
              f"({chars / len(ids):.2f} chars/token)")

    ratio = stats["zh"]["tokens"] / stats["en"]["tokens"]
    print(f"\n  zh/en token ratio for the same content: {ratio:.3f}")
    if ratio < 1:
        print(f"  -> Chinese is {1 / ratio:.2f}x CHEAPER per unit of meaning here")
    else:
        print(f"  -> Chinese is {ratio:.2f}x more expensive per unit of meaning here")

    # A first attempt at this ran the four configs back to back with no cooldown and
    # produced en@512 (5196 tok) FASTER than en@256 (2956 tok) -- physically impossible,
    # and a textbook instance of finding 3 in the README. So: cool down between configs,
    # and visit them twice in opposite orders. If a config measures the same in both
    # passes, the number is real; if not, the sweep drifted and nothing is reported.
    print("\n=== latency at matched token budgets (10 questions, A-B / B-A) ===")
    configs = [("zh", 256), ("en", 256), ("zh", 512), ("en", 512)]
    prepared = []
    for lang, target in configs:
        state = build_state(tok, target, lang)
        n_tok = len(tok(state["body"], add_special_tokens=False)["input_ids"])
        prepared.append((lang, target, state, n_tok))

    rows = []
    for pass_i, order in enumerate((prepared, list(reversed(prepared)))):
        print(f"  -- pass {pass_i} ({'forward' if pass_i == 0 else 'reversed'}) --")
        for lang, target, state, n_tok in order:
            time.sleep(30)
            r = time_steady(agent, state, build_questions(10), "cpu", STEADY_S)
            rows.append({"pass": pass_i, "lang": lang, "target_ctx": target,
                         "state_tokens": n_tok, **r})
            print(f"    {lang} target~{target:<4} state={n_tok:>4} tok  "
                  f"total={r['input_tokens']:>6} tok  ->  p50 {r['p50_ms']:>8.2f} ms "
                  f"({r['p50_ms'] / r['input_tokens']:.3f} ms/tok)")

    print("\n  -- agreement between passes --")
    agree = True
    for lang, target, _, _ in prepared:
        vals = [r["p50_ms"] for r in rows if r["lang"] == lang and r["target_ctx"] == target]
        if len(vals) == 2:
            spread = max(vals) / min(vals)
            ok = spread < 1.20
            agree &= ok
            print(f"    {lang}@{target:<4}: {vals[0]:.0f} vs {vals[1]:.0f} ms  "
                  f"({spread:.2f}x) {'ok' if ok else 'DRIFTED'}")
    print(f"\n  verdict: {'passes agree, numbers usable' if agree else 'sweep drifted, do not report'}")

    out = {
        "meta": {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "steady_s": STEADY_S, "threads": torch.get_num_threads(),
            "torch": torch.__version__, "device": "cpu",
            "encoder": agent.cfg.get("encoder"),
        },
        "tokenization": stats,
        "zh_en_token_ratio": round(ratio, 4),
        "rows": rows,
    }
    out_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
    path = os.path.join(out_dir, "lang_zh_vs_en.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"\nwrote {path}")


if __name__ == "__main__":
    sys.exit(main())
