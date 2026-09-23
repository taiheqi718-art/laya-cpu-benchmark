# What does "33 ms" actually cost? Laya on a CPU

A CPU-only latency study of `laya-multilingual`. No GPU numbers here by design — the
question is what happens when you take the published GPU figure at face value and deploy
a 322M model the way its size invites you to.

[Laya](https://huggingface.co/convaiinnovations/laya) is a 322M non-autoregressive
decision model: give it a state and some typed questions, get back calibrated
probabilities in one forward pass, no text generation. The number everybody repeats is
**32.8 ms**.

That number is from a **Tesla T4 GPU**. The whole point of a 322M model is that you
should not need a GPU — but there is no CPU latency table anywhere in the model card.

The repo does contain one CPU figure, in
`research/results/cpu_51_language_sweep.json`: `1392.5 ms_per_case` at 4 threads. It is
not in the model card, and a "case" is not defined anywhere a user would look (reading
`research/scripts/bench_local.py`, it is one `(state, questions)` pair, with a varying
number of questions per case). The 32.8 ms and the 1392.5 ms cannot be reconciled from
public information.

This repo measures the whole surface with one method so the numbers are comparable.

## TL;DR

On an Intel i9-14900HX (24 cores / 32 threads, 32 GB), `laya-multilingual` in FP32 via
the official `laya` package, measured in **steady state** on Chinese input:

| state length | 1 question | 10 questions | 50 questions |
|---|---|---|---|
| 71 tokens | 50 ms | 401 ms | 4,105 ms |
| 261 tokens | 307 ms | 2,150 ms | 9,103 ms |
| 927 tokens | 1,315 ms | 10,199 ms | 40,131 ms |

Against the published T4 table (32.8 / 72.3 / 337 ms), at a 261-token state that is
**9.4x / 30x / 27x** slower. At a 927-token state — still inside the model's 1024-token
window — a single question costs **1.3 seconds**.

A realistic CPU call is **0.3–2 seconds**, not 33 milliseconds.

Single figures carry a **+/-13%** error bar (see *Known limitations*); the ratios between
cells are tighter.

## Four findings

### 1. GPU batching economics do not transfer to CPU

The published T4 table shows 1 question at 32.8 ms and 10 questions at 72.3 ms: **10x the
questions for 2.2x the time**. That near-free batching is a big part of Laya's appeal —
decompose a judgement into ten sub-questions and pay almost nothing.

On CPU, cost per question is essentially flat:

| state | 1 q | 5 q | 10 q | 25 q | 50 q |
|---|---|---|---|---|---|
| 261 tokens | 307 ms/q | 244 ms/q | 215 ms/q | 200 ms/q | 182 ms/q |
| 551 tokens | 581 ms/q | 532 ms/q | 530 ms/q | 498 ms/q | 531 ms/q |

Batching buys roughly 20–30% going from 1 to 10 questions and nothing after that. The
reason is structural: a GPU at batch 1 is mostly idle, so batching fills it; a CPU is
already saturated.

This matters for design. `laya/agent.py` builds **one sequence per question** — the state
is re-encoded for every question — so compute scales as `questions x sequence_length`.
On GPU that is hidden by parallelism. On CPU you pay all of it.

### 2. Latency is roughly linear in context, and context is the dominant term

At 10 questions: 401 ms (71 tok) -> 1,094 (150) -> 2,150 (261) -> 5,297 (551) ->
10,199 (927). Per total token processed this drifts from 0.56 to 1.10 ms, consistent with
attention's quadratic term starting to bite at the long end.

Practically: **keep the state short**. Going from a 927-token state to a 261-token state
is a 4.7x speedup at the same question count — far more leverage than any batching.

### 3. A short benchmark measures turbo boost, not your machine

This is the one that surprised us, and it generalises beyond Laya.

Running one fixed config (261-token state, 10 questions, 24 threads) continuously from an
idle machine, logging every call with no warmup discarded:

| elapsed | median latency |
|---|---|
| first call | 1,135 ms |
| 0–15 s | 1,135 ms |
| 15–30 s | 1,373 ms |
| 30–60 s | 1,624 ms |
| 60–120 s | 2,658 ms |
| 120–240 s | 2,654 ms |

**2.34x between the first 15 seconds and the plateau, and the decay takes about 90
seconds** — longer than most benchmark runs take in total. A script that warms up and
takes ten samples finishes entirely inside the turbo window and reports a number
production will never see.

The effect is load-dependent: light configs (1 question, short state) never saturate the
package and show no gap, while heavy configs show ~2x. So a benchmark built from light
configs looks trustworthy and a benchmark built from heavy ones is optimistic by half.

Note this is **not** thermal throttling. A 180 s sustained run drifted by -0.9%. Once the
package settles, it stays settled. What takes a long time is *recovery*.

`scripts/repeat.py` measures that directly: soak the machine for 60 s, idle for a varying
period, then measure. Burst capacity comes back as a function of idle time, steady-state
performance does not move:

| idle before measuring | burst half | steady half | ratio |
|---|---|---|---|
| 15 s | 2,577 ms | 2,629 ms | 1.02 |
| 30 s | 2,297 ms | 2,333 ms | 1.02 |
| 60 s | 1,956 ms | 2,528 ms | 1.29 |
| 120 s | 2,512 ms | 2,389 ms | 0.95 |
| 240 s | 1,172 ms | 2,331 ms | 1.99 |
| 480 s | 1,208 ms | 2,999 ms | 2.48 |

**Roughly four minutes of idle to fully restore burst performance**, while the steady-state
figure stays put. Across those six controlled repetitions:

- steady-state spread: **1.29x** (2,331–2,999 ms)
- burst spread: **2.20x** (1,172–2,577 ms)

Steady state is 1.7x more repeatable, and it is the number production will actually see.
That is why every figure in the TL;DR is measured that way: each config runs back to back
for at least 60 s and at least 3 calls, and the reported figure is the median of the calls
in the **last half** of that window.

### 4. More than 8 threads buys nothing

Reference config, steady state, thread count as the only variable:

| threads | latency | speedup vs 1 |
|---|---|---|
| 1 | 6,606 ms | 1.0x |
| 2 | 4,026 ms | 1.6x |
| 4 | 2,159 ms | 3.1x |
| 8 | 1,701 ms | 3.9x |
| 16 | 2,031 ms | 3.3x |
| 24 | 1,675 ms | 3.9x |

24 cores buy 3.9x. Most of that arrives by 4 threads and all of it by 8; 16 measured
*worse* than 8. The workload is memory-bound well before it is core-bound, and this is a
hybrid CPU (8 P-cores + 16 E-cores) where extra threads land on slower cores.

Deployment consequence: **do not give one Laya process 24 threads.** Three 8-thread
processes on a 24-core box will serve roughly three times the throughput of one 24-thread
process.

A 32-thread measurement came out at 3,247 ms, but a drift control taken immediately after
had also degraded by 88%, so that cell is unreliable and is excluded rather than reported
as an oversubscription result.

## Aside: does Chinese cost more?

Latency is a function of token count, so for a zh deployment the question is how many
tokens the same meaning costs under mmBERT's 256k multilingual vocab. On a matched pair of
paragraphs (the same support ticket written in each language):

| | characters | tokens | chars/token |
|---|---|---|---|
| Chinese | 166 | 111 | 1.50 |
| English | 445 | 96 | 4.64 |

**Chinese costs 1.16x the tokens for the same content** — a real penalty but a mild one,
and far smaller than the 2–4x that single-byte-per-character intuitions would suggest. The
256k vocab is doing its job.

Latency itself should not care which language the tokens came from: the encoder sees token
ids, and their identity does not change the FLOPs. A matched spot-check at 256 and 512
token budgets is consistent with that — outside two measurements taken at the extreme ends
of the sweep, every cell landed between 1.06 and 1.32 ms per token with zh and en fully
overlapping. It is not a strong test, though: the A-B/B-A agreement check flagged 1.24–1.34x
drift on two cells, which is larger than any language effect it was trying to detect. Recorded
in `results/lang_zh_vs_en.json`, reported here as "consistent with no effect", not as a
measurement.

The practical takeaway stands on the tokenizer result alone: **budget 1.16x the latency for
Chinese, because you are paying for 1.16x the tokens.**

## Method

- Only `agent.predict()` is timed. Model load and tokenizer construction are excluded.
- CUDA runs call `torch.cuda.synchronize()` around each timed call.
- Steady-state mode: run back to back for >= 60 s and >= 3 calls, report the median of
  the last half of the window.
- Sequence lengths are **measured**, not intended: read back from
  `result["usage"]["input_tokens"]`.
- Questions cycle through all three primitives (`noul`, `choice`, `score`), matching the
  mix in upstream's own benchmark script.
- Input is Chinese, because upstream publishes no zh latency data and CJK tokenization
  differs enough from English to matter. An English comparison is pending.
- Sweeps that vary one factor visit their levels in **randomized order** and re-measure a
  fixed **control config** at the start, middle and end. If the controls disagree, the
  machine drifted during the sweep and the affected cells are thrown out rather than
  reported. This is not optional on a laptop: the first thread sweep run produced a clean
  monotonic-looking table that was entirely an artifact of drift, and only the controls
  caught it.

### Known limitations

- One machine so far. See *Contribute a machine* below.
- **Error bar: +/-13% on any single steady-state figure** (1.29x spread over six controlled
  repetitions of the reference config). Ratios between cells measured within one sweep are
  tighter than that and are the more trustworthy reading.
- Figures measured with the short-burst method earlier in this study ranged over 1.9x for
  one config. Those runs are kept in `results/bench_cpu_zh.json` as an illustration of
  finding 3, and none of them are quoted as results.
- The `927 tok x 5 q` cell (8,406 ms, 1,681 ms/q) is out of line with its neighbours
  (n=10 gives 1,020 ms/q) and only got 9 reps. Probably noise; not yet re-run.
- The 32-thread cell is excluded (see finding 4).
- FP32 only. The official package hard-codes FP32 on CPU and disables autocast there
  (`laya/agent.py`: `elif self.device.type in ("cpu", "mps"): self.dtype = torch.float32`,
  and `use_amp = self.device.type == "cuda"`). Community INT8 ONNX exports exist (~325 MB)
  but the official package does not use them, so "what you get out of the box" and "what
  is achievable" are different questions. Only the first is measured here.

## Environment

| | |
|---|---|
| CPU | Intel i9-14900HX, 24 cores / 32 threads |
| RAM | 32 GB |
| OS | Windows 11 |
| Python | 3.13.5 |
| torch | 2.14.0+cpu |
| laya | 0.3.5 |
| checkpoint | `convaiinnovations/laya`, subfolder `multilingual` |
| encoder | `jhu-clsp/mmBERT-base` |
| params | 321.9M total / 196.6M embedding / **125.3M non-embedding** |
| max_len | 1024 (head_max_len 256) |

## Reproduce

```bash
python -m venv .venv
.venv/bin/pip install laya psutil
# steady-state latency surface: context length x question count
python scripts/bench.py --device cpu --lang zh --only grid --precool 90 --cooldown 0 --steady-s 60

# thread scaling, with randomized order and drift controls
python scripts/bench.py --device cpu --lang zh --only threads --precool 60 --cooldown 10

# turbo decay curve: every call logged from the first, nothing discarded
python scripts/turbo.py

# repeatability and recovery-from-load
python scripts/repeat.py
```

Results land in `results/` as JSON with full metadata (CPU model, thread count, library
versions, measurement policy), so contributed runs are self-describing.

Leave the machine alone while these run. That sounds like superstition; finding 3 is why
it is not.

## Contribute a machine

The interesting version of this table has many rows. If you run it, open a PR with your
`results/*.json` — the metadata block records your CPU, thread count and library versions
automatically. Mac (MPS), older Xeons, Ryzen, and anything ARM are all missing.
