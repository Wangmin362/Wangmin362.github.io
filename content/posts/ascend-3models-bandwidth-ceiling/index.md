---
title: "Same Vendor, Same Generation, One Suffix Apart: Ascend 910B3/910B4/910B4-1 on Real Hardware — 11% Apart on 4B, 309% Apart on 32B"
date: 2026-08-31
draft: true
tags: ["benchmark", "vLLM", "domestic-accelerators", "Ascend", "Huawei", "hardware-selection", "quantization", "goodput", "LLM-inference"]
summary: "Three Ascend boxes, three SKUs, one pinned image digest, 504 measured points. The headline: benchmarking with a 4B model gives you the opposite conclusion from a 32B model — the 910B4 is 11% slower on 4B and 309% slower on 32B, because its effective memory bandwidth flatlines at ~160 GB/s from just 2 GB of weights per card. Plus: I retract a conclusion I published in July — that 'fp16 is 15% slower than bf16' was a measurement-protocol artifact. The real gap is 3%."
ShowToc: true
---

> **TL;DR** — I had three Ascend boxes at once: 910B4 (32G), 910B3 (64G), 910B4-1 (64G). Same image digest, same methodology, 504 measured points. Five numbers worth remembering:
> **(1) Small models give you the opposite answer** — 910B4 vs 910B3 is **11%** slower on 4B, **309%** slower on 32B;
> **(2) The mechanism is an effective-bandwidth ceiling** — 910B4 flatlines at **~160 GB/s** from just 2 GB of weights per card, while the other two climb to **665 GB/s**;
> **(3) Of 8 Ascend-only tuning knobs, not one works across all three SKUs** — `weight_nz_mode` gives three different outcomes (+8.86% / −1.04% / −2.23%), and `enable_flashcomm1` buys +4~5% on two boxes but is dead on the third (all x3-verified);
> **(4) Quantization buys latency, not throughput** — W8A8 gives +13% throughput but **+80% goodput**; meanwhile the card with the *biggest* throughput gain (+29%) gains **zero** usable capacity;
> **(5) I'm retracting a July conclusion** — "Ascend fp16 is 15% slower than bf16" was a **protocol asymmetry**. Same box, same protocol, the real gap is **3%**.

---

## Why do this at all

In July I ran a nine-card cross-vendor sweep answering "how far apart are domestic accelerators?" This is different — I had three Ascend boxes at the same time, SKUs **910B4 / 910B3 / 910B4-1**, names one suffix apart.

Which lets me ask a sharper question:

> **Does a conclusion I tuned on one SKU still hold on another SKU from the same vendor and generation?**

For anyone doing adaptation work this matters far more than "who's #1." Because the real job is: whatever SKU sits in the customer's rack is the SKU you must deliver on.

---

## 1. How it was measured (reproducible)

The three boxes are **identical apart from the silicon** — that's the foundation the whole comparison rests on:

| | 183 | 122 | 15 |
|---|---|---|---|
| Chip | **910B4** `IT21HMDC_Bin6x` | **910B3** `IT21HMDC_Bin6` | **910B4-1** `IT21HMDC_Bin4_1` |
| HBM | **32 GB** | 64 GB | 64 GB |
| AICore | 20 @ **1650** MHz | 20 @ **1800** MHz | 20 @ **1650** MHz |
| driver | 25.5.1 | 26.0.rc1 | 26.0.rc1 |
| **Engine** | **vLLM 0.25.1, digest `cbe01380`** | same | same |

- **Models**: Qwen3-4B / 8B / 32B, byte-for-byte identical (shard sizes verified)
- **Load**: `vllm bench serve --dataset-name random`, baseline `in=512/out=128`
- **Protocol**: every run carries `--goodput ttft:2000 tpot:100` and `--num-warmups 5`; `num-prompts = concurrency × 8`; concurrency swept until two consecutive steps gain <3%; **every headline peak measured 3× independently, median reported**

### ⚠️ One methodology lesson up front

**Image tags lie. Pin the digest.** The images across these boxes:

| tag | digest |
|---|---|
| `nightly-releases-v0.25.1rc` | `cbe01380…` |
| `DeepSeekV4-flash-0731` | `cbe01380…` ← **same image, two unrelated tags** |
| `v0.23.0rc1` | `e964910…` |
| `nightly-releases-v0.23.0` | `dfa205b…` ← **both look like "0.23", actually two different images** |

In a cross-vendor sweep this is a footnote. In a **same-vendor three-SKU comparison** it's fatal — run three different engine versions and "is this difference the silicon or the version?" becomes unanswerable.

---

## 2. The core finding: small models can't see the real gap

Here's the number that surprised me. **910B4 relative to 910B3**:

| Model | Weights/card | Gap |
|---|---:|---:|
| Qwen3-4B | 2.0 GB | **11%** slower |
| Qwen3-8B | 4.0 GB | **86%** slower |
| Qwen3-32B | 15.5 GB | **309%** slower |

**Same two cards. 4B says "basically identical." 32B says "unusable."**

### Mechanism: an effective memory-bandwidth ceiling

During decode, every card must read its full share of the weights once per token. So you can derive **effective bandwidth = weights-per-card ÷ p99 TPOT** (at concurrency 1 — no KV pressure, no queueing):

![Bigger model, bigger gap: the 910B4 hits a ~160 GB/s bandwidth ceiling](/img/ascend3/ascend3-bandwidth-ceiling.png)

```
weights/card    2.0GB    4.0GB    7.75GB   15.5GB
910B4         131      157      147      163  GB/s   <- all four flat at ~150: at the ceiling
910B3         147      292      366      665  GB/s   <- monotonic climb, nowhere near saturated
910B4-1       161      301      350      631  GB/s   <- tracks the 910B3 point-for-point (1~10% apart)
```

**The 910B4 is already at its ceiling with 2 GB per card** — weights grew 7.75×, bandwidth crawled from 131 to 163. Over the same range the 910B3 went 147 → 665.

**4B is too small: neither card reaches its bandwidth ceiling** — at that size you're measuring fixed overhead like kernel launch (both land at 130~150 GB/s), which is why they "look the same."

> **Benchmark domestic accelerators with a 4B model and you will reach the opposite conclusion from a 32B model.**

### The same model also predicted a second dataset

If "the 910B4 is memory-bound" is true, then **it should depend less on graph mode (aclgraph)** — because kernel-launch overhead is a smaller share of total time, so eliminating it saves less.

Measured (throughput lost when graph mode is disabled via `--enforce-eager`):

![Graph-mode gain is inversely related to the bandwidth bottleneck](/img/ascend3/ascend3-graphmode-vs-bandwidth.png)

| | Graph-mode gain (x3) | Effective bandwidth |
|---|---:|---:|
| **910B4** | **1.36×** | **163 GB/s (at ceiling)** |
| 910B4-1 | **2.43×** | 631 GB/s |
| 910B3 | 3.04× (sweep protocol) | 665 GB/s |

**The prediction held.** One model explaining two independently measured datasets is the strongest evidence it's right.

(The 910B3 box was occupied by someone else, so no x3 there. Given the protocol bias measured on the same hardware, its x3 value should sit slightly below 3.04× — which doesn't change the ordering.)

A note for NVIDIA people: **CUDA graphs typically buy 10~30% on NVIDIA; aclgraph on Ascend buys 2.4~3.0×.** Which means:

> **On Ascend, anything that drops you back to eager (a missing operator, dynamic shapes, a debug flag) costs you 26~59% — not 10%.**

---

## 3. Three SKUs of one generation, three different outcomes

Ascend exposes a set of tuning knobs that simply don't exist in CUDA land (`weight_nz_mode`, `enable_flashcomm1`, `enable_cpu_binding`…). I measured every one I could:

![Ascend-only parameters: only one of them actually helps](/img/ascend3/ascend3-param-gains.png)

**Every point measured 3× independently, median reported. The measured x3 spread is only 0.06~0.92%** — which is why these percentages are signal, not noise:

| Parameter | 910B4 (32G) | 910B3 (64G) | 910B4-1 (64G) | Verdict |
|---|---:|---:|---:|---|
| `--enforce-eager` | **−26.32%** (28×) | **−65.87%** (51×) | **−58.80%** (72×) | graph mode is mandatory |
| **`weight_nz_mode = 2`** | **+8.86%** (25×) ⬆️ | −1.04% *(in noise)* | **−2.23%** (3×) ⬇️ | ⚠️ **three SKUs, three outcomes** |
| **`enable_flashcomm1 = true`** | **+5.07%** (12×) | +0.34% *(in noise)* | **+4.01%** (5×) | ⚠️ **works on two, dead on the third** |
| `enable_static_kernel = true` | +0.08% *(in noise)* | — | −0.94% *(in noise)* | ⚪ **no gain** |
| `enable_cpu_binding = false` | +0.3% | −0.5% | −0.1% | ⚪ no effect |

The bracketed figure is the **effect size as a multiple of that point's x3 spread**; anything marked *(in noise)* falls under 2.5× the noise floor, so it can only be reported as "no gain," never as "harmful."

### `weight_nz_mode`: three SKUs of one generation, three different outcomes

This is Ascend's NZ weight memory layout. Three same-generation SKUs give three different answers:

- **910B4: +8.86%** (25× the noise floor — significantly helps)
- **910B4-1: −2.23%** (3× the noise floor — significantly hurts)
- **910B3: −1.04%** (inside the noise — no gain)

**Exactly one SKU benefits; the other two don't, and one of them is measurably slower.**

I published a line earlier this year: the same FP8 weights give **+28%** on a 4090D (Ada, has FP8 cores) and **−18%** on an A100 (Ampere, doesn't) — **GPU generation decides whether quantization is a blessing or a curse.**

This one is worse: that was **across generations**. This is **within a generation, across bins**.

> **Tuning configs don't port across SKUs — not even within one vendor, one generation, one suffix apart.**

### `enable_static_kernel`: the one that sounds most like an optimization buys nothing

The docs say it makes "static shape kernel … **accelerate aclgraph execution**." And we just proved aclgraph is worth 2.4~3×, so accelerating it sounds valuable.

Measured:

```
graph capture time:  72 s  ->  3916 s   (+5340%, i.e. 65 minutes)
inference:           +0.08% / -0.94%    (x3; inside the noise floor = no gain)
```

**65 minutes of cold start, zero return.**

(Honest note: I first wrote this up as a *negative* gain, based on single-sweep numbers. Running x3 showed that −1.7% was inside the noise. **It isn't slower — it's unchanged.** For a flag advertising "accelerates aclgraph execution," unchanged is already bad news.)

And the trap is well hidden: turn this on in production and restarts take an hour with no obvious cause. My own first attempt was killed by the harness's 25-minute timeout and logged as `SERVE_TIMEOUT` — I nearly wrote "this parameter isn't supported." It took reading 37,195 lines of log to find `Graph capturing finished in 3916 secs`.

### Hit rate across 8 Ascend-only parameters

| Outcome | Count | Parameters |
|---|---:|---|
| ⚠️ SKU-dependent | **2** | `enable_flashcomm1` (+5.07% / +4.01% on two SKUs, but **only +0.34% — dead — on the 910B3**); `weight_nz_mode` (three SKUs, three outcomes) |
| 🔴 trap | 1 | `enable_static_kernel` (no gain + 65-minute cold compile) |
| ⚪ no effect | 2 | `enable_cpu_binding`, `fuse_norm_quant` |
| ❌ unusable here | 4 | `enable_kv_nz` (MLA only), `xlite` (C extension not shipped in the image), `finegrained_tp_config` (MoE only, and only on the D node of a PD-disaggregated setup), `--kv-cache-dtype fp8` (operator doesn't accept it) |

> **You are not going to tune your way out of a domestic accelerator's gap. The gap is in memory bandwidth, not in the flags.**
>
> Worse: **even "which flag helps" is SKU-dependent.** Of 8 parameters, the only two that ever produce a gain (`enable_flashcomm1`, `weight_nz_mode`) **each go dead or reverse on one of the three SKUs**.
> Which means **no "Ascend tuning best practices" document ports across SKUs** — every SKU has to be re-measured.

The `finegrained_tp_config` constraints only became clear by **reading the vllm-ascend source** (`ascend_config.py` carries a hard assert: `The finegrained tp sizes can be enabled only for MOE models`), which saved several wasted runs. Which is the broader point: **for domestic-accelerator adaptation, reading the source has a far better ROI than trying flags at random.**

---

## 4. Peak throughput is the easiest metric to fool yourself with

![Peak throughput vs usable capacity](/img/ascend3/ascend3-peak-vs-goodput.png)

Two panels sharing an x-axis. On top, output throughput climbs all the way to c=512. Below, **goodput** (throughput that simultaneously meets p99 TTFT ≤ 2s and p99 TPOT ≤ 100ms) **peaks at c=64 and collapses by c=256**.

```
throughput:  c64=1037 -> c128=1449 -> c256=1638   climbing
goodput:     c64=7.72 -> c128=6.92 -> c256= 0.34  peaks at c64, -96% by c256
```

**Past c=64, every additional unit of concurrency buys throughput your customers cannot use** — at c=256 p99 TTFT is already 12~16 seconds.

Note the orange line (910B4) hugging zero in the lower panel: running 32B/TP4, its goodput is 0 from c=16 onward. Open-loop **max QPS@SLO**:

| | 910B4 | 910B3 | 910B4-1 |
|---|---:|---:|---:|
| 32B **TP4** | **0** 🔴 | 4 | 4 |
| 32B **TP8** | 4 | **6** | **6** |

**On TP4 the 910B4 cannot serve even one request per second** (at rate=1, p99 TPOT is already 110ms, past the 100ms line).

### A test-design trap

My first pass used doubling (rate 1→2→4→8) and scored all three boxes at `max QPS@SLO = 4`. **That looks like "all three SKUs have identical usable capacity" — it's an artifact of insufficient resolution.** Bisecting at rate 5/6/7 separated them into 4 vs 6.

> Doubling finds the interval. **Bisection produces the number.** Skip the second half and your conclusion is wrong in a way you can't see.

---

## 5. Quantization: the win is in latency, not throughput

![W8A8: the card with the biggest throughput gain gains no capacity at all](/img/ascend3/ascend3-quant-goodput.png)

W8A8 vs bf16 (32B / TP4 / concurrency 128):

| | Throughput gain | goodput gain | p99 TPOT |
|---|---:|---:|---|
| 910B3 | +9% | **+61%** | 104ms → **95ms** (crosses the SLO line) |
| 910B4-1 | +13% | **+80%** | 110ms → **95ms** (crosses) |
| **910B4** | **+29%** | **0 → 0** | 226ms → 177ms (**still over**) |

Two things:

**① +13% throughput, +80% goodput** — because quantization pulls p99 TPOT from 110ms to 95ms, **crossing the 100ms SLO line**, and previously-timing-out requests convert wholesale into passing ones. **goodput is a step function, not a linear one.**

**② The card with the biggest throughput gain gains no capacity** — the 910B4 is bandwidth-bound, so halving weight bytes directly relieves its bottleneck and throughput jumps 29% (the largest of the three). But at 177ms it's still **too far from the SLO line** to convert a single request.

> **Looking only at tok/s will both over- and under-state the value of quantization — depending on how far you are from the SLO line.**

---

## 6. Long context: usable capacity goes to zero

![Long-context collapse](/img/ascend3/ascend3-long-context.png)

| Input length | 910B3 throughput | p99 TTFT | goodput |
|---|---:|---:|---:|
| 512 | 476 | 886 ms | 3.72 |
| **8K** | 220 | 13.8 s | 0.46 |
| **16K** | **83** (−83%) | **35.5 s** | **0.00** |

At 16K / concurrency 16 the 910B4's p99 TTFT reaches **91 seconds**.

All three boxes blow past the 2-second TTFT line at **8K input with only 4 concurrent requests**.

> **Running 32B on these three cards, long-context usable capacity is essentially zero.** Not "a bit slow" — **you cannot hold it to an interactive SLO.** For long documents or RAG you either relax the SLO or move to PD disaggregation and split prefill out.

### But the return on adding cards is completely different

| 910B4-1, TP4 → TP8 | Gain |
|---|---:|
| Short QA, in512 c=128 | +38% |
| **Long context, in16k c=16** | **+177%** |

**Same card. The return on adding cards differs by 4.6×, depending on whether your workload is short QA or long documents.**

> "Adding cards isn't worth it," measured on a short-QA workload, is flatly wrong for a long-context business.

---

## 7. Retracting a conclusion I published in July

In my July Ascend deep-dive I wrote:

> ~~Ascend fp16 = 603 vs bf16 = 709, **15% slower**; bf16 is the correct default for the 910B4~~

**That doesn't hold.** Digging out the raw files, those two numbers were measured under **different protocols**:

| | Value | Protocol |
|---|---:|---|
| bf16 | 709.0 | **single point on a sweep curve** |
| fp16 | 603.0 | **independent x3, median** |

And this round I happened to quantify that protocol bias: **a single sweep point runs systematically 13.9% higher than an independent x3 median** (same box 183, same c=128: sweep 1051 vs x3 median 922.7).

Correcting July's number for that bias: `709 ÷ 1.139 ≈ 622` vs fp16 `603` → **−3.1%**.

And measured this round on the same box, same version, same protocol:

| 183 (910B4) 32B TP8 c=128 | run1 | run2 | run3 | median |
|---|---:|---:|---:|---:|
| bf16 | 915.7 | 923.6 | 922.7 | **922.7** |
| fp16 | 889.1 | 893.6 | 894.0 | **893.6** |

**−3.2%** — essentially identical to the −3.1% reconstruction.

```
July's -15%  =  -12% protocol bias (sweep vs x3)  +  -3% real difference
```

**Corrected**: on Ascend, fp16 is about **3%** slower than bf16 (consistent across all three SKUs). bf16 is still the recommended default, but **for numerical-stability reasons, not a 15% performance gap.**

### The lesson is worth more than the number

**Within a single report, some numbers measured by single sweep and others by x3 median, then subtracted across — the difference carries a 14% protocol bias.**

The trap is subtle precisely because **both datasets are individually valid**; they just don't belong in the same subtraction. If I hadn't happened to quantify the protocol bias this round, I'd still believe fp16 was 15% slower.

New rule:

> **Any head-to-head comparison must use the same protocol on both arms. x3 compares only to x3; sweep compares only to sweep.**

---

## 8. One-line summary

If you take away one thing:

> **For domestic-accelerator selection, model scale matters an order of magnitude more than parameter tuning.**
>
> The same two Ascend cards are 11% apart on 4B and 309% apart on 32B — while tuning every Ascend-only flag to its optimum recovers 4%.
> Get the model scale wrong and nothing you tune afterwards matters.

And one for fellow practitioners:

> **Don't benchmark selection decisions with small models, and don't subtract numbers measured under different protocols.**
> I've done both. The second one I published.

---

## Appendix: data and reproducibility

- **504 measured points**: raw JSON plus environment metadata (chip SKU / Board ID / HBM / driver / firmware / image digest / device-visibility env)
- Test matrix configs, batch runner, plotting scripts
- Full incident log (12 methodology rules, 5 falsified hypotheses)

All comparisons in this post use **the same image digest, the same model files, and the same bench protocol**, so cross-SKU conclusions are comparable.

**Known gaps (stated honestly)**: cross-node TP16 was never attempted because a neighbour had a long-running service on the second node; MoE was only measured on one box (the model files differ across boxes, so the numbers aren't comparable) and is kept as an appendix.
