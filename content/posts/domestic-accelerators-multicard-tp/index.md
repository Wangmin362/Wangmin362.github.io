---
title: "Two Cards, Sometimes 3× the Throughput: Multi-Card Tensor-Parallel Scaling Across Domestic and NVIDIA GPUs"
date: 2026-07-11
draft: true
tags: ["benchmark", "vLLM", "tensor-parallel", "domestic-accelerators", "Ascend", "Alibaba-PPU", "MetaX", "Kunlunxin", "Hygon", "Iluvatar", "LLM-inference"]
summary: "Same Qwen2.5-7B, same load, now with 1/2/4 cards each. Six accelerators swept across tensor-parallel sizes on real hardware. The counter-intuitive result: the smallest cards scale super-linearly (a 24 GB 4090D hits 2.34× on two cards, a 32 GB Iluvatar hits 3.24×) while the 98 GB giants barely move. Multi-card is a lifeline for memory-starved cards, and diminishing returns for the big ones."
ShowToc: true
---

> 📌 **给 David 的说明（发布前删掉这段）**：数据表 / TP 扩展图 / 复现命令 / 版本表都是**真机实测**（2026-07-10~11 在 67 + 183 集群跑的），可以直接用。带 `〔观点待核〕` 的段落是我替你起草的判断，请用你自己的话改写核对。全过程 runbook + 每个坑见 vault `400_Experiments/w1-vllm-bench/cross-vendor/multi-card/`（RUNBOOK.md 记了 TP 整除头数、忙卡避让、HIP 不兼容等全部细节）。确认无误后删本段、删所有 `〔观点待核〕`、`draft: false` 发布。

> **TL;DR** — Take the [nine-accelerator single-card benchmark](../domestic-accelerators-9-card-vllm/) and add a dimension: run the *same* models under the *same* load on 1, 2, 4, and **8** cards each (tensor parallel). Three things stood out. **(1) The smallest cards scale best on 7B.** A 24 GB RTX 4090D jumps to **2.34×** on two cards; a 32 GB Iluvatar MR-V100 hits **3.24×** — both *super-linear* — while the 98 GB Kunlunxin only reaches **1.52×** on *four*. Small models on small cards love extra cards; big cards running small models don't. **(2) But big models flip it: Qwen3-32B on Ascend scaled 2.12× from 4 to 8 cards** (583 → **1237 tok/s**) — near-linear, and as far as I can find **the first public real-machine numbers for a domestic accelerator running 32B on 8 cards.** **(3) Tuning does not port across vendors:** `--kv-cache-dtype fp8` is a +38% free lunch on NVIDIA and *silently produces zero tokens* on Ascend. Copy an NVIDIA tuning guide onto domestic hardware and you'll ship a dead endpoint.

This is part two of a real-machine cross-vendor series. [Part one](../domestic-accelerators-9-card-vllm/) ranked nine accelerators on a single card each. This post asks the next question every serving engineer actually faces: **what do I get when I add a second and fourth card?**

## Why single-card rankings mislead you

The single-card post crowned the 98 GB cards (Kunlunxin, PPU) as throughput kings. True — on one card. But nobody serves a real workload on one card and stops there. The moment you go multi-card, the ranking *reshuffles*, because the thing that helps a small card (splitting the model frees KV-cache room) does almost nothing for a card that already had room to spare.

So a single-card benchmark answers "which card is fastest alone" — but the deployment question is "which *configuration* gives me the most tokens per second per dollar," and that only shows up once you sweep tensor-parallel sizes. That's this post.

## The one-glance result

![Multi-card tensor-parallel scaling — peak throughput per TP size, and speedup vs ideal linear](/img/multicard/compare-multicard.png)

Peak output throughput (tok/s), Qwen2.5-7B, `in=512 / out=128`, swept to saturation:

| Card | VRAM | TP1 | TP2 | TP4 | Best speedup |
|---|---|---:|---:|---:|---|
| **Alibaba PPU ZW810E** | 98 GB | 3404 | 5294 | **8641** | TP4 = **2.54×** (keeps climbing) |
| **NVIDIA 4090D** | 24 GB | 1603 | **3753** | 3630 | TP2 = **2.34×** (super-linear, TP4 regresses) |
| **Iluvatar MR-V100** | 32 GB | 422 | **1370** | — | TP2 = **3.24×** (super-linear) |
| **MetaX C500** | 64 GB | 3356 | 4430 | 4815 | TP4 = 1.43× (walls at TP4) |
| **Ascend 910B4** | 32 GB | 827 | 1566 | **2415** | TP4 = **2.92×** (scales all the way to 4) |
| **Kunlunxin P800** | 98 GB | 3547 | 3967 | **5382** | TP4 = 1.52× (sublinear — big VRAM, less to gain) |

> **On the numbers, honestly:** the Ascend and Kunlunxin rows are from a fully-documented re-run (reproducible with the exact command at the bottom of this post); the PPU / MetaX / 4090D / Iluvatar rows are from an earlier broad sweep. **Absolute tok/s is config-sensitive** — it's server-bound and shifts with `max-model-len` and where you cap concurrency, so an earlier run measured some cards up to ~2× higher in absolute terms while the *scaling ratios and ranking held*. So: trust the **per-card scaling ratio** (TP1→TP2→TP4) as the reproducible signal; treat cross-card absolute peaks as indicative, not a leaderboard, since they blend two runs. Every raw JSON is in the repo.

Read the right-hand chart: the dashed line is ideal linear scaling. **Cards that start below the line on a single card tend to shoot above it on two** — because doubling the cards doubles the KV-cache space, and for a memory-bound sweep that's worth more than the raw compute you added.

## The counter-intuitive finding: small VRAM scales best

Here's the mental model. vLLM throughput at high concurrency is gated by how many requests' KV cache you can hold at once. On a single small card, most of the VRAM is eaten by the model weights, leaving a thin sliver for KV — so you saturate at low concurrency. Split that model across two cards (tensor parallel) and each card now holds *half* the weights, freeing a disproportionate amount of room for KV. The concurrency ceiling doesn't just double — it can more than double, because you also gained aggregate memory bandwidth. That's how you get **super-linear** 2.34× and 3.24×.

Now flip it. A 98 GB card running a 7B model (~15 GB in fp16) already had 80+ GB free for KV. It was never memory-bound on this workload. Splitting the model just adds cross-card communication overhead for a bottleneck that barely existed — hence Kunlunxin's weak, sublinear **1.52×** at TP4 (a 24 GB 4090D beat that on *two* cards).

〔观点待核〕 **The takeaway for buyers:** multi-card is *雪中送炭* (a lifeline in the snow) for memory-starved cards and *锦上添花* (diminishing gilding) for big ones. If you're on 24–32 GB cards, a second card can more than pay for itself. If you're on 98 GB cards and running small models, extra cards buy you very little — you'd be better off packing more independent replicas onto single cards.

> **One caveat on the "small VRAM" framing:** it's really *single-card KV headroom + engine maturity*, not VRAM alone. The cleanest proof is two **98 GB** cards scaling completely differently: Alibaba PPU keeps climbing to **2.54×** at TP4 while Kunlunxin stalls at **1.52×** — same capacity, different cross-card engine. Capacity sets the ceiling; the vendor's kernels decide how close you get to it.

## The other big finding: domestic multi-card maturity is wildly uneven

Single-card, most domestic cards "just work." Multi-card is where the cross-card communication stack (each vendor's answer to NCCL) gets stress-tested — and the results separate the mature from the immature fast:

| Vendor | Multi-card stack | 7B multi-card | 32B multi-card | Verdict |
|---|---|---|---|---|
| **Ascend 910B4** | HCCL | ✅ TP2/TP4 | ✅ **TP4=583, TP8=1237** | Most mature — the only domestic card that ran 32B multi-card cleanly, all the way to 8 cards |
| **Alibaba PPU** | NCCL-compatible | ✅ TP2/TP4 (best 7B scaling) | ❌ NCCL error | Great at 7B, cross-card comms break on 32B |
| **Kunlunxin P800** | XCCL | ✅ TP2/TP4 | ❌ worker crash (c10) | 7B fine to TP4; 32B crashes at init — even with graph compilation disabled |
| **MetaX C500** | maca | ✅ TP2/TP4 | (cards busy — not run) | Works on 7B, walls at TP4 (1.43×) |
| **Hygon K100-AI** | HIP/RCCL | ❌ "invalid device pointer" | ❌ | Multi-card unusable on this vLLM fork — TP1 only |

〔观点待核〕 If you're doing platform selection and you *know* you'll need multi-card for 32B+ models, this table matters more than the single-card ranking. The pattern is striking and consistent: **every domestic stack handles 7B across cards, and every one except Ascend fails on 32B** — PPU's NCCL-compatible layer errors out, Kunlunxin's XCCL crashes the worker at init (I even retried with graph compilation off — same crash), Hygon can't do multi-card at all. Three independent cross-card stacks, three different failure modes, one survivor. Huawei's flagship earns its reputation here: HCCL is the only domestic stack that carried both 7B and 32B across cards — cleanly, up to 8 — in my runs.

## Where multi-card actually earns its keep: 32B

7B is small enough that multi-card is often overkill (note how the 4090D actually *regresses* at TP4 — the comms overhead exceeds the benefit — and the big cards barely improve). The real justification for multi-card is a model that **won't fit on one card at all**.

Qwen3-32B in bf16 is ~61 GB of weights. It doesn't fit on a 32 GB card, period. Two cards (64 GB) still can't hold it once you add KV cache and activations — it OOMs. So on 32 GB-class cards, **32B needs TP4 as a floor, not a choice** — and this is where adding *more* cards keeps paying off:

![Ascend 910B4 running Qwen3-32B: TP4 to TP8 scales 2.12×, near-linear; and workload shape swings throughput 4.5×](/img/multicard/compare-32b-tp8.png)

| Model | Card | TP4 (4 cards) | TP8 (8 cards) | Scaling |
|---|---|---:|---:|---|
| Qwen3-32B (bf16) | Ascend 910B4 (32 GB) | **583** | **1237** | **TP8 = 2.12× TP4** (near-linear) |
| Qwen3-32B (bf16) | Alibaba PPU | ❌ NCCL error | — | cross-card comms fail on 32B |

Two things here are, as far as I can find, **the first published real-machine numbers for a domestic accelerator running a 32B model on 4 *and* 8 cards.** You can't fake them, and I ran them on physical 910B4 silicon.

〔观点待核〕 And notice the contrast with the 7B story above: 7B *regressed* going from TP4 to more cards (comms overhead beat the benefit), but 32B **scales 2.12× from 4 to 8 cards** — genuinely near-linear. That's the real rule of thumb: **the bigger the model, the more multi-card pays off.** Small models don't need the cards; big models can't live without them, and reward every one you add.

> **Why 8 and not more, and why not on 7B:** TP size must divide the model's attention-head count. Qwen3-32B has 64 heads → TP8 is legal. Qwen2.5-7B has only 28 heads → I tried TP8 and vLLM rejected it outright: `Total number of attention heads (28) must be divisible by tensor parallel size (8)`. So "just throw 8 cards at it" isn't a free choice — the model architecture decides which TP sizes even exist.

## Tuning vLLM: what moves the needle — and the flag that's a free lunch on NVIDIA but *broken* on Ascend

While the cards were racked I ran one-factor-at-a-time (OFAT) parameter sweeps to separate the tuning myths from the tuning wins. The single most important finding is that **tuning advice does not port across vendors.** Here's the same flag on two different silicons:

| `--kv-cache-dtype fp8` | Baseline | With fp8 KV | Result |
|---|---:|---:|---|
| **NVIDIA 4090D** (standard vLLM) | 1316 (c64) | **1820** | **+38%** — a near-free lunch |
| **Ascend 910B4** (vllm-ascend) | 490 (c64) | **0** | ❌ **accepts the flag, then produces 0 tokens/s** — silently broken |

That's the whole ballgame for anyone tuning domestic hardware: **the #1 "free" throughput trick on NVIDIA is a footgun on Ascend.** vLLM-Ascend accepts `--kv-cache-dtype fp8`, logs a cheerful "reduces memory and boosts performance" message, and then serves nothing. If you'd copied an NVIDIA tuning guide, you'd ship a dead endpoint. You have to re-derive the tuning table per vendor — you can't inherit it.

The rest of the Ascend 7B parameter sweep (TP1, in=512/out=128, c128):

| Config | c128 tok/s | Verdict |
|---|---:|---|
| Baseline | 785 | — |
| `--max-num-seqs 64` | 545 | ⚠️ **Capping the batch too low throttles throughput** (−31%) |
| `--max-num-seqs 512` | ❌ OOM | Too high crashes it — the default is tuned for a reason |
| `--max-num-batched-tokens 16384` | 818 | +4%, marginal |
| `--no-enable-chunked-prefill` | 821 | +4.6% raw throughput, but you pay for it in TTFT |
| `--no-enable-prefix-caching` | 689 | Slightly *lower* on random load — prefix-caching wasn't hurting |
| `--gpu-memory-utilization 0.95` | 785 | No change — 32 GB already holds plenty of KV for a 7B |

And a quantization data point, because 4-bit/8-bit weights are the other big lever on memory-bound cards:

| Ascend 910B4 · Qwen3-32B · TP4 | Peak tok/s | vs bf16 |
|---|---:|---|
| bf16 (baseline) | 583 | — |
| **W8A8 (8-bit weights + activations)** | **736** | **+26%** |

Three things worth internalizing:

1. **Re-derive your tuning table per vendor.** `kv-cache fp8` is +38% on NVIDIA and a silent brick on Ascend. Never assume a flag ports.
2. **Leave `max-num-seqs` alone unless you've measured.** Too low throttles throughput; too high OOMs. The default is usually right.
3. **On memory-bound cards, weight quantization is the reliable win** — W8A8 bought +26% on a 32B, no accuracy cliff in casual checks (measure yours).

## Methodology

- Same as [part one](../domestic-accelerators-9-card-vllm/): `Qwen2.5-7B-Instruct`, fp16, byte-identical checkpoint (verified by hashing config/index/tokenizer + safetensors shard sizes), `random` dataset, `in=512 / out=128`, `--ignore-eos --seed 42`, swept concurrency to saturation with vLLM's own `vllm bench serve`.
- **Tensor parallel via `--tensor-parallel-size N`.** TP size must divide the attention head count: Qwen2.5-7B has **28 heads** → legal TP ∈ {1,2,4,7,14,28}, so **TP8 is architecturally invalid for 7B** (28÷8 isn't an integer — vLLM rejects it). To run TP8 you need a model with a divisible head count, e.g. Qwen3-32B (64 heads).
- **`gpu-memory-utilization` tuned per card to real free memory**, not a blanket 0.90 — some vendors' drivers reserve 15–25 GB, and asking for more than exists just errors out. Every card ran at the highest utilization it could actually sustain; the exact value is logged per run.
- Models were loaded from **local node disk** (hostPath), not network storage — a 32B checkpoint over a slow shared filesystem took 1.6 hours to load; on local disk it's seconds. Load path doesn't affect steady-state throughput, but it makes the difference between "ran the sweep" and "gave up waiting."

> **The same honest caveat as part one:** the model is unified, but each vendor ships its own vLLM fork, and each card's multi-card comms stack is the vendor's own (HCCL / NCCL-compat / HIP). This is *"each card at its practical best on its own software,"* not *"identical engine, different silicon."* That's the only kind of cross-vendor benchmark that's actually runnable today — I just refuse to hide it.

### Reproduce it

Single command per data point, e.g. PPU TP4 at concurrency 64:

```bash
# serve (TP4)
vllm serve /models/Qwen2.5-7B-Instruct \
  --tensor-parallel-size 4 \
  --dtype float16 \
  --gpu-memory-utilization 0.90 \
  --max-model-len 4096 \
  --port 8000

# benchmark (one point of the sweep)
vllm bench serve \
  --model /models/Qwen2.5-7B-Instruct \
  --dataset-name random \
  --random-input-len 512 --random-output-len 128 \
  --num-prompts 400 --max-concurrency 64 \
  --ignore-eos --seed 42 \
  --save-result --result-dir results/ppu-tp4/
```

Swap `--tensor-parallel-size` and `--max-concurrency` for each cell in the tables. Every raw JSON result is archived alongside the runbook.

## What's next

Still on the bench: MetaX 32B multi-card, a Layer-4 quantization comparison (fp16 vs fp8 vs AWQ on the same card), and Moore Threads S4000 (pending an image). If there's a specific card or configuration you want to see, the rig is racked — tell me.

---

*Part of a real-machine cross-vendor LLM-inference benchmark series. Every number here was produced on physical hardware with the command shown. If a vendor's marketing deck disagrees with these numbers, run the command yourself — that's the whole point.*
