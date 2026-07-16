---
title: "Huawei Ascend 910B4 Deep-Dive: The First 8-Card 32B Real-Machine Data on Domestic Silicon — W8A8 +26%, kv-fp8 Silently Yields 0"
date: 2026-07-16
draft: true
tags: ["benchmark", "vLLM", "tensor-parallel", "domestic-accelerators", "Ascend", "Huawei", "quantization", "kv-cache", "LLM-inference"]
summary: "Squeezing a Huawei Ascend 910B4 dry: 32B scaled from TP4 to TP8 (1237 tok/s, 2.12x near-linear — the first 8-card 32B real-machine data on domestic silicon), W8A8 quant +26%, AWQ flat-out unsupported, and kv-fp8 accepted-yet-silently-yields-0. This is the only domestic card that passed all 5 test layers — and the one that best explains what 'domestic-accelerator adaptation' actually means."
ShowToc: true
---


> **TL;DR** — First **single-vendor deep-dive** of the real-machine cross-vendor series. I ran a Huawei Ascend 910B4 through the full 5-layer test spec. Four numbers to remember: **(1) 32B scales TP4→TP8 = 583→1237 tok/s, 2.12x near-linear** — the **first 8-card 32B real-machine data on domestic silicon** I've seen published; **(2) 7B on 8 cards crashes outright** — 28 attention heads aren't divisible by 8, only the 64-head 32B can go 8-card; **(3) W8A8 quant is a +26% free lunch**, but **AWQ is flat-out unsupported** (`not supported in npu`); **(4) `--kv-cache-dtype fp8` is the sneakiest** — the flag is accepted, the server starts fine, and **throughput is 0**. Ascend is the only domestic card in my tests to pass all 5 layers, and the one that best explains what "domestic-accelerator adaptation" really means.

Part four of the real-machine cross-vendor series, and the first **single-vendor deep-dive**. The first three were horizontal: [part one](../domestic-accelerators-9-card-vllm/) ranked nine cards single-card, [part two](../domestic-accelerators-multicard-tp/) swept 7B across cards, [part three](../a100-vs-alibaba-ppu-32b-multicard/) put A100 and Alibaba PPU side by side. From here the approach changes: **one card, exhausted, per post** — TP scaling, param tuning, quantization, workload shape, all measured. Because when selection hits deployment, you don't want "who's #1," you want "what happens when I turn each knob on THIS card to the stop."

## Why Ascend gets the first deep-dive

Because it's the **only domestic card that passed all 5 test layers**. Every other card hit a wall mid-way — Kunlun's 32B multi-card crashes, Hygon's multi-card dies, Cambricon does 18 tok/s on single-card 7B. Only Ascend gave real data at every layer, from 7B to 32B, single-card to 8-card, quant to workload shape. That's a conclusion in itself: **for multi-card 30B+ models today, Ascend has the most mature software stack among domestic cards.**

## How it was measured (reproducible)

- **Hardware**: Huawei Ascend 910B4 ×8 (32GB HBM/card), CANN, npu-smi 25.5.1.
- **Engine**: vllm-ascend 0.18.0.
- **Load**: `vllm bench serve` random, `in=512/out=128` (workload layer sweeps others), concurrency `1/4/16/64/128/256` to saturation, `--ignore-eos --seed 42`, dtype bf16 (fp16 comparison arm kept too).
- **Models from local disk** `/apps/vllm/models` (avoiding slow JuiceFS/NFS loads).
- **⚠️ Honest note**: Ascend's TP1/TP2 absolutes this round were slightly below the prior round (config sensitivity; measured AICore ~90% = server-bound, so numbers are real and internally consistent). **All numbers here are same-methodology comparable**; cross-vendor comparison is in the series comparison post.

## Everything about Ascend, in one chart

![Ascend 910B4 deep-dive: left = 32B multi-card scaling (TP4->TP8 2.12x) + 7B scaling; right = W8A8 quant +26% vs the AWQ / kv-fp8 traps](/img/ascend-deep/ascend-deep.png)

## Finding 1: the first 8-card 32B real-machine data on domestic silicon

The naive problem first: **a 32GB card can't hold 32B**. Qwen3-32B's bf16 weights are ~62GB — a 32G Ascend can't even fit the weights, let alone the KV cache. So Ascend running 32B **must start at TP4** (4 cards, 128GB, enough for weights + KV) — a completely different starting line from 98G cards like PPU/Kunlun that fit 32B on a single card.

Is TP4→TP8 worth it? **Yes, and near-linearly:**

```
Ascend Qwen3-32B:  TP4 = 583  →  TP8 = 1237 tok/s
                            2.12x  near-linear
```

Full TP8 concurrency curve (tok/s): `c1=15.7 / c4=57 / c16=225 / c64=604 / c128=925 / c256=1237`.

**This is data I haven't seen a second copy of anywhere** — a real-machine 8-card 32B throughput curve on domestic silicon. Why so rare? Most people can't assemble 8 identical domestic cards, and those who can lack the patience to debug vLLM from crashing to working on a domestic stack. Which is exactly where **domestic-accelerator adaptation** earns its keep: **Huawei's HCCL (its NCCL-equivalent collective-comms library) runs 32B stably across 8 cards**, while Kunlun's XCCL, Hygon's HIP, and PPU's early NCCL port all crashed on multi-card 32B.

**Why near-linear rather than "wasted cards"?** Because 32B is **both compute- and KV-heavy**: sharding across 8 cards both thins the 62GB of weights to ~8GB/card (freeing lots of KV space for bigger batches) AND parallelizes the matmuls — payoff on both ends, hence near-linear. Small models do the opposite (see Finding 3).

## Finding 2: want 7B on 8 cards? Instant crash

A trap many hit: **not every model can TP8.** I tried Qwen2.5-7B on 8 cards and vLLM threw:

```
Total attention heads (28) must be divisible by TP size (8)
```

**Tensor parallelism shards by attention heads** — splitting N heads evenly across N cards, so TP size must divide the head count. Qwen2.5-7B has 28 heads: 28÷8 doesn't divide → no TP8 (TP1/2/4 fine, since 28÷4=7). Qwen3-32B has 64 heads, 64÷8=8 → 8-card works.

> **📖 De-scaring one term: "sharding by attention heads"**
> Picture multi-head attention as a **workshop with 28 workstations** (28 heads, each computing independently). Tensor parallelism = distributing those 28 stations across cards. 4 cards → 7 stations each, clean; 8 cards → 28÷8=3.5, **you can't have half a station** → crash. So when picking a model, "what does the head count divide by" directly decides "how many cards you can go to." You only need to remember this constraint — the matmul-sharding derivation behind it is entirely skippable. For inference selection, "head count not divisible → can't go that wide" is enough.

**Takeaway**: if your goal is spreading a service across 8 cards, check the model's head count is a multiple of 8 first. A classic 28-head 7B closes the 8-card path outright — not a bug, an architectural constraint.

## Finding 3: big models scale super-linearly, small models waste extra cards

Put 7B and 32B scaling together and the pattern is stark:

| Model | Heads | TP1 | TP2 | TP4 | TP8 | Scaling |
|---|---:|---:|---:|---:|---:|---|
| Qwen2.5-7B (small) | 28 | 827 | 1566 | 2415 | ❌can't | TP4/TP1=2.92x, but 7B isn't card-starved past 4 |
| Qwen3-32B (large) | 64 | can't fit | can't fit | 583 | **1237** | **TP8/TP4=2.12x near-linear** |

**7B does 827 on one card, 2415 on four — but it wasn't KV-starved to begin with, so extra cards are gravy.** 32B can't even start on one card; cards are lifesaving, each genuinely relieving the KV bottleneck. **Conclusion: the bigger the model, the more cards earn their keep.** Eight cards on 7B is waste (and may not even divide); eight cards on 32B is cards well spent.

## Finding 4: quantization — W8A8 is treasure, AWQ is a trap

Fixing Qwen3-32B TP4, measuring quant's throughput impact:

| Config | Peak tok/s | vs bf16 | Verdict |
|---|---:|---:|---|
| Qwen3-32B bf16 (baseline) | 583 | — | reference |
| **Qwen3-32B W8A8** | **736** | **+26%** ✅ | weights to 8-bit → less VRAM → higher concurrency |
| Qwen3-32B AWQ | ❌ | — | `awq quantization is currently not supported in npu` |

**W8A8 (8-bit weights + 8-bit activations) is a genuine +26% on Ascend** — weights from bf16 to int8, VRAM halved, freed space packs bigger batches, throughput rises. Another sign of stack maturity: **it doesn't just "run," it supports mainstream W8A8 quant and actually speeds up.**

But **the same category, AWQ, is flat-out unsupported** — vllm-ascend reports `not supported in npu`. Which leads to the whole series' most expensive lesson...

## Finding 5 (the expensive one): kv-fp8 silently yields 0 — flags don't port across vendors

On NVIDIA, `--kv-cache-dtype fp8` (quantize the KV cache to 8 bits) is a textbook free lunch — +18.5% on A100, +38% on 4090. **So every NVIDIA tuning guide tells you to turn it on.**

I carried that "free lunch" onto Ascend, and got **the worst kind of failure:**

| Flag | Ascend behavior | vs NVIDIA |
|---|---|---|
| `--kv-cache-dtype fp8` | ✅ flag accepted → ✅ server starts fine → ❌ **throughput = 0** | A100 +18.5% / 4090 +38% |

**No error, no crash, health check still green — but throughput is 0.** An endpoint that looks alive but is dead. If you copy an NVIDIA tuning checklist, put kv-fp8 in Ascend's launch flags, and ship without load-testing, you get a service with an **all-green dashboard that emits not a single token.** That's more dangerous than "crashes on launch" — a crash you notice immediately.

Compare the same flag across three vendors (the one table to remember from the whole series):

| Flag | NVIDIA (A100/4090) | Huawei Ascend | Alibaba PPU |
|---|---|---|---|
| **`--kv-cache-dtype fp8`** | ✅ +18.5% / +38% | ❌ **silent 0** | ❌ **crash on launch** (fp8e4nv unsupported) |

**One flag, three fates.** That's what "domestic-accelerator adaptation" actually means: **the moat isn't memorizing eight vendor SDKs, it's the methodology to bring any backend in, test every flag on real hardware, and know which one to turn on and which is a landmine.** You cannot copy this table from NVIDIA's docs — you can only produce it card by card, on real machines.

### Other flags measured (7B TP1, c128 peak)

| Flag | c128 tok/s | vs baseline | Verdict |
|---|---:|---:|---|
| baseline (default) | 785 | — | reference |
| `--max-num-batched-tokens 16384` | 818 | +4% | slight up |
| `--no-enable-chunked-prefill` | 821 | +4.6% | throughput up, hurts TTFT |
| `--no-enable-prefix-caching` | 689 | −12% | prefix cache still helps a bit under random; don't blindly disable |
| `--max-num-seqs 64` (too small) | 545 | **−31%** ❌ | batch cap too small, high-concurrency throughput collapses |
| `--max-num-seqs 512` (too large) | **0** | — | ❌ OOM crash |
| `--gpu-memory-utilization 0.95` | 785 | 0 | 32G card on 7B, KV already enough |

**Param verdict**: ① `max-num-seqs` default is optimal — too small kills throughput, too large OOMs; ② disabling chunked-prefill nudges throughput but sacrifices first-token latency; ③ higher mem-util is a no-op for a 32G card on 7B.

## Bonus: workload shape swings throughput 4.5x

Fixing 7B TP2, varying only the input/output ratio:

| Shape | in/out | Peak tok/s | Scenario |
|---|---|---:|---|
| Generation-heavy | 128/512 | **1778** | writing/continuation (highest, decode-heavy) |
| Short QA (baseline) | 512/128 | 1566 | general chat |
| Very short | 128/16 | 862 | classify/extract |
| Long context | 2048/512 | 703 | long-doc understanding |
| RAG-like | 4096/256 | **396** | retrieval-augmented (lowest, prefill-heavy) |

**Same card, same TP, different shape, 4.5x throughput spread** (gen-heavy 1778 vs RAG 396). **Longer input (prefill-heavy) = lower throughput, longer output (decode-heavy) = higher.** So any "Ascend does X tok/s" number without its in/out ratio is meaningless — **benchmark with your own workload's shape.**

## Wrap-up

Ascend 910B4 has **the most mature software stack among domestic cards**: 32B runs stably to 8 cards over HCCL (1237, near-linear), W8A8 quant is a free +26% — but **AWQ is unsupported, kv-fp8 silently yields 0**, and not one NVIDIA tuning tip ports blindly. **Those three are both the knobs that squeeze Ascend dry and the best footnote to what "domestic-accelerator adaptation" actually adapts.**

Next: **Alibaba PPU vs Huawei Ascend** — two 8-card domestic cards, the same 32B head to head, weighing PPU's 2.6x throughput edge against Ascend's "steadier and more complete."

---

*Methodology, the raw per-config JSON, and the full write-up of turning a user-created inference service into a benchmark harness are in the [notex repo](https://github.com/wangmin362). Every number is from real hardware, none invented.*
