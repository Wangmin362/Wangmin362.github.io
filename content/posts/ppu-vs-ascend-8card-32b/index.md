---
title: "Alibaba PPU vs Huawei Ascend: An 8-Card 32B Showdown — PPU Is 2.6x Faster, but Ascend Is Steadier and More Complete"
date: 2026-07-16
draft: true
tags: ["benchmark", "vLLM", "tensor-parallel", "domestic-accelerators", "Alibaba-PPU", "Ascend", "Huawei", "LLM-inference"]
summary: "Two 8-card domestic cards, the same Qwen3-32B under the same load, head to head. Alibaba PPU hits 3215 tok/s on eight cards — 2.6x Huawei Ascend's 1237; but Ascend's 32G cards run stably over HCCL, support W8A8 quant, and have a more complete stack. More important: kv-fp8 is dead on both (PPU crashes, Ascend silently yields 0) — not one tuning tip ports between them. That's what really matters when picking domestic silicon."
ShowToc: true
---


> **TL;DR** — Part five of the real-machine cross-vendor series. Two **8-card** domestic cards on the table: Alibaba PPU-ZW810E vs Huawei Ascend 910B4, same Qwen3-32B, same load. Three takeaways: **(1) On raw throughput, PPU wins big** — 3215 tok/s on eight cards, **2.6x** Ascend's 1237; **(2) but it's not a fair fight** — PPU's 98G cards fit 32B on one card, Ascend's 32G cards must start at TP4, so the starting lines differ; **(3) Ascend wins on completeness** — W8A8 quant (+26%), a more mature stack. **The expensive lesson: kv-fp8 is dead on both** (PPU crashes on launch, Ascend silently yields 0), and not one NVIDIA tuning tip carries over. Picking domestic silicon isn't just about who's fastest — it's who's steady, who's complete, and whether you can tune it to its best.

Part five of the real-machine cross-vendor series. [The last post](../ascend-910b4-deep-8card-32b/) exhausted Ascend solo; this one pits it against Alibaba PPU — the **only two** domestic cards in my tests to run 8-card 32B on real hardware. One takes the 98G-big-VRAM route, one the 32G-HBM route; together they represent the two technical bets of domestic silicon. Which suits you depends on what you actually care about.

## Hardware first: this is not a fair fight

| | Alibaba PPU-ZW810E | Huawei Ascend 910B4 |
|---|---|---|
| VRAM/card | **98 GB** | 32 GB HBM |
| Fits 32B on one card? | ✅ yes (98G holds 62G weights) | ❌ no, **must start at TP4** |
| Collective-comms lib | NCCL port (PPU is CUDA-compatible) | HCCL (Huawei's own) |
| Engine | asllm + vLLM 0.20.1 | vllm-ascend 0.18.0 |
| Quant support | kv-fp8 ❌ / W8A8 untested | W8A8 ✅+26% / AWQ ❌ / kv-fp8 ❌ |

**The key difference is VRAM.** PPU's 98G/card fits the whole 32B on a single card; Ascend's 32G/card can't even hold 32B's 62G weights, so **32B starts at 4 cards**. That's why in the chart below, Ascend's curve begins at TP4 — not laziness on TP1/2, but **physically can't fit.**

## The showdown, in one chart

![Alibaba PPU vs Huawei Ascend, 8-card Qwen3-32B: left = 1/2/4/8-card scaling (PPU full, Ascend from TP4), right = 8-card peak PPU 3215 vs Ascend 1237](/img/ppu-ascend/ppu-vs-ascend-32b.png)

Same Qwen3-32B, `in=512/out=128`, swept to saturation:

| TP | Alibaba PPU | Huawei Ascend | Note |
|---:|---:|---:|---|
| TP1 | 406 | can't fit | PPU's 98G runs it single-card |
| TP2 | 1128 | can't fit | Ascend's 2×32G=64G still short |
| TP4 | 2005 | 583 | Ascend starts here |
| **TP8** | **3215** | **1237** | **PPU = 2.6x Ascend** |

## Finding 1: on raw throughput, PPU wins by 2.6x

Eight-card 32B: PPU hits **3215 tok/s**, Ascend **1237** — PPU is **2.6x** Ascend. And PPU scales more beautifully: near-linear from 406 at TP1 to 3215 at TP8 (**7.9x**), while Ascend does 2.12x from TP4 to TP8.

**Why is PPU so much faster?** Two compounding reasons: ① **98G VRAM** — never KV-starved, packs huge batches to sustain hundreds of concurrent requests; ② **starts from one card** — PPU is "adding compute" the whole way from 1 to 8 cards, whereas a good chunk of Ascend's first 4 cards goes to *fitting* 32B rather than *adding throughput* — so it spots PPU half a body length at the gun.

If your only KPI is "how many tokens can this pile of cards emit," **PPU wins outright.**

## Finding 2: but Ascend wins on "steady" and "complete"

Raw throughput isn't all of domestic-card selection. Ascend holds two cards PPU can't yet play:

**① Multi-card stability — Ascend was first through 32B multi-card.** Recall [part two](../domestic-accelerators-multicard-tp/): that round, PPU's 32B multi-card **crashed outright** with `NCCL error`; only Ascend ran it stably over HCCL. PPU only caught up this round (new asllm image + vLLM 0.20.1). Meaning **Ascend's multi-card maturity leads by at least one version cycle** — real trust points in production.

**② Quantization — Ascend supports W8A8 (+26%), PPU's line here is unverified.** Ascend's 32B under W8A8 is +26% throughput (583→736) and saves VRAM. I didn't test W8A8 on PPU this round (next time), but its **kv-fp8 crashes outright** (architecture lacks fp8e4nv) — the quant path is at least partly blocked.

**③ Autonomy angle — Ascend is Huawei full-stack in-house.** From chip to CANN to HCCL to vllm-ascend, Ascend is a fully self-developed stack; PPU takes the CUDA-compatible route (fast migration, ecosystem leverage — at the cost of staying bound to CUDA semantics underneath). Under the "self-controllable" policy climate, that difference is a hard requirement for some customers, not something a throughput number replaces.

## Finding 3 (the expensive one): tuning doesn't port even between these two

The one to remember. The same vLLM flag, its fate on each:

| Flag | Alibaba PPU | Huawei Ascend | Portable? |
|---|---|---|---|
| **`--kv-cache-dtype fp8`** | ❌ **crash on launch** (fp8e4nv unsupported) | ❌ **silent 0 tokens** | ❌ **dead on both, differently** |
| `--no-enable-prefix-caching` | −34% | −12% | ⚠️ don't disable either, but magnitudes differ |
| `--max-num-seqs 64` (too small) | −22% | −31% | ⚠️ don't shrink either |
| W8A8 quant | untested | +26% ✅ | ❌ can't assume PPU has it |
| AWQ quant | untested | ❌ unsupported | ❌ vendor-specific |

Look at row one, `kv-fp8`. It's a free lunch on NVIDIA (A100 +18.5%, 4090 +38%), **so every NVIDIA guide tells you to enable it.** Ported to domestic silicon:

- On **PPU** — **crash on launch**: PPU's architecture supports `fp8e4b15/fp8e5` but not vLLM's default `fp8e4nv`, so serve won't come up. At least you **know instantly it's dead.**
- On **Ascend** — **sneakier**: flag accepted, server starts, health check green, then **throughput is 0.** An endpoint that looks alive but is dead.

**Same flag: PPU crashes in your face, Ascend lies that it's fine then emits not a single token.** What does that mean? **You can't tune Ascend with PPU's checklist, or vice versa** — even *which* flag dies differs, let alone *how* it dies after.

That's what "domestic-accelerator adaptation" really means: **the moat isn't memorizing each vendor's SDK, it's the methodology to bring any backend in, test every flag on real hardware, and know which to turn on and which is a landmine.** If PPU and Ascend can't copy each other, NVIDIA's playbook is even further off.

## How to choose? Depends on what you actually care about

| Your need | Pick | Why |
|---|---|---|
| **Max raw throughput / cost-efficiency** | Alibaba PPU | 8-card 32B is 2.6x Ascend, big VRAM sustains high concurrency |
| **Multi-card production stability first** | Huawei Ascend | First through 32B multi-card, HCCL a version cycle ahead |
| **Need quant to cut cost** | Huawei Ascend (today) | W8A8 +26% verified, PPU TBD |
| **Autonomy / policy compliance as hard req** | Huawei Ascend | Chip-to-framework fully in-house |
| **Must fit 32B on a single card** | Alibaba PPU | 98G does it single-card, Ascend needs 4 |

**There's no "one crushes the other" answer.** PPU is "fast but not yet complete," Ascend is "steady but not fast." In real selection these often aren't either/or but "which for which scenario" — PPU for high-throughput inference pools, Ascend for core services needing quant and top stability.

## Wrap-up

**PPU is 2.6x faster; Ascend is a version cycle steadier and holds an extra quant card.** But their real shared conclusion is that kv-fp8 table: **domestic-card tuning doesn't port between vendors, let alone from NVIDIA.** That's the thing to take away when choosing domestic silicon — not "who benchmarks higher," but "whether you can tune any given card to its best on real hardware."

---

*This series: [①9-card single](../domestic-accelerators-9-card-vllm/) · [②7B multi-card TP](../domestic-accelerators-multicard-tp/) · [③A100 vs PPU](../a100-vs-alibaba-ppu-32b-multicard/) · [④Ascend deep-dive](../ascend-910b4-deep-8card-32b/) · ⑤ this post. Methodology and raw JSON in the [notex repo](https://github.com/wangmin362). Every number is from real hardware, none invented.*
