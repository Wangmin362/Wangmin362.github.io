---
title: "kv-fp8: +18% on A100, a Hard Crash on Alibaba PPU — A100 vs Domestic Silicon on 32B Multi-Card"
date: 2026-07-15
draft: true
tags: ["benchmark", "vLLM", "tensor-parallel", "domestic-accelerators", "Alibaba-PPU", "A100", "Ascend", "kv-cache", "LLM-inference"]
summary: "Same Qwen3-32B, same load, now with NVIDIA A100 and Alibaba PPU side by side across 1/2/4/8 cards. Two things worth remembering: (1) Alibaba PPU pushes 32B to 3215 tok/s on eight cards, near-linear 7.9× — after its multi-card runs literally crashed last round; (2) the same `--kv-cache-dtype fp8` is a +18% free lunch on A100, a hard `fp8e4nv not supported` crash on PPU, and silent zero-throughput on Ascend. Tuning does not port across vendors — that's the actual work of domestic-accelerator adaptation."
ShowToc: true
---


> **TL;DR** — Part three of the real-machine cross-vendor series, now adding **NVIDIA A100-80G** as a datacenter-class baseline against **Alibaba PPU-ZW810E**, running the *same* Qwen3-32B under the *same* load on 1/2/4/8 cards. Three things. **(1) Alibaba PPU pushes 32B to 3215 tok/s on eight cards (near-linear 7.9×)** — note that [last round](../domestic-accelerators-multicard-tp/) its 32B multi-card runs *crashed outright* with NCCL errors; a different image stack this round just works, and it's **2.6×** Ascend's 8-card number (1237). **(2) The single-card ceiling is still NVIDIA** (A100 tops out at 482 tok/s on one card); domestic cards catch up with more cards and bigger VRAM. **(3) The most expensive lesson: tuning does not port across vendors.** `--kv-cache-dtype fp8` is a **+18.5%** free lunch on A100, a **hard crash** on PPU (`type fp8e4nv not supported`), and **silent zero-throughput** on Ascend. Copy an NVIDIA tuning checklist onto domestic silicon and you'll ship a dead endpoint.

This is part three of a real-machine cross-vendor series. [Part one](../domestic-accelerators-9-card-vllm/) ranked nine accelerators on a single card; [part two](../domestic-accelerators-multicard-tp/) swept 7B across multiple cards. This post moves to a **bigger model (32B) and a datacenter-class opponent (A100)**, to answer two questions: has domestic multi-card matured in the last six months, and can tuning parameters be reused across vendors?

## How it was measured (reproducible)

- **Models**: Qwen3-32B (dense, 64 heads), Qwen3-4B, Qwen3-Coder-30B-A3B (MoE). All three vendors run the *same* models — that's the only way to compare.
- **Load**: `vllm bench serve`, random dataset, `in=512 / out=128`, concurrency swept `1 / 4 / 16 / 64 / 128 / 256` to saturation, `--ignore-eos --seed 42`, dtype bf16.
- **Hardware**: NVIDIA A100-SXM4-80GB ×2 (NVLink, vLLM 0.20.0) · Alibaba PPU-ZW810E ×8 (98 GB/card, vLLM 0.20.1). Ascend 910B4 numbers come from [part two](../domestic-accelerators-multicard-tp/), same methodology.
- **One critical engineering detail**: models must load from **local disk**. This A100 is a remote cloud node; its NFS mount reads at **6 MB/s** — vLLM loading a 32B model just hangs (weights reach the GPU but the engine never comes up). Switch to node-local disk (1.5–5 GB/s) and it starts in seconds. **Slow storage masquerades as an "engine hang" — don't get tricked into debugging CUDA.**

## 32B multi-card scaling, in one chart

![Cross-vendor Qwen3-32B multi-card scaling: per-vendor 1/2/4/8-card peak throughput and 8-card peak bars](/img/a100-ppu/compare-cross-vendor-32b.png)

Peak output throughput (tok/s) for Qwen3-32B, `in=512 / out=128`, swept to saturation:

| Card | VRAM/card | TP1 | TP2 | TP4 | TP8 | 32B multi-card |
|---|---|---:|---:|---:|---:|---|
| **NVIDIA A100-80G** (NVLink) | 80G | **482** | **1508** | —\* | —\* | ✅ (only 2 cards here) |
| **Alibaba PPU-ZW810E** | 98G | 406 | 1128 | 2005 | **3215** | ✅ all working (new) |
| **Ascend 910B4** | 32G | — | — | 583 | 1237 | ✅ |
| **Kunlunxin P800** | 98G | — | — | ❌ | ❌ | ❌ worker crash |

\* This A100 box only has 2 cards, so TP4/8 aren't testable here.

## Finding 1: Alibaba PPU's multi-card genuinely matured

In [part two](../domestic-accelerators-multicard-tp/), Alibaba PPU's 32B multi-card runs **crashed** — cross-card comms threw NCCL errors; 7B ran, 32B died the moment it went multi-card. The conclusion then was "domestic cards run 7B but fall over on 32B multi-card; only Ascend made it through."

This round, with PPU's own inference image stack (asllm + vLLM 0.20.1), **32B runs from 1 to 8 cards, all successful, and near-linear**:

```
PPU Qwen3-32B:  406 →  1128 →  2005 → 3215 tok/s
                1×     2.78×   4.94×  7.9×
```

3215 tok/s on eight cards — **2.6×** Ascend's 8-card number (1237). PPU's 98 GB of VRAM plus a fixed comms stack, both paying off. **The "domestic cards can't do 32B multi-card" conclusion from six months ago is already stale.** Which is exactly why real-machine benchmarks have to be re-run: vendor stacks iterate fast, and last year's conclusion is not this year's fact.

## Finding 2: big models scale super-linearly, small models waste extra cards

Put all three models' scaling together (PPU, 8-card ÷ 1-card):

| Model | TP1 | TP8 | TP8/TP1 |
|---|---:|---:|---:|
| **Qwen3-32B** (big) | 406 | 3215 | **7.9×** |
| MoE-30B-A3B | 2223 | 7806 | 3.5× |
| **Qwen3-4B** (small) | 3568 | 12104 | 3.4× |

**The bigger the model, the better multi-card scales.** 32B is near-linear on eight cards (7.9×); 4B only reaches 3.4×. Big models are compute-heavy — sharding across cards both thins out the weights (freeing KV space) and parallelizes compute, so the payoff is large. Small models are already fast on one card, so cross-card comms overhead grows as a fraction and returns diminish. A100 shows the same thing (32B 3.13× super-linear on two cards, 4B only 1.53×).

**Selection takeaway**: the bigger the model you serve, the more cards earn their keep; throwing eight cards at a 4B model leaves most of them idling.

## Finding 3 (the expensive one): tuning does not port across vendors

This is the one table to remember from the whole series. The *same* vLLM flag behaves **wildly differently** across vendors:

| Flag | A100 | Alibaba PPU | Ascend |
|---|---|---|---|
| **`--kv-cache-dtype fp8`** | ✅ **+18.5%** | ❌ **crashes the server** (`fp8e4nv not supported`) | ❌ **silent 0 tokens** |
| `--no-enable-prefix-caching` | −30% | −34% | — |
| `--max-num-seqs 64` (too small) | −17% | −22% | throughput collapse |
| `--gpu-memory-utilization 0.95` | +4% | 0 (98G is plenty) | 0 |

Look at row one. `--kv-cache-dtype fp8` (quantize the KV cache to 8 bits) is a textbook free lunch on NVIDIA — **+18.5%** on A100 with negligible quality loss. So every NVIDIA tuning guide tells you to turn it on.

But:

- On **Alibaba PPU**, it **stops the server from starting** — the PPU architecture supports `fp8e4b15 / fp8e5` but not vLLM's default `fp8e4nv`, so serve crashes on launch.
- On **Ascend**, it's sneakier: the flag is accepted, the server starts fine, and then **throughput is 0** — an endpoint that looks alive but is dead.

**If you copy an NVIDIA tuning checklist verbatim onto domestic silicon, best case your performance drops, worst case you ship a crashing or silently-broken service.** Every card's tuning must be re-measured on real hardware — that's not work you can skip by "copying a best-practices doc." **And that's precisely the value of "domestic-accelerator adaptation": the moat isn't memorizing eight vendor SDKs, it's the methodology to bring any backend in, tune it to its best, and know exactly what it costs.**

## Bonus: workload shape swings throughput 4–6.5×

One trap everyone hits — **same card, same TP, different in/out ratio, several-fold difference in throughput** (Qwen3-32B):

| Shape | in/out | A100 (2 cards) | PPU (4 cards) | Scenario |
|---|---|---:|---:|---|
| Long generation | 128/512 | 3831 | 4382 | writing/continuation (highest) |
| Balanced | 1024/1024 | 1728 | 3014 | long chat |
| Short Q&A | 512/128 | 1508 | 2005 | chat |
| Long context | 2048/256 | 587 | 1088 | RAG (lowest) |

**Longer output = higher throughput (decode-heavy); longer input = lower throughput (prefill-heavy)** — consistent across both vendors, up to a 4–6.5× spread. So any "this card does X tok/s" number that doesn't state its in/out ratio is meaningless — **benchmark with your own workload's shape.**

## Wrap-up

Three cross-vendor benchmarks in, one thing is increasingly clear: **conclusions about domestic accelerators have a short shelf life.** Six months ago "PPU can't do 32B multi-card"; today it's near-linear to 3215 tok/s on eight cards. Last year's tuning best-practice might crash your service on a different card today.

So this series keeps going. **Every number is from real hardware, none invented** — because reproducible real-machine data is the one thing you can't fake and can't copy.

*(Methodology, the raw per-config JSON, and the full write-up of turning a user-created inference service into a benchmark harness are in the [notex repo](https://github.com/wangmin362). More vendors' cards are next — I test them as the environments come online.)*
