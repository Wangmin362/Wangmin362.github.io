---
title: "Qwen2.5-7B on RTX 4090D vs Tesla V100: Throughput, Latency, and the Software Cliff"
date: 2026-07-10
draft: true
tags: ["benchmark", "vLLM", "NVIDIA", "RTX-4090", "V100", "LLM-inference"]
summary: "Same model, same load, head-to-head on a consumer RTX 4090D vs a 2017 Tesla V100. Reproducible numbers — every result ships with the exact command. Plus: why the V100 couldn't even run modern vLLM."
ShowToc: true
---


> **TL;DR** — Same Qwen2.5-7B, same load, one card each. At 64 concurrency the RTX 4090D pushes **1316 output tok/s vs the V100's 427 — a 3.1× gap that *widens* with load** (only 1.6× at concurrency 1). But the sharper finding is upstream of any number: **the V100 can't run modern vLLM at all** — its Volta `sm_70` architecture was dropped from the prebuilt PyTorch/vLLM, so it's stuck two years back on the engine. Full numbers and repro steps below.

A table to put the conclusion in your face (Julia Evans' law: most people only scan tables and headings):

| Metric (single card, concurrency 64) | RTX 4090D (24GB) | Tesla V100 (32GB) |
|---|---|---|
| Output throughput (tok/s) | **1316** | 427 |
| TTFT p50 / p99 (ms) | 864 / 3386 | 4152 / 8381 |
| TPOT p50 / p99 (ms) | 40 / 46 | 108 / 150 |
| Engine it could run | vLLM v0.20.0 | vLLM **v0.8.5** (Volta dropped from v0.20) |

---

## Why this benchmark exists

Almost every public LLM inference benchmark runs one card in isolation, on whatever hardware the author happened to have. Head-to-head numbers on the *same model, same load, same measurement* across a modern card and an older one are rarer than they should be — and cross-vendor / cross-generation data is where the interesting decisions actually live. I have a lab with a pile of different accelerators, so I run them side by side and publish the numbers with the exact commands.

This first post is the NVIDIA baseline: a current consumer card (RTX 4090D) against a 2017 datacenter card (V100) that's still extremely common in Chinese clusters. It sets the methodology every later post (domestic accelerators) reuses.

## What's under test

| Dimension | Value |
|---|---|
| Model | `Qwen2.5-7B-Instruct`, dtype **float16** (both cards; V100's Volta has no bf16 hardware, so fp16 is the fair, production-realistic choice) |
| Hardware A | 1× **RTX 4090D**, 24GB, Ada `sm_89`, driver 595.71.05 / CUDA 13.2 |
| Hardware B | 1× **Tesla V100-PCIE**, 32GB, Volta `sm_70`, same cluster |
| Engine A | vLLM **v0.20.0** (torch 2.11.0+cu130) |
| Engine B | vLLM **v0.8.5.post1** (torch 2.6.0+cu124) — *forced*, see gotchas |
| Load | `random` dataset, fixed **input=512 / output=128** tokens, `--ignore-eos` |
| Sweep | max-concurrency ∈ {1, 4, 16, 64}, `--request-rate inf` (closed loop) |

## Methodology

- One warmup run (concurrency 4, 20 prompts) discarded before每 sweep.
- `--ignore-eos` forces every request to emit exactly 128 output tokens → clean, comparable TPOT.
- Fixed `--seed 42`; num-prompts scaled with concurrency (50 / 100 / 200 / 400).
- TTFT = first-token time − request-send time; TPOT = per-output-token time excluding the first; throughput = output tokens / wall-clock.
- Measured with vLLM's own `vllm bench serve`. Both cards ran the identical model weights (same NFS-mounted checkpoint) and identical request shape.

> No methodology, no benchmark. The one caveat that matters: **the two cards ran different engine versions** (the V100 literally cannot run the new one — see below), so this is "each card at its practical best," not "same engine, different silicon." I call that out rather than hide it.

## Results

### Head-to-head

![4090D vs V100 — TTFT p99, TPOT p99, throughput](compare.png)

Full per-card sweeps:

| conc | 4090D tok/s | 4090D TTFT p50/p99 | 4090D TPOT p50/p99 | V100 tok/s | V100 TTFT p50/p99 | V100 TPOT p50/p99 |
|---:|---:|---:|---:|---:|---:|---:|
| 1  | 61.4  | 72.6 / 78.5    | 15.9 / 16.0 | 38.5  | 163.7 / 284.6  | 23.8 / 26.8  |
| 4  | 227.0 | 100.2 / 223.9  | 16.7 / 17.7 | 119.2 | 553.9 / 621.7  | 30.1 / 33.7  |
| 16 | 622.4 | 147.1 / 687.8  | 18.5 / 74.1 | 293.8 | 1172 / 1848    | 42.8 / 54.4  |
| 64 | 1316.3| 864.2 / 3385.9 | 39.9 / 45.9 | 427.4 | 4152 / 8381    | 107.6 / 150.0|

Per-card detail: [4090D](curve-4090.png) · [V100](curve-v100.png)

**What I saw:** The throughput gap *scales with load*. At concurrency 1 the 4090D is only 1.6× the V100 (61 vs 38 tok/s) — if you're serving one stream at a time, the old card is fine. But by concurrency 64 the gap is 3.1× (1316 vs 427). The newer architecture (Ada + FlashAttention-2 + the newer scheduler) only cashes out its advantage when the batch is *full*. Buy new silicon for throughput-bound serving, not for a single low-QPS stream.

### The perf-per-¥ angle

Most benchmarks compare "fast vs slow." The question that actually matters for a platform is "cheap vs expensive per token."

- Rough street prices (**verify these**): RTX 4090D ≈ ¥14k new; Tesla V100 32GB ≈ ¥6–8k used (EOL).
- At concurrency 64: 4090D ≈ 1316 tok/s / ¥14k ≈ 0.094 tok/s per ¥; V100 ≈ 427 / ¥7k ≈ 0.061 tok/s per ¥.
- So the 4090D is roughly **1.5× better tokens-per-¥** *and* you're not two years behind on the engine. The V100 only makes sense if you already own it and the racks are sunk cost.

## What surprised me / gotchas (the part AI can't fake)

- **The V100 fell off a software cliff, not a performance cliff.** Booting Qwen2.5-7B on the current vLLM image failed instantly — not slow, *dead*: `CUDA error: no kernel image is available for execution on the device`, on a plain `torch.zeros`. The image's `torch 2.11.0+cu130` is compiled for `['sm_75, sm_80, sm_86, sm_90, sm_100, sm_120']` — **`sm_70` (Volta) is gone**. The V100 only ran once I dropped back to vLLM **v0.8.5** (torch 2.6, which still ships `sm_70`). The real cost of an old card isn't that it's slower — it's that the modern stack stops shipping kernels for it, and you silently lose two years of engine optimizations. That's invisible in any single-card tok/s number.
- **A TPOT-p99 crossover.** The 4090D's *tail* TPOT actually *dropped* from 74ms at concurrency 16 to 46ms at concurrency 64, while the V100 climbed monotonically to 150ms. My read: at 16 the new scheduler occasionally starves a decode behind a prefill chunk; by 64 the batch is dense enough that it evens out. Worth a dedicated scheduling-params experiment.
- **Getting the weights onto the node was harder than the benchmark.** `docker.io` is blocked, HuggingFace is blocked, and `hf-mirror` doesn't proxy HF's new Xet storage backend (so weight downloads silently time out on `cas-bridge.xethub.hf.co`). ModelScope worked but crawled at ~1.6 MB/s. The fix that actually worked: mount the cluster's existing model PVC (RWX/NFS) read-only and serve from the local path — zero download.

## Reproduce it yourself

```bash
# Serve (inside a GPU pod; model from a local/NFS path avoids the download saga)
vllm serve /models/Qwen2.5-7B-Instruct --served-model-name qwen2.5-7b \
  --dtype float16 --gpu-memory-utilization 0.90 --max-model-len 4096 --port 8000

# Sweep one concurrency level (repeat for 1,4,16,64), 4090D / vLLM v0.20:
vllm bench serve --backend vllm --model qwen2.5-7b \
  --tokenizer /models/Qwen2.5-7B-Instruct --host 127.0.0.1 --port 8000 \
  --dataset-name random --random-input-len 512 --random-output-len 128 --ignore-eos \
  --percentile-metrics ttft,tpot,itl --metric-percentiles 90,99 --seed 42 \
  --max-concurrency 64 --num-prompts 400 --save-result --result-filename c64.json

# On the V100 / vLLM v0.8.5 the bench CLI differs:
#   replace  --backend vllm   with   --endpoint-type openai-comp --endpoint /v1/completions
```

Version lock (an unlocked benchmark is garbage in three months):

| Component | 4090D | V100 |
|---|---|---|
| engine | vLLM 0.20.0 | vLLM 0.8.5.post1 |
| torch / CUDA | 2.11.0+cu130 / driver 595.71.05 (CUDA 13.2) | 2.6.0+cu124 |
| model | Qwen2.5-7B-Instruct (fp16) | same checkpoint |

## Caveats & what I'd test next

- Different engine versions across the two cards (the V100 gave me no choice). A cleaner comparison would pin the same old engine on both — but then the 4090D wouldn't show its real-world best. I chose "each card as you'd actually deploy it."
- Single card, single model, one input/output shape. No TP, no long context, no quantization.
- Next: the same head-to-head on **domestic accelerators** (Ascend / Cambricon / Hygon / MetaX) — the numbers almost nobody publishes, and the whole reason this blog exists.

---

*Spotted a methodology hole? I'd rather be corrected in public than wrong in private — open an issue or ping me.*
