---
# ┌─────────────────────────────────────────────────────────────────┐
# │ 这是你的横评文章模板。用法:                                       │
# │ 1. 复制这个文件，改名为真实标题，如 qwen7b-ascend-vs-a100.md      │
# │ 2. W1 压测数据一出来，把所有 <FILL: ...> 替换掉                    │
# │ 3. 把 draft 改成 false → git push → 自动上线                      │
# │ 铁律:每个 <FILL> 都是真机数据/真实观察,一个都不许编 —— 那是护城河 │
# └─────────────────────────────────────────────────────────────────┘
title: "<FILL: 具体到吓人的标题，如 'Qwen2.5-7B on Ascend 910B vs A100: TTFT, Throughput, and What It Costs'>"
date: 2026-07-12                      # ← 改成发布日期
draft: true                           # ← 填完数据后改成 false
tags: ["benchmark", "vLLM", "Ascend", "NVIDIA", "LLM-inference", "domestic-accelerator"]
summary: "Same model, same load, run head-to-head on <卡A> vs <卡B>. Reproducible numbers — every result ships with the exact command."
ShowToc: true
---

> **TL;DR** — <FILL: 一句话结论，带最关键的一个数字。例如：On a single card, Ascend 910B hit **<X>% of A100's throughput** at 32 concurrency, but TTFT was **<Y>ms higher** because of <原因>. Full numbers and repro steps below.>

一张表把结论砸在读者脸上（Julia Evans 定律：大多数人只扫表格和标题）：

| Metric (single card, bs=<FILL>) | <卡A, 如 A100 80G> | <卡B, 如 Ascend 910B> | <卡C…> |
|---|---|---|---|
| TTFT p50 / p99 (ms) | `<FILL>` | `<FILL>` | `<FILL>` |
| Throughput (output tok/s) | `<FILL>` | `<FILL>` | `<FILL>` |
| TPOT / inter-token (ms) | `<FILL>` | `<FILL>` | `<FILL>` |
| Peak KV-cache mem (GB) | `<FILL>` | `<FILL>` | `<FILL>` |
| Max concurrency before OOM | `<FILL>` | `<FILL>` | `<FILL>` |

---

## Why this benchmark exists（这篇为什么值得存在）

<FILL，但别删这段的骨架——它就是你的护城河宣言：>
几乎所有公开的 LLM 推理 benchmark 都只跑 NVIDIA。国产卡（昇腾 / 寒武纪 / 海光 / 沐曦 / 昆仑芯）的**同口径 head-to-head 数据近乎空白**——不是因为不重要，而是因为**很少有人能同时摸到这些卡**。我恰好能，所以我把它跑出来、公开出来。

> 这一段是你和"AI 水文"的分界线：说清"我有什么别人没有的东西"。

## What's under test（被测对象，越具体越可信）

| 维度 | 值 |
|---|---|
| Model | `<FILL: 如 Qwen2.5-7B-Instruct, 精度 BF16/FP8>` |
| Inference engine | `<FILL: 如 vLLM 0.6.x / vllm-ascend / MindIE，带版本号>` |
| Hardware A | `<FILL: 卡型号 + 显存 + 驱动/CANN/CUDA 版本>` |
| Hardware B | `<FILL>` |
| Dataset / prompts | `<FILL: 如 ShareGPT 500 条 / 固定 input=512 output=128>` |
| Load pattern | `<FILL: 并发数扫描 1→8→32→64，或固定 QPS>` |

## Methodology（可复现性 = 可信度）

<FILL，写清任何人照做能复现的每一步：>
- Warmup: `<FILL>` 轮丢弃
- 每个配置跑 `<FILL>` 次取中位数
- 计时口径: TTFT = 首 token 返回时刻 − 请求发出时刻；Throughput = 稳态输出 token / 墙钟时间
- 测量工具: `<FILL: 如 vllm bench serve / 自写 Go 压测脚本，附链接>`

> 没有 methodology 的 benchmark 等于没有 benchmark。这一节写扎实，比结论更值钱。

## Results

### 1. Time to First Token (TTFT)

`<FILL: 图或表>`

**What I saw:** <FILL: 一句人话观察。如 "Ascend 的 TTFT 在低并发几乎追平 A100，但并发拉到 64 时被 <原因> 拖开差距。">

### 2. Throughput under concurrency

`<FILL: 并发 vs 吞吐曲线的表/图>`

**What I saw:** <FILL>

### 3. 代价：perf-per-card / perf-per-¥（这一节是你的加分项）

<FILL：这正是你职业上最该练、面试最能打的"算代价"。哪怕粗算：>
- 同样打满，达到 `<目标 QPS>` 需要几张 A卡 vs 几张 B卡？
- 按各卡大致单价/功耗，跑同样负载谁的 **每 token 成本 / 每瓦吞吐** 更低？
- 结论要诚实：`<FILL：谁在什么场景更划算，什么场景不划算>`

> 大多数 benchmark 只比"快不快"，不比"值不值"。你比了 —— 这就是"懂内部 + 算得清代价"的公开证据。

## What surprised me / gotchas（AI 编不出来的部分）

<FILL：这一节是"货真价实"的心脏。写你真实踩的坑：>
- <FILL: 如 "vllm-ascend 上 `--enable-prefix-caching` 行为和 NVIDIA 不一致，表现是……">
- <FILL: 如 "精度从 BF16 切 INT8 后，输出对不齐，排查发现……">

> 这些"意外"就是别人复制不了的东西。越具体、越狼狈，越可信。

## Reproduce it yourself（把复现门槛降到最低）

```bash
# <FILL: 从起容器到出数字的完整命令，能 copy-paste 跑通>
# 例:
# docker run --rm -it <镜像:tag> \
#   vllm serve <model> --port 8000 <flags...>
# python bench.py --host ... --concurrency 1,8,32,64 --input-len 512 --output-len 128
```

版本锁定（不锁版本的 benchmark 三个月后就是垃圾）：
| 组件 | 版本 |
|---|---|
| engine | `<FILL>` |
| driver / CANN / CUDA | `<FILL>` |
| model commit | `<FILL>` |

> 配套代码/脚本仓库：<FILL: GitHub 链接，最好就是你的压测脚本 repo>

## Caveats & what I'd test next（诚实收尾，比吹牛可信）

<FILL：>
- 本次**没**测的：`<FILL: 如长上下文 / 多卡 TP / 量化组合>`
- 单卡样本，不代表集群表现
- 下一篇想测：`<FILL>`

---

*Found this useful, or spotted a methodology hole? Open an issue on <repo链接> or ping me on <X/邮箱>. I'd rather be corrected in public than wrong in private.*
