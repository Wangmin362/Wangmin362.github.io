# wangmin362.github.io

我的技术博客 —— 承载「异构推理横评」内容护城河（策略见 vault 笔记《影响力建设-内容护城河策略-2026Q3》）。

- **在线地址**：<https://wangmin362.github.io/> ✅ 已上线
- **框架**：Hugo + PaperMod（GitHub Actions 自动构建部署，本地不用装 Hugo）
- **主线**：内容 = 90 天周作品的公开副产品，不是新增任务。W1 压测数据出来 → 填模板 → `git push` → 自动上线。

---

## 一、现状（一次性设置都已完成，无需再做）

| 项 | 状态 |
|---|---|
| git 仓库 + remote（origin → GitHub） | ✅ 已配好 |
| PaperMod 主题（submodule） | ✅ 已装 |
| 站点配置 `hugo.toml`（baseURL / author / GitHub 链接） | ✅ 已填 |
| GitHub Pages 构建源 = GitHub Actions | ✅ 已开启 |
| Actions 部署工作流 `.github/workflows/hugo.yml` | ✅ 跑通（Hugo 0.146.0） |

> 想验证：`cd /mnt/d/notebook/wangmin362.github.io && git log --oneline` 看提交；浏览器开 <https://wangmin362.github.io/>。

---

## 二、写一篇新文章（这才是你的日常流程）

```bash
cd /mnt/d/notebook/wangmin362.github.io

# 1. 复制横评模板改成真实标题
cp content/posts/0000-benchmark-template.md content/posts/qwen7b-ascend-vs-a100.md
```

2. 打开新文件，把所有 `<FILL: ...>` 换成**真机数据 / 真实观察**（一个都别编 —— 那是护城河）。
3. 写好后把 front matter 里的 `draft: true` 改成 `draft: false`（草稿不会发布，可安心占坑先写一半）。
4. 推送即上线：

```bash
git add -A
git commit -m "post: qwen7b ascend vs a100"
git push
```

`git push` 后 GitHub Actions 自动构建，约 1 分钟后 <https://wangmin362.github.io/> 更新。看构建状态：`gh run list --limit 1`。

---

## 三、可选：本地预览（要装 Hugo；不装也能发）

```bash
# 装 Hugo extended（需 ≥0.146.0，PaperMod 要求）后：
hugo server -D    # -D 连草稿一起预览，浏览器开 http://localhost:1313
```

不想装就跳过，直接 push 靠 Actions 构建即可。

---

## 四、发布后的分发（别只发一处）

- **英文**：X thread（贴 TL;DR 表 + 链接）→ Reddit r/LocalLLaMA → 数据够硬可投 Hacker News
- **中文**：同一篇翻成中文发 掘金 / 知乎 / 公众号（国产卡话题国内更吃香）
- **最高信号**：把压测脚本开源成独立 repo，文章里链过去；相关坑给 vllm-ascend / GAIE 提 issue 或 PR

## 五、防 AI 水文三铁律（贴屏幕上）

1. 每篇必须有一个「只有我能给」的东西（真机数据 / 能跑的仓库 / 真实踩坑），给不出别发。
2. AI 只用来润色排版，不用来生成观点和数据。
3. 写给三个月前的自己，不追赞数；不日更，双周一篇深文即可。

---

## 六、排障（踩过的坑，遇到了照这里做）

- **push 报 `refusing to allow ... workflow ... without workflow scope`**
  改动了 `.github/workflows/` 里的文件时才会触发。需令牌带 `workflow` 权限：`gh auth refresh -h github.com -s workflow`（已做过一次，一般不用再弄）。
- **权限明明够了还是被拒 / push 用了旧令牌**
  git 的 `cache` 凭证助手缓存了旧令牌。清一下即可：`git credential-cache exit`，再 `git push`。
- **Actions 构建失败，日志写 `hugo vX or greater is required for hugo-PaperMod`**
  PaperMod 升级了、要求更高 Hugo 版本。改 `.github/workflows/hugo.yml` 里的 `HUGO_VERSION` 到日志要求的版本，commit push。
- **改了文章但网站没变**
  确认 `draft: false`；`gh run list --limit 1` 看构建是否绿；Actions 成功后 CDN 可能还要缓存几十秒。
