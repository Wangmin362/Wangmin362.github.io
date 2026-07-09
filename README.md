# het-infra-blog

我的技术博客骨架 —— 承载「异构推理横评」内容护城河（策略见 vault 笔记《影响力建设-内容护城河策略-2026Q3》）。

- **框架**：Hugo + PaperMod 主题
- **托管**：GitHub Pages（服务端自动构建，本地不用装 Hugo）
- **主线**：内容 = 90 天周作品的公开副产品，不是新增任务。W1 压测数据出来 → 填模板 → push → 上线。

---

## 一、一次性上线（约 15 分钟，照抄即可）

### 1. 补上 PaperMod 主题（本仓库没带主题，用 submodule 拉）

```bash
cd /mnt/d/notebook/het-infra-blog
git init
git add -A && git commit -m "chore: blog scaffold"
git submodule add --depth=1 https://github.com/adityatelange/hugo-PaperMod themes/PaperMod
git commit -m "chore: add PaperMod theme"
```

### 2. 改 3 个占位符

- `hugo.toml`：`baseURL`、`author`、社交链接里的 `<你的GitHub用户名>`/`<你的handle>`
- `content/about.md` 结尾的 `<FILL>`

改完：
```bash
git commit -am "chore: fill in site identity"
```

### 3. 建 GitHub 仓库并推上去

在 GitHub 新建一个**公开**仓库（名字随意，如 `het-infra-blog`），然后：
```bash
git branch -M main
git remote add origin https://github.com/<你的用户名>/het-infra-blog.git
git push -u origin main
```

### 4. 打开 GitHub Pages 的 Actions 构建

GitHub 仓库页 → **Settings → Pages → Build and deployment → Source** 选 **GitHub Actions**（不是 "Deploy from a branch"）。

推上去后，Actions 页会自动跑 `Deploy Hugo site to Pages`。绿了就能访问 `https://<你的用户名>.github.io/`。

> 之后每次 `git push`，网站自动重建，无需任何本地工具。

---

## 二、写一篇新文章（日常流程）

```bash
# 复制横评模板改名
cp content/posts/0000-benchmark-template.md content/posts/qwen7b-ascend-vs-a100.md
```

1. 打开新文件，把所有 `<FILL: ...>` 换成**真机数据/真实观察**（一个都别编 —— 那是护城河）。
2. `draft: true` 改成 `draft: false`。
3. `git add -A && git commit -m "post: qwen7b ascend vs a100" && git push`。

`draft: true` 的文章不会发布，可以安心占坑先写一半。

---

## 三、可选：本地预览（需要装 Hugo，不装也能发）

```bash
# 装 Hugo extended 后：
hugo server -D        # -D 连草稿一起预览，浏览器开 http://localhost:1313
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
