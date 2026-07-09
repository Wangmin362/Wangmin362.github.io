# wangmin362.github.io

Source for my personal blog. I publish **reproducible LLM-inference benchmarks across heterogeneous / domestic accelerators** — the same model, same load, run head-to-head on NVIDIA vs Ascend / Cambricon / Hygon / Moore Threads / Kunlunxin — plus the routing, KV-cache, and cost trade-offs behind them. Every number ships with the exact command to reproduce it.

🔗 **Live site:** <https://wangmin362.github.io/>

## Stack

- [Hugo](https://gohugo.io/) (extended, ≥ 0.146.0) + [PaperMod](https://github.com/adityatelange/hugo-PaperMod) theme (git submodule)
- Deployed to **GitHub Pages via GitHub Actions** ([`.github/workflows/hugo.yml`](.github/workflows/hugo.yml)) on every push to `main`

## Structure

```
├── hugo.toml                 # site config
├── content/
│   ├── posts/                # blog posts (one file per post)
│   ├── about.md
│   └── search.md
├── archetypes/               # front-matter template for `hugo new`
├── static/                   # static assets
└── themes/PaperMod/          # theme (submodule)
```

## Run locally

```bash
git clone --recurse-submodules https://github.com/Wangmin362/wangmin362.github.io.git
cd wangmin362.github.io
hugo server -D          # -D includes drafts; open http://localhost:1313
```

Posts live in `content/posts/`. A post is published once its front matter has `draft: false`; push to `main` and the Actions workflow builds and deploys it automatically.

## License

Content © Wangmin362, all rights reserved. Code/config in this repo may be reused freely.
