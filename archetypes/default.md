---
title: "{{ replace .Name "-" " " | title }}"
date: {{ .Date }}
draft: true          # true=草稿:只有本地 `hugo server -D` 能看,不上公网。写好改成 false 再 push 发布
tags: []
summary: ""
# ── 想"发到线上、但别人搜不到、只有拿到网址的人能看"?取消下面 3 行注释(并把上面 draft 改成 false)──
# _build:
#   render: always   # 照常生成这一页的网址
#   list: never      # 但从 文章列表/首页/RSS/搜索/站点地图 全隐藏;定稿后删掉整个 _build 块即完全公开
---
