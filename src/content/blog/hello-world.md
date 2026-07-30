---
title: 'Hello world'
description: 'Why I started this blog, and how it is put together.'
pubDate: 2026-07-30
tags: ['meta']
---

This is the first post. If you can read it, the whole pipeline works: Markdown in
a repo, built by GitHub Actions, served by GitHub Pages.

## Writing a new post

Create a file in `src/content/blog/`. The filename becomes the URL, so
`my-second-post.md` publishes at `/blog/my-second-post/`. Every post starts with
a frontmatter block:

```markdown
---
title: 'My second post'
description: 'One sentence, used on the homepage and in the RSS feed.'
pubDate: 2026-08-01
tags: ['notes']
---

Then just write **Markdown**.
```

Only `title`, `description`, and `pubDate` are required. Add `draft: true` to
keep a post out of the published site while you work on it.

## Things that already work

- **Syntax highlighting** in fenced code blocks, with a light and dark theme
- An **RSS feed** at `/rss.xml` and a sitemap for search engines
- **Dark mode**, following your operating system setting
- Social preview tags, so links unfurl properly when shared

## Publishing

Commit and push to `main`. A GitHub Action builds the site and deploys it — no
manual step, and usually under a minute.

That's the whole system. Now the only hard part left is having something to say.
