# nastya236.github.io

My blog. Built with [Astro](https://astro.build), published at
<https://nastya236.github.io>.

## Writing a post

Create a Markdown file in `src/content/blog/`. The filename becomes the URL, so
`src/content/blog/my-post.md` publishes at `/blog/my-post/`.

```markdown
---
title: 'My post'
description: 'One sentence. Shows on the homepage and in the RSS feed.'
pubDate: 2026-08-01
tags: ['notes']
---

Then write **Markdown**.
```

`title`, `description`, and `pubDate` are required. Optional: `tags`,
`updatedDate`, and `draft: true` to hide a post while you work on it.

## Math

LaTeX works in any post, typeset at build time by KaTeX. No JavaScript is sent to
the browser and the fonts are bundled locally, so there is no CDN dependency.

Inline math uses single dollars, display math uses double dollars on their own
lines:

```markdown
The gradient $dx = r(v - c\hat{x})$ is a projection, because

$$
\|\hat{x}\|^2 = \frac{\sum_i x_i^2}{s} = d
$$

and so the subtracted term is exactly $\operatorname{proj}_{\hat{x}}(v)$.
```

A malformed expression renders in red rather than failing the build, so check the
page after writing heavy math. If you ever need a literal dollar sign in prose,
escape it: `\$`.

## Previewing locally

```sh
npm install       # first time only
npm run dev       # http://localhost:4321
```

The dev server reloads as you type, so you can keep it running while writing.

## Publishing

```sh
git add .
git commit -m "New post: my post"
git push
```

Pushing to `main` triggers `.github/workflows/deploy.yml`, which builds the site
and deploys it. Takes about a minute. Progress is visible in the repo's
**Actions** tab.

## Editing the site itself

| What | Where |
| --- | --- |
| Site title, description, nav, social links | `src/consts.ts` |
| Colors, fonts, spacing | `src/styles/global.css` |
| Homepage layout | `src/pages/index.astro` |
| About page | `src/pages/about.astro` |
| Post page layout | `src/layouts/BlogPost.astro` |
| Post frontmatter rules | `src/content.config.ts` |

## Note for future installs on an Apple work machine

Corporate machines resolve npm packages through internal mirrors
(`npm.apple.com`, `artifacts.apple.com`), and those hostnames get written into
`package-lock.json` as `resolved` URLs. GitHub's runners cannot reach them, so
CI fails with fetch errors.

If that happens after a fresh `npm install`, repoint them at the public registry:

```sh
sed -i '' \
  -e 's|https://artifacts\.apple\.com/artifactory/api/npm/npm-apple/|https://registry.npmjs.org/|g' \
  -e 's|https://npm\.apple\.com/|https://registry.npmjs.org/|g' \
  package-lock.json
```

The `integrity` hashes are content hashes and stay valid, so npm still verifies
every package.
