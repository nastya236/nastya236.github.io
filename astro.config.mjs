// @ts-check
import { defineConfig } from 'astro/config';
import mdx from '@astrojs/mdx';
import sitemap from '@astrojs/sitemap';
import remarkMath from 'remark-math';
import rehypeMathjax from 'rehype-mathjax/svg';

// https://astro.build/config
export default defineConfig({
	// Your published URL. Used for sitemap, RSS, and canonical links.
	site: 'https://nastya236.github.io',
	integrations: [mdx(), sitemap()],
	markdown: {
		// $inline$ and $$display$$ math, typeset at build time by MathJax —
		// the same engine Sphinx uses for the MLX docs. SVG output keeps it
		// self-contained: no CDN, no web fonts, no client-side JavaScript.
		remarkPlugins: [remarkMath],
		rehypePlugins: [rehypeMathjax],
		shikiConfig: {
			themes: { light: 'github-light', dark: 'github-dark' },
			wrap: true,
		},
	},
});
