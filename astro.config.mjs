// @ts-check
import { defineConfig } from 'astro/config';
import mdx from '@astrojs/mdx';
import sitemap from '@astrojs/sitemap';
import remarkMath from 'remark-math';
import rehypeKatex from 'rehype-katex';

// https://astro.build/config
export default defineConfig({
	// Your published URL. Used for sitemap, RSS, and canonical links.
	site: 'https://nastya236.github.io',
	integrations: [mdx(), sitemap()],
	markdown: {
		// $inline$ and $$display$$ math, typeset at build time by KaTeX.
		// No JavaScript is shipped to the browser for this.
		remarkPlugins: [remarkMath],
		rehypePlugins: [[rehypeKatex, { strict: false }]],
		shikiConfig: {
			themes: { light: 'github-light', dark: 'github-dark' },
			wrap: true,
		},
	},
});
