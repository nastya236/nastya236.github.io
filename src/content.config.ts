import { defineCollection } from 'astro:content';
import { glob } from 'astro/loaders';
// NOTE: in Astro 7, `z` comes from 'astro/zod' — importing it from
// 'astro:content' was removed.
import { z } from 'astro/zod';

const blog = defineCollection({
	loader: glob({ base: './src/content/blog', pattern: '**/*.{md,mdx}' }),
	schema: z.object({
		title: z.string(),
		description: z.string(),
		pubDate: z.coerce.date(),
		updatedDate: z.coerce.date().optional(),
		// Set `draft: true` to keep a post out of the published site.
		draft: z.boolean().default(false),
		tags: z.array(z.string()).default([]),
	}),
});

export const collections = { blog };
