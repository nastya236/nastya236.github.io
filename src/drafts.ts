/**
 * Drafts are visible while running `npm run dev` so you can read them in the
 * browser, and excluded from `npm run build` so they never reach the published
 * site. `import.meta.env.DEV` is true only for the dev server.
 */
export function isVisible({ data }: { data: { draft?: boolean } }): boolean {
	return import.meta.env.DEV || !data.draft;
}
