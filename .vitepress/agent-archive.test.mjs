import assert from 'node:assert/strict'
import { createHash } from 'node:crypto'
import { readdirSync, readFileSync } from 'node:fs'
import path from 'node:path'
import test from 'node:test'
import { createMarkdownRenderer } from 'vitepress'
import { archiveHref, archiveSource, installAgentArchive } from './agent-archive.mjs'

test('archive links stay in the snapshot while external topics keep their original target', () => {
	const page = 'ai/agent/archive/notes/context/rag-retrieval.md'
	assert.equal(archiveHref('../references.md#source-rag-paper', page), '/ai/agent/archive/notes/references.md#source-rag-paper')
	assert.equal(archiveHref('../../../evaluation/retrieval-evaluation.md', page), '/ai/evaluation/retrieval-evaluation.md')
	assert.equal(archiveHref('/ai/agent/notes/index.md', page), '/ai/agent/archive/notes/index.md')
	assert.equal(archiveHref('assets/interaction.svg', 'ai/agent/archive/notes/index.md'), '/ai/agent/archive/notes/assets/interaction.svg')
	assert.equal(archiveHref('#section', page), '#section')
	assert.equal(archiveHref('../references.md?view=full#source-rag-paper', page), '/ai/agent/archive/notes/references.md?view=full#source-rag-paper')
	assert.equal(archiveHref('https://example.com/a', page), 'https://example.com/a')
	assert.equal(archiveHref('../notes/index.md', 'ai/agent/archive/catalog.md'), '../notes/index.md')
	assert.equal(archiveHref('../references.md', 'ai/agent/notes/tools/skills.md'), '../references.md')
})

test('the notice preserves the historical source and frontmatter', () => {
	const source = '---\ntitle: "Old"\ndate: "2020-01-01"\nsearch: true\n---\n\n# Old\n\nText\n'
	const rendered = archiveSource(source, 'ai/agent/archive/index.md')
	assert.match(rendered, /date: "2020-01-01"/)
	assert.match(rendered, /search: true/)
	assert.equal((rendered.match(/^search:/gm) || []).length, 1)
	assert.match(rendered, /旧版归档/)
	assert.match(rendered, /# Old\n\nText\n$/)
	assert.equal(archiveSource(source, 'ai/agent/notes/index.md'), source)
	assert.equal(archiveSource(source, 'ai/agent/archive/catalog.md'), source)
})

test('VitePress excludes archives from search without rendering metadata as content', async () => {
	const md = await createMarkdownRenderer(process.cwd(), { config: installAgentArchive }, '/computer-science/')
	const source = '---\ntitle: "Old"\ndate: "2020-01-01"\n---\n\n# Old\n\n[Source](../references.md#source-rag-paper)\n'
	const env = {
		path: path.resolve('ai/agent/archive/notes/context/rag-retrieval.md'),
		relativePath: 'ai/agent/archive/notes/context/rag-retrieval.md',
		cleanUrls: true
	}
	const html = md.render(source, env)
	assert.equal(env.frontmatter.search, false)
	assert.equal(env.frontmatter.date, '2020-01-01')
	assert.doesNotMatch(html, /search-false|search: false|date:/)
	assert.match(html, /href="\/computer-science\/ai\/agent\/notes\/"/)
	assert.match(html, /href="\/computer-science\/ai\/agent\/archive\/notes\/references#source-rag-paper"/)
	const current = { path: path.resolve('ai/agent/notes/index.md'), relativePath: 'ai/agent/notes/index.md' }
	assert.doesNotMatch(md.render('# Current', current), /旧版归档/)
	assert.notEqual(current.frontmatter.search, false)
})

test('all 58 archived files remain byte-for-byte at the captured snapshot', () => {
	const root = path.resolve('ai/agent/archive')
	const files = []
	function walk(dir) {
		for (const entry of readdirSync(dir, { withFileTypes: true })) {
			const file = path.join(dir, entry.name)
			if (entry.isDirectory()) walk(file)
			else if (file !== path.join(root, 'catalog.md')) files.push(file)
		}
	}
	walk(root)
	const hash = createHash('sha256')
	for (const file of files.sort()) {
		hash.update(path.relative(root, file) + '\0')
		hash.update(readFileSync(file))
		hash.update('\0')
	}
	assert.equal(files.length, 58)
	assert.equal(hash.digest('hex'), '5fec74aebdcb5a9a48b116d739d8e7beb36c27117ba835f46d4a987ca23eed25')
})
