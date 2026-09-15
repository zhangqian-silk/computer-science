import assert from 'node:assert/strict'
import { createHash } from 'node:crypto'
import { readdirSync, readFileSync } from 'node:fs'
import path from 'node:path'
import test from 'node:test'
import { createMarkdownRenderer } from 'vitepress'
import { archiveHref, archiveSource, installAgentArchive } from './agent-archive.mjs'

test('archive links stay in the same generation while external topics keep their original target', () => {
	const page = 'ai/agent/archive/v1-pre-rewrite/notes/context/rag-retrieval.md'
	assert.equal(archiveHref('../references.md#source-rag-paper', page), '/ai/agent/archive/v1-pre-rewrite/notes/references.md#source-rag-paper')
	assert.equal(archiveHref('../../../evaluation/retrieval-evaluation.md', page), '/ai/evaluation/retrieval-evaluation.md')
	assert.equal(archiveHref('/ai/agent/notes/index.md', page), '/ai/agent/archive/v1-pre-rewrite/notes/index.md')
	assert.equal(archiveHref('assets/interaction.svg', 'ai/agent/archive/v1-pre-rewrite/notes/index.md'), '/ai/agent/archive/v1-pre-rewrite/notes/assets/interaction.svg')
	assert.equal(archiveHref('#section', page), '#section')
	assert.equal(archiveHref('../references.md?view=full#source-rag-paper', page), '/ai/agent/archive/v1-pre-rewrite/notes/references.md?view=full#source-rag-paper')
	assert.equal(archiveHref('https://example.com/a', page), 'https://example.com/a')
	assert.equal(archiveHref('../notes/index.md', 'ai/agent/archive/v1-pre-rewrite/catalog.md'), '../notes/index.md')
	assert.equal(archiveHref('../references.md', 'ai/agent/notes/tools/skills.md'), '../references.md')
})

test('each generation resolves against its own snapshot, and the cross-generation catalog is left alone', () => {
	const second = 'ai/agent/archive/v2-six-modules/notes/tools/skills.md'
	assert.equal(archiveHref('../references.md#source-skills-spec', second), '/ai/agent/archive/v2-six-modules/notes/references.md#source-skills-spec')
	assert.equal(archiveHref('../quality/evaluation.md', second), '/ai/agent/archive/v2-six-modules/notes/quality/evaluation.md')
	// 该代未收录的主题仍留在正文，链接指向现行版本而不是不存在的归档路径。
	assert.equal(archiveHref('../model/llm-api.md#tool-calling', second), '/ai/agent/notes/model/llm-api.md#tool-calling')
	// v1 快照迁入分代目录前写下的旧路径，按代映射到它现在的位置。
	assert.equal(
		archiveHref('../../archive/notes/runtime/evaluation.md', 'ai/agent/archive/v2-six-modules/notes/runtime/evaluation.md'),
		'/ai/agent/archive/v1-pre-rewrite/notes/runtime/evaluation.md'
	)
	// 归档总目录横跨各代，本身不是快照，链接必须保持原样。
	assert.equal(archiveHref('v2-six-modules/catalog.md', 'ai/agent/archive/catalog.md'), 'v2-six-modules/catalog.md')
	assert.equal(archiveHref('../notes/index.md', 'ai/agent/archive/catalog.md'), '../notes/index.md')
})

test('the notice preserves the historical source and frontmatter', () => {
	const source = '---\ntitle: "Old"\ndate: "2020-01-01"\nsearch: true\n---\n\n# Old\n\nText\n'
	const rendered = archiveSource(source, 'ai/agent/archive/v1-pre-rewrite/index.md')
	assert.match(rendered, /date: "2020-01-01"/)
	assert.match(rendered, /search: true/)
	assert.equal((rendered.match(/^search:/gm) || []).length, 1)
	assert.match(rendered, /旧版归档/)
	// 提示语指向所在代的目录，而不是跨代总目录。
	assert.match(rendered, /\/ai\/agent\/archive\/v1-pre-rewrite\/catalog/)
	assert.match(rendered, /# Old\n\nText\n$/)
	assert.match(archiveSource(source, 'ai/agent/archive/v2-six-modules/notes/labs.md'), /\/ai\/agent\/archive\/v2-six-modules\/catalog/)
	assert.equal(archiveSource(source, 'ai/agent/notes/index.md'), source)
	assert.equal(archiveSource(source, 'ai/agent/archive/catalog.md'), source)
	assert.equal(archiveSource(source, 'ai/agent/archive/v1-pre-rewrite/catalog.md'), source)
})

test('VitePress excludes archives from search without rendering metadata as content', async () => {
	const md = await createMarkdownRenderer(process.cwd(), { config: installAgentArchive }, '/computer-science/')
	const source = '---\ntitle: "Old"\ndate: "2020-01-01"\n---\n\n# Old\n\n[Source](../references.md#source-rag-paper)\n'
	const env = {
		path: path.resolve('ai/agent/archive/v1-pre-rewrite/notes/context/rag-retrieval.md'),
		relativePath: 'ai/agent/archive/v1-pre-rewrite/notes/context/rag-retrieval.md',
		cleanUrls: true
	}
	const html = md.render(source, env)
	assert.equal(env.frontmatter.search, false)
	assert.equal(env.frontmatter.date, '2020-01-01')
	assert.doesNotMatch(html, /search-false|search: false|date:/)
	assert.match(html, /href="\/computer-science\/ai\/agent\/notes\/"/)
	assert.match(html, /href="\/computer-science\/ai\/agent\/archive\/v1-pre-rewrite\/notes\/references#source-rag-paper"/)
	const current = { path: path.resolve('ai/agent/notes/index.md'), relativePath: 'ai/agent/notes/index.md' }
	assert.doesNotMatch(md.render('# Current', current), /旧版归档/)
	assert.notEqual(current.frontmatter.search, false)
})

// 每代快照都必须逐字节保持在归档时的状态：哈希变化即意味着历史内容被改写。
const snapshots = [
	{ generation: 'v1-pre-rewrite', files: 58, digest: '5fec74aebdcb5a9a48b116d739d8e7beb36c27117ba835f46d4a987ca23eed25' },
	{ generation: 'v2-six-modules', files: 33, digest: '66a14d552d76677669aaf06680201f78ee114ec2d91141b071ee6f4b51837f4d' }
]

for (const { generation, files: expectedFiles, digest } of snapshots) {
	test(`all ${expectedFiles} files in ${generation} remain byte-for-byte at the captured snapshot`, () => {
		const root = path.resolve('ai/agent/archive', generation)
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
		assert.equal(files.length, expectedFiles)
		assert.equal(hash.digest('hex'), digest)
	})
}
