import { existsSync } from 'node:fs'
import path from 'node:path'

const archiveRoot = 'ai/agent/archive/'
// v1 快照最初直接放在 archive/notes/，分代后迁至 archive/v1-pre-rewrite/；历史文件里的旧路径不改写，在渲染时映射。
const legacyRoot = archiveRoot + 'notes/'
const legacyGeneration = archiveRoot + 'v1-pre-rewrite/notes/'

// 归档按代存放：archive/<generation>/ 是一份独立快照，archive/catalog.md 是跨代总目录。
function generationPrefix(relativePath) {
	if (!relativePath?.startsWith(archiveRoot)) return null
	const rest = relativePath.slice(archiveRoot.length)
	const boundary = rest.indexOf('/')
	return boundary < 0 ? null : archiveRoot + rest.slice(0, boundary) + '/'
}

function isSnapshot(relativePath) {
	const prefix = generationPrefix(relativePath)
	return Boolean(prefix) && relativePath !== prefix + 'catalog.md'
}

export function archiveHref(href, relativePath) {
	if (!isSnapshot(relativePath) || !href || /^(#|\/\/|[a-z][a-z\d+.-]*:)/i.test(href)) return href
	const prefix = generationPrefix(relativePath)
	const boundary = href.search(/[?#]/)
	const pathname = boundary < 0 ? href : href.slice(0, boundary)
	const suffix = boundary < 0 ? '' : href.slice(boundary)
	const originalPage = relativePath.replace(prefix, 'ai/agent/')
	let target = pathname.startsWith('/')
		? pathname.slice(1)
		: path.posix.normalize(path.posix.join(path.posix.dirname(originalPage), pathname))
	if (target.startsWith(legacyRoot)) {
		target = legacyGeneration + target.slice(legacyRoot.length)
	} else if (target.startsWith('ai/agent/') && !target.startsWith(archiveRoot)) {
		// 仅当同代快照真的收录了该页时才留在归档内；主题仍在正文的（如保留下来的模型交互）指向现行版本。
		const withinGeneration = target.replace('ai/agent/', prefix)
		if (existsSync(path.join(process.cwd(), withinGeneration))) {
			target = withinGeneration
		}
	}
	return '/' + target + suffix
}

export function archiveSource(source, relativePath) {
	if (!isSnapshot(relativePath)) return source
	const prefix = generationPrefix(relativePath)
	const frontmatter = source.match(/^---\r?\n([\s\S]*?)\r?\n---(?:\r?\n|$)/)
	const body = frontmatter ? source.slice(frontmatter[0].length) : source
	return `${frontmatter?.[0] ?? ''}\n::: info 旧版归档\n本页是结构调整前的历史内容，不代表新版结构。请阅读[新版学习总览](/ai/agent/notes/)或[本代归档目录](/${prefix}catalog)。\n:::\n\n${body}`
}

export function installAgentArchive(md) {
	md.core.ruler.before('normalize', 'agent_archive_notice', (state) => {
		if (state.env.relativePath?.startsWith(archiveRoot)) {
			// VitePress has already parsed frontmatter before running core rules.
			state.env.frontmatter = { ...state.env.frontmatter, search: false }
		}
		state.src = archiveSource(state.src, state.env.relativePath)
	})
	md.core.ruler.after('inline', 'agent_archive_links', (state) => {
		for (const token of state.tokens) {
			for (const child of token.children ?? []) {
				const attribute = child.type === 'link_open' ? 'href' : child.type === 'image' ? 'src' : null
				if (!attribute) continue
				const href = child.attrGet(attribute)
				if (href === '/ai/agent/notes/' && token.content.startsWith('本页是结构调整前')) continue
				if (href) child.attrSet(attribute, archiveHref(href, state.env.relativePath))
			}
		}
	})
}
