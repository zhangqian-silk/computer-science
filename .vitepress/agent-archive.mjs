import path from 'node:path'

const archivePrefix = 'ai/agent/archive/'

function isSnapshot(relativePath) {
	return relativePath?.startsWith(archivePrefix) && relativePath !== archivePrefix + 'catalog.md'
}

export function archiveHref(href, relativePath) {
	if (!isSnapshot(relativePath) || !href || /^(#|\/\/|[a-z][a-z\d+.-]*:)/i.test(href)) return href
	const boundary = href.search(/[?#]/)
	const pathname = boundary < 0 ? href : href.slice(0, boundary)
	const suffix = boundary < 0 ? '' : href.slice(boundary)
	const originalPage = relativePath.replace(archivePrefix, 'ai/agent/')
	let target = pathname.startsWith('/')
		? pathname.slice(1)
		: path.posix.normalize(path.posix.join(path.posix.dirname(originalPage), pathname))
	if (target.startsWith('ai/agent/') && !target.startsWith(archivePrefix)) {
		target = target.replace('ai/agent/', archivePrefix)
	}
	return '/' + target + suffix
}

export function archiveSource(source, relativePath) {
	if (!isSnapshot(relativePath)) return source
	const frontmatter = source.match(/^---\r?\n([\s\S]*?)\r?\n---(?:\r?\n|$)/)
	const body = frontmatter ? source.slice(frontmatter[0].length) : source
	return `${frontmatter?.[0] ?? ''}\n::: info 旧版归档\n本页是重写前的历史内容，不代表新版结构。请阅读[新版学习总览](/ai/agent/notes/)或[归档目录](/ai/agent/archive/catalog)。\n:::\n\n${body}`
}

export function installAgentArchive(md) {
	md.core.ruler.before('normalize', 'agent_archive_notice', (state) => {
		if (state.env.relativePath?.startsWith(archivePrefix)) {
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
				if (href === '/ai/agent/notes/' && token.content.startsWith('本页是重写前')) continue
				if (href) child.attrSet(attribute, archiveHref(href, state.env.relativePath))
			}
		}
	})
}
