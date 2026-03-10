import { existsSync, readdirSync } from 'node:fs'
import path from 'node:path'
import { defineConfig, type DefaultTheme } from 'vitepress'
import texmath from 'markdown-it-texmath'
import katex from 'katex'
import { withMermaid } from 'vitepress-plugin-mermaid'

const repoName = 'computer-science'

const sectionMeta = [
	{ key: 'ai', text: 'AI', link: '/ai/mechanism/attention' },
	{ key: 'fundamentals', text: '基础', link: '/fundamentals/README' },
	{ key: 'language', text: '语言', link: '/language/golang/README' },
	{ key: 'middleware', text: '中间件', link: '/middleware/redis/redis' },
	{ key: 'distributed', text: '分布式', link: '/distributed/theory' },
	{ key: 'system_design', text: '系统设计', link: '/system_design/README' },
	{ key: 'design_patterns', text: '设计模式', link: '/design_patterns/CreationalPatterns' }
] as const

function shouldIgnoreFile(relativeDir: string, fileName: string) {
	if (fileName === 'AGENTS.md') {
		return true
	}

	return relativeDir === '' && (fileName === 'README.md' || fileName === 'CONTRIBUTING.md')
}

function sortEntries(left: string, right: string) {
	const leftIsDir = existsSync(path.join(process.cwd(), left)) && !path.extname(left)
	const rightIsDir = existsSync(path.join(process.cwd(), right)) && !path.extname(right)

	if (leftIsDir !== rightIsDir) {
		return leftIsDir ? -1 : 1
	}

	return left.localeCompare(right, 'zh-CN')
}

function formatText(name: string) {
	return name
		.replace(/\.md$/i, '')
		.replace(/[-_]/g, ' ')
}

function normalizeLink(relativePath: string) {
	const normalized = relativePath.split(path.sep).join('/')
	const withoutExtension = normalized.replace(/\.md$/i, '')

	if (withoutExtension === 'index') {
		return '/'
	}

	if (withoutExtension.endsWith('/index')) {
		return `/${withoutExtension.replace(/\/(README|index)$/i, '')}/`
	}

	return `/${withoutExtension}`
}

function buildItems(relativeDir: string, depth = 0): DefaultTheme.SidebarItem[] {
	const absoluteDir = path.join(process.cwd(), relativeDir)
	const entries = readdirSync(absoluteDir, { withFileTypes: true })
		.filter((entry) => !entry.name.startsWith('.'))
		.filter((entry) => {
			if (entry.isDirectory()) {
				return true
			}

			return entry.name.endsWith('.md') && !shouldIgnoreFile(relativeDir, entry.name)
		})
		.sort((left, right) => sortEntries(path.join(relativeDir, left.name), path.join(relativeDir, right.name)))

	const items: DefaultTheme.SidebarItem[] = []

	for (const entry of entries) {
		const nextRelativePath = path.join(relativeDir, entry.name)

		if (entry.isDirectory()) {
			const children = buildItems(nextRelativePath, depth + 1)

			if (children.length > 0) {
				items.push({
					text: formatText(entry.name),
					collapsed: depth >= 1,
					items: children
				})
			}

			continue
		}

		items.push({
			text: formatText(entry.name),
			link: normalizeLink(nextRelativePath)
		})
	}

	return items
}

const sidebar = Object.fromEntries(
	sectionMeta.map(({ key }) => [`/${key}/`, buildItems(key)])
)

export default withMermaid(
	defineConfig({
		lang: 'zh-CN',
		title: 'Computer Science',
		description: '计算机科学知识库',
		base: `/${repoName}/`,
		lastUpdated: true,
		cleanUrls: true,
		ignoreDeadLinks: [
			/^https?:\/\//
		],
		markdown: {
			lineNumbers: true,
			config(md) {
				md.use(texmath, {
					engine: katex,
					delimiters: ['brackets', 'dollars', 'beg_end'],
					katexOptions: {
						output: 'html'
					}
				})
			}
		},
		themeConfig: {
			nav: sectionMeta.map(({ text, link }) => ({ text, link })),
			sidebar,
			search: {
				provider: 'local'
			},
			outline: {
				level: [2, 3],
				label: '目录'
			},
			docFooter: {
				prev: '上一页',
				next: '下一页'
			},
			returnToTopLabel: '回到顶部',
			sidebarMenuLabel: '导航',
			darkModeSwitchLabel: '主题',
			lightModeSwitchTitle: '切换到浅色模式',
			darkModeSwitchTitle: '切换到深色模式'
		},
		mermaid: {}
	})
)
