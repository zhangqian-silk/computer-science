#!/usr/bin/env node
/**
 * 设计系统一致性校验
 *
 * 目的：让「新增文件也能保持同一主题风格」成为可检查的约束，而不是口头约定。
 *
 * 校验规则：
 *   1. 组件与样式文件里不得出现字面量色值（#hex / rgb() / hsl() / 具名色）；
 *      唯一例外是 styles/palette.css——原始色板本来就是定义字面量的地方。
 *   2. 组件不得直接引用原始色板变量 --cs-<色相>-<台阶>，必须走语义 token；
 *      只有 styles/themes.css 可以引用。
 *   3. themes.css 里每套主题都必须提供全部必需的语义 token，缺项会静默回退。
 *   4. 组件不得直接读写 data-cs-theme 或主题 localStorage 键，
 *      必须经由 theme-registry.ts，避免出现第二个事实来源。
 *
 * 用法：
 *   node scripts/check-design-tokens.mjs
 * 退出码 0 表示通过，1 表示存在违规。
 */

import { readdirSync, readFileSync, statSync } from 'node:fs'
import path from 'node:path'
import process from 'node:process'

const repoRoot = path.resolve(import.meta.dirname, '..')
const themeDir = path.join(repoRoot, '.vitepress', 'theme')
const stylesDir = path.join(themeDir, 'styles')
const paletteFile = path.join(stylesDir, 'palette.css')
const themesFile = path.join(stylesDir, 'themes.css')
const registryFile = path.join(themeDir, 'theme-registry.ts')

/** 每套主题必须提供的语义 token；缺项会回退到 :root，导致主题不完整 */
const REQUIRED_TOKENS = [
	'--cs-color-brand',
	'--cs-color-brand-hover',
	'--cs-color-brand-active',
	'--cs-color-brand-soft',
	'--cs-color-on-brand',
	'--cs-color-bg',
	'--cs-color-bg-soft',
	'--cs-color-bg-elevated',
	'--cs-color-text',
	'--cs-color-text-muted',
	'--cs-color-text-subtle',
	'--cs-color-border',
	'--cs-color-border-strong',
	'--cs-color-success',
	'--cs-color-success-soft',
	'--cs-color-warning',
	'--cs-color-warning-soft',
	'--cs-color-danger',
	'--cs-color-danger-soft',
	'--cs-color-info',
	'--cs-color-info-soft',
	'--cs-color-neutral-soft',
	'--cs-color-shadow',
	...Array.from({ length: 8 }, (_, index) => `--cs-series-${index + 1}`)
]

const NAMED_COLORS = [
	'white', 'black', 'red', 'green', 'blue', 'yellow', 'orange',
	'purple', 'gray', 'grey', 'pink', 'teal', 'cyan', 'magenta'
]

const violations = []

function record(file, line, rule, detail) {
	violations.push({ file: path.relative(repoRoot, file), line, rule, detail })
}

function walk(dir, extensions) {
	const files = []

	for (const entry of readdirSync(dir, { withFileTypes: true })) {
		const full = path.join(dir, entry.name)

		if (entry.isDirectory()) {
			files.push(...walk(full, extensions))
			continue
		}

		if (extensions.some((extension) => entry.name.endsWith(extension))) {
			files.push(full)
		}
	}

	return files
}

/** 逐行扫描，跳过注释行，避免注释里的示例色值被误判 */
function eachCodeLine(file, visit) {
	const lines = readFileSync(file, 'utf8').split('\n')
	let inBlockComment = false

	lines.forEach((raw, index) => {
		let line = raw

		if (inBlockComment) {
			const end = line.indexOf('*/')

			if (end === -1) {
				return
			}

			line = line.slice(end + 2)
			inBlockComment = false
		}

		// 去掉行内块注释，再判断是否进入跨行块注释
		line = line.replace(/\/\*[^]*?\*\//g, '')

		const blockStart = line.indexOf('/*')

		if (blockStart !== -1) {
			inBlockComment = true
			line = line.slice(0, blockStart)
		}

		// 去掉行尾单行注释（简单形式，足够覆盖 CSS/Vue/TS 的注释写法）
		line = line.replace(/\/\/.*$/, '')

		if (line.trim() !== '') {
			visit(line, index + 1)
		}
	})
}

// ---------- 规则 1 与 2：色值与色板引用 ----------
const scanTargets = [
	...walk(path.join(themeDir, 'components'), ['.vue']),
	...walk(path.join(themeDir, 'system'), ['.vue']),
	...walk(stylesDir, ['.css'])
].filter((file) => file !== paletteFile)

for (const file of scanTargets) {
	const isThemesFile = file === themesFile

	eachCodeLine(file, (line, lineNumber) => {
		const hex = line.match(/#[0-9a-fA-F]{3,8}\b/g)

		if (hex && !isThemesFile) {
			record(file, lineNumber, '字面量色值', `出现 ${hex.join(' ')}，应改用语义 token`)
		}

		const functional = line.match(/\b(?:rgba?|hsla?)\s*\(/g)

		if (functional && !isThemesFile) {
			record(file, lineNumber, '字面量色值', `出现 ${functional.join(' ')}，应改用语义 token`)
		}

		for (const name of NAMED_COLORS) {
			// 只匹配「颜色属性: 具名色」形式，避免误伤 class 名或文案
			const pattern = new RegExp(`(?:^|[\\s;{])(?:color|background|background-color|border-color|fill|stroke)\\s*:\\s*${name}\\b`, 'i')

			if (pattern.test(line)) {
				record(file, lineNumber, '字面量色值', `出现具名色 ${name}，应改用语义 token`)
			}
		}

		// 原始色板变量只允许 themes.css 引用
		const primitive = line.match(/--cs-(?:slate|blue|teal|green|violet|amber|red|pink|white|black)-?\w*/g)

		if (primitive && !isThemesFile) {
			record(file, lineNumber, '越层引用色板', `直接引用 ${[...new Set(primitive)].join(' ')}，应改用语义 token`)
		}
	})
}

// ---------- 规则 3：主题 token 完整性 ----------
const themesSource = readFileSync(themesFile, 'utf8')
const registrySource = readFileSync(registryFile, 'utf8')
const paletteSource = readFileSync(paletteFile, 'utf8')
const registeredThemes = [...registrySource.matchAll(/^\t\tid:\s*'([\w-]+)'/gm)].map((match) => match[1])

if (registeredThemes.length === 0) {
	record(registryFile, 0, '注册表解析失败', '未能从 theme-registry.ts 解析出任何主题 id')
}

/** 取出某个选择器块的声明文本；同一选择器可能出现在多个块（浅色 / 深色） */
function blocksFor(selectorPattern) {
	const blocks = []
	const pattern = new RegExp(`(^|\\n)([^\\n{}]*${selectorPattern}[^{}]*)\\{([^{}]*)\\}`, 'g')

	for (const match of themesSource.matchAll(pattern)) {
		blocks.push(match[3])
	}

	return blocks
}

for (const themeId of registeredThemes) {
	const lightBlocks = blocksFor(`\\[data-cs-theme='${themeId}'\\]`)

	if (lightBlocks.length === 0) {
		record(themesFile, 0, '主题缺失', `注册表登记了 ${themeId}，但 themes.css 没有对应选择器`)
		continue
	}

	// 浅色取值：非 .dark 前缀的块；深色取值：.dark 前缀的块
	const lightText = blocksFor(`\\[data-cs-theme='${themeId}'\\]`)
		.filter((_, index) => index >= 0)
		.join('\n')

	const darkPattern = new RegExp(`\\.dark\\[data-cs-theme='${themeId}'\\][^{}]*\\{([^{}]*)\\}`)
	const darkText = themesSource.match(darkPattern)?.[1] ?? ''

	// blueprint 与 .dark 合并声明，需要把裸 .dark 块也算进去
	const darkFallback = themeId === 'blueprint'
		? (themesSource.match(/(?:^|\n)\.dark,[^{}]*\{([^{}]*)\}/)?.[1] ?? '')
		: ''

	for (const token of REQUIRED_TOKENS) {
		const tokenPattern = new RegExp(`${token}\\s*:`)

		if (!tokenPattern.test(lightText)) {
			record(themesFile, 0, '主题 token 缺失', `${themeId}（浅色）缺少 ${token}`)
		}

		if (!tokenPattern.test(darkText) && !tokenPattern.test(darkFallback)) {
			record(themesFile, 0, '主题 token 缺失', `${themeId}（深色）缺少 ${token}`)
		}
	}
}

// ---------- 规则 4：切换器色点须与主题品牌色一致 ----------
// 切换器要在未应用主题时就展示其颜色，只能写字面量，因此存在漂移风险。
for (const themeId of registeredThemes) {
	const entryPattern = new RegExp(`id:\\s*'${themeId}'[\\s\\S]*?swatch:\\s*'([^']+)'`)
	const swatch = registrySource.match(entryPattern)?.[1]

	if (!swatch) {
		record(registryFile, 0, '切换器色点缺失', `${themeId} 未声明 swatch`)
		continue
	}

	// 取该主题浅色块的 --cs-color-brand，顺着 var() 链求到字面量
	const lightBlock = themesSource.match(
		new RegExp(`(?:^|\\n)(?!\\.dark)[^\\n{}]*\\[data-cs-theme='${themeId}'\\][^{}]*\\{([^{}]*)\\}`)
	)?.[1]

	if (!lightBlock) {
		continue
	}

	const brandRaw = lightBlock.match(/--cs-color-brand\s*:\s*([^;]+);/)?.[1]?.trim()

	if (!brandRaw) {
		continue
	}

	let expected = brandRaw
	const varName = brandRaw.match(/^var\(\s*(--[\w-]+)\s*\)$/)?.[1]

	if (varName) {
		expected = paletteSource.match(new RegExp(`${varName}\\s*:\\s*([^;]+);`))?.[1]?.trim() ?? brandRaw
	}

	if (expected.toLowerCase() !== swatch.toLowerCase()) {
		record(
			registryFile,
			0,
			'切换器色点漂移',
			`${themeId} 的 swatch 为 ${swatch}，但主题品牌色是 ${expected}`
		)
	}
}

// ---------- 规则 5：主题状态只能有一个来源 ----------
const componentFiles = walk(path.join(themeDir, 'components'), ['.vue'])
const switcher = path.join(themeDir, 'system', 'ThemeSwitch.vue')

for (const file of componentFiles) {
	if (file === switcher) {
		continue
	}

	eachCodeLine(file, (line, lineNumber) => {
		if (line.includes('data-cs-theme') || line.includes("'cs-theme'") || line.includes('"cs-theme"')) {
			record(file, lineNumber, '绕过注册表', '组件直接操作主题状态，应通过 theme-registry.ts')
		}
	})
}

// ---------- 输出 ----------
const scanned = scanTargets.length

if (violations.length === 0) {
	console.log(`设计系统校验通过：已扫描 ${scanned} 个文件，主题 ${registeredThemes.length} 套（${registeredThemes.join('、')}）。`)
	process.exit(0)
}

console.error(`设计系统校验失败：${violations.length} 处问题\n`)

const grouped = new Map()

for (const violation of violations) {
	if (!grouped.has(violation.file)) {
		grouped.set(violation.file, [])
	}

	grouped.get(violation.file).push(violation)
}

for (const [file, items] of grouped) {
	console.error(file)

	for (const item of items) {
		const position = item.line > 0 ? `:${item.line}` : ''
		console.error(`  ${position.padEnd(6)} [${item.rule}] ${item.detail}`)
	}

	console.error('')
}

console.error('修复指引见 docs/design-system.md。')
process.exit(1)
