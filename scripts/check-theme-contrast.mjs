#!/usr/bin/env node
/**
 * 主题对比度校验（静态解析，无需浏览器）
 *
 * 自行解析 palette.css 与 themes.css 的 token 引用链
 * （--cs-color-brand → var(--cs-teal-700) → #0f766e），
 * 再按 WCAG 2.1 计算对比度，因此可直接在 CI 里跑，不需要 Chromium。
 *
 * 两个容易算错、这里专门处理的点：
 *   1. 半透明前景（如 rgb(235 235 245 / 38%)）必须先与背景合成，
 *      否则对比度会被显著高估；
 *   2. 深色取值来自 .dark[data-cs-theme='x'] 块，且 blueprint 与裸 .dark
 *      合并声明，需要一并计入。
 *
 * 门槛：正文类 4.5:1（AA 正文），弱化文字与图形类 3:1（AA 大字 / 非文本）。
 * 系列色两两之间要求 CIE76 ΔE ≥ 15，保证图例可区分。
 *
 * 用法：node scripts/check-theme-contrast.mjs
 * 退出码 0 表示通过，1 表示存在不达标组合。
 */

import { readFileSync } from 'node:fs'
import path from 'node:path'
import process from 'node:process'

const repoRoot = path.resolve(import.meta.dirname, '..')
const stylesDir = path.join(repoRoot, '.vitepress', 'theme', 'styles')
const paletteFile = path.join(stylesDir, 'palette.css')
const themesFile = path.join(stylesDir, 'themes.css')
const registryFile = path.join(repoRoot, '.vitepress', 'theme', 'theme-registry.ts')

/** [名称, 前景 token, 背景 token, 门槛] */
const CHECKS = [
	['正文/底', '--cs-color-text', '--cs-color-bg', 4.5],
	['正文/次级底', '--cs-color-text', '--cs-color-bg-soft', 4.5],
	['次要文字/底', '--cs-color-text-muted', '--cs-color-bg', 4.5],
	['次要文字/次级底', '--cs-color-text-muted', '--cs-color-bg-soft', 4.5],
	['弱化文字/底', '--cs-color-text-subtle', '--cs-color-bg', 3.0],
	['弱化文字/次级底', '--cs-color-text-subtle', '--cs-color-bg-soft', 3.0],
	['品牌色/底', '--cs-color-brand', '--cs-color-bg', 3.0],
	['品牌色/次级底', '--cs-color-brand', '--cs-color-bg-soft', 3.0],
	['反色文字/品牌块', '--cs-color-on-brand', '--cs-color-brand', 4.5],
	['成功色/底', '--cs-color-success', '--cs-color-bg', 3.0],
	['成功色/次级底', '--cs-color-success', '--cs-color-bg-soft', 3.0],
	['警告色/底', '--cs-color-warning', '--cs-color-bg', 3.0],
	['失败色/底', '--cs-color-danger', '--cs-color-bg', 3.0],
	['信息色/底', '--cs-color-info', '--cs-color-bg', 3.0]
]

const MIN_SERIES_DELTA_E = 15

/**
 * 语义色之间必须相互可辨。
 *
 * 这是实际踩到过的坑：graphite 曾把品牌色和成功色都设为同一档绿，
 * 于是限流组件里「激活」和「通过」两行看起来一模一样，读者无法区分语义。
 * 门槛取 25——比系列色更严，因为这些颜色承担的是不同含义而非仅仅编号。
 */
const SEMANTIC_PAIRS = [
	['--cs-color-brand', '--cs-color-success'],
	['--cs-color-brand', '--cs-color-warning'],
	['--cs-color-brand', '--cs-color-danger'],
	['--cs-color-success', '--cs-color-warning'],
	['--cs-color-success', '--cs-color-danger'],
	['--cs-color-warning', '--cs-color-danger']
]

const MIN_SEMANTIC_DELTA_E = 25

// ---------- CSS 解析 ----------

/** 剥离注释，避免注释里的示例声明被当成真实取值 */
function stripComments(source) {
	return source.replace(/\/\*[\s\S]*?\*\//g, '')
}

/** 从一段声明文本里取出所有自定义属性 */
function parseDeclarations(text) {
	const declarations = new Map()

	for (const match of text.matchAll(/(--[\w-]+)\s*:\s*([^;]+);/g)) {
		declarations.set(match[1], match[2].trim())
	}

	return declarations
}

/** 取出选择器命中的所有块，并按出现顺序合并声明（后者覆盖前者） */
function collectBlocks(source, predicate) {
	const merged = new Map()

	for (const match of source.matchAll(/([^{}]+)\{([^{}]*)\}/g)) {
		const selector = match[1].trim()

		if (!predicate(selector)) {
			continue
		}

		for (const [key, value] of parseDeclarations(match[2])) {
			merged.set(key, value)
		}
	}

	return merged
}

const paletteSource = stripComments(readFileSync(paletteFile, 'utf8'))
const themesSource = stripComments(readFileSync(themesFile, 'utf8'))
const registrySource = readFileSync(registryFile, 'utf8')

const themes = [...registrySource.matchAll(/^\t\tid:\s*'([\w-]+)'/gm)].map((match) => match[1])

if (themes.length === 0) {
	console.error('未能从 theme-registry.ts 解析出主题列表。')
	process.exit(1)
}

/** 原始色板：selector 含 :root 即视为色板定义 */
const palette = collectBlocks(paletteSource, (selector) => selector.includes(':root'))

/**
 * 构建某个主题 + 明暗模式下的完整取值表。
 *
 * 层叠顺序与浏览器一致：先 :root 默认值，再该主题的浅色块；
 * 深色模式追加裸 .dark 与该主题的 .dark 块。
 */
function resolveScope(themeId, mode) {
	const scope = new Map(palette)

	const apply = (predicate) => {
		for (const [key, value] of collectBlocks(themesSource, predicate)) {
			scope.set(key, value)
		}
	}

	// 浅色：:root 默认值 + 该主题的属性选择器块（排除 .dark 前缀）
	apply((selector) =>
		selector.split(',').some((part) => {
			const one = part.trim()
			if (one.startsWith('.dark')) return false
			return one === ':root' || one.includes(`[data-cs-theme='${themeId}']`)
		})
	)

	if (mode === 'dark') {
		apply((selector) =>
			selector.split(',').some((part) => {
				const one = part.trim()
				if (!one.startsWith('.dark')) return false
				// 裸 .dark 适用于所有主题；带属性选择器的只适用于对应主题
				return one === '.dark' || one.includes(`[data-cs-theme='${themeId}']`)
			})
		)
	}

	return scope
}

// ---------- 颜色解析 ----------

/** 顺着 var() 链求出字面量取值 */
function resolveValue(scope, token, seen = new Set()) {
	if (seen.has(token)) {
		throw new Error(`token 引用出现循环：${token}`)
	}

	seen.add(token)
	const raw = scope.get(token)

	if (raw === undefined) {
		return null
	}

	const varMatch = raw.match(/^var\(\s*(--[\w-]+)\s*(?:,\s*([^)]*))?\)$/)

	if (varMatch) {
		const resolved = resolveValue(scope, varMatch[1], seen)
		return resolved ?? (varMatch[2] ? varMatch[2].trim() : null)
	}

	return raw
}

/** 解析 #rgb / #rrggbb / rgb(r g b / a%) / rgb(r,g,b) */
function parseColor(value) {
	if (!value) {
		return null
	}

	const text = value.trim()
	const hex = text.match(/^#([0-9a-fA-F]{3,8})$/)

	if (hex) {
		let digits = hex[1]

		if (digits.length === 3 || digits.length === 4) {
			digits = [...digits].map((char) => char + char).join('')
		}

		const toInt = (offset) => Number.parseInt(digits.slice(offset, offset + 2), 16)
		return {
			r: toInt(0),
			g: toInt(2),
			b: toInt(4),
			a: digits.length === 8 ? toInt(6) / 255 : 1
		}
	}

	const functional = text.match(/^rgba?\(([^)]+)\)$/)

	if (functional) {
		const body = functional[1].replace(/\//g, ' / ')
		const [colorPart, alphaPart] = body.split('/')
		const channels = colorPart.trim().split(/[\s,]+/).filter(Boolean).map(Number)

		if (channels.length < 3 || channels.some(Number.isNaN)) {
			return null
		}

		let alpha = 1

		if (alphaPart !== undefined) {
			const alphaText = alphaPart.trim()
			alpha = alphaText.endsWith('%') ? Number.parseFloat(alphaText) / 100 : Number.parseFloat(alphaText)
		}

		return { r: channels[0], g: channels[1], b: channels[2], a: Number.isNaN(alpha) ? 1 : alpha }
	}

	return null
}

function relativeLuminance({ r, g, b }) {
	const channel = (value) => {
		const v = value / 255
		return v <= 0.03928 ? v / 12.92 : ((v + 0.055) / 1.055) ** 2.4
	}

	return 0.2126 * channel(r) + 0.7152 * channel(g) + 0.0722 * channel(b)
}

/** 半透明前景与背景合成，得出屏幕上真实呈现的颜色 */
function flatten(fg, bg) {
	return {
		r: fg.r * fg.a + bg.r * (1 - fg.a),
		g: fg.g * fg.a + bg.g * (1 - fg.a),
		b: fg.b * fg.a + bg.b * (1 - fg.a),
		a: 1
	}
}

function contrastRatio(fgColor, bgColor) {
	const fg = fgColor.a < 1 ? flatten(fgColor, bgColor) : fgColor
	const l1 = relativeLuminance(fg)
	const l2 = relativeLuminance(bgColor)
	return (Math.max(l1, l2) + 0.05) / (Math.min(l1, l2) + 0.05)
}

function toLab({ r, g, b }) {
	const linear = [r, g, b].map((value) => {
		const v = value / 255
		return v > 0.04045 ? ((v + 0.055) / 1.055) ** 2.4 : v / 12.92
	})

	const [x, y, z] = linear
	const X = (x * 0.4124 + y * 0.3576 + z * 0.1805) / 0.95047
	const Y = x * 0.2126 + y * 0.7152 + z * 0.0722
	const Z = (x * 0.0193 + y * 0.1192 + z * 0.9505) / 1.08883
	const f = (t) => (t > 0.008856 ? Math.cbrt(t) : 7.787 * t + 16 / 116)

	return [116 * f(Y) - 16, 500 * (f(X) - f(Y)), 200 * (f(Y) - f(Z))]
}

function deltaE(first, second) {
	const a = toLab(first)
	const b = toLab(second)
	return Math.hypot(a[0] - b[0], a[1] - b[1], a[2] - b[2])
}

// ---------- 执行校验 ----------

const contrastRows = []
const seriesRows = []
const semanticRows = []
const missing = []

for (const mode of ['light', 'dark']) {
	for (const themeId of themes) {
		const scope = resolveScope(themeId, mode)

		for (const [name, fgToken, bgToken, need] of CHECKS) {
			const fg = parseColor(resolveValue(scope, fgToken))
			const bg = parseColor(resolveValue(scope, bgToken))

			if (!fg || !bg) {
				missing.push({ mode, theme: themeId, name, token: !fg ? fgToken : bgToken })
				continue
			}

			contrastRows.push({
				mode,
				theme: themeId,
				name,
				need,
				ratio: Number(contrastRatio(fg, bg).toFixed(2))
			})
		}

		const series = []

		for (let index = 1; index <= 8; index += 1) {
			const color = parseColor(resolveValue(scope, `--cs-series-${index}`))

			if (!color) {
				missing.push({ mode, theme: themeId, name: '数据系列色', token: `--cs-series-${index}` })
				continue
			}

			series.push({ slot: index, color })
		}

		let worst = Infinity
		let worstPair = ''

		for (let i = 0; i < series.length; i += 1) {
			for (let j = i + 1; j < series.length; j += 1) {
				const distance = deltaE(series[i].color, series[j].color)

				if (distance < worst) {
					worst = distance
					worstPair = `s${series[i].slot}/s${series[j].slot}`
				}
			}
		}

		seriesRows.push({ mode, theme: themeId, worstPair, deltaE: Number(worst.toFixed(1)) })

		// 语义色两两可辨性
		for (const [firstToken, secondToken] of SEMANTIC_PAIRS) {
			const first = parseColor(resolveValue(scope, firstToken))
			const second = parseColor(resolveValue(scope, secondToken))

			if (!first || !second) {
				missing.push({ mode, theme: themeId, name: '语义色', token: !first ? firstToken : secondToken })
				continue
			}

			semanticRows.push({
				mode,
				theme: themeId,
				pair: `${firstToken.replace('--cs-color-', '')}/${secondToken.replace('--cs-color-', '')}`,
				deltaE: Number(deltaE(first, second).toFixed(1))
			})
		}
	}
}

const contrastFailures = contrastRows.filter((row) => row.ratio < row.need)
const seriesFailures = seriesRows.filter((row) => row.deltaE < MIN_SERIES_DELTA_E)
const semanticFailures = semanticRows.filter((row) => row.deltaE < MIN_SEMANTIC_DELTA_E)

if (
	contrastFailures.length === 0 &&
	seriesFailures.length === 0 &&
	semanticFailures.length === 0 &&
	missing.length === 0
) {
	const tightest = contrastRows.slice().sort((a, b) => a.ratio / a.need - b.ratio / b.need)[0]
	const closest = seriesRows.slice().sort((a, b) => a.deltaE - b.deltaE)[0]
	const closestSemantic = semanticRows.slice().sort((a, b) => a.deltaE - b.deltaE)[0]

	console.log(
		`对比度校验通过：${contrastRows.length} 项对比度 + ${semanticRows.length} 项语义色可辨性，` +
		`覆盖 ${themes.length} 套主题 × 浅/深色。`
	)
	console.log(`  最紧对比度：${tightest.mode} ${tightest.theme} ${tightest.name} = ${tightest.ratio}（需 ${tightest.need}）`)
	console.log(`  语义色最小差异：${closestSemantic.mode} ${closestSemantic.theme} ${closestSemantic.pair} ΔE=${closestSemantic.deltaE}（需 ≥ ${MIN_SEMANTIC_DELTA_E}）`)
	console.log(`  系列色最小差异：${closest.mode} ${closest.theme} ${closest.worstPair} ΔE=${closest.deltaE}（需 ≥ ${MIN_SERIES_DELTA_E}）`)
	process.exit(0)
}

console.error('对比度校验失败\n')

for (const row of missing) {
	console.error(`  ${row.mode} ${row.theme} ${row.name}：无法解析 ${row.token}`)
}

for (const row of contrastFailures) {
	console.error(`  ${row.mode} ${row.theme} ${row.name}：${row.ratio} < ${row.need}`)
}

for (const row of seriesFailures) {
	console.error(`  ${row.mode} ${row.theme} 系列色 ${row.worstPair} ΔE=${row.deltaE} < ${MIN_SERIES_DELTA_E}，难以区分`)
}

for (const row of semanticFailures) {
	console.error(`  ${row.mode} ${row.theme} 语义色 ${row.pair} ΔE=${row.deltaE} < ${MIN_SEMANTIC_DELTA_E}，两种语义会被误读为同一种`)
}

console.error('\n请在 .vitepress/theme/styles/themes.css 调整相应主题的取值。')
process.exit(1)
