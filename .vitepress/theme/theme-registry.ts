/**
 * 主题注册表
 *
 * 这是「一键切换主题」的唯一数据源：切换器 UI、SSR 内联脚本、校验脚本都读它。
 *
 * 新增一套主题：
 *   1. 在 styles/themes.css 追加 [data-cs-theme='<id>'] 与 .dark[data-cs-theme='<id>'] 两个块；
 *   2. 在下面的 csThemes 数组登记同一个 id；
 *   3. 结束——组件只消费语义 token，无需任何改动。
 */

export type CsThemeId = 'blueprint' | 'ocean' | 'graphite' | 'sunset'

export interface CsTheme {
	/** 与 styles/themes.css 中的 data-cs-theme 选择器一一对应 */
	id: CsThemeId
	/** 切换器里展示的名称 */
	label: string
	/** 一句话说明这套主题的适用场景 */
	description: string
	/**
	 * 切换器色点使用的品牌色。
	 *
	 * 这里必须写字面量：切换器要在「尚未应用该主题」时就展示其颜色，
	 * 无法通过 var() 取到目标主题的取值。因此该值须与 themes.css 中
	 * 同名主题的浅色 --cs-color-brand 保持一致，
	 * 由 scripts/check-design-tokens.mjs 校验两处不漂移。
	 */
	swatch: string
}

export const csThemes: readonly CsTheme[] = [
	{
		id: 'blueprint',
		label: '蓝图',
		description: '默认主题，蓝色强调，贴近 VitePress 原生观感',
		swatch: '#2563eb'
	},
	{
		id: 'ocean',
		label: '深海',
		description: '青色强调，长时间阅读时视觉刺激更低',
		swatch: '#0f766e'
	},
	{
		id: 'graphite',
		label: '石墨',
		description: '中性灰强调，适合读代码与公式',
		swatch: '#334155'
	},
	{
		id: 'sunset',
		label: '暮色',
		description: '紫色强调，图示与数据系列对比更强',
		swatch: '#7c3aed'
	}
] as const

export const defaultCsTheme: CsThemeId = 'blueprint'

/** localStorage 键名；SSR 内联脚本与运行时共用，改动需同步 */
export const CS_THEME_STORAGE_KEY = 'cs-theme'

/** DOM 属性名；与 styles/themes.css 的属性选择器共用 */
export const CS_THEME_ATTRIBUTE = 'data-cs-theme'

export function isCsThemeId(value: unknown): value is CsThemeId {
	return typeof value === 'string' && csThemes.some((theme) => theme.id === value)
}

/**
 * 首屏防闪烁脚本。
 *
 * 在 <head> 中同步执行，使 data-cs-theme 在首次绘制前就位；
 * 否则会先按默认主题绘制一帧再切换，产生可见闪烁。
 * 由 config.mts 通过 head 注入，主题 id 列表从本文件派生，避免两处硬编码。
 */
export const csThemeInlineScript = `
;(() => {
	try {
		const allowed = ${JSON.stringify(csThemes.map((theme) => theme.id))}
		const stored = localStorage.getItem('${CS_THEME_STORAGE_KEY}')
		const theme = allowed.includes(stored) ? stored : '${defaultCsTheme}'
		document.documentElement.setAttribute('${CS_THEME_ATTRIBUTE}', theme)
	} catch {
		document.documentElement.setAttribute('${CS_THEME_ATTRIBUTE}', '${defaultCsTheme}')
	}
})()
`.trim()
