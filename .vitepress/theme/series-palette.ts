/**
 * 数据系列色
 *
 * 组件里凡是需要「第 N 条数据用第 N 个颜色」的场合（哈希环节点、位段、通道等），
 * 都从这里取值，而不是自己维护一份 hex 数组。返回的是 CSS 变量引用，
 * 因此颜色随 data-cs-theme 与浅 / 深色一并变化，无需组件重新渲染。
 *
 * 语义色（成功 / 警告 / 失败 / 品牌）不在此列，请直接用对应的语义 token。
 */

/** themes.css 为每套主题提供 8 个系列色 */
export const SERIES_COUNT = 8

/**
 * 取第 index 个系列色（自动按 SERIES_COUNT 取模，可安全传入任意非负整数）。
 * @example seriesColor(0) // 'var(--cs-series-1)'
 */
export function seriesColor(index: number): string {
	const slot = ((Math.trunc(index) % SERIES_COUNT) + SERIES_COUNT) % SERIES_COUNT
	return `var(--cs-series-${slot + 1})`
}

/** 系列色的完整列表，用于图例等需要一次性遍历的场合 */
export const seriesPalette: readonly string[] = Array.from(
	{ length: SERIES_COUNT },
	(_, index) => seriesColor(index)
)

/**
 * 填充块之上的文字色。
 * 系列色填充块用同一套反色文字，保证任何主题下都有对比度。
 */
export const onSeriesColor = 'var(--cs-color-on-brand)'
