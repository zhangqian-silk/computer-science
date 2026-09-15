<script setup lang="ts">
import { computed, ref } from "vue"

/**
 * 三种页面感知方式的对照：无障碍树快照 / 截图 / 原始 DOM。
 *
 * 这张对照要回答的不是「哪个更好」，而是**它们分别把什么变成了可寻址的东西**：
 * 快照给出的是带 ref 的语义节点，截图给出的是像素坐标，DOM 给出的是选择器。
 * 定位方式一旦确定，token 成本、是否需要视觉模型、以及「布局变了会不会失效」
 * 这三件事就跟着被确定了——它们不是可以分别优化的独立维度。
 *
 * 轮次滑块用来呈现一个长会话里的累积效应：单次差距看着不大，
 * 乘上轮次之后，选择哪种感知方式基本决定了这场会话的上下文预算。
 */

type Mode = {
	key: string
	idx: number
	name: string
	/** 单次观察的量级估算（token），用于呈现相对关系而非精确值 */
	perLook: number
	locate: string
	stable: string
	vision: string
	semantic: string
	fit: string
	weak: string
}

const modes: Mode[] = [
	{
		key: "ax",
		idx: 0,
		name: "无障碍树快照",
		perLook: 1_200,
		locate: "每个节点带一个 ref，动作直接指名 ref",
		stable: "确定：同样的结构给出同样的交互",
		vision: "不需要，任意文本模型可用",
		semantic: "role、名称、状态齐全，不可见元素被自然排除",
		fit: "常规网页与表单——绝大多数场景的默认选择。",
		weak: "画布、WebGL 这类自绘界面在无障碍树里没有对应节点，此时它看不见该看的东西。"
	},
	{
		key: "shot",
		idx: 1,
		name: "截图与坐标",
		perLook: 6_000,
		locate: "x / y 坐标，需要模型自己估算位置",
		stable: "易变：布局或视口一变，坐标就失效",
		vision: "必须有视觉模型",
		semantic: "只有像素，语义全靠推断",
		fit: "自绘界面，以及需要判断「看起来对不对」的视觉校验。",
		weak: "坐标要随分辨率归一化才能跨环境使用；单张图的占用比一份结构化快照高一个量级。"
	},
	{
		key: "dom",
		idx: 2,
		name: "原始 DOM",
		perLook: 14_000,
		locate: "CSS 或 XPath 选择器",
		stable: "脆：结构重构就断，且断得无声",
		vision: "不需要",
		semantic: "属性齐全，但混在大量与交互无关的噪声里",
		fit: "需要精确提取结构化数据、而不是与页面交互时。",
		weak: "整页体量最大，且大部分内容对「下一步点哪里」没有帮助。"
	}
]

const turns = ref(12)
const detail = ref(0)
const cur = computed(() => modes[detail.value])

const rows = computed(() => {
	const totals = modes.map((m) => m.perLook * turns.value)
	const max = Math.max(...totals, 1)
	return modes.map((m, i) => ({ ...m, total: totals[i], pct: (totals[i] / max) * 100 }))
})

const ratio = computed(() => {
	const ax = modes[0].perLook
	return { shot: modes[1].perLook / ax, dom: modes[2].perLook / ax }
})

function fmt(n: number) {
	return n >= 1000 ? `${(n / 1000).toFixed(n >= 10_000 ? 0 : 1)}K` : `${Math.round(n)}`
}
</script>

<template>
	<div class="bpm">
		<label class="bpm__ctl">
			<span>会话中的观察次数</span>
			<input v-model.number="turns" type="range" min="1" max="60" step="1" />
			<b class="cs-mono">{{ turns }}</b>
		</label>

		<div class="bpm__rows">
			<div v-for="r in rows" :key="r.key" class="cs-legend-row">
				<i class="cs-legend-row__dot" :style="{ background: `var(--cs-series-${r.idx + 1})` }"></i>
				<span class="cs-legend-row__name">{{ r.name }}</span>
				<span class="cs-legend-row__bar">
					<b :style="{ width: `${r.pct}%`, background: `var(--cs-series-${r.idx + 1})` }"></b>
				</span>
				<strong class="cs-mono">{{ fmt(r.total) }}</strong>
			</div>
		</div>

		<p class="bpm__read">
			按此处的量级估算，截图约为快照的 <b>{{ ratio.shot.toFixed(0) }}</b> 倍、原始 DOM 约为
			<b>{{ ratio.dom.toFixed(0) }}</b> 倍。真正的结论不在倍数本身，而在
			<b>感知方式决定了定位方式</b>：快照指名 ref，截图靠坐标，DOM 靠选择器——
			三者的稳定性来源完全不同，所以这不是一个可以「先选成本再补稳定性」的选择。
		</p>

		<div class="bpm__tabs">
			<button
				v-for="(m, i) in modes"
				:key="m.key"
				type="button"
				class="bpm__tab"
				:class="{ 'is-open': detail === i }"
				:style="{ '--c': `var(--cs-series-${m.idx + 1})` }"
				:aria-pressed="detail === i"
				@click="detail = i"
			>
				<strong>{{ m.name }}</strong>
				<span class="cs-mono">{{ fmt(m.perLook) }} / 次</span>
			</button>
		</div>

		<article class="bpm__panel" :style="{ '--c': `var(--cs-series-${cur.idx + 1})` }">
			<dl class="bpm__dl">
				<div><dt>元素定位</dt><dd>{{ cur.locate }}</dd></div>
				<div><dt>稳定性</dt><dd>{{ cur.stable }}</dd></div>
				<div><dt>视觉模型</dt><dd>{{ cur.vision }}</dd></div>
				<div><dt>语义信息</dt><dd>{{ cur.semantic }}</dd></div>
			</dl>
			<p class="bpm__fit"><i>适合</i>{{ cur.fit }}</p>
			<p class="bpm__weak"><i>短板</i>{{ cur.weak }}</p>
		</article>

		<p class="pc-note">
			条形所用的单次占用为便于对照的<b>量级估算</b>，不对应任何具体页面或产品实测：真实占用取决于页面复杂度、
			截图分辨率与编码方式。可迁移的结论是三者的<b>相对量级</b>与各自的定位方式，而不是这里的具体数字。
			三种方式并非互斥——结构化快照负责交互、截图负责判断视觉效果，是常见的组合用法。
		</p>
	</div>
</template>

<style scoped>
.bpm { margin: 1rem 0; background: var(--cs-color-bg); border: 1px solid var(--cs-color-border); border-radius: var(--cs-radius-lg); padding: var(--cs-space-5) var(--cs-space-6); }
.bpm__ctl { display: grid; grid-template-columns: 9rem 1fr 2.6rem; align-items: center; gap: var(--cs-space-3); min-height: var(--cs-tap-target); font-size: var(--cs-text-sm); }
.bpm__ctl span { color: var(--cs-color-text-muted); }
.bpm__ctl input { width: 100%; accent-color: var(--cs-color-brand); }
.bpm__ctl b { text-align: right; color: var(--cs-color-brand); }
.bpm__rows { display: grid; gap: var(--cs-space-2); margin: var(--cs-space-4) 0; }
.bpm__read { margin: 0 0 var(--cs-space-5); font-size: var(--cs-text-base); line-height: var(--cs-leading-relaxed); color: var(--cs-color-text-muted); }
.bpm__read b { color: var(--cs-color-text); }
.bpm__tabs { display: grid; grid-template-columns: repeat(3, 1fr); gap: var(--cs-space-2); }
.bpm__tab { display: grid; gap: var(--cs-space-1); text-align: left; cursor: pointer; background: var(--cs-color-bg); border: 1px solid var(--cs-color-border); border-top: 3px solid var(--c); border-radius: var(--cs-radius-lg); padding: var(--cs-space-3); min-height: var(--cs-tap-target); transition: var(--cs-transition-colors); }
.bpm__tab:hover { background: var(--cs-color-bg-soft); }
.bpm__tab.is-open { background: color-mix(in srgb, var(--c) 10%, transparent); border-color: var(--c); }
.bpm__tab strong { font-size: var(--cs-text-md); line-height: var(--cs-leading-tight); }
.bpm__tab span { font-size: var(--cs-text-3xs); color: var(--cs-color-text-subtle); }
.bpm__panel { margin-top: var(--cs-space-4); background: var(--cs-color-bg); border: 1px solid var(--cs-color-border); border-left: 3px solid var(--c); border-radius: var(--cs-radius-lg); padding: var(--cs-space-5) var(--cs-space-6); }
.bpm__dl { margin: 0; display: grid; gap: var(--cs-space-2); }
.bpm__dl > div { display: grid; grid-template-columns: 5.5rem minmax(0, 1fr); gap: var(--cs-space-3); }
.bpm__dl dt { font-size: var(--cs-text-2xs); color: var(--cs-color-text-subtle); }
.bpm__dl dd { margin: 0; font-size: var(--cs-text-sm); line-height: var(--cs-leading-normal); color: var(--cs-color-text); }
.bpm__fit, .bpm__weak { margin: var(--cs-space-3) 0 0; padding-top: var(--cs-space-3); border-top: 1px dashed var(--cs-color-border); font-size: var(--cs-text-sm); line-height: var(--cs-leading-normal); color: var(--cs-color-text-muted); }
.bpm__weak { margin-top: var(--cs-space-2); padding-top: var(--cs-space-2); border-top: none; }
.bpm__fit i, .bpm__weak i { display: block; font-family: var(--cs-font-mono); font-size: var(--cs-text-3xs); letter-spacing: .08em; font-style: normal; color: var(--cs-color-text-subtle); margin-bottom: var(--cs-space-1); }
@media (max-width: 720px) {
	.bpm__ctl { grid-template-columns: 7rem 1fr 2.4rem; }
	.bpm__tabs { grid-template-columns: 1fr; }
	.bpm__dl > div { grid-template-columns: 1fr; gap: var(--cs-space-1); }
}
</style>
