<script setup lang="ts">
import { computed, ref } from "vue"

/**
 * 重复压缩下的两种摘要形态，以及信息分辨率的衰减。
 *
 * 左图：摘要体积随压缩次数的变化。层叠式把上一份摘要当作输入的一部分再次纳入，
 *       体积单调上升；改写式每次以旧摘要换新摘要，体积基本持平。
 * 右表：不同类型的信息在多次压缩后的保留程度——凡是精确值都最先流失。
 * 曲线与百分比是说明相对关系的示意，不是实测。
 */

const N = 5
/** 层叠式：每次把前一份摘要连同新历史一起纳入，摘要体积逐次抬高 */
const stacked = [2, 14, 22, 30, 38, 46]
/** 改写式：旧摘要进、新摘要出，体积由预算约束，基本恒定 */
const rewritten = [2, 13, 13, 14, 13, 14]

type Row = { label: string; keep: number[]; note: string }
const rows: Row[] = [
	{ label: "写进文件的内容", keep: [100, 100, 100, 100, 100, 100], note: "不在消息历史里，压缩碰不到它" },
	{ label: "用户原话与任务目标", keep: [100, 98, 96, 94, 93, 92], note: "逐字保留策略下几乎无损" },
	{ label: "来源链接与文件路径", keep: [100, 96, 92, 88, 84, 80], note: "体积极小，容易被摘要带上" },
	{ label: "架构决策与已知边界", keep: [100, 90, 82, 72, 63, 55], note: "被逐层概括，语义还在、依据没了" },
	{ label: "改动文件的完整清单", keep: [100, 76, 60, 46, 36, 28], note: "明细类信息最先流失" },
	{ label: "工具结果里的具体数值", keep: [100, 32, 14, 8, 4, 2], note: "第一次压缩就基本没了" }
]

const at = ref(3)
const mode = ref<"stacked" | "rewritten">("stacked")
const series = computed(() => (mode.value === "stacked" ? stacked : rewritten))

const W = 300
const H = 170
const padL = 34
const padR = 14
const padT = 14
const padB = 30
const plotW = W - padL - padR
const plotH = H - padT - padB
const baseY = padT + plotH
const yMax = 50
const xOf = (i: number) => padL + (i / N) * plotW
const yOf = (v: number) => baseY - (v / yMax) * plotH

const paths = computed(() =>
	[
		{ key: "stacked", label: "层叠式", color: "var(--cs-color-danger)", pts: stacked },
		{ key: "rewritten", label: "改写式", color: "var(--cs-color-success)", pts: rewritten }
	].map((s) => ({
		...s,
		d: s.pts.map((v, i) => `${i === 0 ? "M" : "L"}${xOf(i).toFixed(1)} ${yOf(v).toFixed(1)}`).join(" "),
		dim: s.key !== mode.value
	}))
)

const curSummary = computed(() => series.value[at.value])
const growth = computed(() => (series.value[at.value] / series.value[1]).toFixed(1))
</script>

<template>
	<div class="cfd">
		<div class="cfd__head">
			<div class="cfd__seg" role="tablist" aria-label="选择摘要的组织形态">
				<button
					type="button" role="tab" :aria-selected="mode === 'stacked'"
					:class="{ 'is-on': mode === 'stacked' }" @click="mode = 'stacked'"
				>层叠式</button>
				<button
					type="button" role="tab" :aria-selected="mode === 'rewritten'"
					:class="{ 'is-on': mode === 'rewritten' }" @click="mode = 'rewritten'"
				>改写式</button>
			</div>
			<label class="cfd__slider">
				<span>压缩次数</span>
				<input v-model.number="at" type="range" min="0" :max="N" step="1" />
				<b>{{ at }}</b>
			</label>
		</div>

		<div class="cfd__body">
			<figure class="cfd__fig">
				<svg :viewBox="`0 0 ${W} ${H}`" role="img"
					aria-label="摘要体积随压缩次数的变化：层叠式单调上升，改写式基本持平">
					<g v-for="t in [0, 10, 20, 30, 40, 50]" :key="`y${t}`">
						<line :x1="padL" :y1="yOf(t)" :x2="padL + plotW" :y2="yOf(t)" class="cfd__grid" />
						<text :x="padL - 6" :y="yOf(t) + 3" class="cfd__tick" text-anchor="end">{{ t }}K</text>
					</g>
					<line :x1="padL" :y1="baseY" :x2="padL + plotW" :y2="baseY" class="cfd__axis" />
					<g v-for="i in [0, 1, 2, 3, 4, 5]" :key="`x${i}`">
						<text :x="xOf(i)" :y="baseY + 13" class="cfd__tick" text-anchor="middle">{{ i }}</text>
					</g>
					<text :x="padL + plotW / 2" :y="H - 3" class="cfd__axlab" text-anchor="middle">第 N 次压缩</text>
					<line :x1="xOf(at)" :y1="padT" :x2="xOf(at)" :y2="baseY" class="cfd__cursor" />
					<path v-for="p in paths" :key="p.key" :d="p.d" class="cfd__line" :class="{ 'is-dim': p.dim }" :style="{ stroke: p.color }" />
					<circle
						v-for="p in paths" :key="`d${p.key}`" :cx="xOf(at)" :cy="yOf(p.pts[at])" r="3.2"
						:style="{ fill: p.color }" :class="{ 'is-dim': p.dim }"
					/>
				</svg>
				<figcaption class="cfd__cap">
					第 {{ at }} 次压缩后，摘要自身占 <b>{{ curSummary }}K</b>，
					是首次压缩的 <b>{{ growth }}</b> 倍。
				</figcaption>
			</figure>

			<div class="cfd__rows">
				<div v-for="r in rows" :key="r.label" class="cfd__row">
					<div class="cfd__row-top">
						<span class="cfd__row-lab">{{ r.label }}</span>
						<b class="cfd__row-val">{{ r.keep[at] }}%</b>
					</div>
					<div class="infra-bar">
						<i :style="{ width: `${r.keep[at]}%`, background: r.keep[at] >= 80 ? 'var(--cs-color-success)' : r.keep[at] >= 45 ? 'var(--cs-color-warning)' : 'var(--cs-color-danger)' }"></i>
					</div>
					<span class="cfd__row-note">{{ r.note }}</span>
				</div>
			</div>
		</div>

		<p class="pc-note">
			曲线与百分比为说明相对关系的示意，不是实测数据。两条曲线的差别来自摘要的组织方式：
			<b>层叠式</b>把上一份摘要作为「此前已压缩的上下文」再次纳入输入，于是覆盖的历史跨度越来越长、每段历史留下的字数越来越少；
			<b>改写式</b>把旧摘要喂进去、新摘要出来，体积由预算约束住。右侧六行的排序基本不随实现变化：
			越接近「精确值」的信息流失越快，所以数字、配置、清单、链接应当在压缩之前就落到文件里。
		</p>
	</div>
</template>

<style scoped>
.cfd { margin: 1rem 0; background: var(--cs-color-bg); border: 1px solid var(--cs-color-border); border-radius: var(--cs-radius-lg); padding: var(--cs-space-5) var(--cs-space-6); }
.cfd__head { display: flex; flex-wrap: wrap; align-items: center; justify-content: space-between; gap: var(--cs-space-4); margin-bottom: var(--cs-space-4); }
.cfd__seg { display: inline-flex; border: 1px solid var(--cs-color-border); border-radius: var(--cs-radius-pill); overflow: hidden; }
.cfd__seg button { min-height: var(--cs-tap-target); padding: 0 var(--cs-space-5); font-size: var(--cs-text-sm); background: var(--cs-color-bg); color: var(--cs-color-text-muted); border: 0; cursor: pointer; transition: var(--cs-transition-colors); }
.cfd__seg button.is-on { background: var(--cs-color-brand-soft); color: var(--cs-color-brand); font-weight: 600; }
.cfd__slider { display: inline-grid; grid-template-columns: 4.6rem 8rem 1.4rem; align-items: center; gap: var(--cs-space-2); min-height: var(--cs-tap-target); font-size: var(--cs-text-sm); }
.cfd__slider span { color: var(--cs-color-text-muted); }
.cfd__slider input { width: 100%; accent-color: var(--cs-color-brand); }
.cfd__slider b { font-family: var(--cs-font-mono); color: var(--cs-color-brand); text-align: right; }
.cfd__body { display: grid; grid-template-columns: minmax(0, 5fr) minmax(0, 6fr); gap: var(--cs-space-6); align-items: start; }
.cfd__fig { margin: 0; min-width: 0; }
.cfd__rows { min-width: 0; }
.cfd__fig svg { width: 100%; height: auto; display: block; }
.cfd__grid { stroke: var(--cs-color-border); stroke-width: 1; stroke-dasharray: 3 3; }
.cfd__axis { stroke: var(--cs-color-border-strong); stroke-width: 1; }
.cfd__tick { font-family: var(--cs-font-mono); font-size: 8px; fill: var(--cs-color-text-subtle); }
.cfd__axlab { font-size: 9px; fill: var(--cs-color-text-muted); }
.cfd__cursor { stroke: var(--cs-color-brand); stroke-width: 1; stroke-dasharray: 3 2; opacity: .6; }
.cfd__line { fill: none; stroke-width: 2; stroke-linecap: round; stroke-linejoin: round; }
.cfd__line.is-dim, circle.is-dim { opacity: .28; }
.cfd__cap { margin-top: var(--cs-space-3); font-size: var(--cs-text-2xs); line-height: var(--cs-leading-normal); color: var(--cs-color-text-subtle); }
.cfd__cap b { font-family: var(--cs-font-mono); color: var(--cs-color-text); }
.cfd__row { margin-bottom: var(--cs-space-3); }
.cfd__row-top { display: flex; align-items: baseline; justify-content: space-between; gap: var(--cs-space-2); }
.cfd__row-lab { font-size: var(--cs-text-sm); color: var(--cs-color-text); }
.cfd__row-val { font-family: var(--cs-font-mono); font-size: var(--cs-text-sm); color: var(--cs-color-text); }
.cfd__row-note { display: block; margin-top: var(--cs-space-1); font-size: var(--cs-text-3xs); color: var(--cs-color-text-subtle); }
@media (max-width: 760px) {
	.cfd__body { grid-template-columns: 1fr; }
}
</style>
