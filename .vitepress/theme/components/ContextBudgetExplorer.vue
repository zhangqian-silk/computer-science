<script setup lang="ts">
import { computed, ref, useId } from "vue"
import LearningLab from "./LearningLab.vue"
import { useLabReset } from "../use-lab-reset"
const labId = useId()
const fieldId = (name: string) => `${labId}-${name}`

type Material = { label: string; tokens: number; signal: boolean }

// 候选材料按相关性从高到低排列：前段是高信号（当前任务必需），后段是低信号（冗余或跑题）。
const materials: Material[] = [
	{ label: "当前目标与约束", tokens: 300, signal: true },
	{ label: "配置当前值", tokens: 250, signal: true },
	{ label: "相关代码定义", tokens: 400, signal: true },
	{ label: "最近工具结果", tokens: 350, signal: true },
	{ label: "早期重复的工具原文", tokens: 900, signal: false },
	{ label: "另一环境的旧配置", tokens: 500, signal: false },
	{ label: "泛化的历史寒暄", tokens: 300, signal: false }
]

const included = ref(4)
const rotScale = ref(1600)

const signalTotal = materials.filter((m) => m.signal).length
const tokensOf = (k: number) => materials.slice(0, k).reduce((sum, m) => sum + m.tokens, 0)
const signalOf = (k: number) => materials.slice(0, k).filter((m) => m.signal).length
// context rot：召回随输入长度递减（示意模型，非实测）。
const recall = (tokens: number) => 1 / (1 + (tokens / rotScale.value) ** 2)

const usedTokens = computed(() => tokensOf(included.value))
const signalIncluded = computed(() => signalOf(included.value))
const recallEff = computed(() => recall(usedTokens.value))
const effectiveSignal = computed(() => signalIncluded.value * recallEff.value)
// 注意力两两关系数量级 ~ n²，用于表达注意力被摊薄。
const pairwise = computed(() => included.value * included.value)

const bestIncluded = computed(() => {
	let best = 0
	let bestValue = -1
	for (let k = 0; k <= materials.length; k++) {
		const value = signalValueOf(k)
		if (value > bestValue) {
			bestValue = value
			best = k
		}
	}
	return best
})
const atOptimum = computed(() => included.value === bestIncluded.value)

// 有效信号量曲线：x 为纳入条数 0..7，y 为「纳入高信号条数 × 召回系数」。
// 曲线随滑杆（召回衰减尺度）改变形状，峰值位置也会随之移动。
const signalValueOf = (k: number) => signalOf(k) * recall(tokensOf(k))
const CW = 340
const CH = 200
const cpadL = 30
const cpadR = 14
const cpadT = 20
const cpadB = 36
const cplotW = CW - cpadL - cpadR
const cplotH = CH - cpadT - cpadB
const cbaseY = cpadT + cplotH
const SIG_MAX = signalTotal
const cxOf = (k: number) => cpadL + (k / materials.length) * cplotW
const cyOf = (v: number) => cbaseY - (Math.min(v, SIG_MAX) / SIG_MAX) * cplotH
const curvePts = computed(() =>
	Array.from({ length: materials.length + 1 }, (_, k) => ({ k, v: signalValueOf(k), x: cxOf(k), y: cyOf(signalValueOf(k)) }))
)
const curveLine = computed(() => curvePts.value.map((p, i) => `${i === 0 ? "M" : "L"}${p.x.toFixed(1)} ${p.y.toFixed(1)}`).join(" "))
const curveArea = computed(
	() =>
		`${cpadL},${cbaseY} ${curvePts.value.map((p) => `${p.x.toFixed(1)},${p.y.toFixed(1)}`).join(" ")} ${cpadL + cplotW},${cbaseY}`
)

// 随操作即时变化的结论：空窗口 / 未收全 / 正好在峰值 / 已越过峰值四态。
const verdict = computed<{ tone: "neutral" | "info" | "success" | "danger"; text: string }>(() => {
	const k = included.value
	const peak = bestIncluded.value
	const v = effectiveSignal.value
	if (k === 0) {
		return { tone: "neutral", text: "窗口还是空的：一条材料都没有纳入，有效信号为 0。" }
	}
	if (k < peak) {
		return { tone: "info", text: `已纳入 ${k} 条，其中高信号 ${signalOf(k)} 条；还差 ${peak - k} 条关键材料，有效信号尚未到顶。` }
	}
	if (k === peak) {
		return { tone: "success", text: `当前就是峰值：${peak} 条高信号全部进入，且没有掺入低信号，有效信号 ${v.toFixed(2)} 为全程最高。` }
	}
	const peakVal = signalValueOf(peak)
	return {
		tone: "danger",
		text: `已越过峰值：新增的是低信号材料，有效信号从 ${peakVal.toFixed(2)} 跌到 ${v.toFixed(2)}，输入却涨到 ${usedTokens.value.toLocaleString()} token。`
	}
})

const resetLab = useLabReset(included, rotScale)
</script>

<template>
	<LearningLab topic="ContextBudgetExplorer" @reset="resetLab">
		<p class="infra-lab__hint">材料按相关性从高到低排列：点击某条把「纳入前 N 条」设到该位置，或拖动滑杆。观察输入变长时，有效信号量为何在中途见顶。</p>

		<div class="ctx-chart" role="img" :aria-label="`有效信号量随纳入条数变化的曲线，当前纳入 ${included} 条，有效信号 ${effectiveSignal.toFixed(2)}，峰值出现在第 ${bestIncluded} 条`">
			<svg :viewBox="`0 0 ${CW} ${CH}`">
				<rect :x="cxOf(0)" :y="cpadT" :width="cxOf(signalTotal) - cxOf(0)" :height="cplotH" class="ctx-chart__zone ctx-chart__zone--sig" />
				<rect :x="cxOf(signalTotal)" :y="cpadT" :width="cpadL + cplotW - cxOf(signalTotal)" :height="cplotH" class="ctx-chart__zone ctx-chart__zone--noise" />
				<line :x1="cpadL" :y1="cbaseY" :x2="cpadL + cplotW" :y2="cbaseY" class="ctx-chart__axis" />
				<g v-for="g in [0, 2, 4]" :key="`g${g}`">
					<line :x1="cpadL" :y1="cyOf(g)" :x2="cpadL + cplotW" :y2="cyOf(g)" class="ctx-chart__grid" />
					<text :x="cpadL - 6" :y="cyOf(g) + 3.5" class="ctx-chart__tick" text-anchor="end">{{ g }}</text>
				</g>
				<text :x="(cxOf(0) + cxOf(signalTotal)) / 2" :y="cpadT + 12" class="ctx-chart__zone-lab" text-anchor="middle">高信号材料区</text>
				<text :x="(cxOf(signalTotal) + cxOf(materials.length)) / 2" :y="cpadT + 12" class="ctx-chart__zone-lab" text-anchor="middle">低信号材料区</text>
				<line :x1="cxOf(bestIncluded)" :y1="cpadT" :x2="cxOf(bestIncluded)" :y2="cbaseY" class="ctx-chart__peak" />
				<text :x="cxOf(bestIncluded)" :y="cbaseY + 26" class="ctx-chart__peak-lab" text-anchor="middle">峰值 · {{ bestIncluded }} 条</text>
				<polygon :points="curveArea" class="ctx-chart__area" />
				<polyline :points="curveLine" class="ctx-chart__line" />
				<circle
					v-for="(p, i) in curvePts"
					:key="`p${i}`"
					:cx="p.x"
					:cy="p.y"
					:r="included === i ? 0 : 2.4"
					class="ctx-chart__dot"
				/>
				<circle :cx="cxOf(included)" :cy="cyOf(effectiveSignal)" r="5.5" class="ctx-chart__cur" />
				<text :x="cxOf(included)" :y="cyOf(effectiveSignal) - 10" class="ctx-chart__cur-lab" text-anchor="middle">{{ effectiveSignal.toFixed(1) }}</text>
				<text v-for="k in materials.length + 1" :key="`t${k - 1}`" :x="cxOf(k - 1)" :y="cbaseY + 14" class="ctx-chart__tick" text-anchor="middle">{{ k - 1 }}</text>
				<text :x="cpadL + cplotW" :y="CH - 2" class="ctx-chart__xlab" text-anchor="end">纳入材料条数 →</text>
			</svg>
		</div>

		<p class="ctx-verdict" :class="`ctx-verdict--${verdict.tone}`" aria-live="polite">{{ verdict.text }}</p>

		<div class="ctx-strip" role="group" aria-label="候选材料，按相关性从高到低排列">
			<div v-if="included === 0" class="ctx-cut">
				<span class="ctx-cut__line" />
				<span class="ctx-cut__tag">上下文窗口为空</span>
				<span class="ctx-cut__line" />
			</div>
			<template v-for="(m, index) in materials" :key="index">
				<button
					type="button"
					:class="[
						m.signal ? 'is-signal' : 'is-noise',
						{ 'is-outside': index >= included, 'is-noise-in': !m.signal && index < included }
					]"
					:aria-pressed="index < included"
					:aria-label="`第 ${index + 1} 条：${m.label}，${m.signal ? '高信号' : '低信号'}，${m.tokens} token，点击纳入到此`"
					@click="included = index + 1"
				>
					<span>{{ m.signal ? "高信号" : "低信号" }}</span>
					<strong>{{ m.label }}</strong>
					<span>{{ m.tokens }} tok</span>
				</button>
				<div v-if="index === included - 1" class="ctx-cut">
					<span class="ctx-cut__line" />
					<span class="ctx-cut__tag">本次窗口边界 · 前 {{ included }} 条进入请求</span>
					<span class="ctx-cut__line" />
				</div>
			</template>
		</div>
		<div class="infra-controls">
			<div class="infra-control"><label :for="fieldId('ctx-included')">纳入前 N 条：{{ included }}/{{ materials.length }}</label><input :id="fieldId('ctx-included')" v-model.number="included" type="range" min="0" :max="materials.length"></div>
			<div class="infra-control"><label :for="fieldId('ctx-rot')">召回衰减尺度：{{ rotScale }} tok</label><input :id="fieldId('ctx-rot')" v-model.number="rotScale" type="range" min="600" max="4000" step="200"></div>
		</div>
		<div class="infra-results ctx-results">
			<div class="infra-result"><span>输入 token</span><strong>{{ usedTokens }}</strong></div>
			<div class="infra-result"><span>纳入高信号</span><strong>{{ signalIncluded }}/{{ signalTotal }}</strong></div>
			<div class="infra-result"><span>注意力两两关系 ≈ n²</span><strong>{{ pairwise }}</strong></div>
			<div class="infra-result"><span>有效召回系数</span><strong>{{ recallEff.toFixed(2) }}</strong></div>
			<div class="infra-result"><span>有效信号量</span><strong>{{ effectiveSignal.toFixed(2) }}</strong></div>
			<div class="infra-result"><span>是否最优纳入点</span><strong>{{ atOptimum ? "是" : `否（最优 ${bestIncluded} 条）` }}</strong></div>
		</div>
		<p class="infra-note">有效信号量 = 纳入高信号条数 × 召回系数；召回按 1/(1+(token/尺度)²) 的示意模型随长度递减，非任何模型实测。加入低信号材料只增加长度、不增加信号，峰值出现在「收全高信号且不掺入低信号」处，因此目标是最小高信号集合，而不是纳入越多越好。</p>
	</LearningLab>
</template>

<style scoped>
.ctx-chart {
	margin: var(--cs-space-6) 0 0;
	padding: var(--cs-space-5) var(--cs-space-5) var(--cs-space-3);
	border: var(--cs-border);
	border-radius: var(--cs-radius-md);
	background: var(--cs-color-bg);
}

.ctx-chart svg {
	display: block;
	width: 100%;
	height: auto;
}

.ctx-chart__zone--sig { fill: color-mix(in srgb, var(--cs-color-success) 8%, transparent); }
.ctx-chart__zone--noise { fill: color-mix(in srgb, var(--cs-color-danger) 5%, transparent); }

.ctx-chart__zone-lab {
	fill: var(--cs-color-text-subtle);
	font-size: 9px;
	font-family: var(--cs-font-mono);
	letter-spacing: 0.04em;
}

.ctx-chart__grid { stroke: var(--cs-color-border); stroke-width: 1; stroke-dasharray: 2 3; }
.ctx-chart__axis { stroke: var(--cs-color-border-strong, var(--cs-color-border)); stroke-width: 1; }

.ctx-chart__tick {
	fill: var(--cs-color-text-subtle);
	font-size: 9px;
	font-family: var(--cs-font-mono);
}

.ctx-chart__xlab { fill: var(--cs-color-text-muted); font-size: 10px; }
.ctx-chart__peak { stroke: var(--cs-color-success); stroke-width: 1.2; stroke-dasharray: 4 3; }
.ctx-chart__peak-lab { fill: var(--cs-color-success); font-size: 9px; font-family: var(--cs-font-mono); }
.ctx-chart__area { fill: color-mix(in srgb, var(--cs-color-brand) 12%, transparent); }
.ctx-chart__line { fill: none; stroke: var(--cs-color-brand); stroke-width: 2; stroke-linecap: round; stroke-linejoin: round; }
.ctx-chart__dot { fill: var(--cs-color-brand); opacity: 0.55; }
.ctx-chart__cur { fill: var(--cs-color-brand); stroke: var(--cs-color-bg); stroke-width: 2; }
.ctx-chart__cur-lab { fill: var(--cs-color-brand); font-size: 10px; font-weight: 700; font-family: var(--cs-font-mono); }

/* 随曲线状态变色的结论条 */
.ctx-verdict {
	margin: var(--cs-space-5) 0 0;
	padding: var(--cs-space-4) var(--cs-space-6);
	border-left: var(--cs-border-width-emphasis) solid var(--cs-color-border);
	border-radius: var(--cs-radius-sm);
	background: var(--cs-color-bg-soft);
	font-size: var(--cs-text-sm);
	line-height: var(--cs-leading-relaxed);
	color: var(--cs-color-text-muted);
}

.ctx-verdict--info { border-left-color: var(--cs-color-info); }
.ctx-verdict--success { border-left-color: var(--cs-color-success); }
.ctx-verdict--danger { border-left-color: var(--cs-color-danger); }

/* 候选材料：网格布局，动态在第 N 条后插入「窗口边界」整行分隔 */
.ctx-strip {
	display: grid;
	grid-template-columns: repeat(auto-fill, minmax(138px, 1fr));
	gap: var(--cs-space-3);
	margin: var(--cs-space-6) 0;
}

.ctx-strip button {
	min-height: var(--cs-tap-target);
	padding: var(--cs-space-3) var(--cs-space-4);
	text-align: left;
	border: var(--cs-border);
	border-radius: var(--cs-radius-md);
	background: var(--cs-color-bg);
	color: var(--cs-color-text);
	cursor: pointer;
	transition: var(--cs-transition-colors);
}

.ctx-strip button:hover {
	border-color: var(--cs-color-brand);
}

.ctx-strip button span {
	display: block;
	font-size: var(--cs-text-3xs);
	color: var(--cs-color-text-muted);
}

.ctx-strip button strong {
	display: block;
	margin: var(--cs-space-1) 0;
	font-size: var(--cs-text-sm);
	line-height: var(--cs-leading-tight);
}

.ctx-strip button.is-signal {
	border-color: var(--cs-color-success);
	background: var(--cs-color-success-soft);
}

/* 已进入窗口的低信号材料用告警色标出，直观看出「掺入杂质」的时刻 */
.ctx-strip button.is-noise-in {
	border-color: var(--cs-color-warning);
	background: var(--cs-color-warning-soft);
}

.ctx-strip button.is-outside {
	border-style: dashed;
	opacity: 0.5;
}

.ctx-cut {
	grid-column: 1 / -1;
	display: flex;
	align-items: center;
	gap: var(--cs-space-4);
}

.ctx-cut__line {
	flex: 1;
	border-top: 1px dashed var(--cs-color-brand);
	opacity: 0.55;
}

.ctx-cut__tag {
	flex: 0 0 auto;
	padding: var(--cs-space-1) var(--cs-space-4);
	border: 1px solid var(--cs-color-brand);
	border-radius: var(--cs-radius-pill);
	font-family: var(--cs-font-mono);
	font-size: var(--cs-text-3xs);
	color: var(--cs-color-brand);
	background: var(--cs-color-bg);
}

/* 指标网格：桌面 3 列 × 2 行，窄屏 2 列 × 3 行，避免单列拖长页面 */
.ctx-results {
	grid-template-columns: repeat(3, minmax(0, 1fr));
}

@media (max-width: 640px) {
	.ctx-strip {
		grid-template-columns: repeat(2, minmax(0, 1fr));
	}

	.ctx-results {
		grid-template-columns: repeat(2, minmax(0, 1fr));
	}

	.ctx-chart {
		padding: var(--cs-space-4) var(--cs-space-3) var(--cs-space-2);
	}
}
</style>
