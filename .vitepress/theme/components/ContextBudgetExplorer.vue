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
		const value = signalOf(k) * recall(tokensOf(k))
		if (value > bestValue) {
			bestValue = value
			best = k
		}
	}
	return best
})
const atOptimum = computed(() => included.value === bestIncluded.value)

const resetLab = useLabReset(included, rotScale)
</script>

<template>
	<LearningLab topic="ContextBudgetExplorer" @reset="resetLab">
		<p class="infra-lab__hint">材料按相关性从高到低排列：点击某条把「纳入前 N 条」设到该位置，或拖动滑杆。观察输入变长时，有效信号量为何在中途见顶。</p>
		<div class="ctx-strip" aria-label="候选材料，按相关性从高到低排列">
			<button
				v-for="(m, index) in materials"
				:key="index"
				type="button"
				:class="[m.signal ? 'is-signal' : 'is-noise', { 'is-outside': index >= included }]"
				:aria-label="`第 ${index + 1} 条：${m.label}，${m.signal ? '高信号' : '低信号'}，${m.tokens} token，点击纳入到此`"
				@click="included = index + 1"
			>
				<span>{{ m.signal ? "高信号" : "低信号" }}</span>
				<strong>{{ m.label }}</strong>
				<span>{{ m.tokens }} tok</span>
			</button>
		</div>
		<div class="infra-controls">
			<div class="infra-control"><label :for="fieldId('ctx-included')">纳入前 N 条：{{ included }}/{{ materials.length }}</label><input :id="fieldId('ctx-included')" v-model.number="included" type="range" min="0" :max="materials.length"></div>
			<div class="infra-control"><label :for="fieldId('ctx-rot')">召回衰减尺度：{{ rotScale }} tok</label><input :id="fieldId('ctx-rot')" v-model.number="rotScale" type="range" min="600" max="4000" step="200"></div>
		</div>
		<div class="infra-results">
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
.ctx-strip { display: flex; flex-wrap: wrap; gap: .4rem; margin: .9rem 0; }
.ctx-strip button { flex: 1 1 130px; min-width: 118px; text-align: left; border: 1px solid var(--cs-color-border); border-radius: 8px; background: var(--cs-color-bg); color: var(--cs-color-text); cursor: pointer; padding: .45rem .55rem; }
.ctx-strip button span { display: block; font-size: .68rem; color: var(--cs-color-text-muted); }
.ctx-strip button strong { display: block; font-size: .82rem; margin: .1rem 0; line-height: 1.4; }
.ctx-strip button.is-signal { border-color: var(--cs-color-success); background: var(--cs-color-success-soft); }
.ctx-strip button.is-noise { border-color: var(--cs-color-border); background: var(--cs-color-bg-soft); }
.ctx-strip button.is-outside { opacity: .42; }
</style>
