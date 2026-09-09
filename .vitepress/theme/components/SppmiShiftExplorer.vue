<script setup lang="ts">
import { computed, ref, useId } from "vue"
import LearningLab from "./LearningLab.vue"
import { useLabReset } from "../use-lab-reset"
const labId = useId()
const fieldId = (name: string) => `${labId}-${name}`

const words = ['猫', '狗', '鱼', '汽车', '引擎', '天空']
const contexts = ['宠物', '动物', '水', '交通', '机械']
const counts = [
	[80, 30, 5, 2, 1],
	[90, 40, 8, 1, 2],
	[20, 35, 90, 1, 1],
	[1, 2, 3, 95, 60],
	[1, 1, 2, 70, 85],
	[2, 3, 4, 5, 3]
]

const k = ref(5)

const total = counts.flat().reduce((sum, value) => sum + value, 0)
const wordTotals = words.map((_, i) => contexts.reduce((sum, _, j) => sum + counts[i][j], 0))
const contextTotals = contexts.map((_, j) => words.reduce((sum, _, i) => sum + counts[i][j], 0))

const pmi = (i: number, j: number) => Math.log((counts[i][j] * total) / (wordTotals[i] * contextTotals[j]))
const sppmi = (i: number, j: number) => Math.max(pmi(i, j) - Math.log(k.value), 0)

const pmiMax = Math.max(...words.flatMap((_, i) => contexts.map((_, j) => pmi(i, j))))
const sppmiMax = computed(() => Math.max(...words.flatMap((_, i) => contexts.map((_, j) => sppmi(i, j))), 1e-6))
const logK = computed(() => Math.log(k.value))

const surviving = computed(() => {
	let count = 0
	for (let i = 0; i < words.length; i++) {
		for (let j = 0; j < contexts.length; j++) {
			if (sppmi(i, j) > 0) count++
		}
	}
	return count
})

const cellStyle = (value: number, max: number) => {
	const ratio = value <= 0 ? 0 : Math.min(value / max, 1)
	return {
		backgroundColor: `color-mix(in srgb, var(--vp-c-brand-1) ${(ratio * 24).toFixed(1)}%, var(--vp-c-bg))`,
		color: "var(--vp-c-text-1)"
	}
}

const format = (value: number) => value.toFixed(2)
const resetLab = useLabReset(k)
</script>

<template>
	<LearningLab topic="SppmiShiftExplorer" @reset="resetLab">
		<p class="infra-lab__hint">固定玩具共现计数，调节 k；SPPMI 先从 PMI 减去 log k，再把负值截断为 0。</p>
		<div class="infra-controls">
			<div class="infra-control">
				<label :for="fieldId('sppmi-k')">负采样数 k：{{ k }}（log k ≈ {{ logK.toFixed(2) }}）</label>
				<input :id="fieldId('sppmi-k')" v-model.number="k" type="range" min="1" max="20" step="1">
			</div>
		</div>
		<div class="sppmi-panels">
			<section class="sppmi-panel">
				<p class="sppmi-panel__title">PMI（固定）</p>
				<div class="sppmi-matrix" role="img" aria-label="PMI 矩阵">
					<span class="sppmi-cell is-header"></span>
					<span v-for="ctx in contexts" :key="`pmi-h-${ctx}`" class="sppmi-cell is-header">{{ ctx }}</span>
					<template v-for="(word, i) in words" :key="`pmi-r-${word}`">
						<span class="sppmi-cell is-header">{{ word }}</span>
						<span v-for="(ctx, j) in contexts" :key="`pmi-${word}-${ctx}`" class="sppmi-cell" :style="cellStyle(pmi(i, j), pmiMax)">{{ format(pmi(i, j)) }}</span>
					</template>
				</div>
			</section>
			<section class="sppmi-panel">
				<p class="sppmi-panel__title">SPPMI（k = {{ k }}）</p>
				<div class="sppmi-matrix" role="img" :aria-label="`k=${k} 时的 SPPMI 矩阵`">
					<span class="sppmi-cell is-header"></span>
					<span v-for="ctx in contexts" :key="`sppmi-h-${ctx}`" class="sppmi-cell is-header">{{ ctx }}</span>
					<template v-for="(word, i) in words" :key="`sppmi-r-${word}`">
						<span class="sppmi-cell is-header">{{ word }}</span>
						<span v-for="(ctx, j) in contexts" :key="`sppmi-${word}-${ctx}`" class="sppmi-cell" :style="cellStyle(sppmi(i, j), sppmiMax)">{{ format(sppmi(i, j)) }}</span>
					</template>
				</div>
			</section>
		</div>
		<div class="infra-results">
			<div class="infra-result"><span>幸存非零条目</span><strong>{{ surviving }} / {{ words.length * contexts.length }}</strong></div>
			<div class="infra-result"><span>截断阈值</span><strong>log {{ k }} ≈ {{ logK.toFixed(2) }}</strong></div>
		</div>
		<p class="infra-note">此例「天空」只有 17 次共现，不能据此称为高频词。负 PMI 表示共现低于独立基线；右图才执行平移和截断，颜色深浅不替代数值。</p>
	</LearningLab>
</template>

<style scoped>
.sppmi-panels { display: grid; grid-template-columns: repeat(auto-fit, minmax(min(100%, 280px), 1fr)); gap: 0.9rem; }
.sppmi-panel { min-width: 0; padding: 0.75rem; border: 1px solid var(--vp-c-divider); border-radius: 8px; background: var(--vp-c-bg); }
.sppmi-panel__title { margin: 0 0 0.6rem; font-weight: 600; font-size: 0.88rem; color: var(--vp-c-text-1); }
.sppmi-matrix { display: grid; grid-template-columns: 56px repeat(5, minmax(44px, 1fr)); gap: 4px; overflow-x: auto; }
.sppmi-cell { display: flex; align-items: center; justify-content: center; min-height: 34px; border: 1px solid var(--vp-c-divider); border-radius: 5px; font-family: var(--vp-font-family-mono); font-size: 0.72rem; }
.sppmi-cell.is-header { border: none; background: transparent; font-family: inherit; font-size: 0.75rem; color: var(--vp-c-text-2); }
@media (max-width: 640px) {
	.sppmi-matrix { grid-template-columns: 48px repeat(5, minmax(40px, 1fr)); }
	.sppmi-cell { min-height: 30px; font-size: 0.66rem; }
}
</style>
