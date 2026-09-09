<script setup lang="ts">
import { computed, ref, useId } from "vue"
import LearningLab from "./LearningLab.vue"
import { useLabReset } from "../use-lab-reset"

const labId = useId()
const fieldId = (name: string) => `${labId}-${name}`

const slo = ref(99.9)       // 目标可用性 %
const consumed = ref(40)    // 已消耗的错误预算 %

const MIN_PER_MONTH = 30 * 24 * 60 // 43200
const budgetMin = computed(() => (1 - slo.value / 100) * MIN_PER_MONTH)
const remainingMin = computed(() => budgetMin.value * (1 - consumed.value / 100))

function fmt(mins: number): string {
	if (mins >= 60) return (mins / 60).toFixed(1) + " 小时"
	return mins.toFixed(1) + " 分钟"
}

const action = computed(() => {
	if (consumed.value >= 100) return { text: "预算已耗尽：冻结功能变更，全员投入稳定性与故障复盘。", tone: "danger" }
	if (consumed.value >= 90) return { text: "预算告急：放缓发布、加强灰度与门禁，优先修复可靠性问题。", tone: "warning" }
	if (consumed.value >= 50) return { text: "预算正常消耗：按常规灰度节奏发布，持续观测。", tone: "neutral" }
	return { text: "预算充足：可加快发布、开展试验与风险性变更。", tone: "success" }
})

const resetLab = useLabReset(slo, consumed)
</script>

<template>
	<LearningLab topic="SloErrorBudgetExplorer" @reset="resetLab">
		<p class="infra-lab__hint">错误预算 = (1 − SLO) × 周期（按 30 天算）。它把「可靠性目标」翻译成「可以容忍多少不可用」，并驱动发布节奏。</p>
		<div class="infra-controls">
			<div class="infra-control">
				<label :for="fieldId('slo')">SLO 目标</label>
				<select :id="fieldId('slo')" v-model.number="slo">
					<option :value="99">99%（两个九）</option>
					<option :value="99.9">99.9%（三个九）</option>
					<option :value="99.95">99.95%</option>
					<option :value="99.99">99.99%（四个九）</option>
				</select>
			</div>
			<div class="infra-control"><label :for="fieldId('used')">已消耗错误预算：{{ consumed }}%</label><input :id="fieldId('used')" v-model.number="consumed" type="range" min="0" max="120" step="5"></div>
		</div>
		<div class="seb-bar"><i :class="`t-${action.tone}`" :style="{ width: Math.min(100, consumed) + '%' }"></i></div>
		<div class="infra-results">
			<div class="infra-result"><span>每月错误预算</span><strong>{{ fmt(budgetMin) }}</strong></div>
			<div class="infra-result"><span>剩余预算</span><strong>{{ remainingMin > 0 ? fmt(remainingMin) : "已透支" }}</strong></div>
		</div>
		<div class="infra-result seb-action"><span>建议动作</span><strong>{{ action.text }}</strong></div>
		<p class="infra-note">SLO 每多一个「9」，可容忍停机时间降一个数量级：99%≈7.2 小时/月，99.9%≈43 分钟，99.99%≈4.3 分钟。错误预算把可靠性与迭代速度用同一把尺子对齐。</p>
	</LearningLab>
</template>

<style scoped>
.seb-bar { height: 16px; background: var(--vp-c-bg-soft); border-radius: 8px; overflow: hidden; margin: .8rem 0; }
.seb-bar > i { display: block; height: 100%; }
.t-success { background: var(--vp-c-green-1); }
.t-neutral { background: var(--vp-c-brand-1); }
.t-warning { background: var(--vp-c-warning-1); }
.t-danger { background: var(--vp-c-danger-1); }
.seb-action { margin-top: .5rem; }
</style>
