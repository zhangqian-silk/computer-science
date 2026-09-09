<script setup lang="ts">
import { computed, ref, useId } from "vue"
import LearningLab from "./LearningLab.vue"
import { useLabReset } from "../use-lab-reset"
const labId = useId()
const fieldId = (name: string) => `${labId}-${name}`

const topK = ref(1)
const capacityFactor = ref(1)
const skew = ref(0.2)
const tokenCount = 12
const expertCount = 4

const routes = computed(() => Array.from({ length: tokenCount }, (_, token) => {
	const scores = Array.from({ length: expertCount }, (_, expert) => {
		const structured = Math.sin((token + 1) * (expert + 2) * 1.7) * 0.7
		return structured + (expert === 0 ? skew.value * 2 : 0)
	})
	return scores.map((score, expert) => ({ score, expert })).sort((left, right) => right.score - left.score).slice(0, topK.value)
}))
const capacity = computed(() => Math.ceil(tokenCount * topK.value / expertCount * capacityFactor.value))
const loads = computed(() => Array.from({ length: expertCount }, (_, expert) => routes.value.reduce((sum, route) => sum + Number(route.some((item) => item.expert === expert)), 0)))
const overflow = computed(() => loads.value.reduce((sum, load) => sum + Math.max(0, load - capacity.value), 0))
const utilization = computed(() => loads.value.reduce((sum, load) => sum + Math.min(load, capacity.value), 0) / (capacity.value * expertCount) * 100)
const resetLab = useLabReset(topK, capacityFactor, skew)
</script>

<template>
	<LearningLab topic="MoERouterExplorer" @reset="resetLab">
		<p class="infra-lab__hint">给专家 0 增加偏置，观察 Top-k、容量因子、负载不均和溢出的关系。</p>
		<div class="infra-controls">
			<div class="infra-control"><label :for="fieldId('moe-topk')">Top-k</label><select :id="fieldId('moe-topk')" v-model.number="topK"><option :value="1">1</option><option :value="2">2</option></select></div>
			<div class="infra-control"><label :for="fieldId('moe-capacity')">容量因子：{{ capacityFactor.toFixed(2) }}</label><input :id="fieldId('moe-capacity')" v-model.number="capacityFactor" type="range" min="0.5" max="2" step="0.05"></div>
			<div class="infra-control"><label :for="fieldId('moe-skew')">专家 0 偏置：{{ skew.toFixed(2) }}</label><input :id="fieldId('moe-skew')" v-model.number="skew" type="range" min="0" max="1" step="0.05"></div>
		</div>
		<div class="expert-grid">
			<div v-for="(load, expert) in loads" :key="expert" class="expert-card">
				<span>Expert {{ expert }}</span><strong>{{ load }} / {{ capacity }}</strong>
				<div class="infra-bar"><i :style="{ width: `${Math.min(100, load / capacity * 100)}%` }"></i></div>
				<small>{{ Math.max(0, load - capacity) }} 个溢出分配</small>
			</div>
		</div>
		<div class="infra-results">
			<div class="infra-result"><span>每专家容量</span><strong>{{ capacity }} token 槽</strong></div>
			<div class="infra-result"><span>总溢出分配</span><strong>{{ overflow }}</strong></div>
			<div class="infra-result"><span>已接纳槽位利用率</span><strong>{{ utilization.toFixed(1) }}%</strong></div>
		</div>
		<p class="infra-note">这是确定性的路由示例。真实 Router 概率由模型学习；容量因子只能吸收波动，不能替代辅助损失和拓扑友好的专家布局。</p>
	</LearningLab>
</template>

<style scoped>
.expert-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(130px, 1fr)); gap: .55rem; margin-top: .9rem; }
.expert-card { padding: .65rem; border: 1px solid var(--vp-c-divider); border-radius: 8px; background: var(--vp-c-bg); }
.expert-card span, .expert-card small { display: block; color: var(--vp-c-text-2); font-size: .75rem; }
.expert-card strong { display: block; margin: .15rem 0; color: var(--vp-c-text-1); }
@media (max-width: 640px) { .expert-grid { grid-template-columns: repeat(auto-fit, minmax(100px, 1fr)); } }
</style>
