<script setup lang="ts">
import { computed, ref } from 'vue'

const scores = [ref(2), ref(1), ref(0)]
const values = [-1, 0.5, 2]
const temperature = ref(1)
const weights = computed(() => {
	const scaled = scores.map(score => score.value / temperature.value)
	const maximum = Math.max(...scaled)
	const exponentials = scaled.map(score => Math.exp(score - maximum))
	const denominator = exponentials.reduce((sum, value) => sum + value, 0)
	return exponentials.map(value => value / denominator)
})
const contributions = computed(() => weights.value.map((weight, index) => weight * values[index]))
const output = computed(() => contributions.value.reduce((sum, value) => sum + value, 0))
</script>

<template>
	<div class="infra-lab">
		<p class="infra-lab__title">Attention 读取实验台</p>
		<p class="infra-lab__hint">改变三个 Query-Key 分数和温度，观察 Softmax 如何分配权重，再用权重聚合 Value。</p>
		<div class="infra-controls">
			<div v-for="(score, index) in scores" :key="index" class="infra-control">
				<label :for="`attention-score-${index}`">分数 {{ index + 1 }}：{{ score.value.toFixed(1) }}</label>
				<input :id="`attention-score-${index}`" v-model.number="score.value" type="range" min="-4" max="4" step="0.1">
			</div>
			<div class="infra-control">
				<label for="attention-temperature">温度：{{ temperature.toFixed(1) }}</label>
				<input id="attention-temperature" v-model.number="temperature" type="range" min="0.2" max="3" step="0.1">
			</div>
		</div>
		<div class="metric-list" aria-live="polite">
			<div v-for="(value, index) in values" :key="value" class="metric-row">
				<span>Value {{ index + 1 }} = {{ value.toFixed(1) }}</span>
				<strong>权重 {{ weights[index].toFixed(3) }}</strong>
				<div class="infra-bar"><i :style="{ width: `${weights[index] * 100}%` }" /></div>
				<span>贡献 {{ contributions[index].toFixed(3) }}</span>
			</div>
		</div>
		<div class="infra-results">
			<div class="infra-result"><span>加权读取结果</span><strong>{{ output.toFixed(3) }}</strong></div>
		</div>
		<p class="infra-note">实现时应先应用 mask，再执行数值稳定的 Softmax；减去最大分数只改变计算方式，不改变归一化结果。</p>
	</div>
</template>
