<script setup lang="ts">
import { computed, ref } from 'vue'

const counts = [ref(8), ref(3), ref(1), ref(0)]
const labels = ['模型', '系统', '数据', '未见词']
const alpha = ref(0.5)
const total = computed(() => counts.reduce((sum, count) => sum + count.value, 0))
const denominator = computed(() => total.value + alpha.value * counts.length)
const rows = computed(() => counts.map((count, index) => ({
	label: labels[index],
	count: count.value,
	mle: total.value === 0 ? 0 : count.value / total.value,
	smoothed: (count.value + alpha.value) / denominator.value
})))
</script>

<template>
	<div class="infra-lab">
		<p class="infra-lab__title">N-gram 概率重分配</p>
		<p class="infra-lab__hint">给定同一历史，调整续接计数与加法平滑系数，观察概率质量如何从已见事件转给未见事件。</p>
		<div class="infra-controls">
			<div v-for="(count, index) in counts" :key="labels[index]" class="infra-control">
				<label :for="`ngram-count-${index}`">{{ labels[index] }}：{{ count.value }}</label>
				<input :id="`ngram-count-${index}`" v-model.number="count.value" type="range" min="0" max="20">
			</div>
			<div class="infra-control"><label for="ngram-alpha">平滑 α：{{ alpha.toFixed(2) }}</label><input id="ngram-alpha" v-model.number="alpha" type="range" min="0.05" max="2" step="0.05"></div>
		</div>
		<div class="metric-list">
			<div v-for="row in rows" :key="row.label" class="metric-row">
				<strong>{{ row.label }}</strong>
				<span>MLE {{ (row.mle * 100).toFixed(1) }}%</span>
				<div class="infra-bar"><i :style="{ width: `${row.smoothed * 100}%` }"></i></div>
				<span>平滑后 {{ (row.smoothed * 100).toFixed(1) }}%</span>
			</div>
		</div>
		<p class="infra-note">四个平滑概率之和为 {{ (rows.reduce((sum, row) => sum + row.smoothed, 0) * 100).toFixed(1) }}%。加法平滑直观，但大词表会把过多质量分给大量未见事件。</p>
	</div>
</template>
