<script setup lang="ts">
import { computed, ref } from 'vue'

type Operation = 'all-reduce' | 'all-gather' | 'reduce-scatter' | 'all-to-all'

const operation = ref<Operation>('all-reduce')
const operations: Operation[] = ['all-reduce', 'all-gather', 'reduce-scatter', 'all-to-all']
const ranks = [0, 1, 2, 3]
const operationNames: Record<Operation, string> = {
	'all-reduce': 'All-Reduce',
	'all-gather': 'All-Gather',
	'reduce-scatter': 'Reduce-Scatter',
	'all-to-all': 'All-to-All'
}
const descriptions: Record<Operation, string> = {
	'all-reduce': '先逐元素规约，再把完整结果交给每个 rank。',
	'all-gather': '收集各 rank 的分片，让每个 rank 获得完整序列。',
	'reduce-scatter': '先逐元素规约，再把结果分片交给不同 rank。',
	'all-to-all': '每个 rank 把面向不同目标的分片互换给对应 rank。'
}
const rankRows = computed(() => ranks.map(rank => {
	if (operation.value === 'all-reduce') return { rank, input: `[${rank + 1}]`, output: '[10]' }
	if (operation.value === 'all-gather') return { rank, input: `[${rank + 1}]`, output: '[1, 2, 3, 4]' }
	if (operation.value === 'reduce-scatter') {
		const input = ranks.map(column => rank * 4 + column)
		const reduced = ranks.map(column => ranks.reduce((sum, source) => sum + source * 4 + column, 0))
		return { rank, input: `[${input.join(', ')}]`, output: `[${reduced[rank]}]` }
	}
	return {
		rank,
		input: ranks.map(target => `${rank}→${target}`).join(' · '),
		output: ranks.map(source => `${source}→${rank}`).join(' · ')
	}
}))
</script>

<template>
	<div class="infra-lab">
		<p class="infra-lab__title">Collective 数据流实验台</p>
		<p class="infra-lab__hint">在四个 rank 上切换通信原语，直接比较每个参与者在操作前后持有什么数据。</p>
		<div class="infra-tabs" role="group" aria-label="Collective 操作">
			<button v-for="key in operations" :key="key" type="button" :aria-pressed="operation === key" @click="operation = key">{{ operationNames[key] }}</button>
		</div>
		<p class="collective-description">{{ descriptions[operation] }}</p>
		<div class="infra-cards" aria-live="polite">
			<div v-for="row in rankRows" :key="row.rank" class="infra-card collective-card">
				<strong>Rank {{ row.rank }}</strong>
				<span>操作前</span><code>{{ row.input }}</code>
				<span>操作后</span><code>{{ row.output }}</code>
			</div>
		</div>
		<p class="infra-note">图中只表达张量语义，不暗示具体拓扑或算法。Ring、Tree、分层通信及其分块方式属于实现层选择。</p>
	</div>
</template>

<style scoped>
.collective-description { margin: .55rem 0 .9rem; color: var(--vp-c-text-2); font-size: .88rem; }
.collective-card span { margin-top: .55rem; }
.collective-card code { display: block; white-space: normal; font-size: .72rem; line-height: 1.35; }
@media (max-width: 640px) {
	.collective-card code { font-size: .68rem; }
}
</style>
