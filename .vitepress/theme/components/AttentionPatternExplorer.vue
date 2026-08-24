<script setup lang="ts">
import { computed, ref } from 'vue'

type Pattern = 'bidirectional' | 'causal' | 'sliding' | 'global'

const pattern = ref<Pattern>('causal')
const length = ref(7)
const query = ref(4)
const windowSize = ref(2)

const allowed = (row: number, column: number) => {
	if (pattern.value === 'bidirectional') return true
	if (pattern.value === 'causal') return column <= row
	if (pattern.value === 'sliding') return column <= row && column >= row - windowSize.value
	return column <= row && (column === 0 || column === row || column >= row - windowSize.value)
}
const visibleCount = computed(() => Array.from({ length: length.value }, (_, column) => allowed(query.value, column)).filter(Boolean).length)
const patternName = computed(() => ({
	bidirectional: '双向全局', causal: '因果全局', sliding: '因果滑动窗口', global: '因果局部 + 全局 Key'
})[pattern.value])
</script>

<template>
	<div class="infra-lab">
		<p class="infra-lab__title">Attention 可见性实验台</p>
		<p class="infra-lab__hint">矩阵行是 Query，列是 Key；高亮格表示该 Query 可以读取对应 Key。</p>
		<div class="infra-tabs" role="group" aria-label="Attention 模式">
			<button type="button" :aria-pressed="pattern === 'bidirectional'" @click="pattern = 'bidirectional'">双向</button>
			<button type="button" :aria-pressed="pattern === 'causal'" @click="pattern = 'causal'">因果</button>
			<button type="button" :aria-pressed="pattern === 'sliding'" @click="pattern = 'sliding'">滑动窗口</button>
			<button type="button" :aria-pressed="pattern === 'global'" @click="pattern = 'global'">全局 Key</button>
		</div>
		<div class="infra-controls">
			<div class="infra-control"><label for="attention-length">序列长度：{{ length }}</label><input id="attention-length" v-model.number="length" type="range" min="4" max="10" @input="query = Math.min(query, length - 1)"></div>
			<div class="infra-control"><label for="attention-query">观察 Query：{{ query }}</label><input id="attention-query" v-model.number="query" type="range" min="0" :max="length - 1"></div>
			<div v-if="pattern === 'sliding' || pattern === 'global'" class="infra-control"><label for="attention-window">窗口：{{ windowSize }}</label><input id="attention-window" v-model.number="windowSize" type="range" min="1" :max="length - 1"></div>
		</div>
		<div class="attention-matrix" :style="{ '--matrix-size': length }" role="img" :aria-label="`${patternName}可见性矩阵，Query ${query} 可读取 ${visibleCount} 个位置`">
			<template v-for="row in length" :key="row">
				<span
					v-for="column in length"
					:key="`${row}-${column}`"
					:class="['attention-cell', { 'is-visible': allowed(row - 1, column - 1), 'is-query': row - 1 === query }]"
					:title="`Q${row - 1} → K${column - 1}: ${allowed(row - 1, column - 1) ? '可见' : '屏蔽'}`"
				></span>
			</template>
		</div>
		<div class="infra-results">
			<div class="infra-result"><span>当前模式</span><strong>{{ patternName }}</strong></div>
			<div class="infra-result"><span>Query {{ query }} 可读取</span><strong>{{ visibleCount }} / {{ length }} 个位置</strong></div>
		</div>
		<p class="infra-note">稀疏模式减少的是连接集合；只有 kernel 和数据布局真正跳过被屏蔽连接时，理论稀疏才可能转化为性能收益。</p>
	</div>
</template>

<style scoped>
.attention-matrix { display: grid; grid-template-columns: repeat(var(--matrix-size), minmax(18px, 28px)); gap: 4px; width: fit-content; max-width: 100%; margin: .9rem auto; overflow-x: auto; }
.attention-cell { aspect-ratio: 1; border: 1px solid var(--vp-c-divider); border-radius: 4px; background: var(--vp-c-bg); }
.attention-cell.is-visible { background: var(--vp-c-brand-soft); border-color: var(--vp-c-brand-1); }
.attention-cell.is-query { box-shadow: inset 0 0 0 2px var(--vp-c-warning-1); }
@media (max-width: 640px) { .attention-matrix { grid-template-columns: repeat(var(--matrix-size), minmax(18px, 20px)); margin-block: .65rem; } }
</style>
