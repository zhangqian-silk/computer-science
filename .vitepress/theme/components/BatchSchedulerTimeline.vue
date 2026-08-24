<script setup lang="ts">
import { computed, ref } from 'vue'

const mode = ref<'static' | 'continuous'>('continuous')
const rows = computed(() => mode.value === 'static' ? [
	['A', 'P', 'D', 'D', 'D', '·'], ['B', 'P', 'D', '·', '·', '·'], ['C', 'P', 'D', 'D', 'D', 'D'], ['新请求 D', '等', '等', '等', '等', 'P']
] : [
	['A', 'P', 'D', 'D', 'D', '·'], ['B', 'P', 'D', '·', '·', '·'], ['C', 'P', 'D', 'D', 'D', 'D'], ['新请求 D', '等', '等', 'P', 'D', 'D']
])
</script>

<template>
	<div class="infra-lab">
		<p class="infra-lab__title">Batch 调度时间线</p>
		<p class="infra-lab__hint">P 表示 Prefill，D 表示 Decode，· 表示该序列已结束或槽位空闲。</p>
		<div class="infra-control"><label for="batch-mode">策略</label><select id="batch-mode" v-model="mode"><option value="static">Static Batch</option><option value="continuous">Continuous Batch</option></select></div>
		<div class="timeline">
			<div v-for="row in rows" :key="row[0]" class="timeline-row">
				<strong>{{ row[0] }}</strong><span v-for="(cell, index) in row.slice(1)" :key="index" :class="`cell cell-${cell}`">{{ cell }}</span>
			</div>
		</div>
		<p class="infra-note">Continuous 模式能在 B 完成后引入新请求；真实调度还受 token/KV 预算和 chunked prefill 约束。</p>
	</div>
</template>

<style scoped>
.timeline { margin-top: .9rem; overflow-x: auto; }
.timeline-row { display: grid; grid-template-columns: 100px repeat(5, minmax(44px, 1fr)); gap: .3rem; margin-bottom: .3rem; align-items: center; }
.timeline-row strong { font-size: .78rem; }
.cell { padding: .35rem; border-radius: 5px; text-align: center; background: var(--vp-c-default-soft); }
.cell-P { background: var(--vp-c-brand-soft); color: var(--vp-c-brand-1); }
.cell-D { background: var(--vp-c-green-soft); color: var(--vp-c-green-1); }
.cell-等 { color: var(--vp-c-text-3); }
</style>
