<script setup lang="ts">
import { computed, ref, watch, useId } from "vue"
import LearningLab from "./LearningLab.vue"
import { useLabReset } from "../use-lab-reset"
const labId = useId()
const fieldId = (name: string) => `${labId}-${name}`

const queryHeads = ref(16)
const kvHeads = ref(4)
const layers = ref(32)
const sequenceLength = ref(4096)
const headDimension = ref(128)
const elementBytes = ref(2)

const validKVOptions = computed(() => [1, 2, 4, 8, 16, 32].filter((value) => value <= queryHeads.value && queryHeads.value % value === 0))
watch(queryHeads, () => {
	if (!validKVOptions.value.includes(kvHeads.value)) kvHeads.value = validKVOptions.value.at(-1) ?? 1
})
const groupSize = computed(() => queryHeads.value / kvHeads.value)
const cacheGiB = computed(() => 2 * layers.value * sequenceLength.value * kvHeads.value * headDimension.value * elementBytes.value / 1024 ** 3)
const mhaGiB = computed(() => 2 * layers.value * sequenceLength.value * queryHeads.value * headDimension.value * elementBytes.value / 1024 ** 3)
const mappings = computed(() => Array.from({ length: queryHeads.value }, (_, queryHead) => ({
	queryHead,
	kvHead: Math.floor(queryHead / groupSize.value)
})))
const modeName = computed(() => kvHeads.value === queryHeads.value ? 'MHA' : kvHeads.value === 1 ? 'MQA' : 'GQA')
const resetLab = useLabReset(queryHeads, kvHeads, layers, sequenceLength, headDimension, elementBytes)
</script>

<template>
	<LearningLab topic="HeadSharingExplorer" @reset="resetLab">
		<p class="infra-lab__hint">调整 Query/KV 头数，观察头映射与 KV cache 主体容量；MLA 需要另一套潜表示定义，不纳入此公式。</p>
		<div class="infra-controls">
			<div class="infra-control"><label :for="fieldId('query-heads')">Query 头数</label><select :id="fieldId('query-heads')" v-model.number="queryHeads"><option :value="8">8</option><option :value="16">16</option><option :value="32">32</option></select></div>
			<div class="infra-control"><label :for="fieldId('kv-heads')">KV 头数</label><select :id="fieldId('kv-heads')" v-model.number="kvHeads"><option v-for="option in validKVOptions" :key="option" :value="option">{{ option }}</option></select></div>
			<div class="infra-control"><label :for="fieldId('head-layers')">层数：{{ layers }}</label><input :id="fieldId('head-layers')" v-model.number="layers" type="range" min="8" max="80" step="8"></div>
			<div class="infra-control"><label :for="fieldId('head-sequence')">序列长度：{{ sequenceLength }}</label><input :id="fieldId('head-sequence')" v-model.number="sequenceLength" type="range" min="512" max="32768" step="512"></div>
		</div>
		<div class="head-map" aria-label="Query 头到 KV 头的映射">
			<span v-for="mapping in mappings" :key="mapping.queryHead">Q{{ mapping.queryHead }}→KV{{ mapping.kvHead }}</span>
		</div>
		<div class="infra-results">
			<div class="infra-result"><span>结构</span><strong>{{ modeName }}，每 {{ groupSize }} 个 Q 头共享 KV</strong></div>
			<div class="infra-result"><span>单序列 KV 主体</span><strong>{{ cacheGiB.toFixed(2) }} GiB</strong></div>
			<div class="infra-result"><span>相对 MHA 主项</span><strong>{{ (cacheGiB / mhaGiB * 100).toFixed(1) }}%</strong></div>
		</div>
		<p class="infra-note">容量按 FP16/BF16、head dim 128 估算，不含 block、scale、对齐与并行副本；减少 KV 读量也不保证端到端同比提速。</p>
	</LearningLab>
</template>

<style scoped>
.head-map { display: flex; flex-wrap: wrap; gap: .35rem; margin: .9rem 0; }
.head-map span { padding: .25rem .45rem; border: 1px solid color-mix(in srgb, var(--vp-c-brand-1) 35%, var(--vp-c-divider)); border-radius: 6px; background: var(--vp-c-bg); font-size: .72rem; }
@media (max-width: 640px) {
	.head-map { flex-wrap: nowrap; overflow-x: auto; padding-bottom: .25rem; }
	.head-map span { flex: 0 0 auto; }
}
</style>
