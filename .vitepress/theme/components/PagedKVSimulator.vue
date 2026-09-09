<script setup lang="ts">
import { computed, ref, useId } from "vue"
import LearningLab from "./LearningLab.vue"
import { useLabReset } from "../use-lab-reset"
const labId = useId()
const fieldId = (name: string) => `${labId}-${name}`

const length = ref(2300)
const blockSize = ref(16)
const sequences = ref(12)
const bytesPerTokenKiB = ref(128)
const blocksPerSequence = computed(() => Math.ceil(length.value / blockSize.value))
const allocatedTokens = computed(() => blocksPerSequence.value * blockSize.value * sequences.value)
const usefulTokens = computed(() => length.value * sequences.value)
const wasteTokens = computed(() => allocatedTokens.value - usefulTokens.value)
const utilization = computed(() => usefulTokens.value / allocatedTokens.value * 100)
const allocatedGiB = computed(() => allocatedTokens.value * bytesPerTokenKiB.value / 1024 / 1024)
const resetLab = useLabReset(length, blockSize, sequences, bytesPerTokenKiB)
</script>

<template>
	<LearningLab topic="PagedKVSimulator" @reset="resetLab">
		<p class="infra-lab__hint">固定长度只是便于观察末 block 内部碎片；真实服务应输入长度分布。</p>
		<div class="infra-controls">
			<div class="infra-control"><label :for="fieldId('paged-length')">每序列长度：{{ length }}</label><input :id="fieldId('paged-length')" v-model.number="length" type="range" min="1" max="8192"></div>
			<div class="infra-control"><label :for="fieldId('paged-block')">Block token</label><select :id="fieldId('paged-block')" v-model.number="blockSize"><option :value="8">8</option><option :value="16">16</option><option :value="32">32</option><option :value="64">64</option></select></div>
			<div class="infra-control"><label :for="fieldId('paged-sequences')">序列数：{{ sequences }}</label><input :id="fieldId('paged-sequences')" v-model.number="sequences" type="range" min="1" max="64"></div>
			<div class="infra-control"><label :for="fieldId('paged-kv-bytes')">KV KiB/token：{{ bytesPerTokenKiB }}</label><input :id="fieldId('paged-kv-bytes')" v-model.number="bytesPerTokenKiB" type="range" min="8" max="256" step="8"></div>
		</div>
		<div class="infra-results">
			<div class="infra-result"><span>每序列 block</span><strong>{{ blocksPerSequence }}</strong></div>
			<div class="infra-result"><span>末 block 浪费</span><strong>{{ wasteTokens }} token</strong></div>
			<div class="infra-result"><span>分配 KV 主体</span><strong>{{ allocatedGiB.toFixed(2) }} GiB</strong></div>
		</div>
		<div class="infra-bar"><i :style="{ width: `${utilization}%` }"></i></div>
		<p class="infra-note">Token 槽利用率 {{ utilization.toFixed(2) }}%；不含 block table、scale、对齐和 allocator 保留。</p>
	</LearningLab>
</template>
