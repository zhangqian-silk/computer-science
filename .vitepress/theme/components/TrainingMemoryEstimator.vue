<script setup lang="ts">
import { computed, ref, useId } from "vue"
import LearningLab from "./LearningLab.vue"
import { useLabReset } from "../use-lab-reset"
const labId = useId()
const fieldId = (name: string) => `${labId}-${name}`

const paramsB = ref(7)
const parameterBytes = ref(2)
const gradientBytes = ref(2)
const optimizerBytes = ref(8)
const activationGB = ref(8)
const shardDegree = ref(1)

const parameterGB = computed(() => paramsB.value * 1e9 * parameterBytes.value / 1024 ** 3)
const gradientGB = computed(() => paramsB.value * 1e9 * gradientBytes.value / 1024 ** 3)
const optimizerGB = computed(() => paramsB.value * 1e9 * optimizerBytes.value / 1024 ** 3)
const rankStateGB = computed(() => (parameterGB.value + gradientGB.value + optimizerGB.value) / shardDegree.value)
const totalGB = computed(() => rankStateGB.value + activationGB.value)
const resetLab = useLabReset(paramsB, parameterBytes, gradientBytes, optimizerBytes, activationGB, shardDegree)
</script>

<template>
	<LearningLab topic="TrainingMemoryEstimator" @reset="resetLab">
		<p class="infra-lab__hint">只估算参数、梯度、优化器与给定激活，不包含 workspace、通信 buffer、碎片和框架开销。</p>
		<div class="infra-controls">
			<div class="infra-control"><label :for="fieldId('memory-params')">参数量：{{ paramsB }}B</label><input :id="fieldId('memory-params')" v-model.number="paramsB" type="range" min="1" max="100"></div>
			<div class="infra-control"><label :for="fieldId('memory-param-bytes')">参数字节</label><select :id="fieldId('memory-param-bytes')" v-model.number="parameterBytes"><option :value="1">1</option><option :value="2">2</option><option :value="4">4</option></select></div>
			<div class="infra-control"><label :for="fieldId('memory-grad-bytes')">梯度字节</label><select :id="fieldId('memory-grad-bytes')" v-model.number="gradientBytes"><option :value="2">2</option><option :value="4">4</option></select></div>
			<div class="infra-control"><label :for="fieldId('memory-optimizer-bytes')">优化器字节/参数</label><select :id="fieldId('memory-optimizer-bytes')" v-model.number="optimizerBytes"><option :value="4">4</option><option :value="8">8</option><option :value="12">12</option></select></div>
			<div class="infra-control"><label :for="fieldId('memory-activation')">激活：{{ activationGB }} GiB</label><input :id="fieldId('memory-activation')" v-model.number="activationGB" type="range" min="1" max="80"></div>
			<div class="infra-control"><label :for="fieldId('memory-shard')">状态分片度</label><select :id="fieldId('memory-shard')" v-model.number="shardDegree"><option :value="1">1</option><option :value="2">2</option><option :value="4">4</option><option :value="8">8</option><option :value="16">16</option></select></div>
		</div>
		<div class="infra-results">
			<div class="infra-result"><span>单 rank 模型状态</span><strong>{{ rankStateGB.toFixed(1) }} GiB</strong></div>
			<div class="infra-result"><span>加给定激活后</span><strong>{{ totalGB.toFixed(1) }} GiB</strong></div>
		</div>
		<p class="infra-note">这里假设三类状态按同一度数平均分片；DDP、FSDP、ZeRO 各阶段的实际归属并不相同。</p>
	</LearningLab>
</template>
