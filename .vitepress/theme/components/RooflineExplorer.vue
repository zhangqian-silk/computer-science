<script setup lang="ts">
import { computed, ref, useId } from "vue"
import LearningLab from "./LearningLab.vue"
import { useLabReset } from "../use-lab-reset"
const labId = useId()
const fieldId = (name: string) => `${labId}-${name}`

const peakTflops = ref(100)
const bandwidthGBs = ref(2000)
const flopsG = ref(100)
const bytesG = ref(20)
const intensity = computed(() => flopsG.value / bytesG.value)
const ridge = computed(() => peakTflops.value * 1000 / bandwidthGBs.value)
const bandwidthLimit = computed(() => intensity.value * bandwidthGBs.value / 1000)
const attainable = computed(() => Math.min(peakTflops.value, bandwidthLimit.value))
const bound = computed(() => intensity.value < ridge.value ? '带宽上界更低' : '计算上界更低')
const resetLab = useLabReset(peakTflops, bandwidthGBs, flopsG, bytesG)
</script>

<template>
	<LearningLab topic="RooflineExplorer" @reset="resetLab">
		<p class="infra-lab__hint">FLOPs 与 Bytes 使用同一 workload 口径；这里只给理想上界。</p>
		<div class="infra-controls">
			<div class="infra-control"><label :for="fieldId('roofline-peak')">峰值：{{ peakTflops }} TFLOPS</label><input :id="fieldId('roofline-peak')" v-model.number="peakTflops" type="range" min="1" max="1000"></div>
			<div class="infra-control"><label :for="fieldId('roofline-bandwidth')">带宽：{{ bandwidthGBs }} GB/s</label><input :id="fieldId('roofline-bandwidth')" v-model.number="bandwidthGBs" type="range" min="10" max="5000" step="10"></div>
			<div class="infra-control"><label :for="fieldId('roofline-flops')">工作量：{{ flopsG }} GFLOPs</label><input :id="fieldId('roofline-flops')" v-model.number="flopsG" type="range" min="1" max="1000"></div>
			<div class="infra-control"><label :for="fieldId('roofline-bytes')">搬运：{{ bytesG }} GB</label><input :id="fieldId('roofline-bytes')" v-model.number="bytesG" type="range" min="1" max="200"></div>
		</div>
		<div class="infra-results">
			<div class="infra-result"><span>算术强度</span><strong>{{ intensity.toFixed(2) }} FLOP/byte</strong></div>
			<div class="infra-result"><span>Ridge Point</span><strong>{{ ridge.toFixed(2) }} FLOP/byte</strong></div>
			<div class="infra-result"><span>理想上界</span><strong>{{ attainable.toFixed(2) }} TFLOPS</strong></div>
		</div>
		<p class="infra-note">{{ bound }}；实际还受并行度、cache、launch、shape 和 kernel 效率限制。</p>
	</LearningLab>
</template>
