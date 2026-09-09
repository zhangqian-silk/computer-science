<script setup lang="ts">
import { computed, ref, useId } from "vue"
import LearningLab from "./LearningLab.vue"
import { useLabReset } from "../use-lab-reset"
const labId = useId()
const fieldId = (name: string) => `${labId}-${name}`

const prompt = ref(2048)
const output = ref(256)
const requests = ref(8)
const kvKiB = ref(128)
const prefillPairs = computed(() => requests.value * prompt.value * (prompt.value + 1) / 2)
const decodeReads = computed(() => requests.value * ((output.value - 1) * prompt.value + (output.value - 1) * output.value / 2))
const kvGiB = computed(() => requests.value * (prompt.value + output.value - 1) * kvKiB.value / 1024 / 1024)
const phase = computed(() => prefillPairs.value > decodeReads.value ? 'Attention 连接工作更偏 Prefill' : '累计历史读取更偏 Decode')
const resetLab = useLabReset(prompt, output, requests, kvKiB)
</script>

<template>
	<LearningLab topic="PrefillDecodeExplorer" @reset="resetLab">
		<p class="infra-lab__hint">比较简化 Attention 连接数与 KV 容量；它不是实际延迟预测器。</p>
		<div class="infra-controls">
			<div class="infra-control"><label :for="fieldId('prefill-prompt')">Prompt：{{ prompt }} token</label><input :id="fieldId('prefill-prompt')" v-model.number="prompt" type="range" min="128" max="16384" step="128"></div>
			<div class="infra-control"><label :for="fieldId('prefill-output')">Output：{{ output }} token</label><input :id="fieldId('prefill-output')" v-model.number="output" type="range" min="1" max="2048" step="1"></div>
			<div class="infra-control"><label :for="fieldId('prefill-requests')">请求：{{ requests }}</label><input :id="fieldId('prefill-requests')" v-model.number="requests" type="range" min="1" max="64"></div>
			<div class="infra-control"><label :for="fieldId('prefill-kv')">KV：{{ kvKiB }} KiB/token</label><input :id="fieldId('prefill-kv')" v-model.number="kvKiB" type="range" min="8" max="256" step="8"></div>
		</div>
		<div class="infra-results">
			<div class="infra-result"><span>Prefill 因果可见对（单头口径）</span><strong>{{ prefillPairs.toExponential(2) }}</strong></div>
			<div class="infra-result"><span>Decode 累计历史读取单位</span><strong>{{ decodeReads.toExponential(2) }}</strong></div>
			<div class="infra-result"><span>结束时 KV 主体</span><strong>{{ kvGiB.toFixed(2) }} GiB</strong></div>
		</div>
		<p class="infra-note">{{ phase }}。首 token 由 Prefill 产生，最后输出尚未前向，不计其 KV。计数是可见连接，不是 kernel 实际执行工作；真实瓶颈还取决于线性层、带宽和调度。</p>
	</LearningLab>
</template>
