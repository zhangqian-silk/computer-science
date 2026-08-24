<script setup lang="ts">
import { computed, ref } from 'vue'

const prompt = ref(2048)
const output = ref(256)
const requests = ref(8)
const kvKiB = ref(128)
const prefillPairs = computed(() => requests.value * prompt.value * prompt.value)
const decodeReads = computed(() => requests.value * output.value * (prompt.value + output.value / 2))
const kvGiB = computed(() => requests.value * (prompt.value + output.value) * kvKiB.value / 1024 / 1024)
const phase = computed(() => prefillPairs.value > decodeReads.value ? 'Attention 连接工作更偏 Prefill' : '累计历史读取更偏 Decode')
</script>

<template>
	<div class="infra-lab">
		<p class="infra-lab__title">Prefill / Decode 工作量探索</p>
		<p class="infra-lab__hint">比较简化 Attention 连接数与 KV 容量；它不是实际延迟预测器。</p>
		<div class="infra-controls">
			<div class="infra-control"><label for="prefill-prompt">Prompt：{{ prompt }} token</label><input id="prefill-prompt" v-model.number="prompt" type="range" min="128" max="16384" step="128"></div>
			<div class="infra-control"><label for="prefill-output">Output：{{ output }} token</label><input id="prefill-output" v-model.number="output" type="range" min="16" max="2048" step="16"></div>
			<div class="infra-control"><label for="prefill-requests">请求：{{ requests }}</label><input id="prefill-requests" v-model.number="requests" type="range" min="1" max="64"></div>
			<div class="infra-control"><label for="prefill-kv">KV：{{ kvKiB }} KiB/token</label><input id="prefill-kv" v-model.number="kvKiB" type="range" min="8" max="256" step="8"></div>
		</div>
		<div class="infra-results">
			<div class="infra-result"><span>Prefill query-key 对</span><strong>{{ prefillPairs.toExponential(2) }}</strong></div>
			<div class="infra-result"><span>Decode 累计历史读取单位</span><strong>{{ decodeReads.toExponential(2) }}</strong></div>
			<div class="infra-result"><span>结束时 KV 主体</span><strong>{{ kvGiB.toFixed(2) }} GiB</strong></div>
		</div>
		<p class="infra-note">{{ phase }}。真实瓶颈还取决于线性层、batch、kernel、带宽和调度。</p>
	</div>
</template>
