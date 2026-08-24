<script setup lang="ts">
import { computed, ref } from 'vue'

const queue = ref(40)
const tokenize = ref(5)
const prefill = ref(120)
const perToken = ref(25)
const outputs = ref(100)
const stream = ref(3)
const ttft = computed(() => queue.value + tokenize.value + prefill.value + stream.value)
const e2e = computed(() => ttft.value + Math.max(0, outputs.value - 1) * perToken.value)
const outputPerSecond = computed(() => 1000 / perToken.value)
</script>

<template>
	<div class="infra-lab">
		<p class="infra-lab__title">推理延迟组成</p>
		<p class="infra-lab__hint">以单请求简化时间线理解 TTFT、TPOT 与 E2E，不包含高并发下 iteration 波动。</p>
		<div class="infra-controls">
			<div class="infra-control"><label for="metric-queue">Queue：{{ queue }} ms</label><input id="metric-queue" v-model.number="queue" type="range" min="0" max="1000"></div>
			<div class="infra-control"><label for="metric-tokenize">Tokenize：{{ tokenize }} ms</label><input id="metric-tokenize" v-model.number="tokenize" type="range" min="0" max="100"></div>
			<div class="infra-control"><label for="metric-prefill">Prefill：{{ prefill }} ms</label><input id="metric-prefill" v-model.number="prefill" type="range" min="1" max="2000"></div>
			<div class="infra-control"><label for="metric-tpot">TPOT：{{ perToken }} ms</label><input id="metric-tpot" v-model.number="perToken" type="range" min="1" max="200"></div>
			<div class="infra-control"><label for="metric-output">输出：{{ outputs }} token</label><input id="metric-output" v-model.number="outputs" type="range" min="1" max="1000"></div>
			<div class="infra-control"><label for="metric-stream">首 token 发送：{{ stream }} ms</label><input id="metric-stream" v-model.number="stream" type="range" min="0" max="100"></div>
		</div>
		<div class="infra-results">
			<div class="infra-result"><span>TTFT</span><strong>{{ ttft }} ms</strong></div>
			<div class="infra-result"><span>E2E</span><strong>{{ (e2e / 1000).toFixed(2) }} s</strong></div>
			<div class="infra-result"><span>单请求输出速率</span><strong>{{ outputPerSecond.toFixed(1) }} token/s</strong></div>
		</div>
		<p class="infra-note">降低 TPOT 不会直接消除 queue 或 prefill；优化前应先确认目标 SLO 的主导项。</p>
	</div>
</template>
