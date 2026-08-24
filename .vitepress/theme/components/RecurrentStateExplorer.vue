<script setup lang="ts">
import { computed, ref } from 'vue'

type RecurrentMode = 'rnn' | 'lstm'

const props = withDefaults(defineProps<{ initialMode?: RecurrentMode }>(), {
	initialMode: 'rnn'
})

const mode = ref<RecurrentMode>(props.initialMode)
const input = ref(0.6)
const previousHidden = ref(0.3)
const previousCell = ref(0.5)
const forgetGate = ref(0.9)
const inputGate = ref(0.4)
const outputGate = ref(0.8)
const candidate = ref(-0.2)
const distance = ref(8)

const rnnHidden = computed(() => Math.tanh(0.8 * input.value + 0.6 * previousHidden.value))
const cell = computed(() => forgetGate.value * previousCell.value + inputGate.value * candidate.value)
const lstmHidden = computed(() => outputGate.value * Math.tanh(cell.value))
const retention = computed(() => Math.pow(forgetGate.value, distance.value))
</script>

<template>
	<div class="infra-lab">
		<p class="infra-lab__title">递归状态实验台</p>
		<p class="infra-lab__hint">用单个标量维度比较普通 RNN 的整体重写与 LSTM 的门控加法路径。</p>
		<div class="infra-tabs" role="group" aria-label="递归模型">
			<button type="button" :aria-pressed="mode === 'rnn'" @click="mode = 'rnn'">RNN</button>
			<button type="button" :aria-pressed="mode === 'lstm'" @click="mode = 'lstm'">LSTM</button>
		</div>
		<div class="infra-controls">
			<div class="infra-control"><label for="recurrent-input">当前输入：{{ input.toFixed(2) }}</label><input id="recurrent-input" v-model.number="input" type="range" min="-1" max="1" step="0.05"></div>
			<div class="infra-control"><label for="recurrent-hidden">旧隐藏状态：{{ previousHidden.toFixed(2) }}</label><input id="recurrent-hidden" v-model.number="previousHidden" type="range" min="-1" max="1" step="0.05"></div>
		</div>
		<template v-if="mode === 'lstm'">
			<div class="infra-controls">
				<div class="infra-control"><label for="recurrent-cell">旧细胞状态：{{ previousCell.toFixed(2) }}</label><input id="recurrent-cell" v-model.number="previousCell" type="range" min="-1" max="1" step="0.05"></div>
				<div class="infra-control"><label for="forget-gate">遗忘门：{{ forgetGate.toFixed(2) }}</label><input id="forget-gate" v-model.number="forgetGate" type="range" min="0" max="1" step="0.05"></div>
				<div class="infra-control"><label for="input-gate">输入门：{{ inputGate.toFixed(2) }}</label><input id="input-gate" v-model.number="inputGate" type="range" min="0" max="1" step="0.05"></div>
				<div class="infra-control"><label for="candidate-state">候选记忆：{{ candidate.toFixed(2) }}</label><input id="candidate-state" v-model.number="candidate" type="range" min="-1" max="1" step="0.05"></div>
				<div class="infra-control"><label for="output-gate">输出门：{{ outputGate.toFixed(2) }}</label><input id="output-gate" v-model.number="outputGate" type="range" min="0" max="1" step="0.05"></div>
				<div class="infra-control"><label for="retention-distance">跨越步数：{{ distance }}</label><input id="retention-distance" v-model.number="distance" type="range" min="1" max="40"></div>
			</div>
		</template>
		<div class="infra-results">
			<div v-if="mode === 'rnn'" class="infra-result"><span>新隐藏状态</span><strong>tanh(0.8x + 0.6h) = {{ rnnHidden.toFixed(3) }}</strong></div>
			<template v-else>
				<div class="infra-result"><span>新细胞状态</span><strong>{{ cell.toFixed(3) }}</strong></div>
				<div class="infra-result"><span>新隐藏状态</span><strong>{{ lstmHidden.toFixed(3) }}</strong></div>
				<div class="infra-result"><span>{{ distance }} 步直接保留比例</span><strong>{{ (retention * 100).toFixed(1) }}%</strong></div>
			</template>
		</div>
		<p class="infra-note">这是机制级标量示例，不表示真实模型只使用一个状态维度；门值也不是人工指定，而是由训练学习。</p>
	</div>
</template>
