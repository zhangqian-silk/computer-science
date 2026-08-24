<script setup lang="ts">
import { computed, ref } from 'vue'

const retention = ref(0.8)
const inputGain = ref(0.7)
const outputGain = ref(1)
const step = ref(3)
const sequence = [1, 0.5, -0.25, 0.75, 0]

const states = computed(() => {
	let state = 0
	return sequence.map((input) => {
		state = retention.value * state + inputGain.value * input
		return state
	})
})
const contributions = computed(() => sequence.slice(0, step.value + 1).map((input, index) => ({
	index,
	value: Math.pow(retention.value, step.value - index) * inputGain.value * input
})))
const currentOutput = computed(() => outputGain.value * states.value[step.value])
</script>

<template>
	<div class="infra-lab">
		<p class="infra-lab__title">状态空间递推实验台</p>
		<p class="infra-lab__hint">调节状态保留系数，观察早期输入如何按距离衰减并合成为当前位置状态。</p>
		<div class="infra-controls">
			<div class="infra-control"><label for="ssm-retention">状态保留 A：{{ retention.toFixed(2) }}</label><input id="ssm-retention" v-model.number="retention" type="range" min="0" max="1" step="0.05"></div>
			<div class="infra-control"><label for="ssm-input-gain">输入增益 B：{{ inputGain.toFixed(2) }}</label><input id="ssm-input-gain" v-model.number="inputGain" type="range" min="0" max="1.5" step="0.05"></div>
			<div class="infra-control"><label for="ssm-output-gain">输出增益 C：{{ outputGain.toFixed(2) }}</label><input id="ssm-output-gain" v-model.number="outputGain" type="range" min="0" max="2" step="0.1"></div>
			<div class="infra-control"><label for="ssm-step">观察位置：{{ step }}</label><input id="ssm-step" v-model.number="step" type="range" min="0" :max="sequence.length - 1"></div>
		</div>
		<div class="sequence-track" aria-label="输入序列与状态">
			<div v-for="(input, index) in sequence" :key="index" :class="['sequence-node', { 'is-active': index === step }]">
				<span>x{{ index }}={{ input }}</span><strong>h{{ index }}={{ states[index].toFixed(2) }}</strong>
			</div>
		</div>
		<div class="infra-results">
			<div class="infra-result"><span>当前位置状态</span><strong>{{ states[step].toFixed(3) }}</strong></div>
			<div class="infra-result"><span>当前位置输出</span><strong>{{ currentOutput.toFixed(3) }}</strong></div>
			<div class="infra-result"><span>展开后的贡献和</span><strong>{{ contributions.map((item) => item.value.toFixed(2)).join(' + ') }}</strong></div>
		</div>
		<p class="infra-note">递推逐步得到同一状态；并行 scan 通过组合线性变换重排计算。真实 Selective SSM 的参数还会随输入变化。</p>
	</div>
</template>
