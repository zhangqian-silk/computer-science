<script setup lang="ts">
import { computed, ref, useId } from "vue"
import LearningLab from "./LearningLab.vue"
import { useLabReset } from "../use-lab-reset"
import { beamSteps } from "../learning-algorithms"
const labId = useId()
const fieldId = (name: string) => `${labId}-${name}`

type DecodeMode = 'teacher' | 'greedy' | 'beam'

const mode = ref<DecodeMode>('teacher')
const step = ref(2)
const beamWidth = ref(3)
const target = ['I', 'love', 'AI', '<EOS>']

const decoderInput = computed(() => {
	if (step.value === 0) return '<BOS>'
	return target[step.value - 1]
})
const prefix = computed(() => ["<BOS>", ...target.slice(0, step.value)].join(" "))
const beams = computed(() => beamSteps(mode.value === "greedy" ? 1 : beamWidth.value, step.value + 1)[step.value + 1])
const resetLab = useLabReset(mode, step, beamWidth)
</script>

<template>
	<LearningLab topic="Seq2SeqDecodeExplorer" @reset="resetLab">
		<p class="infra-lab__hint">训练读取给定参考前缀；推理使用独立的玩具概率树实际展开候选，不调用模型。Beam 与 Greedy 使用同一条件分布，无长度惩罚。</p>
		<div class="infra-tabs" role="group" aria-label="解码方式">
			<button type="button" :aria-pressed="mode === 'teacher'" @click="mode = 'teacher'">Teacher Forcing</button>
			<button type="button" :aria-pressed="mode === 'greedy'" @click="mode = 'greedy'">Greedy</button>
			<button type="button" :aria-pressed="mode === 'beam'" @click="mode = 'beam'">Beam</button>
		</div>
		<div class="infra-controls">
			<div class="infra-control"><label :for="fieldId('seq2seq-step')">当前步骤：{{ step + 1 }}</label><input :id="fieldId('seq2seq-step')" v-model.number="step" type="range" min="0" :max="target.length - 1"></div>
			<div v-if="mode === 'beam'" class="infra-control"><label :for="fieldId('beam-width')">Beam width：{{ beamWidth }}</label><input :id="fieldId('beam-width')" v-model.number="beamWidth" type="range" min="2" max="6"></div>
		</div>
		<div v-if="mode === 'teacher'" class="infra-results">
			<div class="infra-result"><span>Decoder 已见前缀</span><strong>{{ prefix }}</strong></div>
			<div class="infra-result"><span>本步输入</span><strong>{{ decoderInput }}</strong></div>
			<div class="infra-result"><span>本步监督目标</span><strong>{{ target[step] }}</strong></div>
		</div>
		<table v-else><caption>按条件概率乘积排序的保留候选</caption><thead><tr><th>排名</th><th>前缀</th><th>累计概率</th></tr></thead><tbody><tr v-for="(beam, i) in beams" :key="beam.tokens.join(' ')"><td>{{ i + 1 }}</td><td>{{ beam.tokens.join(" ") }}</td><td>{{ beam.probability.toFixed(4) }}</td></tr></tbody></table>
		<p class="infra-note">概率树首步 I=0.6、We=0.4，但 We 后 love=0.95，We love 后 AI=0.9。Beam 可保住首步较弱的路径；这不保证在任意树上全局最优。推理没有监督目标可读取。</p>
	</LearningLab>
</template>
