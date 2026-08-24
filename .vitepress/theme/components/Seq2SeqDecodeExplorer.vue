<script setup lang="ts">
import { computed, ref } from 'vue'

type DecodeMode = 'teacher' | 'greedy' | 'beam'

const mode = ref<DecodeMode>('teacher')
const step = ref(2)
const beamWidth = ref(3)
const target = ['I', 'love', 'AI', '<EOS>']
const greedy = ['I', 'like', 'AI', '<EOS>']
const exampleVocabularySize = 6

const decoderInput = computed(() => {
	if (step.value === 0) return '<BOS>'
	return mode.value === 'teacher' ? target[step.value - 1] : greedy[step.value - 1]
})
const prefix = computed(() => {
	const source = mode.value === 'teacher' ? target : greedy
	return ['<BOS>', ...source.slice(0, step.value)].join(' ')
})
const candidateCount = computed(() => (step.value === 0 ? 1 : beamWidth.value) * exampleVocabularySize)
</script>

<template>
	<div class="infra-lab">
		<p class="infra-lab__title">Seq2Seq 解码实验台</p>
		<p class="infra-lab__hint">训练时可读取真实前缀；推理时只能读取自己已经生成的 token，Beam Search 还要保留多个候选。</p>
		<div class="infra-tabs" role="group" aria-label="解码方式">
			<button type="button" :aria-pressed="mode === 'teacher'" @click="mode = 'teacher'">Teacher Forcing</button>
			<button type="button" :aria-pressed="mode === 'greedy'" @click="mode = 'greedy'">Greedy</button>
			<button type="button" :aria-pressed="mode === 'beam'" @click="mode = 'beam'">Beam</button>
		</div>
		<div class="infra-controls">
			<div class="infra-control"><label for="seq2seq-step">当前步骤：{{ step + 1 }}</label><input id="seq2seq-step" v-model.number="step" type="range" min="0" :max="target.length - 1"></div>
			<div v-if="mode === 'beam'" class="infra-control"><label for="beam-width">Beam width：{{ beamWidth }}</label><input id="beam-width" v-model.number="beamWidth" type="range" min="2" max="6"></div>
		</div>
		<div class="infra-results">
			<div class="infra-result"><span>Decoder 已见前缀</span><strong>{{ prefix }}</strong></div>
			<div class="infra-result"><span>本步输入</span><strong>{{ decoderInput }}</strong></div>
			<div class="infra-result"><span>本步监督目标</span><strong>{{ target[step] }}</strong></div>
			<div v-if="mode === 'beam'" class="infra-result"><span>示例词表下本步展开候选</span><strong>{{ candidateCount.toLocaleString() }}</strong></div>
		</div>
		<p class="infra-note">示例故意让 Greedy 第二步产生「like」，用来展示推理错误如何进入后续前缀；Beam 每步从保留前缀扩展完整词表，再只保留累计分数最高的固定数量候选。</p>
	</div>
</template>
