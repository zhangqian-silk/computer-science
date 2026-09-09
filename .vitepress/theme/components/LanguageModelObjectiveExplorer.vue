<script setup lang="ts">
import { computed, ref, useId } from "vue"
import LearningLab from "./LearningLab.vue"
import { useLabReset } from "../use-lab-reset"
const labId = useId()
const fieldId = (name: string) => `${labId}-${name}`

type Objective = 'causal' | 'mlm'

const props = withDefaults(defineProps<{ initialMode?: Objective }>(), {
	initialMode: 'causal'
})

const mode = ref<Objective>(props.initialMode)
const position = ref(3)
const tokens = ['[BOS]', '模型', '学习', '处理', '上下文', '[EOS]']

const visibleTokens = computed(() => mode.value === 'causal'
	? tokens.map((token, index) => ({ token, visible: index < position.value, selected: index === position.value }))
	: tokens.map((token, index) => ({
		token: index === position.value ? '[MASK]' : token,
		visible: true,
		selected: index === position.value
	})))

const target = computed(() => tokens[position.value])
const inputSummary = computed(() => mode.value === 'causal'
	? tokens.slice(0, position.value).join(' ')
	: visibleTokens.value.map((item) => item.token).join(' '))
const lossSummary = computed(() => mode.value === 'causal'
	? `位置 ${position.value - 1} 的状态预测目标位置 ${position.value}：「${target.value}」`
	: `只在被选位置恢复原 token「${target.value}」`)
const resetLab = useLabReset(mode, position)
</script>

<template>
	<LearningLab topic="LanguageModelObjectiveExplorer" @reset="resetLab">
		<p class="infra-lab__hint">切换因果语言模型与 MLM，观察输入破坏、可见性和监督位置如何共同改变。</p>
		<div class="infra-tabs" role="group" aria-label="训练目标">
			<button type="button" :aria-pressed="mode === 'causal'" @click="mode = 'causal'">因果语言模型</button>
			<button type="button" :aria-pressed="mode === 'mlm'" @click="mode = 'mlm'">Masked LM</button>
		</div>
		<div class="infra-control">
			<label :for="fieldId('objective-position')">监督位置：{{ position }}</label>
			<input :id="fieldId('objective-position')" v-model.number="position" type="range" min="1" :max="tokens.length - 2">
		</div>
		<div class="token-strip" aria-label="模型输入 token">
			<span
				v-for="(item, index) in visibleTokens"
				:key="`${index}-${item.token}`"
				:class="['token-chip', { 'is-muted': !item.visible, 'is-selected': item.selected }]"
			>
				{{ item.visible ? item.token : '不可见' }}
			</span>
		</div>
		<div class="infra-results">
			<div class="infra-result"><span>模型实际输入</span><strong>{{ inputSummary }}</strong></div>
			<div class="infra-result"><span>监督信号</span><strong>{{ lossSummary }}</strong></div>
		</div>
		<p class="infra-note">因果输入位置的状态可读自己及左侧，用于预测后一 token；这里仅展示单个目标。MLM 只演示 MASK 分支，原始 BERT 另有随机替换与保持不变分支。</p>
	</LearningLab>
</template>
