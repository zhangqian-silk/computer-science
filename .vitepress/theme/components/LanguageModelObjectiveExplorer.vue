<script setup lang="ts">
import { computed, ref } from 'vue'

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
	? `位置 ${position.value} 预测下一个 token「${target.value}」`
	: `只在被选位置恢复原 token「${target.value}」`)
</script>

<template>
	<div class="infra-lab">
		<p class="infra-lab__title">语言模型目标实验台</p>
		<p class="infra-lab__hint">切换因果语言模型与 MLM，观察输入破坏、可见性和监督位置如何共同改变。</p>
		<div class="infra-tabs" role="group" aria-label="训练目标">
			<button type="button" :aria-pressed="mode === 'causal'" @click="mode = 'causal'">因果语言模型</button>
			<button type="button" :aria-pressed="mode === 'mlm'" @click="mode = 'mlm'">Masked LM</button>
		</div>
		<div class="infra-control">
			<label for="objective-position">监督位置：{{ position }}</label>
			<input id="objective-position" v-model.number="position" type="range" min="1" :max="tokens.length - 2">
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
		<p class="infra-note">因果训练可以并行处理整段序列，但每个位置只能读取左侧；MLM 使用双向上下文，却先破坏被监督位置。</p>
	</div>
</template>
