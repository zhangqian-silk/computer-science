<script setup lang="ts">
import { computed, ref } from 'vue'

const step = ref(0)
const stages = [
	{ rule: '初始字节/字符符号', tokens: ['▁', 'l', 'o', 'w', 'e', 'r'] },
	{ rule: '▁ + l → ▁l', tokens: ['▁l', 'o', 'w', 'e', 'r'] },
	{ rule: '▁l + o → ▁lo', tokens: ['▁lo', 'w', 'e', 'r'] },
	{ rule: '▁lo + w → ▁low', tokens: ['▁low', 'e', 'r'] },
	{ rule: 'e + r → er', tokens: ['▁low', 'er'] },
	{ rule: '▁low + er → ▁lower', tokens: ['▁lower'] }
]
const current = computed(() => stages[step.value])
</script>

<template>
	<div class="infra-lab">
		<p class="infra-lab__title">BPE 合并实验台</p>
		<p class="infra-lab__hint">沿固定合并表处理单词 lower；▁ 表示词边界。训练决定合并表，编码时只执行已有规则。</p>
		<div class="infra-control">
			<label for="token-merge-step">合并步骤：{{ step }} / {{ stages.length - 1 }}</label>
			<input id="token-merge-step" v-model.number="step" type="range" min="0" :max="stages.length - 1">
		</div>
		<div class="token-strip" aria-label="当前子词序列">
			<span v-for="(token, index) in current.tokens" :key="`${index}-${token}`" class="token-chip is-selected">{{ token }}</span>
		</div>
		<div class="infra-results">
			<div class="infra-result"><span>本步规则</span><strong>{{ current.rule }}</strong></div>
			<div class="infra-result"><span>序列长度</span><strong>{{ current.tokens.length }} token</strong></div>
		</div>
		<p class="infra-note">真实 BPE 在语料上按频率学习大量规则；不同词表和正规化配置会让同一文本产生不同 token 数。</p>
	</div>
</template>
