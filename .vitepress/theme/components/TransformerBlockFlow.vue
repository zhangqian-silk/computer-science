<script setup lang="ts">
import { computed, ref } from 'vue'

const stage = ref(0)
const stages = [
	{ name: '输入', shape: '[B, S, H]', note: 'token embedding 与位置机制形成隐藏状态。' },
	{ name: 'Norm + Attention', shape: 'Q/K/V → [B, heads, S, D]', note: '跨 token 读取信息，输出恢复为 [B, S, H]。' },
	{ name: '第一次残差', shape: '[B, S, H]', note: '把 Attention 更新叠加到原隐藏状态。' },
	{ name: 'Norm + FFN', shape: '[B, S, Hff] → [B, S, H]', note: '逐 token 扩展通道、门控并投影回来。' },
	{ name: '第二次残差', shape: '[B, S, H]', note: '形成下一层输入，主干形状保持不变。' }
]
const current = computed(() => stages[stage.value])
</script>

<template>
	<div class="infra-lab">
		<p class="infra-lab__title">Transformer Block 数据流</p>
		<p class="infra-lab__hint">拖动阶段，观察跨位置计算和逐位置计算如何保持统一层间接口。</p>
		<div class="infra-control">
			<label for="block-stage">阶段：{{ stage + 1 }} / {{ stages.length }}</label>
			<input id="block-stage" v-model.number="stage" type="range" min="0" :max="stages.length - 1">
		</div>
		<div class="infra-results">
			<div class="infra-result"><span>当前步骤</span><strong>{{ current.name }}</strong></div>
			<div class="infra-result"><span>主要形状</span><strong>{{ current.shape }}</strong></div>
		</div>
		<p class="infra-note">{{ current.note }}</p>
	</div>
</template>
