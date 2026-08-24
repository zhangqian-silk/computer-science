<script setup lang="ts">
import { computed, ref, watch } from 'vue'

type Scenario = 'success' | 'tool-error' | 'permission-denied'

const scenario = ref<Scenario>('success')
const step = ref(0)
const scenarios = {
	success: [
		{ state: '接收', detail: '验证用户请求与执行范围。', tone: 'neutral' },
		{ state: '规划', detail: '模型选择读取天气工具，不执行外部写入。', tone: 'neutral' },
		{ state: '调用工具', detail: '控制器校验参数并发起只读调用。', tone: 'active' },
		{ state: '观察', detail: '工具返回结构化天气数据。', tone: 'active' },
		{ state: '完成', detail: '模型基于观测生成回答并记录终止状态。', tone: 'success' }
	],
	'tool-error': [
		{ state: '接收', detail: '请求通过输入验证。', tone: 'neutral' },
		{ state: '规划', detail: '模型选择读取工具。', tone: 'neutral' },
		{ state: '调用工具', detail: '工具在超时边界内没有成功返回。', tone: 'active' },
		{ state: '失败判定', detail: '控制器保留错误类型，不把空结果伪装成成功。', tone: 'danger' },
		{ state: '结束', detail: '向用户说明工具失败和未完成范围。', tone: 'warning' }
	],
	'permission-denied': [
		{ state: '接收', detail: '请求包含发送外部消息的动作。', tone: 'neutral' },
		{ state: '规划', detail: '模型提出调用写入型工具。', tone: 'neutral' },
		{ state: '授权检查', detail: '控制器发现缺少明确发送权限。', tone: 'warning' },
		{ state: '暂停', detail: '不执行工具，向用户请求授权。', tone: 'danger' }
	]
} as const
const currentSteps = computed(() => scenarios[scenario.value])
const current = computed(() => currentSteps.value[step.value])
watch(scenario, () => { step.value = 0 })
const advance = () => { step.value = Math.min(step.value + 1, currentSteps.value.length - 1) }
const reset = () => { step.value = 0 }
</script>

<template>
	<div class="infra-lab">
		<p class="infra-lab__title">Agent Runtime 状态实验台</p>
		<p class="infra-lab__hint">切换正常、工具失败和权限不足三条路径，观察模型建议与控制器承诺之间的边界。</p>
		<div class="infra-tabs" role="group" aria-label="Agent 场景">
			<button type="button" :aria-pressed="scenario === 'success'" @click="scenario = 'success'">正常完成</button>
			<button type="button" :aria-pressed="scenario === 'tool-error'" @click="scenario = 'tool-error'">工具失败</button>
			<button type="button" :aria-pressed="scenario === 'permission-denied'" @click="scenario = 'permission-denied'">权限不足</button>
		</div>
		<ol class="state-track" aria-label="Agent 状态序列">
			<li v-for="(item, index) in currentSteps" :key="item.state" :class="[{ 'is-active': index === step, 'is-done': index < step }, `tone-${item.tone}`]">
				<span>{{ index + 1 }}</span><strong>{{ item.state }}</strong>
			</li>
		</ol>
		<div class="infra-result state-detail" aria-live="polite"><span>当前状态：{{ current.state }}</span><strong>{{ current.detail }}</strong></div>
		<div class="infra-actions">
			<button type="button" class="secondary" :disabled="step === 0" @click="reset">重置</button>
			<button type="button" :disabled="step === currentSteps.length - 1" @click="advance">推进一步</button>
		</div>
		<p class="infra-note">失败路径默认采用明确终止或请求授权；重试、恢复 worker 和持久状态只有在外部承诺与证据支持时才应加入。</p>
	</div>
</template>

<style scoped>
.state-track { display: flex; gap: .45rem; margin: 1rem 0; padding: 0; overflow-x: auto; list-style: none; }
.state-track li { min-width: 95px; padding: .55rem; border: 1px solid var(--vp-c-divider); border-radius: 8px; background: var(--vp-c-bg); opacity: .55; }
.state-track li span, .state-track li strong { display: block; }
.state-track li span { color: var(--vp-c-text-3); font-size: .7rem; }
.state-track li strong { font-size: .82rem; }
.state-track li.is-done { opacity: .8; }
.state-track li.is-active { opacity: 1; border-color: var(--vp-c-brand-1); box-shadow: 0 0 0 2px var(--vp-c-brand-soft); }
.state-track li.is-active.tone-success { border-color: var(--vp-c-green-1); }
.state-track li.is-active.tone-warning { border-color: var(--vp-c-warning-1); }
.state-track li.is-active.tone-danger { border-color: var(--vp-c-danger-1); }
.state-detail { margin-top: .8rem; }
</style>
