<script setup lang="ts">
import { computed, ref, useId } from "vue"

type Flow = "user" | "tool"
type Message = {
	role: "system" | "user" | "assistant" | "tool"
	content: string | null
	tool_calls?: { id: string; type: "function"; function: { name: string; arguments: string } }[]
	tool_call_id?: string
}

const id = useId()
const flow = ref<Flow>("user")
const turn = ref(1)
const toolStep = ref(1)
const includeHistory = ref(true)
const systemPrompt = "你是配置解释助手。依据提供的配置回答，先给出字段值，再用一句话解释；区分当前配置与修改建议，缺少字段时明确说明。"
const conversation: Message[] = [
	{ role: "user", content: "配置如下：\nserver:\n  port: 8080\n  timeout_seconds: 30\n\n服务端口是多少？" },
	{ role: "assistant", content: "端口是 8080，服务通过该端口接收请求。" },
	{ role: "user", content: "超时呢？" },
	{ role: "assistant", content: "超时是 30 秒，对应 timeout_seconds。" },
	{ role: "user", content: "如果把它改成 60 呢？" }
]
const toolQuestion: Message = { role: "user", content: "请读取 config.yaml，告诉我服务端口。" }
const toolCall: Message = {
	role: "assistant",
	content: null,
	tool_calls: [{
		id: "call_config",
		type: "function",
		function: { name: "read_file", arguments: "{\"path\":\"config.yaml\"}" }
	}]
}
const toolResult: Message = {
	role: "tool",
	tool_call_id: "call_config",
	content: "server:\n  port: 8080\n  timeout_seconds: 30\n"
}
const currentIndex = computed(() => (turn.value - 1) * 2)
const messages = computed<(Message & { label: string })[]>(() => {
	const system: Message & { label: string } = { role: "system", content: systemPrompt, label: "持续生效的回答要求" }
	if (flow.value === "tool") {
		return [
			{ ...toolQuestion, label: toolStep.value === 1 ? "当前问题" : "原始问题" },
			...(toolStep.value === 2 ? [
				{ ...toolCall, label: "第一次调用返回的工具请求" },
				{ ...toolResult, label: "本轮新增的工具结果" }
			] : [])
		]
	}
	return [
		system,
		...(includeHistory.value
			? conversation.slice(0, currentIndex.value).map(message => ({ ...message, label: "前文" }))
			: []),
		{ ...conversation[currentIndex.value], label: "当前问题" }
	]
})
const inputLabel = computed(() => flow.value === "user"
	? `第 ${turn.value} 轮用户消息对应的模型输入`
	: `第 ${toolStep.value} 次模型调用的输入`)
const observation = computed(() => {
	if (flow.value === "tool") {
		return toolStep.value === 1
			? "用户只提供了文件路径，当前输入没有文件内容。请求另外声明了 read_file 工具，模型可以据此提出读取调用。"
			: "没有新增 user 消息。assistant 中的调用 ID 与 tool_call_id 都是 call_config，配置内容通过 tool 消息进入第二次模型调用。"
	}
	if (turn.value === 1) return "配置和问题都在当前 user 消息中，不依赖之前的回答。"
	if (!includeHistory.value) return "当前请求缺少前文，无法仅凭「超时呢」或「它」确定配置对象及原始值。system prompt 只规定回答方式，不提供配置事实。"
	if (turn.value === 2) return "「超时呢？」依赖第一轮 user 消息中的 timeout_seconds: 30；上一轮只回答端口，不代表其他配置已不可见。"
	return "「它」承接上一轮讨论的 timeout_seconds，60 是新提出的假设值；前文中的当前配置仍是 30 秒。"
})

function selectFlow(value: Flow) {
	flow.value = value
	turn.value = 1
	toolStep.value = 1
	includeHistory.value = true
}

function reset() {
	selectFlow("user")
}
</script>

<template>
	<section class="infra-lab learning-lab model-conversation" :aria-labelledby="`${id}-title`">
		<header class="learning-lab__header">
			<div>
				<span class="learning-lab__eyebrow">下一次模型输入</span>
				<h3 :id="`${id}-title`">用户追问与工具结果，怎样接入对话？</h3>
			</div>
			<button type="button" class="learning-lab__reset" @click="reset">恢复初始值</button>
		</header>
		<div class="learning-lab__body">
			<div class="model-conversation__controls">
				<div class="model-conversation__turns" role="group" aria-label="选择交互方式">
					<button type="button" :aria-pressed="flow === 'user'" @click="selectFlow('user')">用户追问</button>
					<button type="button" :aria-pressed="flow === 'tool'" @click="selectFlow('tool')">工具结果</button>
				</div>
				<div v-if="flow === 'user'" class="model-conversation__turns" role="group" aria-label="选择对话轮次">
					<button
						v-for="number in 3"
						:key="number"
						type="button"
						:aria-pressed="turn === number"
						@click="turn = number"
					>第 {{ number }} 轮</button>
				</div>
				<div v-else class="model-conversation__turns" role="group" aria-label="选择模型调用次数">
					<button
						v-for="number in 2"
						:key="number"
						type="button"
						:aria-pressed="toolStep === number"
						@click="toolStep = number"
					>第 {{ number }} 次调用</button>
				</div>
				<label v-if="flow === 'user'" :for="`${id}-history`">
					<input :id="`${id}-history`" v-model="includeHistory" type="checkbox" :disabled="turn === 1">
					带上此前的 user / assistant 消息
				</label>
			</div>
			<p v-if="flow === 'tool'" class="infra-lab__hint">
				请求还在 <code>tools</code> 中声明 <code>read_file(path: string)</code>。
				下面仅显示 <code>messages</code>，工具定义不属于消息列表。
			</p>
			<ol class="model-conversation__messages" :aria-label="`${inputLabel}，按顺序排列`">
				<li
					v-for="(message, index) in messages"
					:key="`${flow}-${turn}-${toolStep}-${index}`"
					:class="{ 'is-current': index === messages.length - 1 }"
				>
					<div class="model-conversation__meta">
						<code>{{ message.role }}</code>
						<span>{{ message.label }}</span>
					</div>
					<div v-for="call in message.tool_calls" :key="call.id" class="model-conversation__call">
						<p><code>{{ call.function.name }}</code> 参数：<code>{{ call.function.arguments }}</code></p>
						<p>调用 ID：<code>{{ call.id }}</code></p>
					</div>
					<p v-if="message.tool_call_id">tool_call_id：<code>{{ message.tool_call_id }}</code></p>
					<p v-if="message.content">{{ message.content }}</p>
				</li>
			</ol>
			<p class="model-conversation__observation" aria-live="polite">{{ observation }}</p>
			<details class="model-conversation__json">
				<summary>查看本次调用的 messages 数组</summary>
				<pre>{{ JSON.stringify(messages.map(({ label, ...message }) => message), null, "\t") }}</pre>
			</details>
		</div>
	</section>
</template>

<style scoped>
.model-conversation__controls {
	display: grid;
	gap: 0.75rem;
	margin-bottom: 1rem;
}
.model-conversation__turns {
	display: flex;
	flex-wrap: wrap;
	gap: 0.5rem;
}
.model-conversation__turns button {
	min-height: 44px;
	padding: 0.5rem 0.8rem;
	border: 1px solid var(--vp-c-divider);
	border-radius: 5px;
	background: var(--vp-c-bg);
	cursor: pointer;
}
.model-conversation__turns button:hover {
	border-color: var(--lab-accent);
}
.model-conversation__turns button[aria-pressed="true"] {
	border-color: var(--lab-accent);
	color: var(--lab-accent);
	background: var(--vp-c-brand-soft);
	font-weight: 600;
}
.model-conversation__controls label {
	display: flex;
	align-items: center;
	gap: 0.5rem;
	min-height: 44px;
	font-size: 0.85rem;
	line-height: 1.5;
	cursor: pointer;
}
.model-conversation__controls input {
	flex-shrink: 0;
	width: 18px;
	height: 18px;
	accent-color: var(--lab-accent);
}
.model-conversation__messages {
	display: grid;
	gap: 0.65rem;
	margin: 0;
	padding: 0;
	list-style: none;
}
.model-conversation__messages li {
	min-width: 0;
	margin: 0;
	padding: 0.75rem;
	border: 1px solid var(--vp-c-divider);
	border-radius: 5px;
	background: var(--vp-c-bg-soft);
}
.model-conversation__messages li.is-current {
	border-color: var(--lab-accent);
	border-left-width: 3px;
}
.model-conversation__meta {
	display: flex;
	align-items: center;
	flex-wrap: wrap;
	gap: 0.5rem;
	font-size: 0.75rem;
	color: var(--vp-c-text-2);
}
.model-conversation__messages p {
	margin: 0.5rem 0 0;
	white-space: pre-wrap;
	overflow-wrap: anywhere;
	font-size: 0.9rem;
	line-height: 1.65;
}
.model-conversation__observation {
	margin: 1rem 0;
	font-size: 0.9rem;
	line-height: 1.7;
}
.model-conversation__json summary {
	padding: 0.5rem 0;
	cursor: pointer;
	font-size: 0.85rem;
}
.model-conversation__json summary:focus-visible {
	outline: 2px solid var(--lab-accent);
	outline-offset: 3px;
}
.model-conversation__json pre {
	overflow-x: auto;
	max-width: 100%;
	padding: 0.75rem;
	background: var(--vp-c-bg-soft);
	font-size: 0.75rem;
	line-height: 1.6;
}
</style>
