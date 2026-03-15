<script setup lang="ts">
import { computed, ref } from 'vue'
import { renderMath } from '../utils/renderMath'

type Step = {
	id: string
	title: string
	short: string
	formula: string
	description: string
	focus: string
}

const steps: Step[] = [
	{
		id: 'input',
		title: '输入表示',
		short: 'X',
		formula: 'X \\in \\mathbb{R}^{L \\times d_{\\mathrm{model}}}',
		description: '整段输入序列先组织成矩阵 X，每一行对应一个 token 的表示。',
		focus: '原始输入被整理成统一矩阵，后续所有 attention 运算都围绕这份输入展开。'
	},
	{
		id: 'project',
		title: '线性投影',
		short: 'Q \\;/\\; K \\;/\\; V',
		formula: 'Q = XW^Q,\\quad K = XW^K,\\quad V = XW^V',
		description: '同一份输入通过三组可学习矩阵投影到不同子空间，分别承担查询、检索和内容载体的角色。',
		focus: '这是把“我要找什么”和“我真正要读什么”拆开的关键一步。'
	},
	{
		id: 'score',
		title: '相关性打分',
		short: 'QK^\\top',
		formula: 'E = QK^\\top',
		description: 'Q 与 K 的转置相乘，得到所有位置对之间的原始相关性分数矩阵。',
		focus: '每个位置开始和所有位置建立联系，形成全局相关性图。'
	},
	{
		id: 'scale',
		title: '数值缩放',
		short: '/\\sqrt{d_k}',
		formula: 'S = \\frac{QK^\\top}{\\sqrt{d_k}}',
		description: '通过除以 √d_k 控制分数尺度，缓解 softmax 饱和和梯度不稳定。',
		focus: '这一步主要服务于训练稳定性，而不是改变语义排序。'
	},
	{
		id: 'softmax',
		title: '权重归一化',
		short: 'softmax',
		formula: 'A = \\mathrm{softmax}(S)',
		description: '对每一行分数做 softmax，把原始打分转成对所有位置的概率分布。',
		focus: '原始打分在这里变成“我要从哪些位置读多少信息”的权重分布。'
	},
	{
		id: 'output',
		title: '加权聚合',
		short: 'AV',
		formula: 'O = AV',
		description: '利用注意力权重矩阵 A 对所有 value 做加权求和，得到上下文化输出表示。',
		focus: '最终输出不再是原输入本身，而是吸收了上下文后的新表示。'
	}
]

const activeId = ref(steps[0].id)
const activeStep = computed(() => steps.find((step) => step.id === activeId.value) ?? steps[0])
const activeIndex = computed(() => steps.findIndex((step) => step.id === activeId.value))

function renderInlineMath(tex: string) {
	return renderMath(tex, false)
}
</script>

<template>
	<div class="attention-flow">
		<div class="attention-flow__canvas">
			<div
				v-for="(step, index) in steps"
				:key="step.id"
				class="attention-flow__item"
				:class="{
					'attention-flow__item--active': step.id === activeId,
					'attention-flow__item--passed': index <= activeIndex
				}"
			>
				<button class="attention-flow__node" type="button" @click="activeId = step.id">
					<span class="attention-flow__label">{{ step.title }}</span>
					<span class="attention-flow__short" v-html="renderInlineMath(step.short)" />
				</button>

				<div
					v-if="index < steps.length - 1"
					class="attention-flow__arrow"
					:class="{ 'attention-flow__arrow--active': index < activeIndex }"
					aria-hidden="true"
				>
					<span />
				</div>
			</div>
		</div>

		<aside class="attention-flow__detail">
			<div class="attention-flow__detail-head">
				<p class="attention-flow__detail-kicker">当前步骤</p>
				<h4>{{ activeStep.title }}</h4>
			</div>

			<div class="attention-flow__formula" v-html="renderInlineMath(activeStep.formula)" />
			<p class="attention-flow__description">{{ activeStep.description }}</p>
			<p class="attention-flow__focus">{{ activeStep.focus }}</p>

			<div class="attention-flow__mini-map">
				<span
					v-for="(step, index) in steps"
					:key="step.id"
					class="attention-flow__mini-pill"
					:class="{ 'attention-flow__mini-pill--active': index <= activeIndex }"
					v-html="renderInlineMath(step.short)"
				/>
			</div>
		</aside>
	</div>
</template>

<style scoped>
.attention-flow {
	display: grid;
	grid-template-columns: minmax(0, 2fr) minmax(280px, 1fr);
	gap: 1rem;
	margin: 1.5rem 0;
}

.attention-flow__canvas,
.attention-flow__detail {
	border: 1px solid rgba(148, 163, 184, 0.35);
	border-radius: 20px;
	background: linear-gradient(180deg, rgba(248, 250, 252, 0.96), rgba(241, 245, 249, 0.96));
	box-shadow: 0 18px 40px rgba(15, 23, 42, 0.06);
}

.attention-flow__canvas {
	display: flex;
	flex-wrap: wrap;
	align-items: center;
	gap: 0.5rem;
	padding: 1rem;
}

.attention-flow__item {
	display: flex;
	align-items: center;
	gap: 0.5rem;
}

.attention-flow__node {
	min-width: 124px;
	padding: 0.9rem 0.9rem 0.8rem;
	border: 1px solid rgba(148, 163, 184, 0.5);
	border-radius: 16px;
	background: #fff;
	cursor: pointer;
	transition: transform 0.18s ease, border-color 0.18s ease, box-shadow 0.18s ease, background 0.18s ease;
}

.attention-flow__node:hover {
	transform: translateY(-1px);
	border-color: rgba(37, 99, 235, 0.45);
	box-shadow: 0 10px 24px rgba(37, 99, 235, 0.1);
}

.attention-flow__item--active .attention-flow__node {
	border-color: #2563eb;
	background: linear-gradient(180deg, rgba(219, 234, 254, 0.92), rgba(239, 246, 255, 0.96));
	box-shadow: 0 12px 28px rgba(37, 99, 235, 0.16);
}

.attention-flow__item--passed .attention-flow__node {
	border-color: rgba(37, 99, 235, 0.42);
}

.attention-flow__label,
.attention-flow__short {
	display: block;
	text-align: left;
}

.attention-flow__label {
	font-size: 0.84rem;
	color: #475569;
}

.attention-flow__short {
	margin-top: 0.28rem;
	font-size: 1rem;
	font-weight: 700;
	color: #0f172a;
}

.attention-flow__short :deep(.katex) {
	font-size: 1.05em;
}

.attention-flow__arrow {
	display: flex;
	align-items: center;
	justify-content: center;
	width: 32px;
}

.attention-flow__arrow span {
	position: relative;
	display: block;
	width: 100%;
	height: 2px;
	background: linear-gradient(90deg, rgba(100, 116, 139, 0.3), rgba(37, 99, 235, 0.7));
}

.attention-flow__arrow--active span {
	background: linear-gradient(90deg, rgba(37, 99, 235, 0.55), rgba(37, 99, 235, 1));
}

.attention-flow__arrow span::after {
	content: '';
	position: absolute;
	top: -4px;
	right: 0;
	border-top: 5px solid transparent;
	border-bottom: 5px solid transparent;
	border-left: 8px solid rgba(37, 99, 235, 0.75);
}

.attention-flow__detail {
	padding: 1rem 1rem 1.1rem;
}

.attention-flow__detail-kicker {
	margin: 0;
	font-size: 0.76rem;
	letter-spacing: 0.08em;
	text-transform: uppercase;
	color: #64748b;
}

.attention-flow__detail h4 {
	margin: 0.3rem 0 0.8rem;
	font-size: 1.05rem;
	color: #0f172a;
}

.attention-flow__formula {
	margin: 0 0 0.8rem;
	padding: 0.8rem 0.9rem;
	border-radius: 14px;
	background: rgba(15, 23, 42, 0.92);
	color: #e2e8f0;
	overflow-x: auto;
}

.attention-flow__formula :deep(.katex) {
	color: inherit;
	font-size: 1.04em;
}

.attention-flow__formula :deep(.katex-display) {
	margin: 0;
}

.attention-flow__description {
	margin: 0;
	line-height: 1.8;
	color: #334155;
}

.attention-flow__focus {
	margin: 0.8rem 0 0;
	padding: 0.8rem 0.9rem;
	border-radius: 14px;
	background: rgba(37, 99, 235, 0.08);
	color: #1e3a8a;
	line-height: 1.8;
}

.attention-flow__mini-map {
	display: flex;
	flex-wrap: wrap;
	gap: 0.45rem;
	margin-top: 0.9rem;
}

.attention-flow__mini-pill {
	padding: 0.32rem 0.58rem;
	border-radius: 999px;
	background: rgba(148, 163, 184, 0.16);
	color: #64748b;
	font-size: 0.78rem;
	font-weight: 600;
}

.attention-flow__mini-pill--active {
	background: rgba(37, 99, 235, 0.12);
	color: #1d4ed8;
}

.attention-flow__mini-pill :deep(.katex) {
	font-size: 0.96em;
}

@media (max-width: 960px) {
	.attention-flow {
		grid-template-columns: 1fr;
	}
}
</style>
