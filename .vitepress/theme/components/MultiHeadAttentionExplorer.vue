<script setup lang="ts">
import { computed, ref } from 'vue'
import { renderMath } from '../utils/renderMath'

type Head = {
	id: string
	title: string
	focus: string
	description: string
	formula: string
	projection: string
	tokens: string[]
}

const heads: Head[] = [
	{
		id: 'head-1',
		title: 'Head 1',
		focus: '局部搭配',
		description: '这一头更像在观察局部窗口内哪些 token 常一起出现，适合捕捉短距离搭配关系。',
		formula: '\\mathrm{head}_1 = \\mathrm{Attention}(Q_1, K_1, V_1)',
		projection: 'Q_1 = QW_1^Q,\\quad K_1 = KW_1^K,\\quad V_1 = VW_1^V',
		tokens: ['局部依赖', '邻近词', '短距离']
	},
	{
		id: 'head-2',
		title: 'Head 2',
		focus: '句法骨架',
		description: '这一头更像在追踪主谓宾等句法关系，帮助模型在结构层面建立谁修饰谁、谁支配谁。',
		formula: '\\mathrm{head}_2 = \\mathrm{Attention}(Q_2, K_2, V_2)',
		projection: 'Q_2 = QW_2^Q,\\quad K_2 = KW_2^K,\\quad V_2 = VW_2^V',
		tokens: ['主谓宾', '修饰关系', '句法']
	},
	{
		id: 'head-3',
		title: 'Head 3',
		focus: '指代与长依赖',
		description: '这一头更偏向跨越较远距离查找相关实体，例如代词回指、上下文一致性等信息。',
		formula: '\\mathrm{head}_3 = \\mathrm{Attention}(Q_3, K_3, V_3)',
		projection: 'Q_3 = QW_3^Q,\\quad K_3 = KW_3^K,\\quad V_3 = VW_3^V',
		tokens: ['指代', '长距离', '实体关联']
	},
	{
		id: 'head-4',
		title: 'Head 4',
		focus: '语义主题',
		description: '这一头更像在按语义主题进行聚合，把语义相关但不一定相邻的 token 放到一起考虑。',
		formula: '\\mathrm{head}_4 = \\mathrm{Attention}(Q_4, K_4, V_4)',
		projection: 'Q_4 = QW_4^Q,\\quad K_4 = KW_4^K,\\quad V_4 = VW_4^V',
		tokens: ['语义', '主题', '全局线索']
	}
]

const activeId = ref(heads[0].id)
const activeHead = computed(() => heads.find((head) => head.id === activeId.value) ?? heads[0])

const pipeline = [
	'输入 Q / K / V',
	'多组投影矩阵',
	'各头独立 Attention',
	'Concat 拼接',
	'W^O 输出映射'
]

function renderInlineMath(tex: string) {
	return renderMath(tex, false)
}

function renderDisplayMath(tex: string) {
	return renderMath(tex, true)
}
</script>

<template>
	<div class="multi-head">
		<div class="multi-head__top">
			<div class="multi-head__source">
				<p class="multi-head__kicker">共享输入</p>
				<h4>同一份 Q / K / V</h4>
				<p>所有头都从同一份输入出发，但会经过各自独立的投影矩阵，进入不同子空间。</p>
			</div>

			<div class="multi-head__pipeline">
				<span
					v-for="step in pipeline"
					:key="step"
					class="multi-head__pipeline-pill"
				>
					{{ step }}
				</span>
			</div>
		</div>

		<div class="multi-head__body">
			<div class="multi-head__heads">
				<button
					v-for="head in heads"
					:key="head.id"
					type="button"
					class="multi-head__head"
					:class="{ 'multi-head__head--active': head.id === activeId }"
					@click="activeId = head.id"
				>
					<span class="multi-head__head-title">{{ head.title }}</span>
					<span class="multi-head__head-focus">{{ head.focus }}</span>
				</button>
			</div>

			<aside class="multi-head__panel">
				<p class="multi-head__panel-kicker">{{ activeHead.title }}</p>
				<h4>{{ activeHead.focus }}</h4>
				<div class="multi-head__formula" v-html="renderDisplayMath(activeHead.formula)" />
				<div class="multi-head__projection" v-html="renderInlineMath(activeHead.projection)" />
				<p>{{ activeHead.description }}</p>

				<div class="multi-head__tags">
					<span
						v-for="item in activeHead.tokens"
						:key="item"
						class="multi-head__tag"
					>
						{{ item }}
					</span>
				</div>

				<div class="multi-head__merge">
					<span class="multi-head__merge-box" v-html="renderInlineMath('\\mathrm{Concat}(\\mathrm{head}_1, \\ldots, \\mathrm{head}_h)')" />
					<span class="multi-head__merge-arrow" aria-hidden="true">→</span>
					<span class="multi-head__merge-box multi-head__merge-box--accent" v-html="renderInlineMath('W^O')" />
				</div>
				<p class="multi-head__merge-note">最后把所有头的输出拼接起来，再映射回统一的模型维度。</p>
			</aside>
		</div>
	</div>
</template>

<style scoped>
.multi-head {
	margin: 1.5rem 0;
	border: 1px solid rgba(148, 163, 184, 0.35);
	border-radius: 24px;
	background: linear-gradient(180deg, rgba(248, 250, 252, 0.98), rgba(241, 245, 249, 0.96));
	box-shadow: 0 20px 44px rgba(15, 23, 42, 0.06);
	overflow: hidden;
}

.multi-head__top {
	padding: 1rem;
	border-bottom: 1px solid rgba(148, 163, 184, 0.2);
	background: rgba(255, 255, 255, 0.78);
}

.multi-head__source h4 {
	margin: 0.28rem 0 0.55rem;
	font-size: 1.06rem;
	color: #0f172a;
}

.multi-head__source p:last-child {
	margin: 0;
	line-height: 1.8;
	color: #334155;
}

.multi-head__kicker,
.multi-head__panel-kicker {
	margin: 0;
	font-size: 0.76rem;
	letter-spacing: 0.08em;
	text-transform: uppercase;
	color: #64748b;
}

.multi-head__pipeline {
	display: flex;
	flex-wrap: wrap;
	gap: 0.55rem;
	margin-top: 0.9rem;
}

.multi-head__pipeline-pill {
	padding: 0.38rem 0.68rem;
	border-radius: 999px;
	background: rgba(37, 99, 235, 0.08);
	color: #1d4ed8;
	font-size: 0.8rem;
	font-weight: 700;
}

.multi-head__body {
	display: grid;
	grid-template-columns: minmax(0, 1.08fr) minmax(340px, 0.92fr);
	gap: 1rem;
	padding: 1rem;
}

.multi-head__heads {
	display: grid;
	grid-template-columns: repeat(2, minmax(0, 1fr));
	gap: 0.85rem;
	align-content: start;
}

.multi-head__head {
	display: flex;
	flex-direction: column;
	align-items: flex-start;
	gap: 0.35rem;
	padding: 1rem;
	border-radius: 18px;
	border: 1px solid rgba(148, 163, 184, 0.4);
	background: linear-gradient(180deg, rgba(255, 255, 255, 0.98), rgba(248, 250, 252, 0.98));
	text-align: left;
	cursor: pointer;
	transition: all 0.18s ease;
}

.multi-head__head:hover,
.multi-head__head--active {
	transform: translateY(-1px);
	border-color: rgba(14, 165, 233, 0.45);
	box-shadow: 0 12px 28px rgba(14, 165, 233, 0.1);
}

.multi-head__head--active {
	background: linear-gradient(180deg, rgba(224, 242, 254, 0.95), rgba(240, 249, 255, 0.98));
}

.multi-head__head-title {
	font-size: 0.88rem;
	font-weight: 700;
	color: #475569;
}

.multi-head__head-focus {
	font-size: 1rem;
	font-weight: 700;
	color: #0f172a;
}

.multi-head__panel {
	padding: 1rem;
	border-radius: 20px;
	background: rgba(255, 255, 255, 0.84);
	border: 1px solid rgba(148, 163, 184, 0.28);
}

.multi-head__panel h4 {
	margin: 0.3rem 0 0.8rem;
	font-size: 1.08rem;
	color: #0f172a;
}

.multi-head__formula {
	margin: 0 0 0.8rem;
	padding: 0.8rem 0.9rem;
	border-radius: 14px;
	background: rgba(15, 23, 42, 0.92);
	color: #e2e8f0;
	overflow-x: auto;
}

.multi-head__formula :deep(.katex) {
	color: inherit;
	font-size: 1.04em;
}

.multi-head__formula :deep(.katex-display) {
	margin: 0;
}

.multi-head__projection {
	margin: 0 0 0.9rem;
	padding: 0.72rem 0.85rem;
	border-radius: 12px;
	background: rgba(37, 99, 235, 0.08);
	color: #1e3a8a;
	overflow-x: auto;
}

.multi-head__projection :deep(.katex) {
	color: inherit;
	font-size: 0.96em;
}

.multi-head__panel p {
	margin: 0;
	line-height: 1.8;
	color: #334155;
}

.multi-head__tags {
	display: flex;
	flex-wrap: wrap;
	gap: 0.5rem;
	margin-top: 0.9rem;
}

.multi-head__tag {
	padding: 0.36rem 0.64rem;
	border-radius: 999px;
	background: rgba(14, 165, 233, 0.08);
	color: #0369a1;
	font-size: 0.8rem;
	font-weight: 700;
}

.multi-head__merge {
	display: grid;
	grid-template-columns: minmax(0, 1fr) auto auto;
	align-items: center;
	gap: 0.55rem;
	margin-top: 1rem;
}

.multi-head__merge-box {
	padding: 0.6rem 0.8rem;
	border-radius: 14px;
	background: rgba(248, 250, 252, 0.98);
	border: 1px solid rgba(148, 163, 184, 0.32);
	color: #334155;
	font-weight: 700;
}

.multi-head__merge-box--accent {
	background: rgba(219, 234, 254, 0.96);
	border-color: rgba(37, 99, 235, 0.28);
	color: #1d4ed8;
}

.multi-head__merge-box :deep(.katex) {
	font-size: 0.98em;
}

.multi-head__merge-arrow {
	font-size: 1.1rem;
	font-weight: 700;
	color: #64748b;
}

.multi-head__merge-note {
	margin-top: 0.8rem;
	color: #475569;
}

@media (max-width: 960px) {
	.multi-head__body {
		grid-template-columns: 1fr;
	}
}

@media (max-width: 640px) {
	.multi-head__heads {
		grid-template-columns: 1fr;
	}

	.multi-head__merge {
		grid-template-columns: 1fr;
	}

	.multi-head__merge-arrow {
		display: none;
	}
}
</style>
