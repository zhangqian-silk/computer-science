<script setup lang="ts">
import { computed, ref } from 'vue'
import { renderMath } from '../utils/renderMath'

type WorkflowStep = {
	id: string
	title: string
	color: 'slate' | 'blue' | 'amber' | 'green'
	description: string
	formula: string
	badges: string[]
}

type WorkflowMode = {
	id: 'roadmap' | 'training' | 'inference'
	title: string
	summary: string
	steps: WorkflowStep[]
}

const modes: WorkflowMode[] = [
	{
		id: 'roadmap',
		title: '全局路线图',
		summary: '该视图用于概括 N-Gram 的整体逻辑：先以有限上下文近似完整历史，再将条件概率估计转化为计数问题，随后通过平滑与回退缓解数据稀疏性，最后说明其在长程依赖建模中的结构性局限。',
		steps: [
			{
				id: 'roadmap-context',
				title: '有限上下文假设',
				color: 'blue',
				description: 'N-Gram 的理论起点是马尔可夫近似：在预测当前词时，不再显式保留完整历史，而仅保留最近的 $n-1$ 个词作为有效上下文。',
				formula: 'P(w_t\\mid w_1,\\dots,w_{t-1}) \\approx P(w_t\\mid w_{t-n+1}^{t-1})',
				badges: ['马尔可夫近似', '固定窗口']
			},
			{
				id: 'roadmap-mle',
				title: '计数估计概率',
				color: 'slate',
				description: '在上下文窗口固定之后，条件概率便可由语料中的共现频次近似估计，这也是后续概率查表机制的基础。',
				formula: 'P(w_t\\mid h) \\approx \\frac{C(h,w_t)}{C(h)}',
				badges: ['最大似然估计', '计数统计']
			},
			{
				id: 'roadmap-sparse',
				title: '稀疏与零概率',
				color: 'amber',
				description: '随着模型阶数升高，大量未见组合会迅速出现；一旦关键 n-gram 的计数为 0，相应条件概率也将退化为 0。',
				formula: 'C(h,w)=0 \\Rightarrow P(w\\mid h)=0',
				badges: ['数据稀疏', '未登录组合']
			},
			{
				id: 'roadmap-smooth',
				title: '平滑与回退',
				color: 'green',
				description: '为避免零概率，模型需要将部分概率质量从高频已见项重新分配给低频或未见项，并在必要时回退到更低阶分布。',
				formula: 'P_{KN}(w\\mid h)=P_{disc}(w\\mid h)+\\lambda(h)P_{cont}(w)',
				badges: ['平滑', '回退 / 插值']
			},
			{
				id: 'roadmap-limit',
				title: '局限与演化',
				color: 'slate',
				description: '即使平滑设计得较为完善，N-Gram 仍只能建模固定窗口内的局部依赖，这也构成后续神经语言模型继续演化的重要背景。',
				formula: '\\text{context size}=n-1',
				badges: ['局部依赖', '模型演化']
			}
		]
	},
	{
		id: 'training',
		title: '训练与建表',
		summary: '从实现流程看，N-Gram 更接近一种离线统计建模过程：先处理语料，再进行滑窗计数、多阶组织与平滑估计，最终生成可供查询的条件概率表。',
		steps: [
			{
				id: 'training-preprocess',
				title: '语料预处理',
				color: 'blue',
				description: '首先进行清洗与分词，并在句首、句尾补入边界符号，以保证窗口在序列边界处仍具有一致的统计定义。',
				formula: '\\langle s \\rangle\\; w_1\\;\\dots\\; w_m\\; \\langle /s \\rangle',
				badges: ['分词', '边界符号']
			},
			{
				id: 'training-count',
				title: '滑动窗口计数',
				color: 'slate',
				description: '以长度为 $n$ 的窗口扫描语料，累计高阶与低阶 n-gram 的共现次数，为后续概率估计准备原始计数。',
				formula: 'C(w_{t-n+1}^{t})',
				badges: ['滑动窗口', '计数表']
			},
			{
				id: 'training-hierarchy',
				title: '组织多阶统计',
				color: 'slate',
				description: '不仅需要保存 n-gram 计数，也需要同步保留 $(n-1)$-gram、bigram 与 unigram 等低阶统计量，以支撑后续回退或插值。',
				formula: 'n \\rightarrow n-1 \\rightarrow \\dots \\rightarrow 1',
				badges: ['多阶结构', '回退基础']
			},
			{
				id: 'training-smoothing',
				title: '平滑重分配',
				color: 'amber',
				description: '将原始频数转换为更稳健的概率分布：对高阶已见项进行折扣，并为未见项保留回退到低阶模型所需的概率质量。',
				formula: '\\max(C(h,w)-d,0)',
				badges: ['折扣', '概率质量']
			},
			{
				id: 'training-table',
				title: '生成概率表',
				color: 'green',
				description: '最后将历史前缀映射为下一词分布，使推理阶段能够依据上下文直接查询概率表，而无须重新扫描整份语料。',
				formula: 'h \\mapsto P(\\cdot\\mid h)',
				badges: ['概率表', '离线结果']
			}
		]
	},
	{
		id: 'inference',
		title: '推理与回退',
		summary: '推理阶段的核心链路是「截取最近上下文 → 优先查询高阶分布 → 未命中时执行回退 → 输出下一词概率」。这一过程体现了 N-Gram 查询高效的特点，也暴露了其对局部上下文的依赖边界。',
		steps: [
			{
				id: 'inference-context',
				title: '截取最近上下文',
				color: 'blue',
				description: '在生成下一个词时，模型仅保留最近的 $n-1$ 个词作为历史前缀，更早的信息在该时刻已经被窗口截断。',
				formula: 'h = w_{t-n+1}^{t-1}',
				badges: ['当前上下文', '窗口历史']
			},
			{
				id: 'inference-lookup',
				title: '优先查高阶表',
				color: 'slate',
				description: '若当前上下文在高阶表中出现过，则优先使用该条件分布，因为它保留了最具体的局部搭配信息。',
				formula: 'P(w_t\\mid h)',
				badges: ['高阶优先', '局部匹配']
			},
			{
				id: 'inference-backoff',
				title: '未命中则回退',
				color: 'amber',
				description: '当高阶历史未命中时，便缩短历史并回退到更低阶模型继续查找，直至获得可用分布。',
				formula: 'h\\prime = \\mathrm{suffix}(h)',
				badges: ['回退', '低阶模型']
			},
			{
				id: 'inference-mix',
				title: '混合低阶信息',
				color: 'amber',
				description: '在插值或 Kneser-Ney 平滑中，低阶分布并不只在未命中时启用，而是持续参与最终概率的组合。',
				formula: '\\lambda(h)P_{bo}(w\\mid h\\prime)',
				badges: ['插值', '续接概率']
			},
			{
				id: 'inference-decode',
				title: '输出下一词',
				color: 'green',
				description: '得到完整概率分布后，便可通过贪婪选择、束搜索或随机采样生成最终的下一个词。',
				formula: '\\arg\\max_{w} P(w\\mid h)',
				badges: ['解码', '采样 / 束搜索']
			}
		]
	}
]

const activeMode = ref<WorkflowMode['id']>('roadmap')
const activeStepId = ref(modes[0].steps[0].id)

const currentMode = computed(() => modes.find((mode) => mode.id === activeMode.value) ?? modes[0])
const activeStep = computed(() => currentMode.value.steps.find((step) => step.id === activeStepId.value) ?? currentMode.value.steps[0])

function switchMode(nextMode: WorkflowMode['id']) {
	const next = modes.find((mode) => mode.id === nextMode) ?? modes[0]
	activeMode.value = nextMode
	activeStepId.value = next.steps[0]?.id ?? ''
}
</script>

<template>
	<div class="ngram-workflow">
		<div class="ngram-workflow__topbar">
			<button
				v-for="mode in modes"
				:key="mode.id"
				type="button"
				class="ngram-workflow__tab"
				:class="{ 'ngram-workflow__tab--active': mode.id === activeMode }"
				@click="switchMode(mode.id)"
			>
				{{ mode.title }}
			</button>
		</div>

		<div class="ngram-workflow__summary">
			<p class="ngram-workflow__kicker">N-Gram 结构图</p>
			<p>{{ currentMode.summary }}</p>
		</div>

		<div class="ngram-workflow__body">
			<div class="ngram-workflow__diagram">
				<div class="ngram-workflow__rail" />

				<div
					v-for="step in currentMode.steps"
					:key="step.id"
					class="ngram-workflow__step"
					:class="[
						`ngram-workflow__step--${step.color}`,
						{ 'ngram-workflow__step--active': step.id === activeStepId }
					]"
				>
					<button type="button" class="ngram-workflow__node" @click="activeStepId = step.id">
						<span class="ngram-workflow__node-title">{{ step.title }}</span>
					</button>
				</div>
			</div>

			<aside class="ngram-workflow__panel">
				<p class="ngram-workflow__panel-kicker">{{ currentMode.title }}</p>
				<h4>{{ activeStep.title }}</h4>
				<div class="ngram-workflow__badges">
					<span v-for="badge in activeStep.badges" :key="badge" class="ngram-workflow__badge">
						{{ badge }}
					</span>
				</div>
				<div class="ngram-workflow__formula" v-html="renderMath(activeStep.formula, false)" />
				<p>{{ activeStep.description }}</p>
			</aside>
		</div>
	</div>
</template>

<style scoped>
.ngram-workflow {
	margin: 1.5rem 0;
	border: 1px solid rgba(148, 163, 184, 0.35);
	border-radius: 24px;
	background: linear-gradient(180deg, rgba(248, 250, 252, 0.98), rgba(241, 245, 249, 0.96));
	box-shadow: 0 20px 44px rgba(15, 23, 42, 0.06);
	overflow: hidden;
}

.ngram-workflow__topbar {
	display: flex;
	flex-wrap: wrap;
	gap: 0.75rem;
	padding: 1rem;
	border-bottom: 1px solid rgba(148, 163, 184, 0.2);
	background: rgba(255, 255, 255, 0.72);
}

.ngram-workflow__tab {
	padding: 0.7rem 1rem;
	border: 1px solid rgba(148, 163, 184, 0.45);
	border-radius: 999px;
	background: #fff;
	color: #334155;
	cursor: pointer;
	font-weight: 600;
	transition: all 0.18s ease;
}

.ngram-workflow__tab--active {
	border-color: #0f766e;
	background: #0f766e;
	color: #f8fafc;
	box-shadow: 0 10px 24px rgba(15, 118, 110, 0.2);
}

.ngram-workflow__summary {
	padding: 1rem 1rem 0.3rem;
}

.ngram-workflow__kicker {
	margin: 0 0 0.35rem;
	font-size: 0.76rem;
	letter-spacing: 0.08em;
	text-transform: uppercase;
	color: #64748b;
}

.ngram-workflow__summary p:last-child {
	margin: 0;
	line-height: 1.85;
	color: #334155;
}

.ngram-workflow__body {
	display: grid;
	grid-template-columns: minmax(0, 1.4fr) minmax(280px, 1fr);
	gap: 1rem;
	padding: 0.8rem 1rem 1rem;
}

.ngram-workflow__diagram {
	position: relative;
	display: flex;
	flex-direction: column;
	align-items: center;
	gap: 1rem;
	padding: 0.5rem 0.5rem 0.5rem;
}

.ngram-workflow__rail {
	position: absolute;
	top: 1rem;
	bottom: 1rem;
	left: 50%;
	width: 3px;
	transform: translateX(-50%);
	background: linear-gradient(180deg, rgba(15, 118, 110, 0.14), rgba(13, 148, 136, 0.55), rgba(245, 158, 11, 0.38), rgba(15, 118, 110, 0.14));
}

.ngram-workflow__step {
	position: relative;
	z-index: 1;
}

.ngram-workflow__node {
	min-width: 250px;
	padding: 0.95rem 1.15rem;
	border-radius: 18px;
	border: 1px solid rgba(148, 163, 184, 0.45);
	background: #fff;
	cursor: pointer;
	transition: transform 0.18s ease, box-shadow 0.18s ease, border-color 0.18s ease;
}

.ngram-workflow__node:hover,
.ngram-workflow__step--active .ngram-workflow__node {
	transform: translateY(-1px);
	box-shadow: 0 12px 28px rgba(15, 23, 42, 0.1);
}

.ngram-workflow__node-title {
	display: block;
	font-size: 1rem;
	font-weight: 700;
	line-height: 1.5;
	color: #0f172a;
	text-align: left;
}

.ngram-workflow__step--slate .ngram-workflow__node {
	border-color: rgba(100, 116, 139, 0.45);
	background: linear-gradient(180deg, rgba(255, 255, 255, 0.97), rgba(248, 250, 252, 0.97));
}

.ngram-workflow__step--blue .ngram-workflow__node {
	border-color: rgba(14, 165, 233, 0.6);
	background: linear-gradient(180deg, rgba(224, 242, 254, 0.98), rgba(240, 249, 255, 0.98));
}

.ngram-workflow__step--amber .ngram-workflow__node {
	border-color: rgba(245, 158, 11, 0.58);
	background: linear-gradient(180deg, rgba(255, 247, 237, 0.98), rgba(255, 251, 235, 0.98));
}

.ngram-workflow__step--green .ngram-workflow__node {
	border-color: rgba(16, 185, 129, 0.55);
	background: linear-gradient(180deg, rgba(236, 253, 245, 0.98), rgba(240, 253, 244, 0.98));
}

.ngram-workflow__step--active .ngram-workflow__node {
	outline: 3px solid rgba(15, 118, 110, 0.12);
}

.ngram-workflow__panel {
	padding: 1rem 1rem 1.1rem;
	border-radius: 20px;
	background: rgba(255, 255, 255, 0.76);
	border: 1px solid rgba(148, 163, 184, 0.28);
}

.ngram-workflow__panel-kicker {
	margin: 0;
	font-size: 0.76rem;
	letter-spacing: 0.08em;
	text-transform: uppercase;
	color: #64748b;
}

.ngram-workflow__panel h4 {
	margin: 0.28rem 0 0.85rem;
	font-size: 1.08rem;
	color: #0f172a;
}

.ngram-workflow__badges {
	display: flex;
	flex-wrap: wrap;
	gap: 0.45rem;
	margin-bottom: 0.8rem;
}

.ngram-workflow__badge {
	padding: 0.32rem 0.58rem;
	border-radius: 999px;
	background: rgba(15, 118, 110, 0.08);
	color: #0f766e;
	font-size: 0.76rem;
	font-weight: 700;
}

.ngram-workflow__formula {
	margin: 0 0 0.9rem;
	padding: 0.8rem 0.9rem;
	border-radius: 14px;
	background: rgba(15, 23, 42, 0.94);
	color: #e2e8f0;
	overflow-x: auto;
}

.ngram-workflow__formula :deep(.katex) {
	color: inherit;
	font-size: 1.04em;
}

.ngram-workflow__formula :deep(.katex-display) {
	margin: 0;
}

.ngram-workflow__panel p {
	margin: 0;
	line-height: 1.85;
	color: #334155;
}

@media (max-width: 960px) {
	.ngram-workflow__body {
		grid-template-columns: 1fr;
	}

	.ngram-workflow__node {
		min-width: min(100%, 290px);
	}
}
</style>
