<template>
	<div class="word2vec-training-explorer">
		<div class="hero-card">
			<div class="hero-text">
				<p class="eyebrow">Word2Vec Training Explorer</p>
				<h2>CBOW、Skip-gram 与训练加速机制的对比结构图</h2>
				<p class="summary">
					该图把 word2vec 一节中最容易读散的部分收束为一条可切换的信息流：CBOW 如何由上下文预测中心词，Skip-gram 如何由中心词预测上下文，以及 Hierarchical Softmax 与 Negative Sampling 如何只改变训练效率，而不改变“表示来自共现预测”这一核心思想。
				</p>
			</div>
			<div class="legend-grid">
				<div class="legend-item cbow">CBOW 路线</div>
				<div class="legend-item skipgram">Skip-gram 路线</div>
				<div class="legend-item objective">预测目标</div>
				<div class="legend-item speedup">训练加速</div>
			</div>
		</div>

		<div class="controls-card">
			<div class="toggle-group" role="tablist" aria-label="Word2Vec 视图切换">
				<button
					v-for="mode in modes"
					:key="mode.key"
					type="button"
					class="toggle-btn"
					:class="{ active: activeMode === mode.key }"
					@click="activeMode = mode.key"
				>
					{{ mode.label }}
				</button>
			</div>
			<p class="mode-desc">{{ activeDescription }}</p>
		</div>

		<div class="layout-grid">
			<section class="diagram-card">
				<div class="compare-grid">
					<div class="path-card cbow-card" :class="{ focus: activeMode === 'overview' || activeMode === 'cbow' }">
						<div class="path-head">
							<h3>CBOW</h3>
							<span>上下文 → 中心词</span>
						</div>
						<div class="flow-stack">
							<div class="flow-node">$w_{t-m},\dots,w_{t-1},w_{t+1},\dots,w_{t+m}$</div>
							<div class="flow-arrow">↓</div>
							<div class="flow-node">查表得到多个 embedding</div>
							<div class="flow-arrow">↓</div>
							<div class="flow-node formula">$h_t=\frac{1}{2m}\sum x_{t+k}$</div>
							<div class="flow-arrow">↓</div>
							<div class="flow-node">预测中心词 $w_t$</div>
						</div>
						<p>CBOW 先把上下文压成一个平均或求和后的表示，再进行一次中心词预测，因此速度快、结构紧凑。</p>
					</div>

					<div class="path-card skipgram-card" :class="{ focus: activeMode === 'overview' || activeMode === 'skipgram' }">
						<div class="path-head">
							<h3>Skip-gram</h3>
							<span>中心词 → 多个上下文词</span>
						</div>
						<div class="flow-stack">
							<div class="flow-node">输入中心词 $w_t$</div>
							<div class="flow-arrow">↓</div>
							<div class="flow-node">查表得到中心词向量</div>
							<div class="flow-arrow">↓</div>
							<div class="flow-node formula">分别预测 $w_{t+k}$</div>
							<div class="flow-arrow">↓</div>
							<div class="flow-node">累积多个上下文损失</div>
						</div>
						<p>Skip-gram 让一个中心词向多个上下文分发预测信号，因此更擅长低频词和更细粒度的语义关联。</p>
					</div>
				</div>

				<div class="objective-card" :class="{ focus: activeMode === 'objective' }">
					<div class="path-head">
						<h3>共同核心：都在用局部共现预测学习 embedding</h3>
						<span>差异在信息流方向，不在表示本质</span>
					</div>
					<div class="formula-grid">
						<div class="formula-box">CBOW：聚合上下文后预测 $w_t$</div>
						<div class="formula-box">Skip-gram：用 $w_t$ 预测多个 $w_{t+k}$</div>
					</div>
					<p>两者本质上都在利用“词可以由局部上下文刻画”这一分布式假设，差别主要体现在训练样本组织方式与梯度路径上。</p>
				</div>

				<div class="speedup-grid">
					<div class="aux-card" :class="{ focus: activeMode === 'speedup' }">
						<h4>Hierarchical Softmax</h4>
						<p>把大词表预测改写成树路径上的多次二分类，主要解决完整 softmax 过慢的问题。</p>
					</div>
					<div class="aux-card" :class="{ focus: activeMode === 'speedup' }">
						<h4>Negative Sampling</h4>
						<p>只保留真实共现对与少量随机负样本，把大规模多分类近似成少量二分类任务。</p>
					</div>
				</div>
			</section>

			<section class="insight-card">
				<h3>{{ activePanel.title }}</h3>
				<p class="insight-lead">{{ activePanel.lead }}</p>
				<ul>
					<li v-for="item in activePanel.points" :key="item">{{ item }}</li>
				</ul>
				<div class="mini-table">
					<div class="row head">
						<span>维度</span>
						<span>说明</span>
					</div>
					<div v-for="row in activePanel.rows" :key="row.label" class="row">
						<span>{{ row.label }}</span>
						<span>{{ row.value }}</span>
					</div>
				</div>
			</section>
		</div>
	</div>
</template>

<script setup lang="ts">
import { computed, ref } from 'vue'

type ModeKey = 'overview' | 'cbow' | 'skipgram' | 'objective' | 'speedup'

const activeMode = ref<ModeKey>('overview')

const modes: { key: ModeKey; label: string; desc: string }[] = [
	{ key: 'overview', label: '总览', desc: '先看 CBOW 与 Skip-gram 在信息流方向上的总体差异。' },
	{ key: 'cbow', label: 'CBOW', desc: '聚焦 CBOW 如何把多个上下文词先聚合，再预测中心词。' },
	{ key: 'skipgram', label: 'Skip-gram', desc: '聚焦 Skip-gram 如何由中心词向多个上下文词发散预测。' },
	{ key: 'objective', label: '共同目标', desc: '聚焦 CBOW 与 Skip-gram 在训练本质上的共性。' },
	{ key: 'speedup', label: '训练加速', desc: '聚焦 Hierarchical Softmax 与 Negative Sampling 如何缓解大词表训练开销。' }
]

const panels = {
	overview: {
		title: '总览：Word2Vec 的核心不是“某个固定网络”，而是“局部共现预测”这件事',
		lead: 'CBOW 与 Skip-gram 都围绕同一假设展开，只是一个从上下文收缩到中心词，一个从中心词展开到多个上下文词。',
		points: [
			'CBOW 的路径更像“多输入、单输出”，适合快速训练。',
			'Skip-gram 的路径更像“单输入、多输出”，更容易给低频词积累训练信号。',
			'加速技巧并没有改变表示学习的本质，只是在降低大词表 softmax 的计算成本。'
		],
		rows: [
			{ label: 'CBOW', value: '上下文聚合后预测中心词' },
			{ label: 'Skip-gram', value: '中心词分别预测多个上下文词' },
			{ label: '共同点', value: '都在用局部共现关系塑造 embedding 空间' }
		]
	},
	cbow: {
		title: 'CBOW：把局部上下文压成一个表示，再做一次中心词预测',
		lead: 'CBOW 的效率优势来自“先聚合，再预测”：多个上下文 embedding 被平均或求和后，共同服务于一次输出。',
		points: [
			'上下文窗口里的多个词先被查表映射为 embedding。',
			'这些 embedding 被聚合为一个上下文表示。',
			'模型只做一次中心词预测，因此训练速度通常更快。'
		],
		rows: [
			{ label: '输入', value: '窗口内多个上下文词' },
			{ label: '中间表示', value: '对多个上下文 embedding 求和或平均' },
			{ label: '特点', value: '速度快，但对细粒度差异的表达相对平滑' }
		]
	},
	skipgram: {
		title: 'Skip-gram：用一个中心词向多个上下文传播预测信号',
		lead: 'Skip-gram 把训练信号拆得更细，每一个中心词—上下文词对都能独立产生梯度，因此常更利于低频词学习。',
		points: [
			'中心词先被映射成 embedding。',
			'模型分别预测窗口中的多个上下文词。',
			'每一对共现关系都能独立贡献训练信号。'
		],
		rows: [
			{ label: '输入', value: '单个中心词' },
			{ label: '输出', value: '窗口内多个上下文词的条件概率' },
			{ label: '特点', value: '表达更细，但计算通常比 CBOW 更重' }
		]
	},
	objective: {
		title: '共同目标：两条路线都在逼迫向量反映“谁常与谁一起出现”',
		lead: 'Word2Vec 的几何结构并不是直接规定的，而是在上下文预测任务里被间接学出来的。',
		points: [
			'相似上下文会促使相似词获得相近向量。',
			'能够预测相似上下文的词，也会在 embedding 空间中逐渐靠近。',
			'因此语义近邻是训练目标的副产物，而不是手工写入的规则。'
		],
		rows: [
			{ label: '训练信号', value: '局部窗口中的共现关系' },
			{ label: '空间结构', value: '由预测成功率反向塑造出来的几何关系' },
			{ label: '结果', value: '相似词往往在向量空间中彼此更接近' }
		]
	},
	speedup: {
		title: '训练加速：词表太大时，真正难的是输出层而不是 embedding 本身',
		lead: 'Hierarchical Softmax 与 Negative Sampling 的主要价值，在于避免每一步都对整个词表做完整归一化。',
		points: [
			'Hierarchical Softmax 用树路径替代完整多分类。',
			'Negative Sampling 只保留真实样本和少量负样本。',
			'二者都是计算近似，不是新的语义定义方式。'
		],
		rows: [
			{ label: 'Hierarchical Softmax', value: '更像把大类别预测拆成树上的一串二分类' },
			{ label: 'Negative Sampling', value: '更像学习区分“真实共现对”和“随机负样本对”' },
			{ label: '核心意义', value: '降低训练成本，使大规模语料上的词向量学习可行' }
		]
	}
} as const

const activeDescription = computed(() => modes.find((mode) => mode.key === activeMode.value)?.desc ?? '')
const activePanel = computed(() => panels[activeMode.value])
</script>

<style scoped>
.word2vec-training-explorer {
	display: flex;
	flex-direction: column;
	gap: 20px;
	margin: 20px 0 32px;
}

.hero-card,
.controls-card,
.diagram-card,
.insight-card {
	border: 1px solid var(--vp-c-divider);
	border-radius: 20px;
	background: linear-gradient(180deg, color-mix(in srgb, var(--vp-c-bg-soft) 92%, #ffffff 8%), var(--vp-c-bg));
	box-shadow: 0 12px 32px rgb(15 23 42 / 0.06);
}

.hero-card {
	display: grid;
	grid-template-columns: minmax(0, 1.35fr) minmax(260px, 0.95fr);
	gap: 20px;
	padding: 24px;
}

.eyebrow {
	margin: 0 0 8px;
	font-size: 12px;
	font-weight: 700;
	letter-spacing: 0.08em;
	text-transform: uppercase;
	color: var(--vp-c-brand-1);
}

.hero-text h2 {
	margin: 0;
	font-size: 28px;
	line-height: 1.2;
}

.summary {
	margin: 12px 0 0;
	color: var(--vp-c-text-2);
	line-height: 1.75;
}

.legend-grid {
	display: grid;
	grid-template-columns: repeat(2, minmax(0, 1fr));
	gap: 12px;
	align-content: start;
}

.legend-item {
	padding: 14px 16px;
	border-radius: 16px;
	font-size: 14px;
	font-weight: 600;
	border: 1px solid transparent;
}

.legend-item.cbow {
	background: color-mix(in srgb, #06b6d4 14%, var(--vp-c-bg));
	border-color: color-mix(in srgb, #06b6d4 32%, transparent);
}

.legend-item.skipgram {
	background: color-mix(in srgb, #8b5cf6 14%, var(--vp-c-bg));
	border-color: color-mix(in srgb, #8b5cf6 32%, transparent);
}

.legend-item.objective {
	background: color-mix(in srgb, #22c55e 14%, var(--vp-c-bg));
	border-color: color-mix(in srgb, #22c55e 32%, transparent);
}

.legend-item.speedup {
	background: color-mix(in srgb, #f59e0b 14%, var(--vp-c-bg));
	border-color: color-mix(in srgb, #f59e0b 32%, transparent);
}

.controls-card {
	padding: 18px 20px;
}

.toggle-group {
	display: flex;
	gap: 10px;
	flex-wrap: wrap;
}

.toggle-btn {
	border: 1px solid var(--vp-c-divider);
	background: var(--vp-c-bg);
	color: var(--vp-c-text-1);
	padding: 10px 16px;
	border-radius: 999px;
	font-size: 14px;
	font-weight: 600;
	cursor: pointer;
	transition: all 0.2s ease;
}

.toggle-btn.active {
	background: var(--vp-c-brand-1);
	border-color: var(--vp-c-brand-1);
	color: white;
}

.mode-desc {
	margin: 14px 0 0;
	color: var(--vp-c-text-2);
	line-height: 1.7;
}

.layout-grid {
	display: grid;
	grid-template-columns: minmax(0, 1.75fr) minmax(300px, 0.9fr);
	gap: 20px;
}

.diagram-card,
.insight-card {
	padding: 22px;
}

.compare-grid {
	display: grid;
	grid-template-columns: repeat(2, minmax(0, 1fr));
	gap: 16px;
}

.path-card,
.objective-card,
.aux-card {
	padding: 18px;
	border-radius: 18px;
	border: 1px solid var(--vp-c-divider);
	background: color-mix(in srgb, var(--vp-c-bg-soft) 84%, #ffffff 16%);
}

.path-card.focus,
.objective-card.focus,
.aux-card.focus {
	box-shadow: 0 0 0 3px color-mix(in srgb, var(--vp-c-brand-1) 18%, transparent);
}

.cbow-card {
	background: color-mix(in srgb, #06b6d4 8%, var(--vp-c-bg));
}

.skipgram-card {
	background: color-mix(in srgb, #8b5cf6 8%, var(--vp-c-bg));
}

.objective-card {
	margin-top: 16px;
	background: color-mix(in srgb, #22c55e 8%, var(--vp-c-bg));
}

.path-head {
	display: flex;
	justify-content: space-between;
	align-items: baseline;
	gap: 12px;
	margin-bottom: 14px;
}

.path-head h3 {
	margin: 0;
	font-size: 22px;
}

.path-head span {
	font-size: 13px;
	font-weight: 600;
	color: var(--vp-c-text-2);
}

.flow-stack {
	display: grid;
	gap: 8px;
	margin-bottom: 12px;
}

.flow-node,
.flow-arrow,
.formula-box {
	padding: 10px 12px;
	border-radius: 14px;
	background: var(--vp-c-bg);
	border: 1px solid var(--vp-c-divider);
	font-weight: 600;
}

.flow-node.formula,
.formula-box {
	font-family: var(--vp-font-family-mono);
	font-size: 13px;
	word-break: break-word;
}

.flow-arrow {
	background: transparent;
	border: none;
	text-align: center;
	font-size: 18px;
	padding: 0;
}

.path-card p,
.objective-card p,
.aux-card p {
	margin: 0;
	line-height: 1.7;
	color: var(--vp-c-text-2);
}

.formula-grid {
	display: grid;
	grid-template-columns: repeat(2, minmax(0, 1fr));
	gap: 12px;
	margin-bottom: 12px;
}

.speedup-grid {
	display: grid;
	grid-template-columns: repeat(2, minmax(0, 1fr));
	gap: 16px;
	margin-top: 16px;
}

.aux-card h4 {
	margin: 0 0 10px;
	font-size: 18px;
}

.insight-card h3 {
	margin: 0;
	font-size: 22px;
}

.insight-lead {
	margin: 12px 0 14px;
	color: var(--vp-c-text-2);
	line-height: 1.75;
}

.insight-card ul {
	margin: 0;
	padding-left: 18px;
	display: grid;
	gap: 10px;
}

.mini-table {
	margin-top: 18px;
	border: 1px solid var(--vp-c-divider);
	border-radius: 16px;
	overflow: hidden;
}

.row {
	display: grid;
	grid-template-columns: 120px 1fr;
}

.row span {
	padding: 12px 14px;
	border-top: 1px solid var(--vp-c-divider);
}

.row.head span {
	font-weight: 700;
	background: color-mix(in srgb, var(--vp-c-bg-soft) 72%, #ffffff 28%);
	border-top: none;
}

.row span:first-child {
	border-right: 1px solid var(--vp-c-divider);
	font-weight: 600;
}

@media (max-width: 1240px) {
	.hero-card,
	.layout-grid,
	.compare-grid,
	.formula-grid,
	.speedup-grid {
		grid-template-columns: 1fr;
	}
}

@media (max-width: 640px) {
	.hero-card,
	.controls-card,
	.diagram-card,
	.insight-card {
		padding: 18px;
		border-radius: 18px;
	}

	.legend-grid {
		grid-template-columns: 1fr;
	}

	.row {
		grid-template-columns: 96px 1fr;
	}
}
</style>
