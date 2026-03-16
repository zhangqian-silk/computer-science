<template>
	<div class="contrastive-alignment-matrix-explorer">
		<div class="hero-card">
			<div class="hero-text">
				<p class="eyebrow">Contrastive Alignment Matrix Explorer</p>
				<h2>多模态对比学习中的相似度矩阵与正负样本机制</h2>
				<p class="summary">
					该图把第 5.2 节里最关键但也最抽象的部分展开为可视化矩阵：一个 batch 中所有图文组合如何形成相似度表，为什么对角线是正样本，非对角线是负样本，以及困难负样本如何提高 embedding 空间的分辨率。
				</p>
			</div>
			<div class="legend-grid">
				<div class="legend-item positive">正样本对</div>
				<div class="legend-item negative">普通负样本</div>
				<div class="legend-item hard">困难负样本</div>
				<div class="legend-item objective">InfoNCE / 温度</div>
			</div>
		</div>

		<div class="controls-card">
			<div class="toggle-group" role="tablist" aria-label="对比学习视图切换">
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
				<div class="matrix-card" :class="{ focus: activeMode === 'overview' || activeMode === 'matrix' }">
					<div class="matrix-head">
						<h3>Batch 内图文相似度矩阵 $S_{ij}$</h3>
						<span>行：图像　列：文本</span>
					</div>
					<div class="matrix-grid">
						<div class="corner-cell"></div>
						<div class="label-cell">$T_1$</div>
						<div class="label-cell">$T_2$</div>
						<div class="label-cell">$T_3$</div>
						<div class="label-cell">$T_4$</div>

						<div class="label-cell">$I_1$</div>
						<div class="matrix-cell positive">$S_{11}$</div>
						<div class="matrix-cell negative">$S_{12}$</div>
						<div class="matrix-cell negative">$S_{13}$</div>
						<div class="matrix-cell hard">$S_{14}$</div>

						<div class="label-cell">$I_2$</div>
						<div class="matrix-cell negative">$S_{21}$</div>
						<div class="matrix-cell positive">$S_{22}$</div>
						<div class="matrix-cell hard">$S_{23}$</div>
						<div class="matrix-cell negative">$S_{24}$</div>

						<div class="label-cell">$I_3$</div>
						<div class="matrix-cell negative">$S_{31}$</div>
						<div class="matrix-cell hard">$S_{32}$</div>
						<div class="matrix-cell positive">$S_{33}$</div>
						<div class="matrix-cell negative">$S_{34}$</div>

						<div class="label-cell">$I_4$</div>
						<div class="matrix-cell hard">$S_{41}$</div>
						<div class="matrix-cell negative">$S_{42}$</div>
						<div class="matrix-cell negative">$S_{43}$</div>
						<div class="matrix-cell positive">$S_{44}$</div>
					</div>
					<p>对角线表示真实配对样本，目标是把这些位置的相似度推高；其余位置代表不匹配组合，其中有些只是普通负样本，有些则是更难区分的困难负样本。</p>
				</div>

				<div class="flow-grid">
					<div class="flow-card" :class="{ focus: activeMode === 'positive' }">
						<h4>正样本</h4>
						<p>真实匹配的图像—文本对，例如“猫的图片”与“a cat on the grass”。训练目标要求它们在共享空间中彼此靠近。</p>
					</div>
					<div class="flow-card" :class="{ focus: activeMode === 'negative' }">
						<h4>普通负样本</h4>
						<p>语义差异明显的不匹配样本，例如猫图与“波音客机”文本。它们容易区分，提供基础的分离信号。</p>
					</div>
					<div class="flow-card" :class="{ focus: activeMode === 'hard' }">
						<h4>困难负样本</h4>
						<p>看起来很像但其实不匹配的样本，例如柯基图片与“草地上的小型犬”。这类样本最能提高空间分辨率。</p>
					</div>
				</div>

				<div class="objective-card" :class="{ focus: activeMode === 'objective' }">
					<div class="objective-head">
						<h3>InfoNCE 目标</h3>
						<span>让正确匹配在 batch 内胜出</span>
					</div>
					<div class="formula-box">
						$\mathcal{L}_{img\rightarrow text}=-\log\frac{\exp(\operatorname{sim}(z_{v,i},z_{t,i})/\tau)}{\sum_{j=1}^{N}\exp(\operatorname{sim}(z_{v,i},z_{t,j})/\tau)}$
					</div>
					<p>温度参数 $\tau$ 控制 softmax 分布的尖锐程度：温度越低，模型越强调“把真正匹配项从相近负样本中硬拉开”。</p>
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
						<span>元素</span>
						<span>作用</span>
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

type ModeKey = 'overview' | 'matrix' | 'positive' | 'negative' | 'hard' | 'objective'

const activeMode = ref<ModeKey>('overview')

const modes: { key: ModeKey; label: string; desc: string }[] = [
	{ key: 'overview', label: '总览', desc: '先看一个 batch 内图文相似度矩阵的整体结构。' },
	{ key: 'matrix', label: '相似度矩阵', desc: '聚焦对角线、非对角线以及 batch 内竞争关系。' },
	{ key: 'positive', label: '正样本', desc: '聚焦真正匹配的图文对为什么要在共享空间中彼此靠近。' },
	{ key: 'negative', label: '普通负样本', desc: '聚焦明显不匹配样本如何提供基础分离信号。' },
	{ key: 'hard', label: '困难负样本', desc: '聚焦语义边界附近的难例如何提高 embedding 空间分辨率。' },
	{ key: 'objective', label: 'InfoNCE', desc: '聚焦 softmax 归一化、温度参数与 batch 内竞争机制。' }
]

const panels = {
	overview: {
		title: '总览：多模态对比学习本质上是在“重塑整个 batch 的相似度几何”',
		lead: '模型不是只看单个配对样本，而是在一个 batch 内同时比较所有可能组合，让真正匹配项在竞争中胜出。',
		points: [
			'对角线上的真实配对必须拥有更高相似度。',
			'非对角线项构成负样本池，为空间提供分离约束。',
			'困难负样本越多，模型越需要学习更细粒度的跨模态语义。'
		],
		rows: [
			{ label: '对角线', value: '表示真实匹配的图文配对' },
			{ label: '非对角线', value: '表示不匹配的候选组合' },
			{ label: '目标', value: '让匹配项在 batch 内拥有更高归一化概率' }
		]
	},
	matrix: {
		title: '相似度矩阵：一次前向传播就生成了一个“谁更像谁”的全表',
		lead: '矩阵视角能清楚看到，对比学习不是逐对独立训练，而是在同一 batch 中同时塑造所有样本的几何关系。',
		points: [
			'每个图像都要与 batch 中所有文本比较。',
			'每个文本也要与 batch 中所有图像比较。',
			'这让 batch 内其他样本天然成为负样本来源。'
		],
		rows: [
			{ label: '$S_{ij}$', value: '图像 $I_i$ 与文本 $T_j$ 的相似度' },
			{ label: '行方向', value: '以图搜文的竞争关系' },
			{ label: '列方向', value: '以文搜图的竞争关系' }
		]
	},
	positive: {
		title: '正样本：共享空间中的“应该靠近”样本对',
		lead: '正样本定义了多模态对齐的锚点：哪些文本真的对应哪些图像，哪些音频真的对应哪些视频。',
		points: [
			'正样本相似度越高，模型越容易学到跨模态一致性。',
			'这类监督直接决定共享空间里“什么算同一语义对象”。',
			'优质正样本标注比单纯增加样本量更关键。'
		],
		rows: [
			{ label: '来源', value: '真实图文配对、图音配对或其他跨模态匹配数据' },
			{ label: '作用', value: '规定共享空间中的语义锚点' },
			{ label: '结果', value: '匹配样本在向量空间中逐渐靠近' }
		]
	},
	negative: {
		title: '普通负样本：提供基础分离边界，但不一定提升细粒度分辨率',
		lead: '若负样本与正样本差异过大，模型很容易完成区分，却未必能学到更细的跨模态边界。',
		points: [
			'普通负样本常来自 batch 内其他不匹配组合。',
			'它们足以避免一切样本都塌缩到同一区域。',
			'但如果过于容易，模型可能停留在粗粒度区分上。'
		],
		rows: [
			{ label: '优点', value: '稳定、天然、实现简单' },
			{ label: '局限', value: '可能只学到粗粒度差异' },
			{ label: '典型来源', value: '同一 batch 中的其他非匹配样本' }
		]
	},
	 hard: {
		title: '困难负样本：真正迫使模型学习细粒度语义边界的关键来源',
		lead: '困难负样本不是完全错误的东西，而是“看起来像，但其实不完全对”的那些边界案例。',
		points: [
			'它们让模型不能只依赖“动物 / 非动物”这种粗特征。',
			'模型必须进一步学习品种、姿态、场景与细节描述。',
			'但如果误把潜在正样本当成难负样本，就会引入假负样本问题。'
		],
		rows: [
			{ label: '价值', value: '提升 embedding 空间的精细分辨率' },
			{ label: '风险', value: '容易混入假负样本，破坏空间一致性' },
			{ label: '策略', value: '通常与普通负样本混合使用，而不是完全替代' }
		]
	},
	objective: {
		title: 'InfoNCE：让正确匹配项在 softmax 归一化竞争中胜出',
		lead: 'InfoNCE 的关键不是某个公式本身，而是它把“正确项比所有其他候选更大”写成了概率竞争问题。',
		points: [
			'分子只保留正确匹配项。',
			'分母包含 batch 内所有候选。',
			'温度参数决定竞争的尖锐程度与梯度分布。'
		],
		rows: [
			{ label: '分子', value: '正确匹配项的指数相似度' },
			{ label: '分母', value: '同一 batch 内所有候选的归一化总和' },
			{ label: '$\tau$', value: '控制 softmax 温度，影响拉开边界的力度' }
		]
	}
} as const

const activeDescription = computed(() => modes.find((mode) => mode.key === activeMode.value)?.desc ?? '')
const activePanel = computed(() => panels[activeMode.value])
</script>

<style scoped>
.contrastive-alignment-matrix-explorer {
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

.legend-item.positive {
	background: color-mix(in srgb, #22c55e 14%, var(--vp-c-bg));
	border-color: color-mix(in srgb, #22c55e 32%, transparent);
}

.legend-item.negative {
	background: color-mix(in srgb, #94a3b8 18%, var(--vp-c-bg));
	border-color: color-mix(in srgb, #94a3b8 36%, transparent);
}

.legend-item.hard {
	background: color-mix(in srgb, #f59e0b 14%, var(--vp-c-bg));
	border-color: color-mix(in srgb, #f59e0b 32%, transparent);
}

.legend-item.objective {
	background: color-mix(in srgb, #8b5cf6 14%, var(--vp-c-bg));
	border-color: color-mix(in srgb, #8b5cf6 32%, transparent);
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
	grid-template-columns: minmax(0, 1.78fr) minmax(300px, 0.9fr);
	gap: 20px;
}

.diagram-card,
.insight-card {
	padding: 22px;
}

.matrix-card,
.flow-card,
.objective-card {
	padding: 18px;
	border-radius: 18px;
	border: 1px solid var(--vp-c-divider);
	background: color-mix(in srgb, var(--vp-c-bg-soft) 84%, #ffffff 16%);
}

.matrix-card.focus,
.flow-card.focus,
.objective-card.focus {
	box-shadow: 0 0 0 3px color-mix(in srgb, var(--vp-c-brand-1) 18%, transparent);
}

.matrix-head,
.objective-head {
	display: flex;
	justify-content: space-between;
	align-items: baseline;
	gap: 12px;
	margin-bottom: 14px;
}

.matrix-head h3,
.objective-head h3 {
	margin: 0;
	font-size: 22px;
}

.matrix-head span,
.objective-head span {
	font-size: 13px;
	font-weight: 600;
	color: var(--vp-c-text-2);
}

.matrix-grid {
	display: grid;
	grid-template-columns: 72px repeat(4, minmax(72px, 1fr));
	gap: 8px;
	margin-bottom: 12px;
}

.corner-cell,
.label-cell,
.matrix-cell {
	padding: 12px 10px;
	border-radius: 12px;
	text-align: center;
	font-weight: 700;
	border: 1px solid var(--vp-c-divider);
	background: var(--vp-c-bg);
}

.label-cell {
	background: color-mix(in srgb, var(--vp-c-bg-soft) 72%, #ffffff 28%);
}

.matrix-cell.positive {
	background: color-mix(in srgb, #22c55e 14%, var(--vp-c-bg));
	border-color: color-mix(in srgb, #22c55e 32%, transparent);
}

.matrix-cell.negative {
	background: color-mix(in srgb, #94a3b8 12%, var(--vp-c-bg));
	border-color: color-mix(in srgb, #94a3b8 30%, transparent);
}

.matrix-cell.hard {
	background: color-mix(in srgb, #f59e0b 14%, var(--vp-c-bg));
	border-color: color-mix(in srgb, #f59e0b 32%, transparent);
}

.matrix-card p,
.flow-card p,
.objective-card p {
	margin: 0;
	line-height: 1.7;
	color: var(--vp-c-text-2);
}

.flow-grid {
	display: grid;
	grid-template-columns: repeat(3, minmax(0, 1fr));
	gap: 16px;
	margin-top: 16px;
}

.flow-card h4 {
	margin: 0 0 10px;
	font-size: 18px;
}

.objective-card {
	margin-top: 16px;
	background: color-mix(in srgb, #8b5cf6 8%, var(--vp-c-bg));
}

.formula-box {
	padding: 12px 14px;
	border-radius: 14px;
	background: var(--vp-c-bg);
	border: 1px solid var(--vp-c-divider);
	font-family: var(--vp-font-family-mono);
	font-size: 13px;
	word-break: break-word;
	margin-bottom: 12px;
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
	.flow-grid {
		grid-template-columns: 1fr;
	}
}

@media (max-width: 760px) {
	.matrix-grid {
		grid-template-columns: 56px repeat(4, minmax(56px, 1fr));
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
