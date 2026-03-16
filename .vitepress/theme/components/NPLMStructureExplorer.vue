<template>
	<div class="nplm-structure-explorer">
		<div class="hero-card">
			<div class="hero-text">
				<p class="eyebrow">NPLM Structure Explorer</p>
				<h2>从离散窗口到连续概率分布的前馈流水线</h2>
				<p class="summary">
					该图把 NPLM 拆成「固定窗口 → 词向量查表 → 向量拼接 → 隐藏层组合 → 全词表 softmax」五个阶段，突出它如何在保留有限上下文假设的同时，用连续表示与共享参数缓解 n-gram 的稀疏问题。
				</p>
			</div>
			<div class="legend-grid">
				<div class="legend-item window">上下文窗口</div>
				<div class="legend-item embedding">词向量矩阵 $C$</div>
				<div class="legend-item hidden">隐藏表示 $h_t$</div>
				<div class="legend-item output">输出打分与 softmax</div>
			</div>
		</div>

		<div class="controls-card">
			<div class="toggle-group" role="tablist" aria-label="NPLM 视图切换">
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
				<div class="pipeline-grid">
					<div class="stage-card window-stage" :class="{ focus: activeMode === 'overview' || activeMode === 'window' }">
						<div class="stage-head">
							<h3>1. 固定窗口</h3>
							<span>有限上下文近似</span>
						</div>
						<div class="token-row">
							<span class="token">$w_{t-m}$</span>
							<span class="token">$\dots$</span>
							<span class="token">$w_{t-1}$</span>
						</div>
						<p>仍然只看最近的 $m$ 个词，但不再为每个离散组合单独造表。</p>
					</div>

					<div class="stage-card embedding-stage" :class="{ focus: activeMode === 'overview' || activeMode === 'embedding' }">
						<div class="stage-head">
							<h3>2. Embedding</h3>
							<span>查表并拼接</span>
						</div>
						<div class="token-row">
							<span class="state-chip">$C(w_{t-m})$</span>
							<span class="state-chip">$\dots$</span>
							<span class="state-chip">$C(w_{t-1})$</span>
						</div>
						<div class="formula-box">$x_t=[C(w_{t-m});\dots;C(w_{t-1})]$</div>
						<p>词先变成连续向量，再拼接成上下文输入。</p>
					</div>

					<div class="stage-card hidden-stage" :class="{ focus: activeMode === 'overview' || activeMode === 'hidden' }">
						<div class="stage-head">
							<h3>3. 隐藏层</h3>
							<span>局部组合加工</span>
						</div>
						<div class="formula-box">$h_t=\tanh(Hx_t+d_h)$</div>
						<p>把多个词的局部搭配关系压成可用于预测的组合特征。</p>
					</div>

					<div class="stage-card output-stage" :class="{ focus: activeMode === 'overview' || activeMode === 'output' }">
						<div class="stage-head">
							<h3>4. 输出层</h3>
							<span>全词表竞争</span>
						</div>
						<div class="formula-box">$y_t=b+Wx_t+Uh_t$</div>
						<div class="formula-box softmax">$P(w_t=j|c_t)=\text{softmax}(y_t)_j$</div>
						<p>每个词都与当前上下文表示做匹配，再经 softmax 得到条件概率。</p>
					</div>
				</div>

				<div class="aux-grid">
					<div class="aux-card" :class="{ focus: activeMode === 'generalize' }">
						<h4>连续空间泛化</h4>
						<p>相似词与相似上下文可以通过 embedding 与共享参数迁移统计强度，而不是只依赖精确计数命中。</p>
					</div>
					<div class="aux-card" :class="{ focus: activeMode === 'output' }">
						<h4>历史意义</h4>
						<p>NPLM 仍是“有限窗口语言模型”，但把离散计数表升级成了连续表示 + 参数化前馈网络。</p>
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
						<span>结构位点</span>
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

type ModeKey = 'overview' | 'window' | 'embedding' | 'hidden' | 'output' | 'generalize'

const activeMode = ref<ModeKey>('overview')

const modes: { key: ModeKey; label: string; desc: string }[] = [
	{ key: 'overview', label: '总览', desc: '先看 NPLM 如何把 n-gram 式固定窗口改写成神经网络流水线。' },
	{ key: 'window', label: '上下文窗口', desc: '聚焦 NPLM 仍然保留的有限上下文假设。' },
	{ key: 'embedding', label: '词向量层', desc: '聚焦离散词如何先映射到连续空间，再拼接成输入。' },
	{ key: 'hidden', label: '隐藏层', desc: '聚焦前馈网络如何把多个上下文词的组合模式提炼成局部特征。' },
	{ key: 'output', label: '输出层', desc: '聚焦全词表 logits 与 softmax 如何形成下一词条件概率分布。' },
	{ key: 'generalize', label: '泛化机制', desc: '聚焦 NPLM 相比 n-gram 的关键提升：连续空间中的统计迁移。' }
]

const panels = {
	overview: {
		title: '总览：NPLM 是“神经网络参数化的 n-gram”',
		lead: '它没有放弃有限窗口假设，而是把离散计数表替换为 embedding 与前馈网络。',
		points: [
			'NPLM 仍然只看最近的若干个词，但每个词先被映射到低维连续向量。',
			'上下文不再通过硬计数直接查表，而是通过共享参数的网络完成条件概率估计。',
			'它的历史意义在于把“连续表示”正式引入语言模型主干。'
		],
		rows: [
			{ label: '窗口', value: '保留 n-gram 的有限上下文近似' },
			{ label: '表示', value: '把离散词映射到可学习 embedding' },
			{ label: '预测', value: '通过前馈网络输出全词表概率分布' }
		]
	},
	window: {
		title: '上下文窗口：限制历史范围，但不再限制表达方式',
		lead: 'NPLM 依旧认为预测当前词时最重要的是最近的有限上下文，只是换了表示与计算方式。',
		points: [
			'窗口长度通常为 $m=n-1$，与 n-gram 的局部建模思路一致。',
			'这让模型的输入规模固定，适合前馈网络处理。',
			'它没有解决“无限长依赖”，而是先解决“离散稀疏与泛化不足”。'
		],
		rows: [
			{ label: '$w_{t-m:t-1}$', value: '用于预测当前词的固定长度历史窗口' },
			{ label: '优点', value: '输入维度固定，易于建模和训练' },
			{ label: '限制', value: '长程依赖仍无法直接显式表达' }
		]
	},
	embedding: {
		title: '词向量层：离散符号第一次被放进共享几何空间',
		lead: 'Embedding 让“猫”“狗”“狐狸”之类相似词不再彼此孤立，而能在连续空间中靠近。',
		points: [
			'词表中的每个词都在矩阵 $C$ 中拥有一行连续向量。',
			'上下文输入由多个词向量按顺序拼接得到，而不是 one-hot 直接拼接。',
			'这是 NPLM 泛化能力的第一层来源。'
		],
		rows: [
			{ label: '$C$', value: '输入词向量矩阵，每一行对应一个词的 embedding' },
			{ label: '$x_t$', value: '由多个上下文词向量拼接而成的输入向量' },
			{ label: '几何意义', value: '相似词可在连续空间中共享统计结构' }
		]
	},
	hidden: {
		title: '隐藏层：把上下文拼接向量变成局部语义组合特征',
		lead: '隐藏层不是简单压缩，而是在学习“哪些局部搭配模式对下一词预测最重要”。',
		points: [
			'线性层负责重组上下文特征，tanh 负责引入非线性。',
			'隐藏表示可以视为若干局部模式探测器的激活结果。',
			'这一步让模型超越了纯线性相加的限制。'
		],
		rows: [
			{ label: '$H$', value: '隐藏层权重矩阵' },
			{ label: '$h_t$', value: '上下文经过非线性加工后的中间表示' },
			{ label: '作用', value: '提炼局部词序与语义搭配特征' }
		]
	},
	output: {
		title: '输出层：全词表候选在当前上下文下竞争',
		lead: 'NPLM 最终仍是语言模型，因此目标是得到整个词表上“谁最可能是下一个词”的条件概率分布。',
		points: [
			'原始 NPLM 同时保留了输入到输出的直接线性路径和隐藏层到输出的非线性路径。',
			'每个词在输出端都对应一个参数模板，与当前上下文表示做匹配。',
			'softmax 让所有候选词形成归一化竞争。'
		],
		rows: [
			{ label: '$y_t$', value: '未归一化的全词表打分（logits）' },
			{ label: '$W,U,b$', value: '输出层参数，决定词表竞争方式' },
			{ label: 'softmax', value: '把 logits 归一化为条件概率分布' }
		]
	},
	generalize: {
		title: '泛化机制：连续空间中的迁移，而不是显式回退',
		lead: 'n-gram 面对未见上下文时主要依赖折扣和平滑，NPLM 则依赖 embedding 与共享参数做连续空间泛化。',
		points: [
			'相似词可以共享词向量结构。',
			'相似上下文会在隐藏层中激活相近的局部模式。',
			'因此未见组合也可能从见过的近邻样本中“借到”概率质量。'
		],
		rows: [
			{ label: 'n-gram', value: '主要依赖显式平滑与回退' },
			{ label: 'NPLM', value: '主要依赖连续表示与网络参数共享' },
			{ label: '提升点', value: '未见组合也能通过相似性获得合理预测' }
		]
	}
} as const

const activeDescription = computed(() => modes.find((mode) => mode.key === activeMode.value)?.desc ?? '')
const activePanel = computed(() => panels[activeMode.value])
</script>

<style scoped>
.nplm-structure-explorer {
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

.legend-item.window {
	background: color-mix(in srgb, #8b5cf6 14%, var(--vp-c-bg));
	border-color: color-mix(in srgb, #8b5cf6 32%, transparent);
}

.legend-item.embedding {
	background: color-mix(in srgb, #06b6d4 14%, var(--vp-c-bg));
	border-color: color-mix(in srgb, #06b6d4 32%, transparent);
}

.legend-item.hidden {
	background: color-mix(in srgb, #22c55e 14%, var(--vp-c-bg));
	border-color: color-mix(in srgb, #22c55e 32%, transparent);
}

.legend-item.output {
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
	grid-template-columns: minmax(0, 1.75fr) minmax(290px, 0.9fr);
	gap: 20px;
}

.diagram-card,
.insight-card {
	padding: 22px;
}

.pipeline-grid {
	display: grid;
	grid-template-columns: repeat(4, minmax(0, 1fr));
	gap: 16px;
}

.stage-card,
.aux-card {
	padding: 18px;
	border-radius: 18px;
	border: 1px solid var(--vp-c-divider);
	background: color-mix(in srgb, var(--vp-c-bg-soft) 84%, #ffffff 16%);
}

.stage-card.focus,
.aux-card.focus {
	box-shadow: 0 0 0 3px color-mix(in srgb, var(--vp-c-brand-1) 18%, transparent);
}

.window-stage {
	background: color-mix(in srgb, #8b5cf6 8%, var(--vp-c-bg));
}

.embedding-stage {
	background: color-mix(in srgb, #06b6d4 8%, var(--vp-c-bg));
}

.hidden-stage {
	background: color-mix(in srgb, #22c55e 8%, var(--vp-c-bg));
}

.output-stage {
	background: color-mix(in srgb, #f59e0b 8%, var(--vp-c-bg));
}

.stage-head {
	display: flex;
	justify-content: space-between;
	align-items: baseline;
	gap: 10px;
	margin-bottom: 14px;
}

.stage-head h3 {
	margin: 0;
	font-size: 20px;
}

.stage-head span {
	font-size: 13px;
	font-weight: 600;
	color: var(--vp-c-text-2);
}

.token-row {
	display: flex;
	flex-wrap: wrap;
	gap: 8px;
	margin-bottom: 12px;
}

.token,
.state-chip,
.formula-box {
	padding: 10px 12px;
	border-radius: 14px;
	background: var(--vp-c-bg);
	border: 1px solid var(--vp-c-divider);
	font-weight: 600;
}

.state-chip {
	background: color-mix(in srgb, #06b6d4 10%, var(--vp-c-bg));
}

.formula-box {
	margin-bottom: 10px;
	font-family: var(--vp-font-family-mono);
	font-size: 13px;
	word-break: break-word;
}

.formula-box.softmax {
	background: color-mix(in srgb, #f59e0b 10%, var(--vp-c-bg));
}

.stage-card p,
.aux-card p {
	margin: 0;
	line-height: 1.7;
	color: var(--vp-c-text-2);
}

.aux-grid {
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
	.pipeline-grid,
	.aux-grid {
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
