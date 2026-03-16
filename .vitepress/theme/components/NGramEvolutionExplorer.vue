<template>
	<div class="ngram-evolution-explorer">
		<div class="hero-card">
			<div class="hero-text">
				<p class="eyebrow">N-Gram Evolution Explorer</p>
				<h2>N-Gram 到 RNN、LSTM、Transformer 的演化对比图</h2>
				<p class="summary">
					该图把文档后半段的“典型局限与后续模型”整理成一条清晰演化链：N-Gram 如何依赖固定窗口，RNN 如何用状态传递替代计数表，LSTM 如何通过门控缓解长程依赖衰减，Transformer 又如何用全局注意力进一步缩短依赖路径。
				</p>
			</div>
			<div class="legend-grid">
				<div class="legend-item ngram">N-Gram</div>
				<div class="legend-item rnn">RNN</div>
				<div class="legend-item lstm">LSTM</div>
				<div class="legend-item transformer">Transformer</div>
			</div>
		</div>

		<div class="controls-card">
			<div class="toggle-group" role="tablist" aria-label="模型演化视图切换">
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
				<div class="timeline-row">
					<div class="model-card ngram-card" :class="{ focus: activeMode === 'overview' || activeMode === 'ngram' }">
						<h3>N-Gram</h3>
						<span>固定窗口计数</span>
						<p>只看最近 $n-1$ 个词，用计数表和概率表估计下一个词。</p>
					</div>
					<div class="arrow">→</div>
					<div class="model-card rnn-card" :class="{ focus: activeMode === 'overview' || activeMode === 'rnn' }">
						<h3>RNN</h3>
						<span>连续状态传递</span>
						<p>用隐藏状态压缩历史，不再依赖固定窗口大小。</p>
					</div>
					<div class="arrow">→</div>
					<div class="model-card lstm-card" :class="{ focus: activeMode === 'overview' || activeMode === 'lstm' }">
						<h3>LSTM</h3>
						<span>门控记忆</span>
						<p>通过输入门、遗忘门、输出门缓解长程依赖衰减。</p>
					</div>
					<div class="arrow">→</div>
					<div class="model-card transformer-card" :class="{ focus: activeMode === 'overview' || activeMode === 'transformer' }">
						<h3>Transformer</h3>
						<span>全局注意力</span>
						<p>直接在整段序列上建立位置间交互，显著缩短依赖路径。</p>
					</div>
				</div>

				<div class="compare-card" :class="{ focus: activeMode === 'compare' || activeMode === 'overview' }">
					<div class="section-head">
						<h3>核心演化逻辑</h3>
						<span>不是简单替代，而是逐步放宽上下文表达能力</span>
					</div>
					<div class="compare-grid">
						<div class="compare-item">
							<h4>上下文表示</h4>
							<p>N-Gram 用显式计数窗口；RNN/LSTM 用连续状态；Transformer 用注意力直接关联全局位置。</p>
						</div>
						<div class="compare-item">
							<h4>长程依赖</h4>
							<p>从固定窗口 → 状态传递 → 门控增强 → 全局直接访问，依赖路径逐步缩短。</p>
						</div>
						<div class="compare-item">
							<h4>代价与限制</h4>
							<p>从稀疏与参数爆炸，转向递归训练困难，再转向注意力计算与显存成本。</p>
						</div>
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
						<span>模型</span>
						<span>关键变化</span>
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

type ModeKey = 'overview' | 'ngram' | 'rnn' | 'lstm' | 'transformer' | 'compare'

const activeMode = ref<ModeKey>('overview')

const modes: { key: ModeKey; label: string; desc: string }[] = [
	{ key: 'overview', label: '总览', desc: '先看 N-Gram 到 Transformer 的整体演化主线。' },
	{ key: 'ngram', label: 'N-Gram', desc: '聚焦固定窗口计数模型的表达边界。' },
	{ key: 'rnn', label: 'RNN', desc: '聚焦连续状态如何替代显式计数窗口。' },
	{ key: 'lstm', label: 'LSTM', desc: '聚焦门控机制如何缓解长程信息衰减。' },
	{ key: 'transformer', label: 'Transformer', desc: '聚焦全局注意力如何进一步缩短依赖路径。' },
	{ key: 'compare', label: '对比总结', desc: '聚焦上下文表示方式、依赖路径与工程代价的整体变化。' }
]

const panels = {
	overview: {
		title: '总览：语言模型的演化主线，就是上下文表示能力不断增强的过程',
		lead: '从 N-Gram 到 Transformer，并不是问题变了，而是“如何表达与利用上下文”这件事越来越强。',
		points: [
			'N-Gram 只会显式记住有限窗口内的统计。',
			'RNN 把历史压进连续状态，理论上不再受固定窗口限制。',
			'LSTM 用门控改善长期信息保持，Transformer 则进一步用全局注意力缩短依赖路径。'
		],
		rows: [
			{ label: 'N-Gram', value: '局部计数统计，窗口固定' },
			{ label: 'RNN / LSTM', value: '历史被压缩进状态向量，递归传递' },
			{ label: 'Transformer', value: '任意位置可直接交互，上下文范围显著扩展' }
		]
	},
	ngram: {
		title: 'N-Gram：最可解释，但上下文范围最受限',
		lead: 'N-Gram 的优势是清晰、可查表、可解释；它的边界也同样清楚：只能稳定处理极短上下文。',
		points: [
			'历史被截断成固定长度窗口。',
			'参数本质上是离散计数表及其平滑变体。',
			'当阶数增大时，组合爆炸与稀疏性迅速恶化。'
		],
		rows: [
			{ label: '表示', value: '显式窗口计数' },
			{ label: '优点', value: '可解释、实现简单、训练不依赖梯度下降' },
			{ label: '局限', value: '长程依赖难覆盖，参数和数据稀疏问题突出' }
		]
	},
	rnn: {
		title: 'RNN：把离散统计窗口改写成连续状态传递',
		lead: 'RNN 的关键变化，是不再为每种局部组合单独存表，而是把历史压缩进隐藏状态。',
		points: [
			'每一步都读取当前输入并更新隐藏状态。',
			'理论上可以累积任意长度历史。',
			'但长距离信息在递归中仍可能逐渐衰减。'
		],
		rows: [
			{ label: '变化点', value: '从离散计数转向连续状态' },
			{ label: '收益', value: '不再依赖固定窗口大小' },
			{ label: '问题', value: '梯度消失 / 爆炸与长依赖衰减' }
		]
	},
	lstm: {
		title: 'LSTM：在 RNN 上加入门控，提升长期信息保持能力',
		lead: 'LSTM 并没有改变递归主干，而是在状态更新中加入门控，使模型更有能力决定“保留什么、遗忘什么”。',
		points: [
			'遗忘门控制旧信息保留多少。',
			'输入门控制新信息写入多少。',
			'输出门控制当前状态如何暴露给下一层或下一个时间步。'
		],
		rows: [
			{ label: '改进点', value: '在递归更新里加入可学习门控结构' },
			{ label: '收益', value: '长程依赖建模通常优于普通 RNN' },
			{ label: '局限', value: '仍依赖顺序递归，难以高度并行' }
		]
	},
	transformer: {
		title: 'Transformer：把“逐步传递历史”改成“直接查看全局位置”',
		lead: 'Transformer 最关键的变化，是不再让信息沿时间步一层层传，而是用注意力直接建立位置间连接。',
		points: [
			'任意两个位置之间都可直接交互。',
			'依赖路径大幅缩短，更适合捕捉远距离关系。',
			'计算可以高度矩阵化并行，但长序列成本通常随长度平方增长。'
		],
		rows: [
			{ label: '机制', value: '全局注意力而非递归状态传递' },
			{ label: '优势', value: '并行性强，长距离依赖建模能力更强' },
			{ label: '代价', value: '长上下文时注意力计算和显存成本较高' }
		]
	},
	compare: {
		title: '对比总结：模型更强，不是因为“更复杂”，而是因为上下文组织方式变了',
		lead: '从窗口计数到连续状态，再到全局注意力，语言模型每一次升级都在改变历史信息被表示和访问的方式。',
		points: [
			'上下文表示方式决定了模型能记住什么。',
			'依赖路径长度决定了长程信息多难被利用。',
			'工程代价则决定了这些能力能否在现实系统中被高效使用。'
		],
		rows: [
			{ label: '上下文表示', value: '窗口计数 → 状态向量 → 全局注意力' },
			{ label: '依赖路径', value: '固定窗口 → 递归传递 → 直接跨位置连接' },
			{ label: '主瓶颈', value: '稀疏性 → 递归训练困难 → 注意力计算成本' }
		]
	}
} as const

const activeDescription = computed(() => modes.find((mode) => mode.key === activeMode.value)?.desc ?? '')
const activePanel = computed(() => panels[activeMode.value])
</script>

<style scoped>
.ngram-evolution-explorer {
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

.legend-item.ngram {
	background: color-mix(in srgb, #94a3b8 18%, var(--vp-c-bg));
	border-color: color-mix(in srgb, #94a3b8 36%, transparent);
}

.legend-item.rnn {
	background: color-mix(in srgb, #06b6d4 14%, var(--vp-c-bg));
	border-color: color-mix(in srgb, #06b6d4 32%, transparent);
}

.legend-item.lstm {
	background: color-mix(in srgb, #f59e0b 14%, var(--vp-c-bg));
	border-color: color-mix(in srgb, #f59e0b 32%, transparent);
}

.legend-item.transformer {
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

.timeline-row {
	display: grid;
	grid-template-columns: repeat(4, minmax(0, 1fr));
	gap: 12px;
	align-items: stretch;
}

.model-card,
.compare-card,
.compare-item {
	padding: 18px;
	border-radius: 18px;
	border: 1px solid var(--vp-c-divider);
	background: color-mix(in srgb, var(--vp-c-bg-soft) 84%, #ffffff 16%);
}

.model-card.focus,
.compare-card.focus {
	box-shadow: 0 0 0 3px color-mix(in srgb, var(--vp-c-brand-1) 18%, transparent);
}

.model-card h3 {
	margin: 0;
	font-size: 22px;
}

.model-card span {
	display: block;
	margin: 8px 0 10px;
	font-size: 13px;
	font-weight: 700;
	color: var(--vp-c-text-2);
}

.model-card p,
.compare-item p {
	margin: 0;
	line-height: 1.7;
	color: var(--vp-c-text-2);
}

.ngram-card {
	background: color-mix(in srgb, #94a3b8 10%, var(--vp-c-bg));
}

.rnn-card {
	background: color-mix(in srgb, #06b6d4 8%, var(--vp-c-bg));
}

.lstm-card {
	background: color-mix(in srgb, #f59e0b 8%, var(--vp-c-bg));
}

.transformer-card {
	background: color-mix(in srgb, #8b5cf6 8%, var(--vp-c-bg));
}

.arrow {
	display: none;
}

.compare-card {
	margin-top: 16px;
}

.section-head {
	display: flex;
	justify-content: space-between;
	align-items: baseline;
	gap: 12px;
	margin-bottom: 14px;
}

.section-head h3 {
	margin: 0;
	font-size: 22px;
}

.section-head span {
	font-size: 13px;
	font-weight: 600;
	color: var(--vp-c-text-2);
}

.compare-grid {
	display: grid;
	grid-template-columns: repeat(3, minmax(0, 1fr));
	gap: 16px;
}

.compare-item h4 {
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

@media (min-width: 1280px) {
	.timeline-row {
		grid-template-columns: minmax(0,1fr) auto minmax(0,1fr) auto minmax(0,1fr) auto minmax(0,1fr);
	}

	.arrow {
		display: flex;
		align-items: center;
		justify-content: center;
		font-size: 20px;
		color: var(--vp-c-text-2);
	}
}

@media (max-width: 1240px) {
	.hero-card,
	.layout-grid,
	.compare-grid,
	.timeline-row {
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
