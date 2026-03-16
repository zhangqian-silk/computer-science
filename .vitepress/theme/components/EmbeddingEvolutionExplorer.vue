<template>
	<div class="embedding-evolution-explorer">
		<div class="hero-card">
			<div class="hero-text">
				<p class="eyebrow">Embedding Evolution Explorer</p>
				<h2>静态词向量到上下文化表示的演化图</h2>
				<p class="summary">
					该图聚焦 Embedding 文档中最关键的一条演化主线：为什么固定词向量不够、上下文化表示究竟改进了什么，以及不同训练目标如何塑造不同的向量空间结构。
				</p>
			</div>
			<div class="legend-grid">
				<div class="legend-item static">静态表示</div>
				<div class="legend-item contextual">上下文化表示</div>
				<div class="legend-item objective">训练目标</div>
				<div class="legend-item downstream">下游作用</div>
			</div>
		</div>

		<div class="controls-card">
			<div class="toggle-group" role="tablist" aria-label="Embedding 演化视图切换">
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
			<section class="compare-card">
				<div class="compare-columns">
					<div class="model-card static-card" :class="{ focus: activeMode === 'overview' || activeMode === 'static' || activeMode === 'objective' }">
						<div class="model-head">
							<h3>静态 Embedding</h3>
							<span>one token → one vector</span>
						</div>
						<div class="example-box">
							<div class="example-title">例：bank</div>
							<div class="example-chip">river bank → 同一向量</div>
							<div class="example-chip">bank account → 同一向量</div>
						</div>
						<ul>
							<li>同一个词无论出现在什么上下文中，都共享同一个固定表示。</li>
							<li>训练目标通常围绕局部预测或全局共现统计展开。</li>
							<li>优点是简单、高效、易缓存；缺点是难区分一词多义。</li>
						</ul>
					</div>

					<div class="model-card contextual-card" :class="{ focus: activeMode === 'overview' || activeMode === 'contextual' || activeMode === 'objective' }">
						<div class="model-head">
							<h3>上下文化 Embedding</h3>
							<span>one token + context → dynamic vector</span>
						</div>
						<div class="example-box">
							<div class="example-title">例：bank</div>
							<div class="example-chip dynamic">river bank → 河岸语义向量</div>
							<div class="example-chip dynamic">bank account → 金融语义向量</div>
						</div>
						<ul>
							<li>词表示不再是查表常量，而是上下文模型内部生成的动态状态。</li>
							<li>训练目标常来自语言模型、掩码预测或自回归建模。</li>
							<li>优点是语义表达更细腻；代价是模型更重、依赖上下文主干。</li>
						</ul>
					</div>
				</div>

				<div class="timeline-card" :class="{ focus: activeMode === 'timeline' }">
					<h4>演化路线</h4>
					<div class="timeline-row">
						<div class="timeline-node">Word2Vec / GloVe / FastText</div>
						<div class="timeline-arrow">→</div>
						<div class="timeline-node">ELMo</div>
						<div class="timeline-arrow">→</div>
						<div class="timeline-node">BERT / GPT / 现代表示</div>
					</div>
					<p>核心变化不是“向量维度更大”，而是表示从静态参数表转向上下文驱动的动态生成。</p>
				</div>

				<div class="formula-grid">
					<div class="formula-card" :class="{ focus: activeMode === 'static' }">
						<span class="formula-label">静态查表</span>
						<code>$x_i=E^\top e_i$</code>
					</div>
					<div class="formula-card" :class="{ focus: activeMode === 'contextual' }">
						<span class="formula-label">上下文化表示</span>
						<code>$x_t = g_\theta(w_t,\text{context})$</code>
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

type ModeKey = 'overview' | 'static' | 'contextual' | 'objective' | 'timeline'

const activeMode = ref<ModeKey>('overview')

const modes: { key: ModeKey; label: string; desc: string }[] = [
	{ key: 'overview', label: '总览', desc: '先看固定词向量与上下文化表示在“表示单位”和“生成方式”上的根本差异。' },
	{ key: 'static', label: '静态表示', desc: '聚焦为什么 word2vec / GloVe / FastText 风格表示简单高效，却难处理一词多义。' },
	{ key: 'contextual', label: '上下文化', desc: '聚焦 ELMo / BERT / GPT 一类表示如何让同一词在不同上下文中产生不同向量。' },
	{ key: 'objective', label: '训练目标', desc: '聚焦不同训练目标如何塑造不同的 embedding 几何结构。' },
	{ key: 'timeline', label: '演化路线', desc: '聚焦 Embedding 发展从静态参数表走向动态上下文表示的历史脉络。' }
]

const panels = {
	overview: {
		title: '总览：Embedding 的一次关键升级，是“查表向量”变成“上下文状态”',
		lead: '静态与上下文化表示之间最重要的差异，不是维度大小，而是表示是否依赖当前上下文。',
		points: [
			'静态 embedding 适合把词看成稳定对象；上下文化 embedding 适合把词看成语境中的动态角色。',
			'二者都属于表示学习，但可表达的语义细粒度明显不同。',
			'现代大模型内部大量使用的其实是上下文化表示，而非传统静态词向量。'
		],
		rows: [
			{ label: '静态', value: '对象一旦映射，就始终对应固定向量' },
			{ label: '动态', value: '表示由对象和上下文共同决定' },
			{ label: '升级点', value: '从“记住词是什么”走向“理解词现在是什么意思”' }
		]
	},
	static: {
		title: '静态表示：高效、稳定，但语义分辨率有限',
		lead: '静态 embedding 最大的优点是结构简单，最大的代价是一个词通常只能拥有一个主向量。',
		points: [
			'训练完成后，每个词都在矩阵里有一个固定位置。',
			'这让检索、缓存、部署与下游初始化都很方便。',
			'但当一个词在不同语境中扮演不同角色时，固定向量就会混合这些语义。'
		],
		rows: [
			{ label: '代表方法', value: 'word2vec / GloVe / FastText' },
			{ label: '优势', value: '简单、高效、可独立预训练和复用' },
			{ label: '局限', value: '对多义词、语境变化与复杂上下文敏感度不足' }
		]
	},
	contextual: {
		title: '上下文化表示：同一个词在不同句子里成为不同向量',
		lead: '上下文化 embedding 把表示学习从“词表查表”推进到“语境驱动的动态生成”。',
		points: [
			'词表示会随着上下文变化，因此能够区分不同语义用法。',
			'这种表示通常来自双向或自回归语言模型内部的隐藏状态。',
			'它的提升不只是词义消歧，还包括对句法角色与局部语义关系的更细粒度表达。'
		],
		rows: [
			{ label: '代表路线', value: 'ELMo → BERT / GPT / 现代 Transformer 表示' },
			{ label: '输入条件', value: '当前 token 与其上下文共同决定表示' },
			{ label: '代价', value: '需要更强的上下文建模主干与更高计算成本' }
		]
	},
	objective: {
		title: '训练目标：Embedding 学成什么样，最终取决于它被要求完成什么任务',
		lead: '向量空间的几何结构不是先验给定的，而是在预测、对比或生成目标驱动下逐步形成的。',
		points: [
			'局部上下文预测会让相似共现词更接近。',
			'掩码语言模型会让表示更依赖双向上下文。',
			'下游分类、检索、推荐任务也会把自身结构写进 embedding 空间。'
		],
		rows: [
			{ label: '局部预测', value: '典型代表为 word2vec 一类静态词向量训练' },
			{ label: '掩码 / 自回归', value: '典型代表为 BERT / GPT 一类上下文化表示学习' },
			{ label: '结论', value: 'Embedding 几何结构本质上是目标函数的副产物' }
		]
	},
	timeline: {
		title: '演化路线：Embedding 的升级不是替换，而是能力边界不断外扩',
		lead: '静态 embedding 仍有工程价值，但上下文化表示把表达边界显著往前推进了一大步。',
		points: [
			'静态词向量解决了离散稀疏与几何相似性问题。',
			'上下文化表示进一步解决了语义随上下文变化的问题。',
			'现代系统常把二者结合使用：底层仍可有查表层，高层则生成动态表示。'
		],
		rows: [
			{ label: '第一阶段', value: '静态词向量：让词拥有可比较的几何位置' },
			{ label: '第二阶段', value: '上下文化表示：让词义随语境动态变化' },
			{ label: '今天', value: '更多系统在静态输入表示之上叠加动态上下文化建模' }
		]
	}
} as const

const activeDescription = computed(() => modes.find((mode) => mode.key === activeMode.value)?.desc ?? '')
const activePanel = computed(() => panels[activeMode.value])
</script>

<style scoped>
.embedding-evolution-explorer {
	display: flex;
	flex-direction: column;
	gap: 20px;
	margin: 20px 0 32px;
}

.hero-card,
.controls-card,
.compare-card,
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

.legend-item.static {
	background: color-mix(in srgb, #06b6d4 14%, var(--vp-c-bg));
	border-color: color-mix(in srgb, #06b6d4 32%, transparent);
}

.legend-item.contextual {
	background: color-mix(in srgb, #8b5cf6 14%, var(--vp-c-bg));
	border-color: color-mix(in srgb, #8b5cf6 32%, transparent);
}

.legend-item.objective {
	background: color-mix(in srgb, #22c55e 14%, var(--vp-c-bg));
	border-color: color-mix(in srgb, #22c55e 32%, transparent);
}

.legend-item.downstream {
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
	grid-template-columns: minmax(0, 1.7fr) minmax(290px, 0.9fr);
	gap: 20px;
}

.compare-card,
.insight-card {
	padding: 22px;
}

.compare-columns {
	display: grid;
	grid-template-columns: repeat(2, minmax(0, 1fr));
	gap: 16px;
}

.model-card,
.timeline-card {
	padding: 18px;
	border-radius: 18px;
	border: 1px solid var(--vp-c-divider);
	background: color-mix(in srgb, var(--vp-c-bg-soft) 84%, #ffffff 16%);
}

.model-card.focus,
.timeline-card.focus,
.formula-card.focus {
	box-shadow: 0 0 0 3px color-mix(in srgb, var(--vp-c-brand-1) 18%, transparent);
}

.static-card {
	background: color-mix(in srgb, #06b6d4 8%, var(--vp-c-bg));
}

.contextual-card {
	background: color-mix(in srgb, #8b5cf6 8%, var(--vp-c-bg));
}

.model-head {
	display: flex;
	justify-content: space-between;
	align-items: baseline;
	gap: 12px;
	margin-bottom: 14px;
}

.model-head h3 {
	margin: 0;
	font-size: 24px;
}

.model-head span {
	font-size: 13px;
	font-weight: 600;
	color: var(--vp-c-text-2);
}

.example-box {
	padding: 14px;
	border-radius: 16px;
	background: var(--vp-c-bg);
	border: 1px solid var(--vp-c-divider);
	margin-bottom: 14px;
}

.example-title {
	font-size: 13px;
	font-weight: 700;
	color: var(--vp-c-text-2);
	margin-bottom: 8px;
}

.example-chip {
	padding: 10px 12px;
	border-radius: 12px;
	background: color-mix(in srgb, #06b6d4 12%, var(--vp-c-bg));
	border: 1px solid color-mix(in srgb, #06b6d4 32%, transparent);
	font-weight: 600;
	margin-top: 8px;
}

.example-chip.dynamic {
	background: color-mix(in srgb, #8b5cf6 12%, var(--vp-c-bg));
	border-color: color-mix(in srgb, #8b5cf6 32%, transparent);
}

.model-card ul {
	margin: 0;
	padding-left: 18px;
	display: grid;
	gap: 10px;
}

.timeline-card {
	margin-top: 16px;
}

.timeline-card h4 {
	margin: 0 0 12px;
	font-size: 18px;
}

.timeline-row {
	display: grid;
	grid-template-columns: minmax(0, 1fr) auto minmax(0, 0.7fr) auto minmax(0, 1fr);
	gap: 10px;
	align-items: center;
}

.timeline-node,
.timeline-arrow {
	padding: 10px 12px;
	border-radius: 14px;
	background: var(--vp-c-bg);
	border: 1px solid var(--vp-c-divider);
	text-align: center;
	font-weight: 600;
}

.timeline-arrow {
	background: transparent;
	border: none;
	font-size: 20px;
}

.timeline-card p {
	margin: 12px 0 0;
	color: var(--vp-c-text-2);
	line-height: 1.7;
}

.formula-grid {
	display: grid;
	grid-template-columns: repeat(auto-fit, minmax(240px, 1fr));
	gap: 12px;
	margin-top: 16px;
}

.formula-card {
	padding: 14px 16px;
	border-radius: 16px;
	background: var(--vp-c-bg-soft);
	border: 1px solid var(--vp-c-divider);
}

.formula-label {
	display: block;
	font-size: 12px;
	font-weight: 700;
	letter-spacing: 0.06em;
	text-transform: uppercase;
	color: var(--vp-c-text-2);
	margin-bottom: 8px;
}

.formula-card code {
	white-space: normal;
	word-break: break-word;
	font-size: 13px;
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

@media (max-width: 1180px) {
	.hero-card,
	.layout-grid,
	.compare-columns,
	.timeline-row {
		grid-template-columns: 1fr;
	}

	.timeline-arrow {
		display: none;
	}
}

@media (max-width: 640px) {
	.hero-card,
	.controls-card,
	.compare-card,
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
