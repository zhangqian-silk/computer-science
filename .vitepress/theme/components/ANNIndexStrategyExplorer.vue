<template>
	<div class="ann-index-strategy-explorer">
		<div class="hero-card">
			<div class="hero-text">
				<p class="eyebrow">ANN Index Strategy Explorer</p>
				<h2>Flat、IVF、PQ 与 HNSW 的结构思路与工程取舍</h2>
				<p class="summary">
					该图把第 6.3 节最容易混淆的几类 ANN 方案放在同一视图里比较：谁是全量精确搜索、谁是先分桶、谁是先压缩、谁是走近邻图。重点不是背术语，而是建立“它们分别靠什么减少搜索成本”的整体理解。
				</p>
			</div>
			<div class="legend-grid">
				<div class="legend-item flat">Flat / 精确基线</div>
				<div class="legend-item ivf">IVF / 分桶</div>
				<div class="legend-item pq">PQ / 压缩</div>
				<div class="legend-item hnsw">HNSW / 图搜索</div>
			</div>
		</div>

		<div class="controls-card">
			<div class="toggle-group" role="tablist" aria-label="ANN 方法视图切换">
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
				<div class="strategy-grid">
					<div class="strategy-card flat-card" :class="{ focus: activeMode === 'overview' || activeMode === 'flat' }">
						<div class="strategy-head">
							<h3>Flat</h3>
							<span>全量比较</span>
						</div>
						<div class="mini-flow">
							<div class="chip">查询向量</div>
							<div class="arrow">→</div>
							<div class="chip">与所有 $x_i$ 比较</div>
						</div>
						<p>结果最精确，但复杂度高。更像基线方案，而不是大规模在线服务的最终形态。</p>
					</div>

					<div class="strategy-card ivf-card" :class="{ focus: activeMode === 'overview' || activeMode === 'ivf' }">
						<div class="strategy-head">
							<h3>IVF</h3>
							<span>先分桶再局部搜索</span>
						</div>
						<div class="mini-flow vertical">
							<div class="chip">查询向量</div>
							<div class="arrow">↓</div>
							<div class="chip">找到最相关桶</div>
							<div class="arrow">↓</div>
							<div class="chip">只搜少量桶内候选</div>
						</div>
						<p>核心思想是减少真正需要比较的候选数量，适合大规模分桶式索引。</p>
					</div>

					<div class="strategy-card pq-card" :class="{ focus: activeMode === 'overview' || activeMode === 'pq' }">
						<div class="strategy-head">
							<h3>PQ</h3>
							<span>先压缩再近似距离</span>
						</div>
						<div class="mini-flow vertical">
							<div class="chip">原始高维向量</div>
							<div class="arrow">↓</div>
							<div class="chip">拆子空间并量化</div>
							<div class="arrow">↓</div>
							<div class="chip">用压缩码近似比较</div>
						</div>
						<p>重点在降低存储和距离计算成本，常与 IVF 组合为 IVF-PQ。</p>
					</div>

					<div class="strategy-card hnsw-card" :class="{ focus: activeMode === 'overview' || activeMode === 'hnsw' }">
						<div class="strategy-head">
							<h3>HNSW</h3>
							<span>分层近邻图遍历</span>
						</div>
						<div class="mini-flow vertical">
							<div class="chip">高层粗导航</div>
							<div class="arrow">↓</div>
							<div class="chip">逐层逼近局部邻域</div>
							<div class="arrow">↓</div>
							<div class="chip">底层精细近邻</div>
						</div>
						<p>像从高速路一路下钻到街道，常在高召回、低延迟场景里表现很强。</p>
					</div>
				</div>

				<div class="tradeoff-card" :class="{ focus: activeMode === 'tradeoff' || activeMode === 'overview' }">
					<div class="strategy-head">
						<h3>工程取舍总览</h3>
						<span>召回、延迟、内存三角关系</span>
					</div>
					<div class="tradeoff-table">
						<div class="table-row head">
							<span>方法</span>
							<span>核心思路</span>
							<span>典型优点</span>
						</div>
						<div class="table-row">
							<span>Flat</span>
							<span>全量比较</span>
							<span>精确、简单、便于做基线</span>
						</div>
						<div class="table-row">
							<span>IVF</span>
							<span>先分桶再搜索</span>
							<span>大规模检索更可扩展</span>
						</div>
						<div class="table-row">
							<span>PQ</span>
							<span>压缩向量编码</span>
							<span>省内存、省带宽</span>
						</div>
						<div class="table-row">
							<span>HNSW</span>
							<span>近邻图导航</span>
							<span>高召回、低延迟常见强方案</span>
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
						<span>方法</span>
						<span>定位</span>
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

type ModeKey = 'overview' | 'flat' | 'ivf' | 'pq' | 'hnsw' | 'tradeoff'

const activeMode = ref<ModeKey>('overview')

const modes: { key: ModeKey; label: string; desc: string }[] = [
	{ key: 'overview', label: '总览', desc: '先看几类 ANN 方法分别靠什么机制减少搜索成本。' },
	{ key: 'flat', label: 'Flat', desc: '聚焦为什么 Flat 更像精确基线，而不是大规模在线检索的主力方案。' },
	{ key: 'ivf', label: 'IVF', desc: '聚焦“先分桶再搜索”的思路如何减少候选规模。' },
	{ key: 'pq', label: 'PQ', desc: '聚焦向量压缩编码如何节省存储与近似计算成本。' },
	{ key: 'hnsw', label: 'HNSW', desc: '聚焦分层近邻图如何通过导航式搜索逼近目标。' },
	{ key: 'tradeoff', label: '工程取舍', desc: '聚焦召回、延迟、内存与构建复杂度之间的现实权衡。' }
]

const panels = {
	overview: {
		title: '总览：所有 ANN 方法本质上都在回答同一个问题——如何少比一些向量',
		lead: '差别不在于“谁更神秘”，而在于它们分别用分桶、压缩、图遍历等不同机制减少精确比较的数量。',
		points: [
			'Flat 不减少比较数量，因此最精确，也最慢。',
			'IVF 通过先找相关桶，减少真正要比的候选。',
			'PQ 通过压缩向量，减少存储与距离计算成本；HNSW 则通过近邻图快速导航。'
		],
		rows: [
			{ label: 'Flat', value: '精确基线，简单但扩展性弱' },
			{ label: 'IVF / PQ', value: '典型的工业大规模折中路线' },
			{ label: 'HNSW', value: '图搜索路线，常在高召回场景里表现突出' }
		]
	},
	flat: {
		title: 'Flat：不做近似，直接和所有向量比较',
		lead: 'Flat 的价值不在于速度，而在于它给出了精确搜索的基线效果。',
		points: [
			'实现最直接，调试与验证最容易。',
			'结果精确，不引入索引近似误差。',
			'当规模变大时，延迟和成本会迅速成为瓶颈。'
		],
		rows: [
			{ label: '优点', value: '简单、精确、适合基线评估' },
			{ label: '缺点', value: '复杂度高，难支撑超大规模在线检索' },
			{ label: '适用', value: '小规模库、离线分析或精度对照实验' }
		]
	},
	ivf: {
		title: 'IVF：先粗分桶，再在局部桶内细搜',
		lead: 'IVF 的关键在于：查询不需要访问整个库，只要进入最相关的少数几个桶即可。',
		points: [
			'先通过聚类把向量划入多个桶。',
			'查询阶段先找最近的桶中心。',
			'只对少量桶内向量做后续比较，因此显著降低搜索量。'
		],
		rows: [
			{ label: '核心', value: '以“先缩小搜索区域”来换取速度' },
			{ label: '优点', value: '适合大规模向量库，结构清晰，易与 PQ 组合' },
			{ label: '局限', value: '若桶划分不佳，可能漏掉应访问的相关候选' }
		]
	},
	pq: {
		title: 'PQ：把高维向量压缩成更短的编码，再做近似距离比较',
		lead: 'PQ 的主战场是存储与带宽优化，它让“大量向量怎么存、怎么比”都变得更便宜。',
		points: [
			'高维向量被拆成多个子空间。',
			'每个子空间分别量化编码。',
			'最终距离比较变成在压缩码上的近似计算。'
		],
		rows: [
			{ label: '优点', value: '节省内存，适合超大规模向量库' },
			{ label: '局限', value: '会引入量化误差，精度与压缩率需权衡' },
			{ label: '常见搭配', value: '经常与 IVF 组合成 IVF-PQ' }
		]
	},
	 hnsw: {
		title: 'HNSW：把向量组织成分层近邻图，再通过图导航逼近目标',
		lead: 'HNSW 的直觉不是“先压缩”或“先分桶”，而是“先走到附近，再在局部继续逼近”。',
		points: [
			'高层图负责快速粗定位。',
			'低层图负责更细的局部邻域搜索。',
			'这种多层导航方式常带来很强的召回—延迟平衡。'
		],
		rows: [
			{ label: '优点', value: '高召回、低延迟，常见实战表现强' },
			{ label: '局限', value: '索引构建和内存开销通常不低' },
			{ label: '适用', value: '追求在线性能与高质量近邻搜索的场景' }
		]
	},
	tradeoff: {
		title: '工程取舍：没有万能索引，只有围绕业务目标的折中',
		lead: '检索系统最终要在召回率、延迟、内存、构建成本与更新频率之间找到最合适的平衡。',
		points: [
			'小规模任务可能根本不需要复杂近似索引。',
			'大规模在线系统往往必须引入近似方法。',
			'真实系统里，索引策略要和 embedding 质量、相似度度量、重排模型一起联合考虑。'
		],
		rows: [
			{ label: '高精度优先', value: 'Flat 或高召回 HNSW 更常见' },
			{ label: '大规模省内存', value: 'IVF / PQ 系列更有吸引力' },
			{ label: '最终原则', value: '索引只是系统链条一环，要与模型和业务目标协同设计' }
		]
	}
} as const

const activeDescription = computed(() => modes.find((mode) => mode.key === activeMode.value)?.desc ?? '')
const activePanel = computed(() => panels[activeMode.value])
</script>

<style scoped>
.ann-index-strategy-explorer {
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

.legend-item.flat {
	background: color-mix(in srgb, #94a3b8 18%, var(--vp-c-bg));
	border-color: color-mix(in srgb, #94a3b8 36%, transparent);
}

.legend-item.ivf {
	background: color-mix(in srgb, #06b6d4 14%, var(--vp-c-bg));
	border-color: color-mix(in srgb, #06b6d4 32%, transparent);
}

.legend-item.pq {
	background: color-mix(in srgb, #f59e0b 14%, var(--vp-c-bg));
	border-color: color-mix(in srgb, #f59e0b 32%, transparent);
}

.legend-item.hnsw {
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

.strategy-grid {
	display: grid;
	grid-template-columns: repeat(2, minmax(0, 1fr));
	gap: 16px;
}

.strategy-card,
.tradeoff-card {
	padding: 18px;
	border-radius: 18px;
	border: 1px solid var(--vp-c-divider);
	background: color-mix(in srgb, var(--vp-c-bg-soft) 84%, #ffffff 16%);
}

.strategy-card.focus,
.tradeoff-card.focus {
	box-shadow: 0 0 0 3px color-mix(in srgb, var(--vp-c-brand-1) 18%, transparent);
}

.flat-card {
	background: color-mix(in srgb, #94a3b8 10%, var(--vp-c-bg));
}

.ivf-card {
	background: color-mix(in srgb, #06b6d4 8%, var(--vp-c-bg));
}

.pq-card {
	background: color-mix(in srgb, #f59e0b 8%, var(--vp-c-bg));
}

.hnsw-card {
	background: color-mix(in srgb, #8b5cf6 8%, var(--vp-c-bg));
}

.tradeoff-card {
	margin-top: 16px;
}

.strategy-head {
	display: flex;
	justify-content: space-between;
	align-items: baseline;
	gap: 12px;
	margin-bottom: 14px;
}

.strategy-head h3 {
	margin: 0;
	font-size: 22px;
}

.strategy-head span {
	font-size: 13px;
	font-weight: 600;
	color: var(--vp-c-text-2);
}

.mini-flow {
	display: flex;
	align-items: center;
	gap: 10px;
	flex-wrap: wrap;
	margin-bottom: 12px;
}

.mini-flow.vertical {
	flex-direction: column;
	align-items: stretch;
}

.chip,
.arrow {
	padding: 10px 12px;
	border-radius: 14px;
	background: var(--vp-c-bg);
	border: 1px solid var(--vp-c-divider);
	font-weight: 600;
}

.arrow {
	background: transparent;
	border: none;
	padding: 0;
	font-size: 18px;
}

.strategy-card p,
.tradeoff-card p {
	margin: 0;
	line-height: 1.7;
	color: var(--vp-c-text-2);
}

.tradeoff-table {
	border: 1px solid var(--vp-c-divider);
	border-radius: 16px;
	overflow: hidden;
}

.table-row {
	display: grid;
	grid-template-columns: 88px 1fr 1fr;
}

.table-row span {
	padding: 12px 14px;
	border-top: 1px solid var(--vp-c-divider);
}

.table-row.head span {
	font-weight: 700;
	background: color-mix(in srgb, var(--vp-c-bg-soft) 72%, #ffffff 28%);
	border-top: none;
}

.table-row span:not(:last-child) {
	border-right: 1px solid var(--vp-c-divider);
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
	.strategy-grid {
		grid-template-columns: 1fr;
	}
}

@media (max-width: 720px) {
	.table-row {
		grid-template-columns: 72px 1fr 1fr;
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
