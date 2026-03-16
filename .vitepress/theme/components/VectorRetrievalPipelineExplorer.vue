<template>
	<div class="vector-retrieval-pipeline-explorer">
		<div class="hero-card">
			<div class="hero-text">
				<p class="eyebrow">Vector Retrieval Pipeline Explorer</p>
				<h2>Embedding 在检索系统中的多阶段流水线</h2>
				<p class="summary">
					该图把第 6 节中的部署逻辑收束为一条完整链路：查询如何编码为向量、如何经 ANN 索引召回候选、如何与稀疏检索混合、再如何经过重排得到最终结果。这样可以把“embedding 训练”和“embedding 落地”明确接起来。
				</p>
			</div>
			<div class="legend-grid">
				<div class="legend-item query">查询表示</div>
				<div class="legend-item ann">ANN 召回</div>
				<div class="legend-item hybrid">混合检索</div>
				<div class="legend-item rerank">重排与最终输出</div>
			</div>
		</div>

		<div class="controls-card">
			<div class="toggle-group" role="tablist" aria-label="向量检索视图切换">
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
				<div class="pipeline-row">
					<div class="stage-card query-stage" :class="{ focus: activeMode === 'overview' || activeMode === 'query' }">
						<div class="stage-head">
							<h3>1. 查询编码</h3>
							<span>text → vector</span>
						</div>
						<div class="formula-box">$q = f_{enc}(x)$</div>
						<p>用户查询先进入编码器，变成可与文档向量直接比较的查询 embedding。</p>
					</div>

					<div class="arrow">→</div>

					<div class="stage-card ann-stage" :class="{ focus: activeMode === 'overview' || activeMode === 'ann' }">
						<div class="stage-head">
							<h3>2. ANN 召回</h3>
							<span>fast top-k</span>
						</div>
						<div class="formula-box">$\operatorname*{topk}_i\ s(q,x_i)$</div>
						<p>在大规模向量库上，系统通常用 IVF / PQ / HNSW 等索引近似寻找最近邻。</p>
					</div>

					<div class="arrow">→</div>

					<div class="stage-card hybrid-stage" :class="{ focus: activeMode === 'overview' || activeMode === 'hybrid' }">
						<div class="stage-head">
							<h3>3. 混合召回</h3>
							<span>sparse + dense</span>
						</div>
						<div class="token-row">
							<span class="token">BM25</span>
							<span class="token">Dense</span>
							<span class="token">Fusion</span>
						</div>
						<p>实际系统常把关键词命中与语义近邻结合，避免单一路线在某些查询类型上失真。</p>
					</div>

					<div class="arrow">→</div>

					<div class="stage-card rerank-stage" :class="{ focus: activeMode === 'overview' || activeMode === 'rerank' }">
						<div class="stage-head">
							<h3>4. 重排输出</h3>
							<span>higher precision</span>
						</div>
						<div class="token-row">
							<span class="token">Cross-Encoder</span>
							<span class="token">Late Interaction</span>
						</div>
						<p>粗召回保证覆盖面，重排阶段再用更重的模型精细比较查询与候选的真实相关性。</p>
					</div>
				</div>

				<div class="aux-grid">
					<div class="aux-card" :class="{ focus: activeMode === 'ann' }">
						<h4>ANN 的角色</h4>
						<p>ANN 解决的是“如何更快找到近邻”，而不是“哪些对象本来就应该互为近邻”。后者仍由 embedding 质量决定。</p>
					</div>
					<div class="aux-card" :class="{ focus: activeMode === 'hybrid' }">
						<h4>混合检索的价值</h4>
						<p>稠密检索擅长语义近邻，稀疏检索擅长关键词精确命中，混合方案更接近真实系统需求。</p>
					</div>
					<div class="aux-card" :class="{ focus: activeMode === 'rerank' }">
						<h4>重排的意义</h4>
						<p>第一阶段强调低延迟和高召回，第二阶段再把算力花在少量候选上，提高最终精度。</p>
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
						<span>阶段</span>
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

type ModeKey = 'overview' | 'query' | 'ann' | 'hybrid' | 'rerank'

const activeMode = ref<ModeKey>('overview')

const modes: { key: ModeKey; label: string; desc: string }[] = [
	{ key: 'overview', label: '总览', desc: '先看 embedding 在真实检索系统中从查询到结果的多阶段流水线。' },
	{ key: 'query', label: '查询表示', desc: '聚焦自然语言查询如何先被编码成向量。' },
	{ key: 'ann', label: 'ANN 召回', desc: '聚焦 IVF / PQ / HNSW 等索引如何在大规模向量库中加速最近邻搜索。' },
	{ key: 'hybrid', label: '混合检索', desc: '聚焦为什么真实系统常把稀疏检索与稠密检索结合。' },
	{ key: 'rerank', label: '重排阶段', desc: '聚焦为什么召回后还需要更重的精排模型。' }
]

const panels = {
	overview: {
		title: '总览：Embedding 的落地不是“算个相似度”就结束，而是一条系统流水线',
		lead: '训练好的向量要真正服务检索系统，通常还要经过索引、融合、重排与业务规则等多个阶段。',
		points: [
			'查询先被编码为向量，进入 ANN 索引做高召回、低延迟的候选搜索。',
			'候选集合常与 BM25 一类稀疏检索结果融合，兼顾语义与关键词命中。',
			'最终再交给更重的重排模型，以更高精度决定输出顺序。'
		],
		rows: [
			{ label: '查询', value: '把文本或其他输入编码成向量表示' },
			{ label: '召回', value: '在大规模库上快速找出可能相关的少量候选' },
			{ label: '精排', value: '用更细粒度模型决定最终相关性顺序' }
		]
	},
	query: {
		title: '查询表示：检索系统首先要把问题写成同一种向量语言',
		lead: '如果查询与文档不在同一几何空间里，再高效的索引也无法得到正确语义近邻。',
		points: [
			'查询编码器与文档编码器往往需要协同训练。',
			'点积、余弦或欧氏距离的选择会影响整个检索体系设计。',
			'向量归一化方式也会直接影响检索稳定性与排序行为。'
		],
		rows: [
			{ label: '输入', value: '文本查询、图像查询或其他模态查询' },
			{ label: '输出', value: '与文档向量可直接比较的查询 embedding' },
			{ label: '前提', value: '查询空间与候选空间需要共享几何语义' }
		]
	},
	ann: {
		title: 'ANN 召回：牺牲少量精确性，换取数量级上的速度提升',
		lead: '在百万到亿级向量库里，全量精确比较通常太慢，因此系统必须依赖近似索引。',
		points: [
			'IVF 依赖分桶，PQ 依赖压缩，HNSW 依赖近邻图遍历。',
			'这些方法都在减少真正需要精确比较的候选数量。',
			'ANN 优化的是搜索效率，而不是 embedding 本身的语义质量。'
		],
		rows: [
			{ label: 'IVF', value: '先分桶，再在少数相关桶里搜索' },
			{ label: 'PQ', value: '压缩表示并近似距离计算，节省存储与算力' },
			{ label: 'HNSW', value: '通过分层近邻图快速逼近目标邻域' }
		]
	},
	hybrid: {
		title: '混合检索：现实查询往往同时需要语义理解与关键词命中',
		lead: '单独依赖 dense 或 sparse 都容易偏科，混合召回更能覆盖真实用户查询的多样性。',
		points: [
			'稀疏检索对专有名词、精确词面匹配与罕见字符串更稳定。',
			'稠密检索对同义改写、语义相近表达与弱词面重叠更有优势。',
			'融合策略常采用打分加权、候选合并或分阶段路由。'
		],
		rows: [
			{ label: 'Sparse', value: 'BM25 等，擅长显式词项命中' },
			{ label: 'Dense', value: 'Embedding 近邻，擅长语义相似与改写鲁棒性' },
			{ label: 'Hybrid', value: '联合二者，提升整体召回覆盖面' }
		]
	},
	rerank: {
		title: '重排阶段：把算力集中花在少量候选上，换取更高最终精度',
		lead: '召回阶段更像是“先别漏掉”，重排阶段才真正决定“谁应该排在前面”。',
		points: [
			'Cross-Encoder 能直接联合读取查询与文档，相关性判断更细。',
			'Late Interaction 保留 token 级匹配能力，在效果与成本之间折中。',
			'这是多阶段检索系统里最常见的精度放大器。'
		],
		rows: [
			{ label: '输入', value: '召回得到的少量候选集合' },
			{ label: '处理', value: '用更重模型细致比较查询与候选关系' },
			{ label: '输出', value: '更高精度的最终排序结果' }
		]
	}
} as const

const activeDescription = computed(() => modes.find((mode) => mode.key === activeMode.value)?.desc ?? '')
const activePanel = computed(() => panels[activeMode.value])
</script>

<style scoped>
.vector-retrieval-pipeline-explorer {
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

.legend-item.query {
	background: color-mix(in srgb, #06b6d4 14%, var(--vp-c-bg));
	border-color: color-mix(in srgb, #06b6d4 32%, transparent);
}

.legend-item.ann {
	background: color-mix(in srgb, #8b5cf6 14%, var(--vp-c-bg));
	border-color: color-mix(in srgb, #8b5cf6 32%, transparent);
}

.legend-item.hybrid {
	background: color-mix(in srgb, #22c55e 14%, var(--vp-c-bg));
	border-color: color-mix(in srgb, #22c55e 32%, transparent);
}

.legend-item.rerank {
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
	grid-template-columns: minmax(0, 1.8fr) minmax(300px, 0.9fr);
	gap: 20px;
}

.diagram-card,
.insight-card {
	padding: 22px;
}

.pipeline-row {
	display: grid;
	grid-template-columns: minmax(0, 1fr) auto minmax(0, 1fr) auto minmax(0, 1fr) auto minmax(0, 1fr);
	gap: 12px;
	align-items: stretch;
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

.query-stage {
	background: color-mix(in srgb, #06b6d4 8%, var(--vp-c-bg));
}

.ann-stage {
	background: color-mix(in srgb, #8b5cf6 8%, var(--vp-c-bg));
}

.hybrid-stage {
	background: color-mix(in srgb, #22c55e 8%, var(--vp-c-bg));
}

.rerank-stage {
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

.formula-box,
.token,
.arrow {
	padding: 10px 12px;
	border-radius: 14px;
	background: var(--vp-c-bg);
	border: 1px solid var(--vp-c-divider);
	font-weight: 600;
}

.formula-box {
	margin-bottom: 10px;
	font-family: var(--vp-font-family-mono);
	font-size: 13px;
	word-break: break-word;
}

.arrow {
	background: transparent;
	border: none;
	font-size: 20px;
	display: flex;
	align-items: center;
	justify-content: center;
}

.token-row {
	display: flex;
	flex-wrap: wrap;
	gap: 8px;
	margin-bottom: 12px;
}

.stage-card p,
.aux-card p {
	margin: 0;
	line-height: 1.7;
	color: var(--vp-c-text-2);
}

.aux-grid {
	display: grid;
	grid-template-columns: repeat(3, minmax(0, 1fr));
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

@media (max-width: 1320px) {
	.hero-card,
	.layout-grid,
	.pipeline-row,
	.aux-grid {
		grid-template-columns: 1fr;
	}

	.arrow {
		display: none;
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
