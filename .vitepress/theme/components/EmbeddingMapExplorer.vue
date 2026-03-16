<template>
	<div class="embedding-map-explorer">
		<div class="hero-card">
			<div class="hero-text">
				<p class="eyebrow">Embedding Map Explorer</p>
				<h2>从离散符号到连续表示空间的结构图</h2>
				<p class="summary">
					该图把 Embedding 拆成「离散对象 → one-hot / 索引 → 查表矩阵 → 连续向量 → 相似度与下游任务」的主链路，同时补上静态表示与上下文化表示的分支，帮助读者先建立整体地图，再进入细节推导。
				</p>
			</div>
			<div class="legend-grid">
				<div class="legend-item discrete">离散对象 / ID</div>
				<div class="legend-item table">Embedding 矩阵 $E$</div>
				<div class="legend-item vector">连续向量 $x_i$</div>
				<div class="legend-item task">几何关系与任务用途</div>
			</div>
		</div>

		<div class="controls-card">
			<div class="toggle-group" role="tablist" aria-label="Embedding 视图切换">
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
					<div class="stage-card discrete-stage" :class="{ focus: activeMode === 'overview' || activeMode === 'discrete' }">
						<div class="stage-head">
							<h3>1. 离散对象</h3>
							<span>词、子词、用户、物品、标签</span>
						</div>
						<div class="token-row">
							<span class="token">猫</span>
							<span class="token">狗</span>
							<span class="token">商品A</span>
							<span class="token">用户42</span>
						</div>
						<p>这些对象原本只是符号或 ID，本身没有可直接参与数值计算的结构。</p>
					</div>

					<div class="stage-card table-stage" :class="{ focus: activeMode === 'overview' || activeMode === 'lookup' }">
						<div class="stage-head">
							<h3>2. 查表映射</h3>
							<span>one-hot / index → row lookup</span>
						</div>
						<div class="formula-box">$x_i=E^\top e_i$</div>
						<div class="matrix-box">
							<span>$E$</span>
							<span>每一行对应一个对象的可学习向量</span>
						</div>
						<p>数学上可写成矩阵乘法，工程上通常等价于按索引取 embedding 矩阵中的某一行。</p>
					</div>

					<div class="stage-card vector-stage" :class="{ focus: activeMode === 'overview' || activeMode === 'geometry' }">
						<div class="stage-head">
							<h3>3. 连续向量空间</h3>
							<span>低维、稠密、可比较</span>
						</div>
						<div class="token-row">
							<span class="state-chip">$x_{cat}$</span>
							<span class="state-chip">$x_{dog}$</span>
							<span class="state-chip">$x_{apple}$</span>
						</div>
						<div class="formula-box">$s(i,j)=x_i^\top x_j$</div>
						<p>相似对象在向量空间中更靠近，不同对象也可以通过方向与距离表达关系。</p>
					</div>

					<div class="stage-card task-stage" :class="{ focus: activeMode === 'overview' || activeMode === 'task' }">
						<div class="stage-head">
							<h3>4. 下游使用</h3>
							<span>分类、生成、检索、推荐</span>
						</div>
						<div class="formula-box">$h=f(Wx_i+b)$</div>
						<p>Embedding 自身通常只是表示层，真正的语义结构由训练目标与下游任务共同塑造。</p>
					</div>
				</div>

				<div class="aux-grid">
					<div class="aux-card" :class="{ focus: activeMode === 'static' }">
						<h4>静态 Embedding</h4>
						<p>同一个对象无论出现在什么上下文中，都对应同一个固定向量，例如 word2vec、GloVe、FastText。</p>
					</div>
					<div class="aux-card" :class="{ focus: activeMode === 'contextual' }">
						<h4>上下文化 Embedding</h4>
						<p>同一个词在不同上下文下生成不同表示，例如 ELMo、BERT、GPT 的隐藏状态表示。</p>
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

type ModeKey = 'overview' | 'discrete' | 'lookup' | 'geometry' | 'task' | 'static' | 'contextual'

const activeMode = ref<ModeKey>('overview')

const modes: { key: ModeKey; label: string; desc: string }[] = [
	{ key: 'overview', label: '总览', desc: '先看 Embedding 从离散对象到连续向量、再到下游任务的完整主链。' },
	{ key: 'discrete', label: '离散对象', desc: '聚焦 Embedding 为什么必须从“不可比较的符号”起步。' },
	{ key: 'lookup', label: '查表机制', desc: '聚焦 one-hot / 索引如何映射到 embedding 矩阵中的某一行。' },
	{ key: 'geometry', label: '几何结构', desc: '聚焦相似度、距离与方向关系如何赋予向量空间语义。' },
	{ key: 'task', label: '任务作用', desc: '聚焦 Embedding 为什么通常只是表示层，而不是完整任务模型。' },
	{ key: 'static', label: '静态表示', desc: '聚焦固定向量式 embedding 的优点与边界。' },
	{ key: 'contextual', label: '上下文化', desc: '聚焦同一 token 在不同上下文下如何生成不同表示。' }
]

const panels = {
	overview: {
		title: '总览：Embedding 的本质是“把符号放进可学习的几何空间”',
		lead: '它不只是压缩维度，更关键的是把“相似、可替换、可组合”这些关系变成向量空间中的可计算结构。',
		points: [
			'离散对象本身无法直接比较或优化，Embedding 先把它们映射为实数向量。',
			'这些向量通常由某个任务目标训练出来，因此几何结构具有任务相关性。',
			'从词向量到推荐 ID 表示，再到多模态特征，Embedding 都是基础表示层。'
		],
		rows: [
			{ label: '输入', value: '离散符号、ID 或类别标签' },
			{ label: '映射', value: '通过可学习矩阵把离散对象变成稠密向量' },
			{ label: '输出', value: '供相似度计算、下游神经网络或检索系统使用' }
		]
	},
	discrete: {
		title: '离散对象：符号本身没有可比较的几何结构',
		lead: '“猫”和“狗”在原始符号层只是两个不同字符串，计算机无法仅凭符号本身知道它们语义接近。',
		points: [
			'one-hot 虽然能区分对象身份，但无法表达语义近邻关系。',
			'这就是为何需要先把符号投影到连续空间。',
			'Embedding 解决的首先是“如何表示”，其次才是“如何利用”。'
		],
		rows: [
			{ label: '符号层', value: '只能告诉模型“是不是同一个对象”' },
			{ label: '缺失项', value: '缺少相似性、层次性、可替换性等结构' },
			{ label: '目标', value: '把离散对象变成可比较、可优化的数值表示' }
		]
	},
	lookup: {
		title: '查表机制：Embedding 在工程上常常就是“按索引取一行参数”',
		lead: '虽然公式可以写成矩阵乘法，但真正实现时常直接通过 ID 查询 embedding 矩阵中的对应行。',
		points: [
			'one-hot 只在数学表达上帮助我们理解“为什么等价于取一行”。',
			'每个对象都在矩阵 $E$ 中拥有自己的向量参数。',
			'训练时，真正被访问到的行会根据损失函数接收梯度并更新。'
		],
		rows: [
			{ label: '$e_i$', value: '对象 $i$ 的 one-hot 向量' },
			{ label: '$E$', value: 'embedding 矩阵，每一行对应一个对象向量' },
			{ label: '$x_i$', value: '对象 $i$ 被查出来的连续表示' }
		]
	},
	geometry: {
		title: '几何结构：相似度不再是“标签是否相等”，而是向量之间如何相互作用',
		lead: 'Embedding 的真正价值，在于向量空间中的距离、方向与点积开始承载任务所需的关系模式。',
		points: [
			'相似词、相似用户、相似商品可以在空间中彼此靠近。',
			'向量的方向差有时还能表达某种关系模式。',
			'这使得相似性检索、聚类、近邻迁移和对比学习都成为可能。'
		],
		rows: [
			{ label: '点积', value: '常用于衡量两个表示在当前空间中的匹配程度' },
			{ label: '余弦相似度', value: '常用于关注方向而弱化向量模长影响' },
			{ label: '几何意义', value: '距离近往往表示语义、功能或行为更接近' }
		]
	},
	task: {
		title: '任务作用：Embedding 通常只是入口层，真正的语义结构由训练目标塑造',
		lead: '单独看一个 embedding 向量本身，常常无法解释全部意义；它最终学成什么样，取决于模型被要求完成什么任务。',
		points: [
			'语言模型会把有助于预测上下文的结构写进 embedding。',
			'推荐系统会把用户—物品交互模式写进 embedding。',
			'分类、检索、排序、多模态对齐都会塑造不同的几何空间。'
		],
		rows: [
			{ label: '语言建模', value: '更关注上下文预测与词间分布结构' },
			{ label: '推荐 / 检索', value: '更关注用户、物品、查询、文档之间的匹配关系' },
			{ label: '结论', value: 'Embedding 的几何形状总是受目标函数牵引' }
		]
	},
	static: {
		title: '静态表示：同一个对象始终对应同一个固定向量',
		lead: '静态 embedding 高效、稳定、易复用，但在一词多义和复杂上下文依赖上能力有限。',
		points: [
			'word2vec、GloVe、FastText 都属于典型静态表示路线。',
			'同一个词无论出现在哪个句子中，通常都使用同一个向量。',
			'优点是简单高效，缺点是难以区分语境差异。'
		],
		rows: [
			{ label: '优点', value: '易训练、易缓存、部署代价低' },
			{ label: '局限', value: '难处理多义词和复杂语境差异' },
			{ label: '代表', value: 'word2vec / GloVe / FastText' }
		]
	},
	contextual: {
		title: '上下文化表示：同一词在不同语境中生成不同向量',
		lead: '上下文化 embedding 把词表示从“查表参数”推进到“上下文驱动的动态状态”。',
		points: [
			'同一个 token 在不同句子里可以获得不同表示。',
			'这使模型能区分一词多义、角色变化和更细粒度的语境差异。',
			'ELMo、BERT、GPT 隐层表示都推动了这一路线成为主流。'
		],
		rows: [
			{ label: '输入条件', value: '词本身 + 周围上下文共同决定表示' },
			{ label: '优势', value: '能表达动态语义与复杂语境差异' },
			{ label: '代价', value: '通常更重、更依赖上下文模型主干' }
		]
	}
} as const

const activeDescription = computed(() => modes.find((mode) => mode.key === activeMode.value)?.desc ?? '')
const activePanel = computed(() => panels[activeMode.value])
</script>

<style scoped>
.embedding-map-explorer {
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

.legend-item.discrete {
	background: color-mix(in srgb, #8b5cf6 14%, var(--vp-c-bg));
	border-color: color-mix(in srgb, #8b5cf6 32%, transparent);
}

.legend-item.table {
	background: color-mix(in srgb, #06b6d4 14%, var(--vp-c-bg));
	border-color: color-mix(in srgb, #06b6d4 32%, transparent);
}

.legend-item.vector {
	background: color-mix(in srgb, #22c55e 14%, var(--vp-c-bg));
	border-color: color-mix(in srgb, #22c55e 32%, transparent);
}

.legend-item.task {
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

.discrete-stage {
	background: color-mix(in srgb, #8b5cf6 8%, var(--vp-c-bg));
}

.table-stage {
	background: color-mix(in srgb, #06b6d4 8%, var(--vp-c-bg));
}

.vector-stage {
	background: color-mix(in srgb, #22c55e 8%, var(--vp-c-bg));
}

.task-stage {
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
.formula-box,
.matrix-box {
	padding: 10px 12px;
	border-radius: 14px;
	background: var(--vp-c-bg);
	border: 1px solid var(--vp-c-divider);
	font-weight: 600;
}

.state-chip {
	background: color-mix(in srgb, #22c55e 10%, var(--vp-c-bg));
}

.formula-box,
.matrix-box {
	margin-bottom: 10px;
	font-family: var(--vp-font-family-mono);
	font-size: 13px;
	word-break: break-word;
}

.matrix-box span:first-child {
	display: block;
	font-weight: 700;
	margin-bottom: 6px;
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
