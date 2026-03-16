<template>
	<div class="w2v-landscape">
		<div class="header">
			<div>
				<p class="eyebrow">Word2Vec 全文总览</p>
				<h3>从局部预测任务到静态词向量结构</h3>
				<p class="desc">
					这个总览图把当前文档中的核心内容压缩为一条主线：先定义局部上下文预测任务，再区分
					CBOW 与 Skip-gram 两种训练方向，随后引入近似训练技巧，最终得到可用于相似度、类比与召回系统的静态词向量表示。
				</p>
			</div>
			<div class="mode-switch">
				<button
					v-for="item in modes"
					:key="item.key"
					:type="item.key === activeMode ? 'button' : 'button'"
					:class="['mode-btn', { active: item.key === activeMode }]"
					@click="activeMode = item.key"
				>
					{{ item.label }}
				</button>
			</div>
		</div>

		<div class="overview-grid">
			<div class="main-panel">
				<template v-if="activeMode === 'overview'">
					<div class="pipeline">
						<div class="stage stage-problem">
							<h4>1. 任务定义</h4>
							<p>从中心词与局部上下文的共现关系中构造监督信号。</p>
							<ul>
								<li>中心词 $w_t$</li>
								<li>上下文集合 $C_t$</li>
								<li>窗口半径 $m$</li>
							</ul>
						</div>
						<div class="arrow">→</div>
						<div class="stage stage-architectures">
							<h4>2. 两大架构</h4>
							<p>CBOW 与 Skip-gram 分别对应两种方向相反的局部预测任务。</p>
							<div class="mini-branches">
								<span>CBOW：上下文 → 中心词</span>
								<span>Skip-gram：中心词 → 上下文</span>
							</div>
						</div>
						<div class="arrow">→</div>
						<div class="stage stage-training">
							<h4>3. 训练近似</h4>
							<p>用近似优化方法解决大词表 softmax 的计算瓶颈。</p>
							<div class="mini-branches">
								<span>Full Softmax</span>
								<span>Negative Sampling</span>
								<span>Hierarchical Softmax</span>
								<span>Subsampling</span>
							</div>
						</div>
						<div class="arrow">→</div>
						<div class="stage stage-geometry">
							<h4>4. 结果形态</h4>
							<p>训练结果表现为静态词向量及其几何结构。</p>
							<ul>
								<li>语义邻近</li>
								<li>类比位移</li>
								<li>余弦相似度</li>
							</ul>
						</div>
						<div class="arrow">→</div>
						<div class="stage stage-legacy">
							<h4>5. 方法遗产</h4>
							<p>作为 embedding 学习范式被扩展到推荐、图和召回系统。</p>
							<div class="mini-branches">
								<span>item2vec</span>
								<span>node2vec</span>
								<span>召回检索</span>
							</div>
						</div>
					</div>
				</template>

				<template v-else-if="activeMode === 'architectures'">
					<div class="compare-grid two-col">
						<div class="card cbow">
							<h4>CBOW</h4>
							<p class="formula">$P(w_t\mid C_t)$</p>
							<p>先聚合多个上下文词向量，再预测中间的中心词。</p>
							<ul>
								<li>更像完形填空</li>
								<li>训练通常更快</li>
								<li>高频模式更稳定</li>
								<li>上下文内部顺序被弱化</li>
							</ul>
						</div>
						<div class="card skipgram">
							<h4>Skip-gram</h4>
							<p class="formula">$P(w_o\mid w_t)$</p>
							<p>给定中心词，分别预测窗口中的多个上下文词。</p>
							<ul>
								<li>更像语境展开</li>
								<li>一次出现可生成更多监督样本</li>
								<li>对低频词更友好</li>
								<li>是 SGNS 的典型载体</li>
							</ul>
						</div>
					</div>
				</template>

				<template v-else-if="activeMode === 'training'">
					<div class="training-map">
						<div class="lane full-softmax">
							<h4>Full Softmax</h4>
							<p>完整条件概率模型，所有词参与归一化竞争。</p>
							<p class="metric">复杂度：$O(|V|)$</p>
						</div>
						<div class="lane neg-sampling">
							<h4>Negative Sampling</h4>
							<p>把全词表多分类改写为“真实词对 vs 噪声词对”的局部二分类。</p>
							<p class="metric">复杂度：$O(K)$</p>
						</div>
						<div class="lane hs">
							<h4>Hierarchical Softmax</h4>
							<p>将词表组织为树，在从根到叶的路径上逐步完成决策。</p>
							<p class="metric">复杂度：$O(\log |V|)$</p>
						</div>
						<div class="lane subsampling">
							<h4>Subsampling</h4>
							<p>丢弃部分高频功能词，降低训练成本并减少噪声主导。</p>
							<p class="metric">高频词越常见，越容易被丢弃</p>
						</div>
					</div>
				</template>

				<template v-else-if="activeMode === 'geometry'">
					<div class="geometry-grid">
						<div class="card">
							<h4>语义邻近</h4>
							<p>上下文相似的词在向量空间中彼此靠近。</p>
							<div class="chips">
								<span>猫 ↔ 狗</span>
								<span>医生 ↔ 护士</span>
								<span>苹果 ↔ 香蕉</span>
							</div>
						</div>
						<div class="card emphasis">
							<h4>类比位移</h4>
							<p class="formula">$q=v_b-v_a+v_c$</p>
							<p>某些稳定关系会在向量空间中表现为近似平行的位移方向。</p>
						</div>
						<div class="card">
							<h4>统计本质</h4>
							<p>SGNS 在理想化条件下可理解为对 shifted PMI 的隐式分解。</p>
							<p class="formula">$u_c^\top v_w \approx PMI(w,c)-\log k$</p>
						</div>
					</div>
				</template>

				<template v-else>
					<div class="legacy-grid">
						<div class="card">
							<h4>推荐系统</h4>
							<p>把商品、内容、广告 ID 当作“词”，把行为序列当作“句子”。</p>
						</div>
						<div class="card">
							<h4>图表示学习</h4>
							<p>通过随机游走把节点序列化，再交给 word2vec 风格框架学习表示。</p>
						</div>
						<div class="card">
							<h4>检索与召回</h4>
							<p>把离散对象压缩为向量后，可直接进入 ANN 检索与相似度召回链路。</p>
						</div>
						<div class="card warning">
							<h4>边界</h4>
							<p>它学习的是静态词向量与共现结构，不是完整上下文化语义理解。</p>
						</div>
					</div>
				</template>
			</div>

			<div class="side-panel">
				<h4>{{ sidePanel.title }}</h4>
				<p>{{ sidePanel.summary }}</p>
				<ul>
					<li v-for="point in sidePanel.points" :key="point">{{ point }}</li>
				</ul>
			</div>
		</div>
	</div>
</template>

<script setup lang="ts">
import { computed, ref } from 'vue'

type ModeKey = 'overview' | 'architectures' | 'training' | 'geometry' | 'legacy'

const activeMode = ref<ModeKey>('overview')

const modes = [
	{ key: 'overview', label: '总览主线' },
	{ key: 'architectures', label: '两大架构' },
	{ key: 'training', label: '训练近似' },
	{ key: 'geometry', label: '几何结果' },
	{ key: 'legacy', label: '方法遗产' },
] as const

const sidePanelMap: Record<ModeKey, { title: string; summary: string; points: string[] }> = {
	overview: {
		title: '如何阅读这篇文档',
		summary: '先看任务定义和两大架构，再理解大词表训练近似，最后回到几何结构与应用遗产。',
		points: [
			'第 1 节回答“为什么词能变成向量”',
			'第 2、3 节回答“模型怎么训练起来”',
			'第 4 节回答“训练后为什么会有几何结构”',
			'第 5 至 7 节回答“它今天还在哪里有用、边界在哪里”',
		],
	},
	architectures: {
		title: 'CBOW 与 Skip-gram 的关系',
		summary: '它们共享词向量内积打分框架，但监督方向相反：一个从上下文收缩到中心词，一个从中心词发散到上下文。',
		points: [
			'CBOW 更偏聚合与压缩',
			'Skip-gram 更偏展开与多样本监督',
			'两者本质上都在学习“什么词会一起出现”',
		],
	},
	training: {
		title: '为什么需要训练近似',
		summary: '真正的工程瓶颈不是向量本身，而是大词表条件下的归一化成本。',
		points: [
			'Full softmax 语义最完整，但最贵',
			'Negative Sampling 更像对比式表示学习',
			'Hierarchical Softmax 保留合法概率结构',
			'Subsampling 用于控制高频词噪声',
		],
	},
	geometry: {
		title: '训练结果为什么看起来有几何规律',
		summary: '因为模型优化的是词与上下文的相对兼容性，稳定共现结构会沉淀为向量邻近与方向关系。',
		points: [
			'邻近关系对应相似上下文分布',
			'类比位移反映稳定的统计差异模式',
			'SGNS 可从 shifted PMI 角度理解',
		],
	},
	legacy: {
		title: '为什么 word2vec 今天仍然重要',
		summary: '重要的不只是词向量本身，而是“局部共现 → embedding”的学习范式。',
		points: [
			'它适合超大规模离散序列',
			'输出向量天然适合相似度检索',
			'推荐、图、召回系统都仍在继承这套思路',
		],
	},
}

const sidePanel = computed(() => sidePanelMap[activeMode.value])
</script>

<style scoped>
.w2v-landscape {
	margin: 24px 0 32px;
	padding: 24px;
	border: 1px solid var(--vp-c-divider);
	border-radius: 20px;
	background: linear-gradient(180deg, rgba(46, 160, 67, 0.08), rgba(9, 105, 218, 0.04));
}

.header {
	display: flex;
	justify-content: space-between;
	gap: 20px;
	align-items: flex-start;
	margin-bottom: 20px;
	flex-wrap: wrap;
}

.eyebrow {
	margin: 0 0 6px;
	font-size: 12px;
	font-weight: 700;
	letter-spacing: 0.08em;
	text-transform: uppercase;
	color: var(--vp-c-brand-1);
}

.header h3 {
	margin: 0 0 10px;
	font-size: 24px;
}

.desc {
	margin: 0;
	max-width: 860px;
	line-height: 1.75;
	color: var(--vp-c-text-2);
}

.mode-switch {
	display: flex;
	gap: 10px;
	flex-wrap: wrap;
}

.mode-btn {
	border: 1px solid var(--vp-c-divider);
	background: var(--vp-c-bg-soft);
	border-radius: 999px;
	padding: 8px 14px;
	font-size: 13px;
	cursor: pointer;
	transition: all 0.2s ease;
}

.mode-btn.active {
	background: var(--vp-c-brand-1);
	color: white;
	border-color: var(--vp-c-brand-1);
}

.overview-grid {
	display: grid;
	grid-template-columns: minmax(0, 2.3fr) minmax(260px, 1fr);
	gap: 18px;
}

.main-panel,
.side-panel {
	padding: 18px;
	border-radius: 16px;
	background: rgba(255, 255, 255, 0.72);
	border: 1px solid rgba(148, 163, 184, 0.25);
	backdrop-filter: blur(6px);
}

.dark .main-panel,
.dark .side-panel {
	background: rgba(15, 23, 42, 0.72);
	border-color: rgba(148, 163, 184, 0.18);
}

.pipeline {
	display: flex;
	align-items: stretch;
	gap: 10px;
	flex-wrap: wrap;
}

.stage {
	flex: 1 1 180px;
	border-radius: 14px;
	padding: 16px;
	border: 1px solid rgba(148, 163, 184, 0.22);
}

.stage h4,
.card h4,
.side-panel h4,
.lane h4 {
	margin: 0 0 10px;
	font-size: 16px;
}

.stage p,
.card p,
.side-panel p,
.lane p {
	margin: 0 0 10px;
	line-height: 1.7;
	color: var(--vp-c-text-2);
}

.stage ul,
.side-panel ul {
	margin: 0;
	padding-left: 18px;
}

.mini-branches {
	display: flex;
	flex-direction: column;
	gap: 8px;
	margin-top: 10px;
}

.mini-branches span,
.chips span {
	display: inline-flex;
	align-items: center;
	padding: 6px 10px;
	border-radius: 999px;
	background: rgba(59, 130, 246, 0.1);
	font-size: 12px;
	color: var(--vp-c-text-1);
	width: fit-content;
}

.arrow {
	display: flex;
	align-items: center;
	justify-content: center;
	font-size: 22px;
	font-weight: 700;
	color: var(--vp-c-text-3);
	padding: 0 4px;
}

.compare-grid,
.geometry-grid,
.legacy-grid {
	display: grid;
	gap: 14px;
}

.two-col {
	grid-template-columns: repeat(2, minmax(0, 1fr));
}

.geometry-grid {
	grid-template-columns: repeat(3, minmax(0, 1fr));
}

.legacy-grid {
	grid-template-columns: repeat(2, minmax(0, 1fr));
}

.card,
.lane {
	padding: 16px;
	border-radius: 14px;
	border: 1px solid rgba(148, 163, 184, 0.22);
	background: rgba(255, 255, 255, 0.5);
}

.dark .card,
.dark .lane {
	background: rgba(2, 6, 23, 0.28);
}

.formula,
.metric {
	font-weight: 700;
	color: var(--vp-c-brand-1) !important;
}

.training-map {
	display: grid;
	grid-template-columns: repeat(2, minmax(0, 1fr));
	gap: 14px;
}

.chips {
	display: flex;
	gap: 8px;
	flex-wrap: wrap;
	margin-top: 8px;
}

.stage-problem {
	background: rgba(34, 197, 94, 0.08);
}

.stage-architectures,
.cbow {
	background: rgba(59, 130, 246, 0.08);
}

.stage-training,
.neg-sampling {
	background: rgba(245, 158, 11, 0.08);
}

.stage-geometry,
.emphasis {
	background: rgba(168, 85, 247, 0.08);
}

.stage-legacy,
.warning {
	background: rgba(239, 68, 68, 0.08);
}

@media (max-width: 1100px) {
	.overview-grid,
	.two-col,
	.geometry-grid,
	.legacy-grid,
	.training-map {
		grid-template-columns: 1fr;
	}
}
</style>
