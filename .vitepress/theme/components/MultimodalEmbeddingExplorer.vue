<template>
	<div class="multimodal-embedding-explorer">
		<div class="hero-card">
			<div class="hero-text">
				<p class="eyebrow">Multimodal Embedding Explorer</p>
				<h2>多模态对齐与连接器的整体结构图</h2>
				<p class="summary">
					该图把第 5 节的主线统一起来：不同模态如何分别编码、如何投影到同一向量空间、如何通过对比学习对齐，以及连接器怎样把视觉表征继续桥接到语言模型的 token 空间。
				</p>
			</div>
			<div class="legend-grid">
				<div class="legend-item encoder">模态编码器</div>
				<div class="legend-item align">共享向量空间</div>
				<div class="legend-item objective">对比学习目标</div>
				<div class="legend-item connector">连接器 / 语言接口</div>
			</div>
		</div>

		<div class="controls-card">
			<div class="toggle-group" role="tablist" aria-label="多模态 embedding 视图切换">
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
				<div class="top-grid">
					<div class="stage-card encoder-stage" :class="{ focus: activeMode === 'overview' || activeMode === 'encoders' }">
						<div class="stage-head">
							<h3>1. 模态编码器</h3>
							<span>各自读懂本模态</span>
						</div>
						<div class="token-row">
							<span class="token">图像塔</span>
							<span class="token">文本塔</span>
							<span class="token">音频塔</span>
						</div>
						<p>不同模态先由各自擅长的编码器抽取内部特征，例如 ViT、Transformer、音频编码器等。</p>
					</div>

					<div class="stage-card align-stage" :class="{ focus: activeMode === 'overview' || activeMode === 'space' }">
						<div class="stage-head">
							<h3>2. 共享向量空间</h3>
							<span>跨模态可比较</span>
						</div>
						<div class="formula-box">$z_v=P_v f_{img}(I),\ z_t=P_t f_{text}(T)$</div>
						<p>投影层把不同模态压到同一维度，使图像、文本、音频等在同一空间里可以直接比较。</p>
					</div>

					<div class="stage-card objective-stage" :class="{ focus: activeMode === 'overview' || activeMode === 'objective' }">
						<div class="stage-head">
							<h3>3. 对比学习</h3>
							<span>匹配对靠近，非匹配对分离</span>
						</div>
						<div class="formula-box">$S_{ij}=\operatorname{sim}(z_{v,i},z_{t,j})$</div>
						<div class="formula-box">$\mathcal{L}_{InfoNCE}$</div>
						<p>训练目标不是单模态分类，而是塑造“谁和谁应该靠近”的统一几何关系。</p>
					</div>
				</div>

				<div class="connector-card" :class="{ focus: activeMode === 'connector' || activeMode === 'generation' }">
					<div class="connector-head">
						<h3>4. 连接器：从视觉 embedding 到语言 token</h3>
						<span>对齐之后继续进入多模态大模型</span>
					</div>
					<div class="connector-flow">
						<div class="flow-node visual">视觉特征 $h_v$</div>
						<div class="flow-arrow">→</div>
						<div class="flow-node bridge">MLP / Q-Former / Resampler</div>
						<div class="flow-arrow">→</div>
						<div class="flow-node llm">语言 token 空间 $z$</div>
					</div>
					<p>连接器不是重新理解整张图像，而是把视觉侧的连续特征翻译成语言模型更容易消费的 token 形式。</p>
				</div>

				<div class="aux-grid">
					<div class="aux-card" :class="{ focus: activeMode === 'connector' }">
						<h4>连接器分层</h4>
						<p>从简单线性投影到 Q-Former / Resampler，本质差异在于是否主动完成压缩、筛选与重组。</p>
					</div>
					<div class="aux-card" :class="{ focus: activeMode === 'generation' }">
						<h4>对齐后的用途</h4>
						<p>统一空间不仅能做检索，还能作为检测、分类、问答与生成模型的跨模态语义接口。</p>
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
						<span>模块</span>
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

type ModeKey = 'overview' | 'encoders' | 'space' | 'objective' | 'connector' | 'generation'

const activeMode = ref<ModeKey>('overview')

const modes: { key: ModeKey; label: string; desc: string }[] = [
	{ key: 'overview', label: '总览', desc: '先看多模态 embedding 从各自编码到统一空间、再到语言接口的完整主链。' },
	{ key: 'encoders', label: '模态编码器', desc: '聚焦不同模态如何先被各自擅长的编码器处理。' },
	{ key: 'space', label: '共享空间', desc: '聚焦为什么跨模态检索必须先把不同模态投影到同一几何空间。' },
	{ key: 'objective', label: '对比目标', desc: '聚焦 InfoNCE 一类目标如何塑造“匹配对靠近、不匹配对远离”的几何关系。' },
	{ key: 'connector', label: '连接器', desc: '聚焦视觉表征如何继续桥接到语言模型 token 空间。' },
	{ key: 'generation', label: '生成接口', desc: '聚焦统一向量空间为什么能继续服务问答、生成与多模态推理。' }
]

const panels = {
	overview: {
		title: '总览：多模态 embedding 的核心是“先统一几何语言，再统一任务接口”',
		lead: '多模态系统不是直接让图像和文本互相替代，而是先让它们进入同一个可比较的连续空间。',
		points: [
			'每种模态先用自己的编码器提取表示，再经过投影层进入共享空间。',
			'对比学习负责塑造“匹配对接近、不匹配对远离”的跨模态几何结构。',
			'若还要接入语言模型，则需要连接器把视觉表示转换为语言 token 风格的输入。'
		],
		rows: [
			{ label: '编码', value: '先在各自模态内部形成高质量表征' },
			{ label: '对齐', value: '再把不同模态压到统一向量空间' },
			{ label: '扩展', value: '进一步通过连接器接入语言或生成模型' }
		]
	},
	encoders: {
		title: '模态编码器：每个模态都先在本领域被理解',
		lead: '统一空间不是从原始像素和原始文本直接生长出来的，而是建立在各模态内部表征之上。',
		points: [
			'图像常由 ResNet / ViT 之类视觉编码器处理。',
			'文本常由 Transformer 编码器处理。',
			'音频、视频、深度等模态也需要各自的专用编码器。'
		],
		rows: [
			{ label: '图像塔', value: '提取视觉纹理、形状、空间布局等特征' },
			{ label: '文本塔', value: '提取词序、句法与语义特征' },
			{ label: '其他模态', value: '按本模态规律先形成内部高层表示' }
		]
	},
	space: {
		title: '共享空间：跨模态比较必须先把不同模态说成“同一种几何语言”',
		lead: '只有当不同模态的表示被压到同一维度、同一度量下，跨模态检索与匹配才真正有意义。',
		points: [
			'投影层的作用是把不同编码器输出变成可直接比较的向量。',
			'统一空间中的接近关系代表跨模态语义一致性。',
			'这一步让“猫的文字”和“猫的图片”可以直接做最近邻比较。'
		],
		rows: [
			{ label: '$z_v$', value: '图像经过编码与投影后的共享空间表示' },
			{ label: '$z_t$', value: '文本经过编码与投影后的共享空间表示' },
			{ label: '意义', value: '不同模态第一次获得可直接比较的坐标系' }
		]
	},
	objective: {
		title: '对比目标：统一空间的几何不是天然存在，而是靠训练目标挤压出来的',
		lead: 'InfoNCE 一类目标通过 batch 内竞争，让匹配项成为高相似度对角线，其他组合作为负样本被推远。',
		points: [
			'正样本负责定义“哪些跨模态对象应该靠近”。',
			'负样本负责提升空间分辨率，防止一切都挤在一起。',
			'困难负样本越接近语义边界，越能迫使模型学习细粒度对齐。'
		],
		rows: [
			{ label: '相似度矩阵', value: '刻画一个 batch 中所有跨模态两两组合' },
			{ label: '正样本', value: '匹配的图文 / 图音 / 文音对' },
			{ label: '负样本', value: '不匹配但可能相近的候选，用于提高分辨率' }
		]
	},
	connector: {
		title: '连接器：共享空间解决“能比较”，连接器解决“能和 LLM 协作”',
		lead: '对齐后的视觉向量并不天然适合语言模型消费，因此还需要一层桥接结构把它们翻译成 token 风格输入。',
		points: [
			'简单方案是线性层或 MLP 直接投影。',
			'更复杂方案如 Q-Former / Resampler 能主动筛选和压缩视觉信息。',
			'连接器决定了多模态模型在 token 长度、信息密度与训练成本之间的权衡。'
		],
		rows: [
			{ label: 'MLP', value: '简单直接，适合快速对齐与实现' },
			{ label: 'Q-Former', value: '以可学习查询主动抽取少量高价值视觉 token' },
			{ label: 'Resampler', value: '把大量视觉特征压缩为固定长度潜变量' }
		]
	},
	generation: {
		title: '生成接口：统一空间一旦形成，就不只是检索工具，而是跨模态语义接口',
		lead: '对齐后的 embedding 可以继续作为问答、检测、生成与指令跟随模型的条件输入。',
		points: [
			'文本可以检索图像，图像也能检索文本或视频。',
			'音频或图像 embedding 还可以作为检测、分类或生成模型的条件信号。',
			'这说明 embedding 已经从“表示层”扩展成“系统接口层”。'
		],
		rows: [
			{ label: '检索', value: '跨模态最近邻搜索' },
			{ label: '理解', value: '图像问答、匹配、分类、检测' },
			{ label: '生成', value: '把视觉或音频条件输入语言模型或扩散模型' }
		]
	}
} as const

const activeDescription = computed(() => modes.find((mode) => mode.key === activeMode.value)?.desc ?? '')
const activePanel = computed(() => panels[activeMode.value])
</script>

<style scoped>
.multimodal-embedding-explorer {
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

.legend-item.encoder {
	background: color-mix(in srgb, #06b6d4 14%, var(--vp-c-bg));
	border-color: color-mix(in srgb, #06b6d4 32%, transparent);
}

.legend-item.align {
	background: color-mix(in srgb, #8b5cf6 14%, var(--vp-c-bg));
	border-color: color-mix(in srgb, #8b5cf6 32%, transparent);
}

.legend-item.objective {
	background: color-mix(in srgb, #22c55e 14%, var(--vp-c-bg));
	border-color: color-mix(in srgb, #22c55e 32%, transparent);
}

.legend-item.connector {
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
	grid-template-columns: minmax(0, 1.72fr) minmax(300px, 0.9fr);
	gap: 20px;
}

.diagram-card,
.insight-card {
	padding: 22px;
}

.top-grid {
	display: grid;
	grid-template-columns: repeat(3, minmax(0, 1fr));
	gap: 16px;
}

.stage-card,
.connector-card,
.aux-card {
	padding: 18px;
	border-radius: 18px;
	border: 1px solid var(--vp-c-divider);
	background: color-mix(in srgb, var(--vp-c-bg-soft) 84%, #ffffff 16%);
}

.stage-card.focus,
.connector-card.focus,
.aux-card.focus {
	box-shadow: 0 0 0 3px color-mix(in srgb, var(--vp-c-brand-1) 18%, transparent);
}

.encoder-stage {
	background: color-mix(in srgb, #06b6d4 8%, var(--vp-c-bg));
}

.align-stage {
	background: color-mix(in srgb, #8b5cf6 8%, var(--vp-c-bg));
}

.objective-stage {
	background: color-mix(in srgb, #22c55e 8%, var(--vp-c-bg));
}

.connector-card {
	margin-top: 16px;
	background: color-mix(in srgb, #f59e0b 8%, var(--vp-c-bg));
}

.stage-head,
.connector-head {
	display: flex;
	justify-content: space-between;
	align-items: baseline;
	gap: 10px;
	margin-bottom: 14px;
}

.stage-head h3,
.connector-head h3 {
	margin: 0;
	font-size: 20px;
}

.stage-head span,
.connector-head span {
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
.formula-box,
.flow-node,
.flow-arrow {
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

.connector-flow {
	display: grid;
	grid-template-columns: minmax(0, 1fr) auto minmax(0, 1.2fr) auto minmax(0, 1fr);
	gap: 12px;
	align-items: center;
	margin-bottom: 12px;
}

.flow-node.visual {
	background: color-mix(in srgb, #06b6d4 10%, var(--vp-c-bg));
}

.flow-node.bridge {
	background: color-mix(in srgb, #f59e0b 10%, var(--vp-c-bg));
}

.flow-node.llm {
	background: color-mix(in srgb, #8b5cf6 10%, var(--vp-c-bg));
}

.flow-arrow {
	background: transparent;
	border: none;
	font-size: 18px;
}

.stage-card p,
.connector-card p,
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
	.top-grid,
	.aux-grid,
	.connector-flow {
		grid-template-columns: 1fr;
	}

	.flow-arrow {
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
