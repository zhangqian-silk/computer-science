<template>
	<div class="seq2seq-workflow-explorer">
		<div class="hero-card">
			<div class="hero-text">
				<p class="eyebrow">Seq2Seq Workflow Explorer</p>
				<h2>编码器—解码器到 Attention 的整体路线图</h2>
				<p class="summary">
					该图把 Seq2Seq 的核心链路拆成「输入编码 → 条件传递 → 自回归解码 → 动态对齐 → 输出搜索」五个层次，帮助读者先建立完整任务结构，再进入固定上下文、attention 与 Transformer 的细节演化。
				</p>
			</div>
			<div class="legend-grid">
				<div class="legend-item encoder">编码器表征：$h_i$</div>
				<div class="legend-item context">上下文条件：$c / c_t$</div>
				<div class="legend-item decoder">解码状态：$s_t$</div>
				<div class="legend-item search">解码与搜索策略</div>
			</div>
		</div>

		<div class="controls-card">
			<div class="toggle-group" role="tablist" aria-label="Seq2Seq 视图切换">
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
					<div class="stage-card encoder-stage" :class="{ focus: activeMode === 'overview' || activeMode === 'encoder' }">
						<div class="stage-head">
							<h3>1. Encoder</h3>
							<span>读取输入序列</span>
						</div>
						<div class="token-row">
							<span class="token">$x_1$</span>
							<span class="token">$x_2$</span>
							<span class="token">$\dots$</span>
							<span class="token">$x_n$</span>
						</div>
						<div class="state-row">
							<span class="state-chip">$h_1$</span>
							<span class="state-chip">$h_2$</span>
							<span class="state-chip">$\dots$</span>
							<span class="state-chip">$h_n$</span>
						</div>
						<p>把输入序列编码成一串可供后续读取的表征。</p>
					</div>

					<div class="stage-card context-stage" :class="{ focus: activeMode === 'overview' || activeMode === 'context' || activeMode === 'attention' }">
						<div class="stage-head">
							<h3>2. 条件传递</h3>
							<span>固定或动态上下文</span>
						</div>
						<div class="context-box fixed">固定长度版本：$c=h_n$</div>
						<div class="context-box dynamic">Attention 版本：$c_t=\sum_i \alpha_{t,i}h_i$</div>
						<p>从“整句压缩成一个向量”演化到“按输出步动态读取输入”。</p>
					</div>

					<div class="stage-card decoder-stage" :class="{ focus: activeMode === 'overview' || activeMode === 'decoder' }">
						<div class="stage-head">
							<h3>3. Decoder</h3>
							<span>逐步生成输出</span>
						</div>
						<div class="token-row">
							<span class="token bos">&lt;BOS&gt;</span>
							<span class="token">$y_1$</span>
							<span class="token">$y_2$</span>
							<span class="token">$\dots$</span>
							<span class="token eos">&lt;EOS&gt;</span>
						</div>
						<div class="state-row">
							<span class="state-chip">$s_1$</span>
							<span class="state-chip">$s_2$</span>
							<span class="state-chip">$\dots$</span>
							<span class="state-chip">$s_t$</span>
						</div>
						<p>输出通过自回归方式逐步生成，每一步都依赖输入条件与既往输出前缀。</p>
					</div>
				</div>

				<div class="aux-grid">
					<div class="aux-card" :class="{ focus: activeMode === 'attention' }">
						<h4>4. Attention 与对齐</h4>
						<p>解码步 $t$ 用匹配分数 $e_{t,i}$ 和权重 $\alpha_{t,i}$ 在输入状态序列上做软选择。</p>
						<code>$\alpha_{t,i}=\frac{\exp(e_{t,i})}{\sum_j \exp(e_{t,j})}$</code>
					</div>
					<div class="aux-card" :class="{ focus: activeMode === 'search' }">
						<h4>5. Decoding 与搜索</h4>
						<p>训练时常用教师强制，推理时可用 Greedy 或 Beam Search 近似寻找高概率输出序列。</p>
						<code v-pre>$S(Y_{1:t})=S(Y_{1:t-1})+\log P(y_t|y_{&lt;t},X)$</code>
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

type ModeKey = 'overview' | 'encoder' | 'context' | 'decoder' | 'attention' | 'search'

const activeMode = ref<ModeKey>('overview')

const modes: { key: ModeKey; label: string; desc: string }[] = [
	{ key: 'overview', label: '总览', desc: '先看 Seq2Seq 从输入编码到输出生成的完整数据流。' },
	{ key: 'encoder', label: '编码器', desc: '聚焦输入如何被编码成一串隐藏状态，供后续解码器读取。' },
	{ key: 'context', label: '条件传递', desc: '聚焦固定上下文向量与动态上下文向量之间的演化关系。' },
	{ key: 'decoder', label: '解码器', desc: '聚焦输出序列如何在输入条件下自回归生成。' },
	{ key: 'attention', label: '注意力', desc: '聚焦解码步如何对输入位置做动态对齐，而不是只依赖单一上下文向量。' },
	{ key: 'search', label: '搜索策略', desc: '聚焦训练与推理的差异，以及 Greedy / Beam Search 的角色。' }
]

const panels = {
	overview: {
		title: '总览：Seq2Seq 是条件序列生成框架，而不是单一网络层',
		lead: '它的核心不是某个特定算子，而是“先编码输入，再在输入条件下逐步生成输出”的统一任务组织方式。',
		points: [
			'输入和输出都可以是变长序列，因此它天然适合翻译、摘要、对话与 ASR 等任务。',
			'Encoder 负责建模输入，Decoder 负责在条件下逐步生成目标序列。',
			'Attention 与 Transformer 改变了底层实现，但没有改变这条主线。'
		],
		rows: [
			{ label: 'Encoder', value: '把输入序列转换为可读表示' },
			{ label: 'Context', value: '把输入条件传递给解码器' },
			{ label: 'Decoder', value: '逐步生成目标输出并决定何时结束' }
		]
	},
	encoder: {
		title: '编码器：把输入变成后续可以读取的状态序列',
		lead: '经典 RNN/LSTM 编码器按顺序读取输入，每一步都把新信息压入状态。',
		points: [
			'原始 Seq2Seq 常把最后状态直接当作整句摘要。',
			'更现代的视角则把全部 $h_i$ 保留下来，供 attention 动态读取。',
			'编码器的职责不是输出答案，而是准备条件表示。'
		],
		rows: [
			{ label: '$x_t$', value: '输入序列在位置 $t$ 的向量表示' },
			{ label: '$h_t$', value: '编码器读到位置 $t$ 后的隐藏状态' },
			{ label: '状态序列', value: 'attention 版本真正读取的是整串 $h_i$，而不是仅最后一步' }
		]
	},
	context: {
		title: '条件传递：从固定长度上下文瓶颈走向动态读取',
		lead: 'Seq2Seq 的关键问题之一，是解码器究竟如何接收输入条件。',
		points: [
			'固定长度版本把整句压进一个向量 $c$，长句时容易形成瓶颈。',
			'Attention 版本让每个解码步都能动态生成自己的上下文 $c_t$。',
			'这一步的演化，直接推动了神经机器翻译质量的大幅提升。'
		],
		rows: [
			{ label: '$c$', value: '固定长度上下文，原始 Seq2Seq 的整句摘要' },
			{ label: '$c_t$', value: '解码步 $t$ 专属的动态上下文向量' },
			{ label: '瓶颈变化', value: '从“句子压缩”走向“按需读取”' }
		]
	},
	decoder: {
		title: '解码器：在输入条件下做自回归生成',
		lead: '解码器既不是简单分类器，也不是无条件语言模型，而是“条件语言模型”。',
		points: [
			'每一步生成都依赖输入条件 $X$ 与已生成前缀 $y_{<t}$。',
			'起始通常由 `<BOS>` 提供，终止由 `<EOS>` 决定。',
			'这使得 Seq2Seq 可以自然处理输入输出长度不一致。'
		],
		rows: [
			{ label: '$s_t$', value: '解码步 $t$ 的隐藏状态' },
			{ label: '$y_{<t}$', value: '当前时刻之前已经生成的输出前缀' },
			{ label: '终止机制', value: '通过生成 `<EOS>` 主动结束输出' }
		]
	},
	attention: {
		title: '注意力：输出步对输入位置做动态对齐',
		lead: 'Attention 让解码器在每一步都能问：“我现在应该更关注输入的哪一部分？”',
		points: [
			'匹配分数 $e_{t,i}$ 衡量当前解码状态与输入位置 $i$ 的相关性。',
			'注意力权重 $\alpha_{t,i}$ 把这种相关性归一化成概率分布。',
			'动态上下文 $c_t$ 因此不再固定，而会随输出步变化。'
		],
		rows: [
			{ label: '$e_{t,i}$', value: '解码步 $t$ 对输入位置 $i$ 的匹配分数' },
			{ label: '$\alpha_{t,i}$', value: '归一化后的注意力权重' },
			{ label: '$c_t$', value: '按当前解码需要动态汇聚得到的上下文向量' }
		]
	},
	search: {
		title: '搜索策略：训练目标与推理目标并不完全相同',
		lead: '训练时通常知道真实前缀，推理时则要在巨大的输出空间里近似搜索高概率序列。',
		points: [
			'教师强制让训练稳定，但会带来训练—推理分布不一致问题。',
			'Greedy Search 简单快速，但容易局部最优。',
			'Beam Search 通过保留多个候选前缀改善全局输出质量。'
		],
		rows: [
			{ label: 'Teacher Forcing', value: '训练时用真实前缀驱动下一步预测' },
			{ label: 'Greedy', value: '每一步只保留当前最优 token' },
			{ label: 'Beam Search', value: '保留多个候选路径近似全局最优' }
		]
	}
} as const

const activeDescription = computed(() => modes.find((mode) => mode.key === activeMode.value)?.desc ?? '')
const activePanel = computed(() => panels[activeMode.value])
</script>

<style scoped>
.seq2seq-workflow-explorer {
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

.legend-item.context {
	background: color-mix(in srgb, #8b5cf6 14%, var(--vp-c-bg));
	border-color: color-mix(in srgb, #8b5cf6 32%, transparent);
}

.legend-item.decoder {
	background: color-mix(in srgb, #22c55e 14%, var(--vp-c-bg));
	border-color: color-mix(in srgb, #22c55e 32%, transparent);
}

.legend-item.search {
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

.diagram-card,
.insight-card {
	padding: 22px;
}

.pipeline-grid {
	display: grid;
	grid-template-columns: repeat(3, minmax(0, 1fr));
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

.encoder-stage {
	background: color-mix(in srgb, #06b6d4 8%, var(--vp-c-bg));
}

.context-stage {
	background: color-mix(in srgb, #8b5cf6 8%, var(--vp-c-bg));
}

.decoder-stage {
	background: color-mix(in srgb, #22c55e 8%, var(--vp-c-bg));
}

.stage-head {
	display: flex;
	justify-content: space-between;
	align-items: baseline;
	gap: 12px;
	margin-bottom: 14px;
}

.stage-head h3 {
	margin: 0;
	font-size: 22px;
}

.stage-head span {
	font-size: 13px;
	font-weight: 600;
	color: var(--vp-c-text-2);
}

.token-row,
.state-row {
	display: flex;
	flex-wrap: wrap;
	gap: 8px;
	margin-bottom: 12px;
}

.token,
.state-chip,
.context-box {
	padding: 10px 12px;
	border-radius: 14px;
	background: var(--vp-c-bg);
	border: 1px solid var(--vp-c-divider);
	font-weight: 600;
}

.token.bos,
.token.eos {
	background: color-mix(in srgb, #f59e0b 12%, var(--vp-c-bg));
}

.state-chip {
	background: color-mix(in srgb, #06b6d4 10%, var(--vp-c-bg));
}

.context-box.fixed {
	margin-bottom: 10px;
	background: color-mix(in srgb, #8b5cf6 12%, var(--vp-c-bg));
}

.context-box.dynamic {
	background: color-mix(in srgb, #a855f7 12%, var(--vp-c-bg));
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

.aux-card code {
	display: block;
	margin-top: 12px;
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
