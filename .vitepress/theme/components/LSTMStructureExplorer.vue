<template>
	<div class="lstm-structure-explorer">
		<div class="hero-card">
			<div class="hero-text">
				<p class="eyebrow">LSTM Structure Explorer</p>
				<h2>门控记忆单元的单步结构图</h2>
				<p class="summary">
					该图把标准 LSTM 的单步前向传播拆成「遗忘、写入、更新、读出」四个阶段，突出细胞状态 $c_t$ 作为长期记忆通道的结构地位，并展示它与隐藏状态 $h_t$ 的功能分工。
				</p>
			</div>
			<div class="legend-grid">
				<div class="legend-item gate">门控向量：$f_t,i_t,o_t$</div>
				<div class="legend-item cell">细胞状态：$c_t$</div>
				<div class="legend-item hidden">隐藏状态：$h_t$</div>
				<div class="legend-item candidate">候选记忆：$\tilde{c}_t$</div>
			</div>
		</div>

		<div class="controls-card">
			<div class="toggle-group" role="tablist" aria-label="LSTM 视图切换">
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
				<div class="flow-grid">
					<div class="col source-col">
						<div class="node source-node">
							<span class="node-title">输入拼接</span>
							<strong>$z_t=[h_{t-1};x_t]$</strong>
							<small>当前输入与上一时刻隐藏状态共同决定四个门控分支。</small>
						</div>
					</div>

					<div class="col gate-col">
						<div class="node gate-node" :class="{ focus: activeMode === 'forget' || activeMode === 'memory' }">
							<span class="node-title">遗忘门</span>
							<strong>$f_t$</strong>
							<small>决定旧记忆保留多少。</small>
						</div>
						<div class="node gate-node" :class="{ focus: activeMode === 'write' || activeMode === 'memory' }">
							<span class="node-title">输入门</span>
							<strong>$i_t$</strong>
							<small>决定候选内容写入多少。</small>
						</div>
						<div class="node candidate-node" :class="{ focus: activeMode === 'write' || activeMode === 'memory' }">
							<span class="node-title">候选记忆</span>
							<strong>$\tilde{c}_t$</strong>
							<small>生成当前时刻可写入的内容。</small>
						</div>
						<div class="node gate-node" :class="{ focus: activeMode === 'output' }">
							<span class="node-title">输出门</span>
							<strong>$o_t$</strong>
							<small>控制内部记忆向外暴露多少。</small>
						</div>
					</div>

					<div class="col state-col">
						<div class="memory-lane" :class="{ focus: activeMode === 'memory' || activeMode === 'forget' || activeMode === 'write' }">
							<div class="memory-head">长期记忆通道</div>
							<div class="memory-item prev">旧记忆：$c_{t-1}$</div>
							<div class="memory-op">$f_t \odot c_{t-1}$</div>
							<div class="memory-op">$i_t \odot \tilde{c}_t$</div>
							<div class="memory-item next">新记忆：$c_t$</div>
						</div>
						<div class="node hidden-node" :class="{ focus: activeMode === 'output' }">
							<span class="node-title">隐藏状态</span>
							<strong>$h_t=o_t\odot\tanh(c_t)$</strong>
							<small>作为当前时刻对外可读的摘要表示。</small>
						</div>
					</div>
				</div>

				<div class="formula-strip">
					<div class="formula-card">
						<span class="formula-label">遗忘</span>
						<code>$f_t=\sigma(W_fz_t+b_f)$</code>
					</div>
					<div class="formula-card">
						<span class="formula-label">写入</span>
						<code>$i_t=\sigma(W_iz_t+b_i),\ \tilde{c}_t=\tanh(W_cz_t+b_c)$</code>
					</div>
					<div class="formula-card">
						<span class="formula-label">更新</span>
						<code>$c_t=f_t\odot c_{t-1}+i_t\odot\tilde{c}_t$</code>
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

type ModeKey = 'overview' | 'forget' | 'write' | 'memory' | 'output'

const activeMode = ref<ModeKey>('overview')

const modes: { key: ModeKey; label: string; desc: string }[] = [
	{ key: 'overview', label: '总览', desc: '先看标准 LSTM 的整体拆分：同一输入分支生成四类门控量，再共同决定细胞状态和隐藏状态。' },
	{ key: 'forget', label: '遗忘门', desc: '聚焦 $f_t$ 如何筛选旧记忆，理解哪些历史信息会被继续保留。' },
	{ key: 'write', label: '写入路径', desc: '聚焦 $i_t$ 与 $\tilde{c}_t$ 如何协同，理解新信息是怎样被门控写入细胞状态的。' },
	{ key: 'memory', label: '记忆通道', desc: '聚焦 $c_{t-1} \rightarrow c_t$ 的加法式更新，理解 LSTM 为何比普通 RNN 更适合长期依赖。' },
	{ key: 'output', label: '输出读取', desc: '聚焦 $o_t$ 与 $h_t$，理解内部记忆如何在当前时刻被选择性暴露。' }
]

const panels = {
	overview: {
		title: '总览：LSTM 把状态更新改写成受控的读写过程',
		lead: '与普通 RNN 一次性压缩历史不同，LSTM 明确区分了长期记忆通道 $c_t$ 与对外输出状态 $h_t$。',
		points: [
			'同一个输入拼接向量 $z_t$ 会分流到遗忘门、输入门、候选记忆和输出门。',
			'真正承担长期记忆的是细胞状态，而不是隐藏状态本身。',
			'门控的作用不是增加复杂度，而是把“保留、写入、暴露”这三件事显式拆开。'
		],
		rows: [
			{ label: '$z_t$', value: '统一承载当前输入与过去隐藏状态的信息' },
			{ label: '$c_t$', value: '长期记忆主通道，适合跨时间稳定传播' },
			{ label: '$h_t$', value: '当前时刻对外可读的摘要状态' }
		]
	},
	forget: {
		title: '遗忘门：不是机械丢弃，而是按维度筛选旧记忆',
		lead: '遗忘门 $f_t$ 决定 $c_{t-1}$ 中各维信息在当前时刻应保留多少。',
		points: [
			'当某一维 $f_t$ 接近 1 时，对应旧记忆可以继续向后传递。',
			'当某一维 $f_t$ 接近 0 时，对应旧记忆会被显著抑制。',
			'这使得模型能区分“应该长期保留的语义线索”和“只在局部有用的临时信息”。'
		],
		rows: [
			{ label: '$f_t$', value: '控制旧记忆保留率的门控向量' },
			{ label: '$c_{t-1}$', value: '上一时刻长期记忆内容' },
			{ label: '$f_t\odot c_{t-1}$', value: '经过筛选后的旧记忆保留项' }
		]
	},
	write: {
		title: '写入路径：候选内容与输入门共同决定“写什么、写多少”',
		lead: 'LSTM 不会把当前输入直接塞进状态，而是先生成候选记忆，再由输入门控制写入强度。',
		points: [
			'候选记忆 $\tilde{c}_t$ 负责提供可写入的新内容。',
			'输入门 $i_t$ 负责按维度决定这些内容是否值得写入。',
			'这使得 LSTM 的状态更新更接近“带权限控制的写缓存”。'
		],
		rows: [
			{ label: '$i_t$', value: '控制新信息写入比例' },
			{ label: '$\tilde{c}_t$', value: '由当前上下文生成的候选记忆内容' },
			{ label: '$i_t\odot\tilde{c}_t$', value: '真正被写入细胞状态的新信息项' }
		]
	},
	memory: {
		title: '记忆通道：LSTM 的关键不在门多，而在加法式状态更新',
		lead: '细胞状态的更新写成“旧记忆保留项 + 新记忆写入项”，这是它缓解长程梯度问题的结构基础。',
		points: [
			'与普通 RNN 的整体非线性压缩不同，$c_t$ 的主路径更接近线性传输。',
			'梯度在这条路径上不必每步都穿过同样强烈的非线性压缩。',
			'门控让长期依赖是否保留变成了可学习的选择，而不是偶然结果。'
		],
		rows: [
			{ label: '$c_t$', value: '长期记忆的新状态' },
			{ label: '加法结构', value: '让旧信息与新信息可以并列地参与更新' },
			{ label: '长期依赖', value: '是否保留，主要受遗忘门持续控制' }
		]
	},
	output: {
		title: '输出读取：内部记忆不会全部暴露给外部',
		lead: 'LSTM 将内部长期记忆与对外输出分离，输出门决定当前时刻哪些内部信息值得被读取。',
		points: [
			'细胞状态可保留比当前输出更丰富的内部信息。',
			'输出门 $o_t$ 决定当前时刻哪些信息适合对外暴露。',
			'因此 $h_t$ 更像“当前可读摘要”，而 $c_t$ 更像“内部记忆缓存”。'
		],
		rows: [
			{ label: '$o_t$', value: '控制内部记忆向外暴露的门控' },
			{ label: '$\tanh(c_t)$', value: '把细胞状态映射到适合输出的数值范围' },
			{ label: '$h_t$', value: '供后续层或输出头读取的公开状态' }
		]
	}
} as const

const activeDescription = computed(() => modes.find((mode) => mode.key === activeMode.value)?.desc ?? '')
const activePanel = computed(() => panels[activeMode.value])
</script>

<style scoped>
.lstm-structure-explorer {
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

.legend-item.gate {
	background: color-mix(in srgb, #a855f7 14%, var(--vp-c-bg));
	border-color: color-mix(in srgb, #a855f7 32%, transparent);
}

.legend-item.cell {
	background: color-mix(in srgb, #06b6d4 14%, var(--vp-c-bg));
	border-color: color-mix(in srgb, #06b6d4 32%, transparent);
}

.legend-item.hidden {
	background: color-mix(in srgb, #22c55e 14%, var(--vp-c-bg));
	border-color: color-mix(in srgb, #22c55e 32%, transparent);
}

.legend-item.candidate {
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

.flow-grid {
	display: grid;
	grid-template-columns: minmax(220px, 0.95fr) minmax(240px, 1fr) minmax(260px, 1.1fr);
	gap: 18px;
	align-items: stretch;
}

.col {
	display: flex;
	flex-direction: column;
	gap: 14px;
}

.node {
	display: flex;
	flex-direction: column;
	gap: 6px;
	padding: 16px;
	border-radius: 16px;
	border: 1px solid transparent;
}

.node-title {
	font-size: 12px;
	font-weight: 700;
	letter-spacing: 0.05em;
	text-transform: uppercase;
}

.node strong {
	font-size: 20px;
	line-height: 1.35;
}

.node small {
	color: var(--vp-c-text-2);
	line-height: 1.6;
}

.source-node {
	background: color-mix(in srgb, #64748b 12%, var(--vp-c-bg));
	border-color: color-mix(in srgb, #64748b 32%, transparent);
	justify-content: center;
	min-height: 100%;
}

.gate-node {
	background: color-mix(in srgb, #a855f7 14%, var(--vp-c-bg));
	border-color: color-mix(in srgb, #a855f7 32%, transparent);
}

.candidate-node {
	background: color-mix(in srgb, #f59e0b 14%, var(--vp-c-bg));
	border-color: color-mix(in srgb, #f59e0b 32%, transparent);
}

.hidden-node {
	background: color-mix(in srgb, #22c55e 14%, var(--vp-c-bg));
	border-color: color-mix(in srgb, #22c55e 32%, transparent);
}

.node.focus,
.memory-lane.focus {
	box-shadow: 0 0 0 3px color-mix(in srgb, var(--vp-c-brand-1) 18%, transparent);
}

.memory-lane {
	border-radius: 18px;
	padding: 16px;
	background: color-mix(in srgb, #06b6d4 12%, var(--vp-c-bg));
	border: 1px solid color-mix(in srgb, #06b6d4 32%, transparent);
	display: grid;
	gap: 10px;
}

.memory-head {
	font-size: 12px;
	font-weight: 700;
	letter-spacing: 0.08em;
	text-transform: uppercase;
	color: var(--vp-c-text-2);
}

.memory-item,
.memory-op {
	padding: 12px 14px;
	border-radius: 14px;
	background: color-mix(in srgb, var(--vp-c-bg) 82%, #ffffff 18%);
	border: 1px solid var(--vp-c-divider);
	font-weight: 600;
}

.memory-op {
	font-family: var(--vp-font-family-mono);
	color: var(--vp-c-text-2);
}

.formula-strip {
	display: grid;
	grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
	gap: 12px;
	margin-top: 18px;
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
	.flow-grid {
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
