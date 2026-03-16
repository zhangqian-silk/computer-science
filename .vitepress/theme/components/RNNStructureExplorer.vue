<template>
	<div class="rnn-structure-explorer">
		<div class="hero-card">
			<div class="hero-text">
				<p class="eyebrow">RNN Structure Explorer</p>
				<h2>循环序列建模的时间展开图</h2>
				<p class="summary">
					该图把最基础的单向 RNN 拆成「输入 → 状态更新 → 输出读取」三层，并显式标出跨时间步共享参数与隐藏状态传递，帮助读者先建立整体结构感，再进入公式与训练细节。
				</p>
			</div>
			<div class="legend-grid">
				<div class="legend-item input">输入向量 $x_t$</div>
				<div class="legend-item state">隐藏状态 $h_t$</div>
				<div class="legend-item output">输出 $\hat{y}_t$</div>
				<div class="legend-item recur">循环连接 / 参数共享</div>
			</div>
		</div>

		<div class="controls-card">
			<div class="toggle-group" role="tablist" aria-label="RNN 视图切换">
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

		<div class="panel-grid">
			<section class="diagram-card">
				<div class="time-grid">
					<div v-for="step in steps" :key="step.id" class="time-step">
						<div class="time-label">时间步 {{ step.id }}</div>
						<div class="node input-node">
							<span class="node-title">输入</span>
							<strong>{{ step.input }}</strong>
						</div>
						<div class="vertical-arrow">↓</div>
						<div class="node state-node" :class="{ focus: activeMode === 'state' || activeMode === 'train' }">
							<span class="node-title">隐藏状态</span>
							<strong>{{ step.state }}</strong>
							<small>{{ step.stateNote }}</small>
						</div>
						<div class="vertical-arrow" :class="{ faint: activeMode === 'state' }">↓</div>
						<div class="node output-node" :class="{ focus: activeMode === 'output' }">
							<span class="node-title">输出</span>
							<strong>{{ step.output }}</strong>
							<small>{{ step.outputNote }}</small>
						</div>
						<div
							v-if="step.id < steps.length"
							class="recurrent-arrow"
							:class="{ focus: activeMode === 'state' || activeMode === 'train' }"
						>
							<span>共享 $W_{hh}$ 传递到下一步</span>
						</div>
					</div>
				</div>

				<div class="formula-strip">
					<div class="formula-card">
						<span class="formula-label">状态更新</span>
						<code>$h_t=\phi(W_{xh}x_t+W_{hh}h_{t-1}+b_h)$</code>
					</div>
					<div class="formula-card">
						<span class="formula-label">输出读取</span>
						<code>$\hat{y}_t=g(W_{hy}h_t+b_y)$</code>
					</div>
					<div class="formula-card" v-if="activeMode === 'train'">
						<span class="formula-label">时间展开训练</span>
						<code>$\mathcal{L}=\sum_{t=1}^{T}\mathcal{L}_t(\hat{y}_t,y_t^*)$</code>
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

type ModeKey = 'overview' | 'state' | 'output' | 'train'

const activeMode = ref<ModeKey>('overview')

const modes: { key: ModeKey; label: string; desc: string }[] = [
	{ key: 'overview', label: '总览', desc: '先看单向 RNN 在时间维上的展开：每一步读取当前输入，结合上一时刻状态，再产生当前输出。' },
	{ key: 'state', label: '状态传递', desc: '聚焦 $h_{t-1} \rightarrow h_t$ 的递归链，理解 RNN 如何把历史压缩为动态状态。' },
	{ key: 'output', label: '输出读取', desc: '聚焦每一步如何由隐藏状态映射到输出层，适合理解序列标注和逐步预测。' },
	{ key: 'train', label: '训练展开', desc: '聚焦时间展开后的损失累积与 BPTT 路径，理解梯度为何会沿长链传播。' }
]

const steps = [
	{ id: 1, input: '$x_1$', state: '$h_1$', output: '$\\hat{y}_1$', stateNote: '由初始状态 $h_0$ 与首个输入共同得到', outputNote: '可对应第一个时间步的预测' },
	{ id: 2, input: '$x_2$', state: '$h_2$', output: '$\\hat{y}_2$', stateNote: '融合当前输入与上一步历史摘要', outputNote: '继续读取序列条件分布' },
	{ id: 3, input: '$x_3$', state: '$h_3$', output: '$\\hat{y}_3$', stateNote: '较早历史只能通过状态链间接保留', outputNote: '最后一步可作为序列级任务的读取点' }
]

const panels = {
	overview: {
		title: '总览：RNN 把历史压进一个会演化的状态',
		lead: '最基础的 RNN 不是显式保存全部过去，而是把过去压缩为隐藏状态，并在每个时间步重复同一套状态更新规则。',
		points: [
			'时间展开后可以看成多个共享参数的同构单元串联在一起。',
			'每一步只读取当前输入，但通过隐藏状态间接接触全部过去。',
			'输出层可以每步都读，也可以只读最后一步，具体取决于任务接口。'
		],
		rows: [
			{ label: '输入节点', value: '负责注入当前时刻的新信息 $x_t$' },
			{ label: '状态节点', value: '负责保存对前缀历史的任务相关摘要 $h_t$' },
			{ label: '输出节点', value: '负责把状态映射成分类、回归或词分布' }
		]
	},
	state: {
		title: '状态传递：RNN 的核心不是输出，而是递归状态更新',
		lead: '隐藏状态链 $h_0 \rightarrow h_1 \rightarrow h_2 \rightarrow h_3$ 是 RNN 建模顺序依赖的核心机制。',
		points: [
			'矩阵 $W_{hh}$ 在所有时间步共享，因此同一套转移规则被反复应用。',
			'越久远的信息，需要穿过越长的状态链才能影响当前时刻。',
			'这也解释了普通 RNN 容易遇到长程依赖衰减与梯度消失问题。'
		],
		rows: [
			{ label: '$h_{t-1}$', value: '上一时刻留下的历史摘要' },
			{ label: '$W_{hh}$', value: '决定过去信息如何进入当前状态' },
			{ label: '$\\phi$', value: '提供非线性，使状态转移不退化为纯线性系统' }
		]
	},
	output: {
		title: '输出读取：状态本身不等于答案，还要经过任务头映射',
		lead: 'RNN 的隐藏状态只是中间表征；最终预测还要通过 $W_{hy}$ 和任务相关输出函数 $g(\cdot)$ 来读取。',
		points: [
			'在语言建模中，输出通常是 softmax 后的词分布。',
			'在序列标注中，每个时间步都可读取一个标签预测。',
			'在多对一任务中，常直接使用最后一步状态或其聚合结果。'
		],
		rows: [
			{ label: '$W_{hy}$', value: '把隐藏空间映射到输出空间' },
			{ label: '$g(\\cdot)$', value: '根据任务选择 softmax / sigmoid / 线性层' },
			{ label: '读取策略', value: '逐步读取或末步读取，决定任务接口类型' }
		]
	},
	train: {
		title: '训练展开：BPTT 要沿时间链回传梯度',
		lead: '训练时会把循环结构在时间上展开，并对所有时间步损失求和，再让梯度沿时间反向传播。',
		points: [
			'越长的序列意味着越长的反向路径，梯度更容易衰减或爆炸。',
			'截断 BPTT、梯度裁剪等技巧，本质上都在控制这条时间链的训练稳定性。',
			'后续 LSTM、GRU 的门控结构，就是在此瓶颈上做出的结构性改进。'
		],
		rows: [
			{ label: '$\\mathcal{L}$', value: '由全序列各时间步损失累积得到' },
			{ label: '反向路径', value: '梯度需跨多个时间步回流到较早状态' },
			{ label: '训练难点', value: '长链乘积导致梯度消失或爆炸' }
		]
	}
} as const

const activeDescription = computed(() => modes.find((mode) => mode.key === activeMode.value)?.desc ?? '')
const activePanel = computed(() => panels[activeMode.value])
</script>

<style scoped>
.rnn-structure-explorer {
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
	grid-template-columns: minmax(0, 1.4fr) minmax(260px, 0.9fr);
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

.legend-item.input {
	background: color-mix(in srgb, #8b5cf6 14%, var(--vp-c-bg));
	border-color: color-mix(in srgb, #8b5cf6 32%, transparent);
}

.legend-item.state {
	background: color-mix(in srgb, #06b6d4 14%, var(--vp-c-bg));
	border-color: color-mix(in srgb, #06b6d4 32%, transparent);
}

.legend-item.output {
	background: color-mix(in srgb, #22c55e 14%, var(--vp-c-bg));
	border-color: color-mix(in srgb, #22c55e 32%, transparent);
}

.legend-item.recur {
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

.panel-grid {
	display: grid;
	grid-template-columns: minmax(0, 1.6fr) minmax(280px, 0.9fr);
	gap: 20px;
}

.diagram-card,
.insight-card {
	padding: 22px;
}

.time-grid {
	display: grid;
	grid-template-columns: repeat(3, minmax(0, 1fr));
	gap: 16px;
}

.time-step {
	position: relative;
	padding: 18px 14px 54px;
	border-radius: 18px;
	background: color-mix(in srgb, var(--vp-c-bg-soft) 82%, #ffffff 18%);
	border: 1px solid var(--vp-c-divider);
}

.time-label {
	font-size: 13px;
	font-weight: 700;
	color: var(--vp-c-text-2);
	margin-bottom: 12px;
}

.node {
	display: flex;
	flex-direction: column;
	gap: 6px;
	padding: 14px;
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
}

.node small {
	color: var(--vp-c-text-2);
	line-height: 1.55;
}

.input-node {
	background: color-mix(in srgb, #8b5cf6 14%, var(--vp-c-bg));
	border-color: color-mix(in srgb, #8b5cf6 32%, transparent);
}

.state-node {
	background: color-mix(in srgb, #06b6d4 14%, var(--vp-c-bg));
	border-color: color-mix(in srgb, #06b6d4 32%, transparent);
}

.output-node {
	background: color-mix(in srgb, #22c55e 14%, var(--vp-c-bg));
	border-color: color-mix(in srgb, #22c55e 32%, transparent);
}

.node.focus,
.recurrent-arrow.focus {
	box-shadow: 0 0 0 3px color-mix(in srgb, var(--vp-c-brand-1) 18%, transparent);
}

.vertical-arrow {
	text-align: center;
	font-size: 22px;
	font-weight: 700;
	color: var(--vp-c-text-3);
	padding: 6px 0;
}

.vertical-arrow.faint {
	opacity: 0.35;
}

.recurrent-arrow {
	position: absolute;
	left: calc(100% - 14px);
	top: 50%;
	transform: translateY(-50%);
	width: 28px;
	height: 2px;
	background: linear-gradient(90deg, #f59e0b, #fb7185);
}

.recurrent-arrow span {
	position: absolute;
	top: -38px;
	left: 50%;
	transform: translateX(-50%);
	width: 130px;
	font-size: 12px;
	line-height: 1.35;
	text-align: center;
	color: var(--vp-c-text-2);
}

.recurrent-arrow::after {
	content: '';
	position: absolute;
	right: -2px;
	top: -4px;
	border-left: 8px solid #fb7185;
	border-top: 5px solid transparent;
	border-bottom: 5px solid transparent;
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
	gap: 0;
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

@media (max-width: 1100px) {
	.hero-card,
	.panel-grid {
		grid-template-columns: 1fr;
	}
}

@media (max-width: 860px) {
	.time-grid {
		grid-template-columns: 1fr;
	}

	.recurrent-arrow {
		position: static;
		width: 2px;
		height: 24px;
		margin: 12px auto 0;
		transform: none;
		background: linear-gradient(180deg, #f59e0b, #fb7185);
	}

	.recurrent-arrow span {
		position: static;
		transform: none;
		display: block;
		width: auto;
		margin-top: 10px;
	}

	.recurrent-arrow::after {
		left: -4px;
		right: auto;
		top: auto;
		bottom: -2px;
		border-left: 5px solid transparent;
		border-right: 5px solid transparent;
		border-top: 8px solid #fb7185;
		border-bottom: none;
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
		grid-template-columns: 92px 1fr;
	}
}
</style>
