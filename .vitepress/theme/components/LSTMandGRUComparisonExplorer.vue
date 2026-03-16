<template>
	<div class="lstm-gru-comparison-explorer">
		<div class="hero-card">
			<div>
				<p class="eyebrow">LSTM vs GRU Explorer</p>
				<h2>门控结构与状态设计的并列对比图</h2>
				<p class="summary">
					该图把 LSTM 与 GRU 放在同一观察框架下，对比它们在状态数量、门控职责、更新路径与工程取舍上的差异，帮助读者理解为什么 GRU 常被视为 LSTM 的轻量化门控变体。
				</p>
			</div>
			<div class="legend-grid">
				<div class="legend-item lstm">LSTM：双状态 + 三门控</div>
				<div class="legend-item gru">GRU：单状态 + 双门控</div>
				<div class="legend-item memory">长期保留路径</div>
				<div class="legend-item tradeoff">工程权衡与适用场景</div>
			</div>
		</div>

		<div class="controls-card">
			<div class="toggle-group" role="tablist" aria-label="LSTM GRU 对比视图切换">
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
			<section class="compare-card">
				<div class="compare-columns">
					<div class="model-card lstm-card" :class="{ focus: activeMode === 'overview' || activeMode === 'state' || activeMode === 'tradeoff' }">
						<div class="model-head">
							<h3>LSTM</h3>
							<span>双状态结构</span>
						</div>
						<div class="pill-row">
							<span class="pill">遗忘门 $f_t$</span>
							<span class="pill">输入门 $i_t$</span>
							<span class="pill">输出门 $o_t$</span>
						</div>
						<div class="state-box">
							<div class="state-chip cell">细胞状态 $c_t$：长期记忆通道</div>
							<div class="state-chip hidden">隐藏状态 $h_t$：当前可读摘要</div>
						</div>
						<ul>
							<li>把“保留、写入、读出”明确拆成三类门控行为。</li>
							<li>长期记忆路径更清晰，适合解释长程依赖如何被保留。</li>
							<li>参数更多，结构也更重，但状态控制更细。</li>
						</ul>
					</div>

					<div class="model-card gru-card" :class="{ focus: activeMode === 'overview' || activeMode === 'state' || activeMode === 'tradeoff' }">
						<div class="model-head">
							<h3>GRU</h3>
							<span>单状态结构</span>
						</div>
						<div class="pill-row">
							<span class="pill">更新门 $z_t$</span>
							<span class="pill">重置门 $r_t$</span>
						</div>
						<div class="state-box single">
							<div class="state-chip hidden">隐藏状态 $h_t$：同时承担记忆与输出</div>
						</div>
						<ul>
							<li>把长期保留与新信息混合写回收敛到单一状态更新式。</li>
							<li>结构更紧凑，工程实现和训练速度通常更友好。</li>
							<li>状态解释比 LSTM 更简洁，但控制粒度略粗。</li>
						</ul>
					</div>
				</div>

				<div class="formula-grid">
					<div class="formula-card">
						<span class="formula-label">LSTM 记忆更新</span>
						<code>$c_t=f_t\odot c_{t-1}+i_t\odot\tilde{c}_t$</code>
					</div>
					<div class="formula-card">
						<span class="formula-label">GRU 状态更新</span>
						<code>$h_t=z_t\odot h_{t-1}+(1-z_t)\odot\tilde{h}_t$</code>
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
						<span>比较维度</span>
						<span>结论</span>
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

type ModeKey = 'overview' | 'state' | 'gate' | 'tradeoff'

const activeMode = ref<ModeKey>('overview')

const modes: { key: ModeKey; label: string; desc: string }[] = [
	{ key: 'overview', label: '总览', desc: '先看两种门控循环单元在状态设计和公式形态上的整体差异。' },
	{ key: 'state', label: '状态设计', desc: '聚焦 LSTM 的双状态结构与 GRU 的单状态结构，理解它们的抽象差异。' },
	{ key: 'gate', label: '门控职责', desc: '聚焦各自门控到底在控制什么：LSTM 更细，GRU 更紧凑。' },
	{ key: 'tradeoff', label: '工程取舍', desc: '聚焦参数量、训练速度、控制粒度与典型使用策略之间的权衡。' }
]

const panels = {
	overview: {
		title: '总览：GRU 不是“简化版 RNN”，而是“压缩版 LSTM”',
		lead: '它们都属于门控循环模型，但 LSTM 把状态拆得更细，而 GRU 把状态和门控逻辑压缩得更紧。',
		points: [
			'LSTM 显式区分长期记忆 $c_t$ 与当前输出 $h_t$。',
			'GRU 只保留单一隐藏状态，用更新门和重置门完成记忆管理。',
			'二者目标一致：让长期依赖比普通 RNN 更容易学习。'
		],
		rows: [
			{ label: '模型目标', value: '都在缓解普通 RNN 的长期依赖与梯度问题' },
			{ label: '结构路线', value: 'LSTM 细分职责，GRU 合并职责' },
			{ label: '理解方式', value: '可把 GRU 看成 LSTM 的更紧凑门控近亲' }
		]
	},
	state: {
		title: '状态设计：LSTM 双状态，GRU 单状态',
		lead: '最本质的结构区别，不是门的个数，而是是否把内部长期记忆与对外输出状态分开。',
		points: [
			'LSTM 的 $c_t$ 更偏内部记忆缓存，$h_t$ 更偏当前可读摘要。',
			'GRU 把记忆与输出统一到 $h_t$，减少了状态接口复杂度。',
			'这使得 LSTM 更适合强调“状态分工”的解释，GRU 更适合强调“更新效率”。'
		],
		rows: [
			{ label: 'LSTM', value: '两套状态：长期记忆 + 对外摘要' },
			{ label: 'GRU', value: '一套状态：内部保留与外部输出共用' },
			{ label: '影响', value: 'LSTM 控制更细，GRU 结构更简' }
		]
	},
	gate: {
		title: '门控职责：LSTM 分工更细，GRU 将职责合并',
		lead: 'LSTM 分别控制遗忘、写入和读出；GRU 则主要控制旧状态保留比例与候选状态生成方式。',
		points: [
			'LSTM 的输出门让“内部记忆是否对外暴露”也成为可学习决策。',
			'GRU 没有独立输出门，状态更新完成后即可直接作为外部输出。',
			'这种差异决定了两者在表达粒度和实现复杂度上的不同。'
		],
		rows: [
			{ label: 'LSTM 门控', value: '遗忘门 + 输入门 + 输出门' },
			{ label: 'GRU 门控', value: '更新门 + 重置门' },
			{ label: '职责压缩', value: 'GRU 将部分遗忘/写入逻辑折叠进更新门中' }
		]
	},
	tradeoff: {
		title: '工程取舍：LSTM 更细粒度，GRU 更轻量',
		lead: '实际选择往往不是谁“理论更先进”，而是谁更适合当前资源约束、任务复杂度和上线要求。',
		points: [
			'若更关心参数量、推理速度和部署成本，GRU 往往更讨喜。',
			'若更关心长期状态控制与内部机制解释，LSTM 更直观。',
			'在很多中等复杂度任务上，两者效果接近，差别往往由数据与工程环境决定。'
		],
		rows: [
			{ label: '参数量', value: 'GRU 通常低于 LSTM' },
			{ label: '训练速度', value: 'GRU 常更快，LSTM 常更重' },
			{ label: '适用偏好', value: '复杂状态控制偏 LSTM，资源敏感场景偏 GRU' }
		]
	}
} as const

const activeDescription = computed(() => modes.find((mode) => mode.key === activeMode.value)?.desc ?? '')
const activePanel = computed(() => panels[activeMode.value])
</script>

<style scoped>
.lstm-gru-comparison-explorer {
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

.hero-card h2 {
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

.legend-item.lstm {
	background: color-mix(in srgb, #06b6d4 14%, var(--vp-c-bg));
	border-color: color-mix(in srgb, #06b6d4 32%, transparent);
}

.legend-item.gru {
	background: color-mix(in srgb, #f59e0b 14%, var(--vp-c-bg));
	border-color: color-mix(in srgb, #f59e0b 32%, transparent);
}

.legend-item.memory {
	background: color-mix(in srgb, #22c55e 14%, var(--vp-c-bg));
	border-color: color-mix(in srgb, #22c55e 32%, transparent);
}

.legend-item.tradeoff {
	background: color-mix(in srgb, #a855f7 14%, var(--vp-c-bg));
	border-color: color-mix(in srgb, #a855f7 32%, transparent);
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
	grid-template-columns: minmax(0, 1.6fr) minmax(290px, 0.9fr);
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

.model-card {
	padding: 18px;
	border-radius: 18px;
	border: 1px solid var(--vp-c-divider);
	background: color-mix(in srgb, var(--vp-c-bg-soft) 84%, #ffffff 16%);
}

.model-card.focus {
	box-shadow: 0 0 0 3px color-mix(in srgb, var(--vp-c-brand-1) 18%, transparent);
}

.lstm-card {
	background: color-mix(in srgb, #06b6d4 8%, var(--vp-c-bg));
}

.gru-card {
	background: color-mix(in srgb, #f59e0b 8%, var(--vp-c-bg));
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

.pill-row {
	display: flex;
	flex-wrap: wrap;
	gap: 8px;
	margin-bottom: 14px;
}

.pill {
	padding: 8px 12px;
	border-radius: 999px;
	background: var(--vp-c-bg);
	border: 1px solid var(--vp-c-divider);
	font-size: 13px;
	font-weight: 600;
}

.state-box {
	display: grid;
	gap: 10px;
	margin-bottom: 14px;
}

.state-chip {
	padding: 12px 14px;
	border-radius: 14px;
	border: 1px solid transparent;
	font-weight: 600;
}

.state-chip.cell {
	background: color-mix(in srgb, #06b6d4 14%, var(--vp-c-bg));
	border-color: color-mix(in srgb, #06b6d4 32%, transparent);
}

.state-chip.hidden {
	background: color-mix(in srgb, #22c55e 14%, var(--vp-c-bg));
	border-color: color-mix(in srgb, #22c55e 32%, transparent);
}

.state-box.single .state-chip.hidden {
	background: color-mix(in srgb, #f59e0b 14%, var(--vp-c-bg));
	border-color: color-mix(in srgb, #f59e0b 32%, transparent);
}

.model-card ul {
	margin: 0;
	padding-left: 18px;
	display: grid;
	gap: 10px;
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
	.panel-grid,
	.compare-columns {
		grid-template-columns: 1fr;
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
