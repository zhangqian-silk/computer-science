<template>
	<div class="rnn-dynamics">
		<div class="header">
			<div>
				<p class="eyebrow">RNN 训练与推理机制图</p>
				<h3>前向递推、时间回传与推理分叉</h3>
				<p class="desc">
					这一组件把当前文档中最容易混在一起的几条链路拆开：前向状态更新、BPTT 的时间反传、截断与裁剪等稳定化手段，以及 teacher forcing 与自回归推理之间的分叉关系。
				</p>
			</div>
			<div class="mode-switch">
				<button
					v-for="item in modes"
					:key="item.key"
					:class="['mode-btn', { active: item.key === activeMode }]"
					@click="activeMode = item.key"
				>
					{{ item.label }}
				</button>
			</div>
		</div>

		<div class="layout">
			<div class="main-panel">
				<template v-if="activeMode === 'forward'">
					<div class="pipeline-row">
						<div class="stage input">
							<h4>输入 $x_t$</h4>
							<p>当前时间步只接收当前输入，而不是一次性看到整段序列。</p>
						</div>
						<div class="arrow">→</div>
						<div class="stage state">
							<h4>状态更新 $h_t$</h4>
							<p class="formula">$h_t=\phi(W_{xh}x_t+W_{hh}h_{t-1}+b_h)$</p>
							<p>把当前输入与历史摘要融合成新的隐藏状态。</p>
						</div>
						<div class="arrow">→</div>
						<div class="stage output">
							<h4>输出读取 $\hat{y}_t$</h4>
							<p class="formula">$\hat{y}_t=g(W_{hy}h_t+b_y)$</p>
							<p>根据任务不同，可以每步输出，也可以只在最后一步读取。</p>
						</div>
						<div class="arrow">→</div>
						<div class="stage time">
							<h4>传给下一步</h4>
							<p>当前状态 $h_t$ 会成为下一时刻的历史摘要。</p>
						</div>
					</div>
					<div class="timeline-card">
						<div class="timeline-node" v-for="step in forwardSteps" :key="step.title">
							<h5>{{ step.title }}</h5>
							<p>{{ step.text }}</p>
						</div>
					</div>
				</template>

				<template v-else-if="activeMode === 'bptt'">
					<div class="bptt-grid">
						<div class="card loss-card">
							<h4>1. 总损失来自所有时间步</h4>
							<p class="formula">$\mathcal{L}=\sum_{t=1}^{T}\mathcal{L}_t$</p>
							<p>同一组参数在所有时间步重复使用，因此梯度必须累加所有时间位置的责任。</p>
						</div>
						<div class="card back-card">
							<h4>2. 误差沿时间倒序回传</h4>
							<p>从最后时刻开始，把输出误差先传回隐藏状态，再顺着 $h_t \to h_{t-1}$ 链条向前追责。</p>
						</div>
						<div class="card share-card">
							<h4>3. 参数共享导致梯度求和</h4>
							<p class="formula">$\frac{\partial \mathcal{L}}{\partial W_{hh}}=\sum_{t=1}^{T}\delta_t h_{t-1}^{\top}$</p>
							<p>不是每步更新一套新参数，而是把所有展开副本对同一矩阵的贡献累加后统一更新。</p>
						</div>
					</div>
					<div class="chain-row">
						<div class="chain-node" v-for="node in bpttChain" :key="node.title">
							<h5>{{ node.title }}</h5>
							<p>{{ node.text }}</p>
						</div>
					</div>
				</template>

				<template v-else-if="activeMode === 'stability'">
					<div class="compare-grid">
						<div class="card vanishing">
							<h4>梯度消失</h4>
							<p>局部雅可比连乘长期小于 1 时，早期状态几乎收不到后期损失信号。</p>
							<p class="formula">$\left\|\frac{\partial h_t}{\partial h_k}\right\|\to 0$</p>
						</div>
						<div class="card exploding">
							<h4>梯度爆炸</h4>
							<p>若循环矩阵与局部导数连乘整体偏大，误差信号会在长链上传播放大。</p>
							<p class="formula">$\left\|\frac{\partial h_t}{\partial h_k}\right\|\to \infty$</p>
						</div>
						<div class="card truncation">
							<h4>截断 BPTT</h4>
							<p>只让梯度在最近的固定窗口内回传，降低显存占用与长链数值不稳定。</p>
						</div>
						<div class="card clipping">
							<h4>梯度裁剪</h4>
							<p class="formula">$g\leftarrow \frac{\tau}{\|g\|}g$</p>
							<p>在梯度范数超过阈值时只缩放长度，不改变方向。</p>
						</div>
					</div>
				</template>

				<template v-else>
					<div class="fork-layout">
						<div class="fork-source">
							<h4>同一条件建模目标</h4>
							<p class="formula">$P(y_{1:T})=\prod_{t=1}^{T}P(y_t\mid y_{1:t-1}, context)$</p>
							<p>训练和推理共享同一条件分布目标，但喂给模型的历史来源不同。</p>
						</div>
						<div class="fork-branches">
							<div class="branch teacher">
								<h4>Teacher Forcing（训练）</h4>
								<p>下一步输入用真实历史 $y_{t-1}^\ast$，让每步都在“正确上下文”下学习预测。</p>
							</div>
							<div class="branch inference">
								<h4>Autoregressive Inference（推理）</h4>
								<p>下一步输入改为模型自己刚刚生成的结果，错误会重新回灌后续链条。</p>
							</div>
						</div>
						<div class="exposure-box">
							<h4>曝光偏差（Exposure Bias）</h4>
							<p>训练分布使用真实历史，推理分布使用模型历史。两者不一致时，早期错误会在后续时间步不断累积。</p>
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

type ModeKey = 'forward' | 'bptt' | 'stability' | 'train-infer'

const activeMode = ref<ModeKey>('forward')

const modes = [
	{ key: 'forward', label: '前向递推' },
	{ key: 'bptt', label: '时间回传' },
	{ key: 'stability', label: '稳定化' },
	{ key: 'train-infer', label: '训练 vs 推理' },
] as const

const forwardSteps = [
	{ title: '状态不是原始历史', text: 'RNN 不显式保存全部过去，而是把前缀压缩成固定维度的隐藏状态。' },
	{ title: '时间展开像深层网络', text: '从实现上看像一个循环单元反复调用；从训练上看则像按时间展开的共享参数深层网络。' },
	{ title: '输出读取取决于任务', text: '语言建模常每步输出，情感分类常只读取最后状态或其聚合结果。' },
]

const bpttChain = [
	{ title: '输出误差', text: '先由当前步的 softmax / 交叉熵或其他任务损失生成输出误差信号。' },
	{ title: '回到隐藏状态', text: '输出误差先回传到当前隐藏状态，再进入时间链条。' },
	{ title: '穿过循环边', text: '误差顺着 $h_t \rightarrow h_{t-1}$ 的递归连接不断向早期传播。' },
	{ title: '累加共享参数梯度', text: '所有时间步对同一组 $W_{xh},W_{hh},W_{hy}$ 的贡献会被统一加总。' },
]

const sidePanelMap: Record<ModeKey, { title: string; summary: string; points: string[] }> = {
	forward: {
		title: '前向递推的本质',
		summary: 'RNN 不是一次性读取整段序列，而是在每个时间步把新输入与已有状态融合成新的历史摘要。',
		points: [
			'隐藏状态是对前缀的压缩摘要',
			'状态会沿时间持续传递',
			'输出可按任务在每步或最后一步读取',
		],
	},
	bptt: {
		title: 'BPTT 真正做了什么',
		summary: '它把循环结构沿时间展开后，对共享参数执行一次“跨时间求和”的反向传播。',
		points: [
			'总损失是所有时间步损失的累加',
			'输出误差先回到当前隐藏状态',
			'时间链越长，梯度路径越长',
		],
	},
	stability: {
		title: '为什么普通 RNN 难训',
		summary: '同一条时间链既承载历史记忆，也承载梯度回传；链条一长，消失与爆炸就更容易同时出现。',
		points: [
			'截断 BPTT 用于限制回传长度',
			'梯度裁剪用于抑制异常大更新',
			'LSTM / GRU 正是为长期依赖稳定性而生',
		],
	},
	'train-infer': {
		title: '训练与推理为什么不一样',
		summary: '训练时喂给模型真实历史，推理时喂给模型自身历史，这种输入分布差异会导致曝光偏差。',
		points: [
			'Teacher forcing 让训练更稳',
			'自回归推理更接近真实部署环境',
			'早期错误会在生成中不断滚动放大',
		],
	},
}

const sidePanel = computed(() => sidePanelMap[activeMode.value])
</script>

<style scoped>
.rnn-dynamics {
	margin: 24px 0 32px;
	padding: 24px;
	border: 1px solid var(--vp-c-divider);
	border-radius: 20px;
	background: linear-gradient(180deg, rgba(99, 102, 241, 0.08), rgba(16, 185, 129, 0.04));
}

.header {
	display: flex;
	justify-content: space-between;
	gap: 20px;
	align-items: flex-start;
	flex-wrap: wrap;
	margin-bottom: 20px;
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

.layout {
	display: grid;
	grid-template-columns: minmax(0, 2.35fr) minmax(260px, 1fr);
	gap: 18px;
}

.main-panel,
.side-panel {
	padding: 18px;
	border-radius: 16px;
	background: rgba(255, 255, 255, 0.72);
	border: 1px solid rgba(148, 163, 184, 0.24);
	backdrop-filter: blur(6px);
}

.dark .main-panel,
.dark .side-panel {
	background: rgba(15, 23, 42, 0.72);
	border-color: rgba(148, 163, 184, 0.18);
}

.pipeline-row {
	display: flex;
	gap: 10px;
	align-items: stretch;
	flex-wrap: wrap;
	margin-bottom: 16px;
}

.stage,
.card,
.branch,
.fork-source,
.exposure-box,
.timeline-node,
.chain-node {
	border-radius: 14px;
	padding: 16px;
	border: 1px solid rgba(148, 163, 184, 0.22);
	background: rgba(255, 255, 255, 0.52);
}

.dark .stage,
.dark .card,
.dark .branch,
.dark .fork-source,
.dark .exposure-box,
.dark .timeline-node,
.dark .chain-node {
	background: rgba(2, 6, 23, 0.28);
}

.stage {
	flex: 1 1 180px;
}

.stage h4,
.card h4,
.branch h4,
.fork-source h4,
.exposure-box h4,
.side-panel h4,
.timeline-node h5,
.chain-node h5 {
	margin: 0 0 10px;
}

.stage p,
.card p,
.branch p,
.fork-source p,
.exposure-box p,
.side-panel p,
.timeline-node p,
.chain-node p {
	margin: 0;
	line-height: 1.7;
	color: var(--vp-c-text-2);
}

.formula {
	font-weight: 700;
	color: var(--vp-c-brand-1) !important;
	margin-bottom: 10px !important;
}

.arrow {
	display: flex;
	align-items: center;
	justify-content: center;
	padding: 0 4px;
	font-size: 22px;
	font-weight: 700;
	color: var(--vp-c-text-3);
}

.timeline-card,
.chain-row {
	display: grid;
	gap: 12px;
}

.timeline-card {
	grid-template-columns: repeat(3, minmax(0, 1fr));
}

.bptt-grid,
.compare-grid {
	display: grid;
	grid-template-columns: repeat(2, minmax(0, 1fr));
	gap: 14px;
	margin-bottom: 14px;
}

.chain-row {
	grid-template-columns: repeat(4, minmax(0, 1fr));
}

.fork-layout {
	display: grid;
	gap: 14px;
}

.fork-branches {
	display: grid;
	grid-template-columns: repeat(2, minmax(0, 1fr));
	gap: 14px;
}

.input { background: rgba(59, 130, 246, 0.08); }
.state { background: rgba(16, 185, 129, 0.08); }
.output { background: rgba(245, 158, 11, 0.08); }
.time { background: rgba(168, 85, 247, 0.08); }
.vanishing { background: rgba(59, 130, 246, 0.08); }
.exploding { background: rgba(239, 68, 68, 0.08); }
.truncation { background: rgba(16, 185, 129, 0.08); }
.clipping { background: rgba(168, 85, 247, 0.08); }
.teacher { background: rgba(16, 185, 129, 0.08); }
.inference { background: rgba(245, 158, 11, 0.08); }

.side-panel ul {
	margin: 12px 0 0;
	padding-left: 18px;
}

@media (max-width: 1100px) {
	.layout,
	.timeline-card,
	.bptt-grid,
	.compare-grid,
	.chain-row,
	.fork-branches {
		grid-template-columns: 1fr;
	}
}
</style>
