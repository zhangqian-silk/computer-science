<script setup lang="ts">
import { computed, ref } from 'vue'
import { renderMath } from '../utils/renderMath'

type DiagramStep = {
	id: string
	title: string
	color: string
	description: string
	formula: string
	badges: string[]
}

type DiagramMode = {
	id: 'encoder' | 'decoder'
	title: string
	steps: DiagramStep[]
}

const modes: DiagramMode[] = [
	{
		id: 'encoder',
		title: '编码器块',
		steps: [
			{
				id: 'enc-in',
				title: '输入 H^(l-1)',
				color: 'slate',
				description: '上一层输出进入当前编码器块，作为本层的起始表示。',
				formula: 'H^{(\\ell-1)}',
				badges: ['source: previous block', 'shape: L x d_model']
			},
			{
				id: 'enc-ln1',
				title: 'LayerNorm',
				color: 'slate',
				description: '先做层归一化，稳定数值分布，再送入自注意力子层。',
				formula: '\\mathrm{LN}\\big(H^{(\\ell-1)}\\big)',
				badges: ['purpose: stabilize scale']
			},
			{
				id: 'enc-attn',
				title: 'Multi-Head Self-Attention',
				color: 'blue',
				description: '在全序列范围内建立 token 之间的相关性，让每个位置读取其他位置的信息。',
				formula: '\\mathrm{MHA}(H, H, H)',
				badges: ['Q <- encoder', 'K <- encoder', 'V <- encoder']
			},
			{
				id: 'enc-add1',
				title: '残差相加',
				color: 'slate',
				description: '把子层输出与原输入相加，保留恒等信息通路。',
				formula: 'H^{(\\ell-1)} + \\mathrm{AttnOut}',
				badges: ['path: skip connection']
			},
			{
				id: 'enc-ln2',
				title: 'LayerNorm',
				color: 'slate',
				description: '在进入前馈网络前再次归一化，减小深层训练中的尺度漂移。',
				formula: '\\mathrm{LN}\\big(Z^{(\\ell)}\\big)',
				badges: ['purpose: stabilize FFN input']
			},
			{
				id: 'enc-ffn',
				title: 'FFN',
				color: 'violet',
				description: '对每个位置做独立的非线性重加工，增强单位置表示能力。',
				formula: 'W_2\\,\\phi(W_1x + b_1) + b_2',
				badges: ['position-wise', 'non-linear']
			},
			{
				id: 'enc-add2',
				title: '残差相加',
				color: 'slate',
				description: '再次执行残差融合，得到当前编码器块的最终输出。',
				formula: 'Z^{(\\ell)} + \\mathrm{FFNOut}',
				badges: ['path: skip connection']
			},
			{
				id: 'enc-out',
				title: '输出 H^(l)',
				color: 'green',
				description: '当前编码器块的输出将作为下一层编码器块或解码器 cross-attention 的输入。',
				formula: 'H^{(\\ell)}',
				badges: ['to: next encoder', 'to: decoder cross-attn']
			}
		]
	},
	{
		id: 'decoder',
		title: '解码器块',
		steps: [
			{
				id: 'dec-in',
				title: '输入 Y^(l-1)',
				color: 'slate',
				description: '上一层解码器输出进入当前块，作为当前生成步的上下文表示。',
				formula: 'Y^{(\\ell-1)}',
				badges: ['source: previous decoder block']
			},
			{
				id: 'dec-ln1',
				title: 'LayerNorm',
				color: 'slate',
				description: '先做层归一化，再进入带因果约束的自注意力。',
				formula: '\\mathrm{LN}\\big(Y^{(\\ell-1)}\\big)',
				badges: ['purpose: stabilize masked attn']
			},
			{
				id: 'dec-mask',
				title: 'Masked Self-Attention',
				color: 'blue',
				description: '只能读取当前位置及其历史前缀，保证自回归生成不偷看未来 token。',
				formula: '\\mathrm{MaskedMHA}(Y, Y, Y)',
				badges: ['Q <- decoder', 'K <- decoder', 'V <- decoder']
			},
			{
				id: 'dec-add1',
				title: '残差相加',
				color: 'slate',
				description: '保留原始解码器输入通路，并融合 masked self-attention 的结果。',
				formula: 'Y^{(\\ell-1)} + \\mathrm{SelfAttnOut}',
				badges: ['path: skip connection']
			},
			{
				id: 'dec-ln2',
				title: 'LayerNorm',
				color: 'slate',
				description: '在 cross-attention 前再次归一化。',
				formula: '\\mathrm{LN}\\big(U^{(\\ell)}\\big)',
				badges: ['purpose: prepare cross-attn']
			},
			{
				id: 'dec-cross',
				title: 'Cross-Attention',
				color: 'amber',
				description: 'Query 来自解码器当前状态，Key / Value 来自编码器输出，用于读取条件输入信息。',
				formula: '\\mathrm{MHA}(U, H_{\\mathrm{enc}}, H_{\\mathrm{enc}})',
				badges: ['Q <- decoder', 'K <- encoder', 'V <- encoder']
			},
			{
				id: 'dec-add2',
				title: '残差相加',
				color: 'slate',
				description: '将 cross-attention 读到的条件信息与当前解码器状态融合。',
				formula: 'U^{(\\ell)} + \\mathrm{CrossAttnOut}',
				badges: ['path: skip connection']
			},
			{
				id: 'dec-ln3',
				title: 'LayerNorm',
				color: 'slate',
				description: '在送入前馈网络前再次归一化。',
				formula: '\\mathrm{LN}\\big(V^{(\\ell)}\\big)',
				badges: ['purpose: stabilize FFN input']
			},
			{
				id: 'dec-ffn',
				title: 'FFN',
				color: 'violet',
				description: '对当前位置表示做逐位置非线性重构。',
				formula: 'W_2\\,\\phi(W_1x + b_1) + b_2',
				badges: ['position-wise', 'non-linear']
			},
			{
				id: 'dec-add3',
				title: '残差相加',
				color: 'slate',
				description: '再次融合原路径与 FFN 输出，形成最终表示。',
				formula: 'V^{(\\ell)} + \\mathrm{FFNOut}',
				badges: ['path: skip connection']
			},
			{
				id: 'dec-out',
				title: '输出 Y^(l)',
				color: 'green',
				description: '当前解码器块的输出将流向下一层，最终用于词表预测。',
				formula: 'Y^{(\\ell)}',
				badges: ['to: next decoder', 'to: vocabulary head']
			}
		]
	}
]

const activeMode = ref<DiagramMode['id']>('encoder')
const activeStepId = ref(modes[0].steps[2].id)

const currentMode = computed(() => modes.find((mode) => mode.id === activeMode.value) ?? modes[0])
const activeStep = computed(() => currentMode.value.steps.find((step) => step.id === activeStepId.value) ?? currentMode.value.steps[0])

function switchMode(nextMode: DiagramMode['id']) {
	activeMode.value = nextMode
	activeStepId.value = (modes.find((mode) => mode.id === nextMode) ?? modes[0]).steps[2]?.id ?? (modes.find((mode) => mode.id === nextMode) ?? modes[0]).steps[0].id
}

function renderInlineMath(tex: string) {
	return renderMath(tex, false)
}
</script>

<template>
	<div class="block-explorer">
		<div class="block-explorer__topbar">
			<button
				v-for="mode in modes"
				:key="mode.id"
				type="button"
				class="block-explorer__tab"
				:class="{ 'block-explorer__tab--active': mode.id === activeMode }"
				@click="switchMode(mode.id)"
			>
				{{ mode.title }}
			</button>
		</div>

		<div class="block-explorer__body">
			<div class="block-explorer__diagram">
				<div class="block-explorer__rail" />

				<div
					v-for="step in currentMode.steps"
					:key="step.id"
					class="block-explorer__step"
					:class="[
						`block-explorer__step--${step.color}`,
						{ 'block-explorer__step--active': step.id === activeStepId }
					]"
				>
					<button type="button" class="block-explorer__node" @click="activeStepId = step.id">
						{{ step.title }}
					</button>
				</div>
			</div>

			<aside class="block-explorer__panel">
				<p class="block-explorer__panel-kicker">{{ currentMode.title }}</p>
				<h4>{{ activeStep.title }}</h4>
				<div class="block-explorer__badges">
					<span v-for="badge in activeStep.badges" :key="badge" class="block-explorer__badge">
						{{ badge }}
					</span>
				</div>
				<div class="block-explorer__formula" v-html="renderInlineMath(activeStep.formula)" />
				<p>{{ activeStep.description }}</p>
			</aside>
		</div>
	</div>
</template>

<style scoped>
.block-explorer {
	margin: 1.5rem 0;
	border: 1px solid rgba(148, 163, 184, 0.35);
	border-radius: 24px;
	background: linear-gradient(180deg, rgba(248, 250, 252, 0.97), rgba(241, 245, 249, 0.96));
	box-shadow: 0 20px 44px rgba(15, 23, 42, 0.06);
	overflow: hidden;
}

.block-explorer__topbar {
	display: flex;
	gap: 0.75rem;
	padding: 1rem;
	border-bottom: 1px solid rgba(148, 163, 184, 0.2);
	background: rgba(255, 255, 255, 0.72);
}

.block-explorer__tab {
	padding: 0.7rem 1rem;
	border: 1px solid rgba(148, 163, 184, 0.45);
	border-radius: 999px;
	background: #fff;
	color: #334155;
	cursor: pointer;
	font-weight: 600;
	transition: all 0.18s ease;
}

.block-explorer__tab--active {
	border-color: #2563eb;
	background: #2563eb;
	color: #fff;
	box-shadow: 0 10px 24px rgba(37, 99, 235, 0.2);
}

.block-explorer__body {
	display: grid;
	grid-template-columns: minmax(0, 1.7fr) minmax(280px, 1fr);
	gap: 1rem;
	padding: 1rem;
}

.block-explorer__diagram {
	position: relative;
	display: flex;
	flex-direction: column;
	align-items: center;
	gap: 1rem;
	padding: 0.5rem 0.5rem 0.5rem;
}

.block-explorer__rail {
	position: absolute;
	top: 1rem;
	bottom: 1rem;
	left: 50%;
	width: 3px;
	transform: translateX(-50%);
	background: linear-gradient(180deg, rgba(100, 116, 139, 0.25), rgba(37, 99, 235, 0.35), rgba(100, 116, 139, 0.25));
}

.block-explorer__step {
	position: relative;
	z-index: 1;
}

.block-explorer__node {
	min-width: 240px;
	padding: 1rem 1.2rem;
	border-radius: 18px;
	border: 1px solid rgba(148, 163, 184, 0.45);
	background: #fff;
	color: #0f172a;
	font-size: 1.02rem;
	font-weight: 600;
	cursor: pointer;
	transition: transform 0.18s ease, box-shadow 0.18s ease, border-color 0.18s ease;
}

.block-explorer__step--active .block-explorer__node,
.block-explorer__node:hover {
	transform: translateY(-1px);
	box-shadow: 0 12px 28px rgba(15, 23, 42, 0.1);
}

.block-explorer__step--slate .block-explorer__node {
	border-color: rgba(100, 116, 139, 0.45);
	background: linear-gradient(180deg, rgba(255, 255, 255, 0.97), rgba(248, 250, 252, 0.97));
}

.block-explorer__step--blue .block-explorer__node {
	border-color: rgba(14, 165, 233, 0.65);
	background: linear-gradient(180deg, rgba(224, 242, 254, 0.98), rgba(240, 249, 255, 0.98));
	color: #0c4a6e;
}

.block-explorer__step--violet .block-explorer__node {
	border-color: rgba(147, 51, 234, 0.55);
	background: linear-gradient(180deg, rgba(245, 243, 255, 0.98), rgba(250, 245, 255, 0.98));
	color: #581c87;
}

.block-explorer__step--green .block-explorer__node {
	border-color: rgba(34, 197, 94, 0.6);
	background: linear-gradient(180deg, rgba(236, 253, 245, 0.98), rgba(240, 253, 244, 0.98));
	color: #14532d;
}

.block-explorer__step--amber .block-explorer__node {
	border-color: rgba(245, 158, 11, 0.6);
	background: linear-gradient(180deg, rgba(255, 247, 237, 0.98), rgba(255, 251, 235, 0.98));
	color: #78350f;
}

.block-explorer__step--active .block-explorer__node {
	outline: 3px solid rgba(37, 99, 235, 0.12);
}

.block-explorer__panel {
	padding: 1rem 1rem 1.1rem;
	border-radius: 20px;
	background: rgba(255, 255, 255, 0.76);
	border: 1px solid rgba(148, 163, 184, 0.28);
}

.block-explorer__panel-kicker {
	margin: 0;
	font-size: 0.76rem;
	letter-spacing: 0.08em;
	text-transform: uppercase;
	color: #64748b;
}

.block-explorer__panel h4 {
	margin: 0.28rem 0 0.85rem;
	font-size: 1.08rem;
	color: #0f172a;
}

.block-explorer__badges {
	display: flex;
	flex-wrap: wrap;
	gap: 0.45rem;
	margin-bottom: 0.8rem;
}

.block-explorer__badge {
	padding: 0.32rem 0.58rem;
	border-radius: 999px;
	background: rgba(37, 99, 235, 0.08);
	color: #1d4ed8;
	font-size: 0.76rem;
	font-weight: 700;
}

.block-explorer__formula {
	margin: 0 0 0.9rem;
	padding: 0.8rem 0.9rem;
	border-radius: 14px;
	background: rgba(15, 23, 42, 0.92);
	color: #e2e8f0;
	overflow-x: auto;
}

.block-explorer__formula :deep(.katex) {
	color: inherit;
	font-size: 1.04em;
}

.block-explorer__formula :deep(.katex-display) {
	margin: 0;
}

.block-explorer__panel p {
	margin: 0;
	line-height: 1.85;
	color: #334155;
}

@media (max-width: 960px) {
	.block-explorer__body {
		grid-template-columns: 1fr;
	}

	.block-explorer__node {
		min-width: min(100%, 280px);
	}
}
</style>
