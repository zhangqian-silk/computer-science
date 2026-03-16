<template>
	<div class="ngram-smoothing-explorer">
		<div class="hero-card">
			<div class="hero-text">
				<p class="eyebrow">N-Gram Smoothing Explorer</p>
				<h2>平滑、回退、插值与 Kneser-Ney 的机制对比图</h2>
				<p class="summary">
					该图把 N-Gram 文档中最容易读散的部分集中到同一视图：零概率问题为什么会出现，拉普拉斯平滑、回退、插值分别在重分配什么概率质量，以及 Kneser-Ney 为什么不只看词频，还要看续接能力。
				</p>
			</div>
			<div class="legend-grid">
				<div class="legend-item issue">零概率 / 稀疏性</div>
				<div class="legend-item smoothing">平滑与插值</div>
				<div class="legend-item backoff">回退机制</div>
				<div class="legend-item kn">Kneser-Ney</div>
			</div>
		</div>

		<div class="controls-card">
			<div class="toggle-group" role="tablist" aria-label="N-Gram 平滑视图切换">
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
				<div class="problem-card" :class="{ focus: activeMode === 'overview' || activeMode === 'problem' }">
					<div class="section-head">
						<h3>1. 问题起点：MLE 遇到未见组合时会直接给出 0</h3>
						<span>零概率会把整句概率直接压成 0</span>
					</div>
					<div class="formula-box">$P(w_t\mid h)=\frac{C(h,w_t)}{C(h)}$</div>
					<div class="mini-flow">
						<div class="chip">若 $C(h,w_t)=0$</div>
						<div class="arrow">→</div>
						<div class="chip">则 $P(w_t\mid h)=0$</div>
						<div class="arrow">→</div>
						<div class="chip">整句概率可能退化为 0</div>
					</div>
					<p>平滑的出发点不是“让公式更复杂”，而是重新分配高频组合占据的部分概率质量，使未见组合获得非零估计。</p>
				</div>

				<div class="method-grid">
					<div class="method-card laplace-card" :class="{ focus: activeMode === 'laplace' }">
						<h4>拉普拉斯平滑</h4>
						<div class="formula-box">$\frac{C(h,w)+1}{C(h)+|V|}$</div>
						<p>对所有候选统一加 1，最简单，但大词表下容易过平滑。</p>
					</div>
					<div class="method-card backoff-card" :class="{ focus: activeMode === 'backoff' }">
						<h4>回退</h4>
						<div class="mini-flow vertical">
							<div class="chip">查高阶 $P(w\mid h)$</div>
							<div class="arrow">↓</div>
							<div class="chip">若未命中则去掉最左历史</div>
							<div class="arrow">↓</div>
							<div class="chip">继续查低阶模型</div>
						</div>
						<p>高阶优先，缺失时回到低阶模型。</p>
					</div>
					<div class="method-card interp-card" :class="{ focus: activeMode === 'interpolation' }">
						<h4>插值</h4>
						<div class="formula-box">$P=\lambda_3P_3+\lambda_2P_2+\lambda_1P_1$</div>
						<p>不是“高阶失效才启用低阶”，而是多个阶数始终共同参与估计。</p>
					</div>
				</div>

				<div class="kn-card" :class="{ focus: activeMode === 'kn' || activeMode === 'overview' }">
					<div class="section-head">
						<h3>2. Kneser-Ney：低阶概率不只看出现次数，更看续接能力</h3>
						<span>频率高 ≠ 在新上下文里更合理</span>
					</div>
					<div class="formula-box">$P_{KN}(w\mid h)=\frac{\max(C(h,w)-d,0)}{C(h)}+\lambda(h)P_{cont}(w)$</div>
					<div class="compare-grid">
						<div class="compare-card">
							<h5>普通低阶直觉</h5>
							<p>看 $C(w)$：一个词总出现次数越高，低阶概率越大。</p>
						</div>
						<div class="compare-card emphasis">
							<h5>KN 低阶直觉</h5>
							<p>看 $N_{1+}(\ast,w)$：一个词能接在多少种不同前词后，续接能力越强，低阶概率越大。</p>
						</div>
					</div>
					<p>这就是为什么「旧金山」里的“金山”虽然总频率高，却未必应该在陌生新上下文里获得更高概率。</p>
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
						<span>方法</span>
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

type ModeKey = 'overview' | 'problem' | 'laplace' | 'backoff' | 'interpolation' | 'kn'

const activeMode = ref<ModeKey>('overview')

const modes: { key: ModeKey; label: string; desc: string }[] = [
	{ key: 'overview', label: '总览', desc: '先看 N-Gram 稀疏性问题如何催生平滑、回退、插值与 KN。' },
	{ key: 'problem', label: '零概率问题', desc: '聚焦 MLE 在未见组合上为何会失效。' },
	{ key: 'laplace', label: '拉普拉斯', desc: '聚焦最直接的统一加一平滑，以及它为何容易过平滑。' },
	{ key: 'backoff', label: '回退', desc: '聚焦高阶缺失时如何逐级退回低阶模型。' },
	{ key: 'interpolation', label: '插值', desc: '聚焦不同阶数如何共同参与最终概率估计。' },
	{ key: 'kn', label: 'Kneser-Ney', desc: '聚焦续接概率如何修正低阶分布。' }
]

const panels = {
	overview: {
		title: '总览：N-Gram 的真正难点不在计数本身，而在如何合理分配未见事件的概率质量',
		lead: '平滑方法都在回答同一个问题：高频组合占据的大量概率，应该怎样让渡给低频或未见组合。',
		points: [
			'拉普拉斯最简单，但往往把高频项稀释得过重。',
			'回退强调“高阶优先，缺失时退低阶”；插值强调“多阶同时参与”。',
			'Kneser-Ney 进一步修正低阶概率来源，使其更像“上下文多样性”而非“纯词频”。'
		],
		rows: [
			{ label: '拉普拉斯', value: '统一加一，简单直接但容易过平滑' },
			{ label: '回退 / 插值', value: '在不同阶数之间重新组织概率质量' },
			{ label: 'Kneser-Ney', value: '通过续接能力定义更合理的低阶分布' }
		]
	},
	problem: {
		title: '零概率问题：未见组合会让 MLE 直接失效',
		lead: '统计模型的第一道坎，不是公式写不出来，而是很多合理组合在训练语料里根本没出现过。',
		points: [
			'MLE 只依赖观测计数，因此未见即为零。',
			'句子概率是条件概率乘积，只要一项为零，整句就会退化。',
			'平滑方法的本质是避免这种“未见即不可能”的极端判断。'
		],
		rows: [
			{ label: '原因', value: '语料有限而组合空间巨大' },
			{ label: '表现', value: '未见 N-Gram 被赋值为 0' },
			{ label: '后果', value: '整句概率被某个未见组合直接归零' }
		]
	},
	laplace: {
		title: '拉普拉斯平滑：最直观，但也最容易过度平均化',
		lead: '加一平滑的优点是概念简单，缺点是它把所有未见组合都当成同等候选，从而过度稀释已见高频项。',
		points: [
			'所有候选项统一加 1。',
			'分母也相应加上词表大小。',
			'在大词表下，真实高频项和极不合理项的差距会被压得过小。'
		],
		rows: [
			{ label: '优点', value: '实现最简单，适合教学演示' },
			{ label: '问题', value: '默认所有未见词组先验相同，容易过平滑' },
			{ label: '结论', value: '更适合作为入门方法，而非高质量统计语言模型主力方案' }
		]
	},
	backoff: {
		title: '回退：让高阶统计先说话，缺失时再去问低阶模型',
		lead: '回退的逻辑是“高阶更具体，但也更稀疏”，因此只有在高阶失效时才把部分概率质量交给低阶模型。',
		points: [
			'如果高阶组合见过，就优先使用高阶统计。',
			'如果高阶组合未见，就删掉最左历史继续查低阶分布。',
			'回退权重负责把释放出的概率质量交给低阶模型。'
		],
		rows: [
			{ label: '优势', value: '结构清楚，符合“具体上下文优先”的直觉' },
			{ label: '局限', value: '高阶与低阶信息常是切换式，而非同时利用' },
			{ label: '代表', value: 'Katz Backoff 等经典统计语言模型方案' }
		]
	},
	interpolation: {
		title: '插值：高阶、中阶、低阶始终一起参与最终估计',
		lead: '插值的核心不是回退，而是融合：哪怕高阶组合已经出现，低阶统计仍然保留稳定补充信息。',
		points: [
			'所有阶数都用权重参与最终概率。',
			'权重通常通过验证集估计。',
			'相比硬回退，它更平滑，也更稳健。'
		],
		rows: [
			{ label: '核心', value: '多阶分布共同参与，而不是失效后才启用低阶' },
			{ label: '优势', value: '估计更稳定，抗稀疏性更强' },
			{ label: '代价', value: '需要估计合适权重，并管理多阶概率融合' }
		]
	},
	kn: {
		title: 'Kneser-Ney：把低阶概率从“词频”改成“续接能力”',
		lead: 'KN 的高明之处不是又发明一个折扣常数，而是重新定义了低阶分布的来源。',
		points: [
			'高阶部分仍然先做绝对折扣。',
			'释放出的概率质量通过回退权重交给低阶模型。',
			'低阶模型不再简单按词频分配，而是看一个词可出现在多少种不同前缀之后。'
		],
		rows: [
			{ label: '折扣项', value: '对已见组合减去固定折扣，腾出概率质量' },
			{ label: '续接概率', value: '强调词的上下文多样性，而不是总频率' },
			{ label: '意义', value: '在稀疏场景下通常比简单平滑更合理' }
		]
	}
} as const

const activeDescription = computed(() => modes.find((mode) => mode.key === activeMode.value)?.desc ?? '')
const activePanel = computed(() => panels[activeMode.value])
</script>

<style scoped>
.ngram-smoothing-explorer {
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

.legend-item.issue {
	background: color-mix(in srgb, #ef4444 12%, var(--vp-c-bg));
	border-color: color-mix(in srgb, #ef4444 28%, transparent);
}

.legend-item.smoothing {
	background: color-mix(in srgb, #06b6d4 14%, var(--vp-c-bg));
	border-color: color-mix(in srgb, #06b6d4 32%, transparent);
}

.legend-item.backoff {
	background: color-mix(in srgb, #f59e0b 14%, var(--vp-c-bg));
	border-color: color-mix(in srgb, #f59e0b 32%, transparent);
}

.legend-item.kn {
	background: color-mix(in srgb, #8b5cf6 14%, var(--vp-c-bg));
	border-color: color-mix(in srgb, #8b5cf6 32%, transparent);
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
	grid-template-columns: minmax(0, 1.78fr) minmax(300px, 0.9fr);
	gap: 20px;
}

.diagram-card,
.insight-card {
	padding: 22px;
}

.problem-card,
.method-card,
.kn-card,
.compare-card {
	padding: 18px;
	border-radius: 18px;
	border: 1px solid var(--vp-c-divider);
	background: color-mix(in srgb, var(--vp-c-bg-soft) 84%, #ffffff 16%);
}

.problem-card.focus,
.method-card.focus,
.kn-card.focus {
	box-shadow: 0 0 0 3px color-mix(in srgb, var(--vp-c-brand-1) 18%, transparent);
}

.section-head {
	display: flex;
	justify-content: space-between;
	align-items: baseline;
	gap: 12px;
	margin-bottom: 14px;
}

.section-head h3 {
	margin: 0;
	font-size: 22px;
}

.section-head span {
	font-size: 13px;
	font-weight: 600;
	color: var(--vp-c-text-2);
}

.formula-box,
.chip,
.arrow {
	padding: 10px 12px;
	border-radius: 14px;
	background: var(--vp-c-bg);
	border: 1px solid var(--vp-c-divider);
	font-weight: 600;
}

.formula-box {
	margin-bottom: 12px;
	font-family: var(--vp-font-family-mono);
	font-size: 13px;
	word-break: break-word;
}

.mini-flow {
	display: flex;
	align-items: center;
	gap: 10px;
	flex-wrap: wrap;
	margin-bottom: 12px;
}

.mini-flow.vertical {
	flex-direction: column;
	align-items: stretch;
}

.arrow {
	background: transparent;
	border: none;
	padding: 0;
	font-size: 18px;
}

.problem-card p,
.method-card p,
.kn-card p,
.compare-card p {
	margin: 0;
	line-height: 1.7;
	color: var(--vp-c-text-2);
}

.method-grid,
.compare-grid {
	display: grid;
	grid-template-columns: repeat(3, minmax(0, 1fr));
	gap: 16px;
	margin-top: 16px;
}

.laplace-card {
	background: color-mix(in srgb, #06b6d4 8%, var(--vp-c-bg));
}

.backoff-card {
	background: color-mix(in srgb, #f59e0b 8%, var(--vp-c-bg));
}

.interp-card {
	background: color-mix(in srgb, #10b981 8%, var(--vp-c-bg));
}

.kn-card {
	margin-top: 16px;
	background: color-mix(in srgb, #8b5cf6 8%, var(--vp-c-bg));
}

.compare-card.emphasis {
	background: color-mix(in srgb, #8b5cf6 10%, var(--vp-c-bg));
}

.compare-card h5,
.method-card h4 {
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
	.method-grid,
	.compare-grid {
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
