<script setup lang="ts">
import { computed, ref } from 'vue'
import { renderMath } from '../utils/renderMath'

type Mode = {
	id: 'plain' | 'causal'
	title: string
	description: string
}

const modes: Mode[] = [
	{
		id: 'plain',
		title: '无掩码',
		description: '所有位置都可参与匹配，适合理解完整输入时的全局建模。'
	},
	{
		id: 'causal',
		title: '因果掩码',
		description: '未来位置会被遮盖，适合自回归生成场景。'
	}
]

const tokens = ['x1', 'x2', 'x3', 'x4']

const baseScores = [
	[2.6, 1.1, 0.7, 0.2],
	[1.0, 2.4, 1.7, 0.8],
	[0.3, 1.2, 2.7, 1.9],
	[0.1, 0.8, 1.5, 2.8]
]

const activeMode = ref<Mode['id']>('plain')
const activeRow = ref(1)

const maskedScores = computed(() =>
	baseScores.map((row, rowIndex) =>
		row.map((value, columnIndex) => {
			if (activeMode.value === 'causal' && columnIndex > rowIndex) {
				return Number.NEGATIVE_INFINITY
			}

			return value
		})
	)
)

const currentMode = computed(() => modes.find((mode) => mode.id === activeMode.value) ?? modes[0])
const currentRawRow = computed(() => baseScores[activeRow.value])
const currentMaskedRow = computed(() => maskedScores.value[activeRow.value])

function softmax(row: number[]) {
	const finiteValues = row.filter((value) => Number.isFinite(value))
	const maxValue = Math.max(...finiteValues)
	const exps = row.map((value) => (Number.isFinite(value) ? Math.exp(value - maxValue) : 0))
	const sum = exps.reduce((total, value) => total + value, 0)

	return exps.map((value) => value / sum)
}

const currentWeights = computed(() => softmax(currentMaskedRow.value))

const visibleTargets = computed(() =>
	tokens.filter((_, index) => currentWeights.value[index] > 0)
)

const outputExpression = computed(() =>
	currentWeights.value
		.map((weight, index) => `${weight.toFixed(2)}v_{${index + 1}}`)
		.join(' + ')
)

function formatScore(value: number) {
	return Number.isFinite(value) ? value.toFixed(1) : '-inf'
}

function weightPercent(value: number) {
	return `${Math.max(6, value * 100)}%`
}

const outputFormula = computed(() => `o_{${activeRow.value + 1}} = ${outputExpression.value}`)

function renderInlineMath(tex: string) {
	return renderMath(tex, false)
}
</script>

<template>
	<div class="matrix-explorer">
		<div class="matrix-explorer__toolbar">
			<div class="matrix-explorer__group">
				<span class="matrix-explorer__label">模式</span>
				<div class="matrix-explorer__chips">
					<button
						v-for="mode in modes"
						:key="mode.id"
						type="button"
						class="matrix-explorer__chip"
						:class="{ 'matrix-explorer__chip--active': mode.id === activeMode }"
						@click="activeMode = mode.id"
					>
						{{ mode.title }}
					</button>
				</div>
			</div>

			<div class="matrix-explorer__group">
				<span class="matrix-explorer__label">观察行</span>
				<div class="matrix-explorer__chips">
					<button
						v-for="(token, rowIndex) in tokens"
						:key="token"
						type="button"
						class="matrix-explorer__chip matrix-explorer__chip--muted"
						:class="{ 'matrix-explorer__chip--active': rowIndex === activeRow }"
						@click="activeRow = rowIndex"
					>
						q{{ rowIndex + 1 }}
					</button>
				</div>
			</div>
		</div>

		<p class="matrix-explorer__lead">
			当前处于「{{ currentMode.title }}」模式，正在观察第 {{ activeRow + 1 }} 行 query。
			{{ currentMode.description }}
		</p>

		<div class="matrix-explorer__grid">
			<section class="matrix-explorer__card">
				<div class="matrix-explorer__card-head">
					<p class="matrix-explorer__kicker">Step 1</p>
					<h4>原始分数矩阵</h4>
				</div>

				<div class="matrix-explorer__table-wrap">
					<table class="matrix-explorer__table">
						<thead>
							<tr>
								<th />
								<th v-for="(token, columnIndex) in tokens" :key="token">k{{ columnIndex + 1 }}</th>
							</tr>
						</thead>
						<tbody>
							<tr
								v-for="(row, rowIndex) in baseScores"
								:key="`raw-${rowIndex}`"
								:class="{ 'matrix-explorer__row--active': rowIndex === activeRow }"
							>
								<th>
									<button type="button" @click="activeRow = rowIndex">q{{ rowIndex + 1 }}</button>
								</th>
								<td v-for="(value, columnIndex) in row" :key="`raw-${rowIndex}-${columnIndex}`">
									{{ formatScore(value) }}
								</td>
							</tr>
						</tbody>
					</table>
				</div>

				<p class="matrix-explorer__hint">
					第 {{ activeRow + 1 }} 行表示当前 query 与所有 key 的原始相关性。
				</p>
			</section>

			<section class="matrix-explorer__card">
				<div class="matrix-explorer__card-head">
					<p class="matrix-explorer__kicker">Step 2</p>
					<h4>加掩码后的矩阵</h4>
				</div>

				<div class="matrix-explorer__table-wrap">
					<table class="matrix-explorer__table">
						<thead>
							<tr>
								<th />
								<th v-for="(token, columnIndex) in tokens" :key="token">k{{ columnIndex + 1 }}</th>
							</tr>
						</thead>
						<tbody>
							<tr
								v-for="(row, rowIndex) in maskedScores"
								:key="`masked-${rowIndex}`"
								:class="{ 'matrix-explorer__row--active': rowIndex === activeRow }"
							>
								<th>
									<button type="button" @click="activeRow = rowIndex">q{{ rowIndex + 1 }}</button>
								</th>
								<td
									v-for="(value, columnIndex) in row"
									:key="`masked-${rowIndex}-${columnIndex}`"
									:class="{ 'matrix-explorer__cell--blocked': !Number.isFinite(value) }"
								>
									{{ formatScore(value) }}
								</td>
							</tr>
						</tbody>
					</table>
				</div>

				<p class="matrix-explorer__hint">
					在「因果掩码」模式下，右上角未来位置会被直接置为 <code>-inf</code>。
				</p>
			</section>

			<section class="matrix-explorer__card">
				<div class="matrix-explorer__card-head">
					<p class="matrix-explorer__kicker">Step 3</p>
					<h4>softmax 后的权重</h4>
				</div>

				<div class="matrix-explorer__bars">
					<div
						v-for="(weight, index) in currentWeights"
						:key="`weight-${index}`"
						class="matrix-explorer__bar-row"
					>
						<span class="matrix-explorer__bar-label">a{{ activeRow + 1 }}{{ index + 1 }}</span>
						<div class="matrix-explorer__bar-track">
							<div class="matrix-explorer__bar-fill" :style="{ width: weightPercent(weight) }" />
						</div>
						<span class="matrix-explorer__bar-value">{{ weight.toFixed(2) }}</span>
					</div>
				</div>

				<p class="matrix-explorer__hint">
					softmax 把第 {{ activeRow + 1 }} 行分数压成概率分布，行和恒为 1。
				</p>
			</section>

			<section class="matrix-explorer__card matrix-explorer__card--accent">
				<div class="matrix-explorer__card-head">
					<p class="matrix-explorer__kicker">Step 4</p>
					<h4>加权聚合输出</h4>
				</div>

				<div class="matrix-explorer__value-list">
					<span
						v-for="(token, index) in tokens"
						:key="`value-${token}`"
						class="matrix-explorer__value-chip"
						:class="{ 'matrix-explorer__value-chip--active': currentWeights[index] > 0.2 }"
					>
						{{ token }} -> v{{ index + 1 }}
					</span>
				</div>

				<div class="matrix-explorer__formula" v-html="renderInlineMath(outputFormula)" />
				<p class="matrix-explorer__hint">
					当前输出主要由 {{ visibleTargets.join('、') }} 对应的 value 共同组成，只是占比不同。
				</p>
			</section>
		</div>
	</div>
</template>

<style scoped>
.matrix-explorer {
	margin: 1.5rem 0;
	padding: 1rem;
	border: 1px solid rgba(148, 163, 184, 0.35);
	border-radius: 24px;
	background: linear-gradient(180deg, rgba(248, 250, 252, 0.98), rgba(241, 245, 249, 0.96));
	box-shadow: 0 20px 44px rgba(15, 23, 42, 0.06);
}

.matrix-explorer__toolbar {
	display: flex;
	flex-wrap: wrap;
	gap: 1rem;
	justify-content: space-between;
}

.matrix-explorer__group {
	display: flex;
	flex-wrap: wrap;
	align-items: center;
	gap: 0.75rem;
}

.matrix-explorer__label {
	font-size: 0.84rem;
	font-weight: 700;
	color: #475569;
}

.matrix-explorer__chips {
	display: flex;
	flex-wrap: wrap;
	gap: 0.55rem;
}

.matrix-explorer__chip {
	padding: 0.62rem 0.9rem;
	border: 1px solid rgba(148, 163, 184, 0.45);
	border-radius: 999px;
	background: #fff;
	color: #334155;
	font-weight: 600;
	cursor: pointer;
	transition: all 0.18s ease;
}

.matrix-explorer__chip--muted {
	background: rgba(255, 255, 255, 0.82);
}

.matrix-explorer__chip--active {
	border-color: #2563eb;
	background: #2563eb;
	color: #fff;
	box-shadow: 0 10px 24px rgba(37, 99, 235, 0.2);
}

.matrix-explorer__lead {
	margin: 1rem 0 0;
	line-height: 1.85;
	color: #334155;
}

.matrix-explorer__grid {
	display: grid;
	grid-template-columns: repeat(2, minmax(0, 1fr));
	gap: 1rem;
	margin-top: 1rem;
}

.matrix-explorer__card {
	padding: 1rem;
	border-radius: 20px;
	border: 1px solid rgba(148, 163, 184, 0.28);
	background: rgba(255, 255, 255, 0.84);
}

.matrix-explorer__card--accent {
	background: linear-gradient(180deg, rgba(239, 246, 255, 0.98), rgba(224, 242, 254, 0.98));
	border-color: rgba(37, 99, 235, 0.22);
}

.matrix-explorer__card-head {
	display: flex;
	align-items: baseline;
	justify-content: space-between;
	gap: 1rem;
	margin-bottom: 0.85rem;
}

.matrix-explorer__kicker {
	margin: 0;
	font-size: 0.74rem;
	letter-spacing: 0.08em;
	text-transform: uppercase;
	color: #64748b;
}

.matrix-explorer__card-head h4 {
	margin: 0;
	font-size: 1rem;
	color: #0f172a;
}

.matrix-explorer__table-wrap {
	overflow-x: auto;
}

.matrix-explorer__table {
	width: 100%;
	border-collapse: separate;
	border-spacing: 0.35rem;
}

.matrix-explorer__table th,
.matrix-explorer__table td {
	min-width: 3.25rem;
	padding: 0.6rem 0.45rem;
	border-radius: 12px;
	text-align: center;
	font-size: 0.88rem;
}

.matrix-explorer__table thead th {
	color: #64748b;
	font-weight: 700;
}

.matrix-explorer__table tbody th button {
	width: 100%;
	padding: 0.6rem 0.45rem;
	border: 1px solid rgba(148, 163, 184, 0.34);
	border-radius: 12px;
	background: rgba(248, 250, 252, 0.98);
	color: #334155;
	font-weight: 700;
	cursor: pointer;
}

.matrix-explorer__table td {
	border: 1px solid rgba(148, 163, 184, 0.24);
	background: rgba(248, 250, 252, 0.92);
	color: #0f172a;
}

.matrix-explorer__row--active td,
.matrix-explorer__row--active th button {
	border-color: rgba(37, 99, 235, 0.45);
	background: rgba(219, 234, 254, 0.8);
}

.matrix-explorer__cell--blocked {
	background: rgba(254, 226, 226, 0.82) !important;
	border-color: rgba(239, 68, 68, 0.28) !important;
	color: #b91c1c !important;
}

.matrix-explorer__bars {
	display: flex;
	flex-direction: column;
	gap: 0.72rem;
}

.matrix-explorer__bar-row {
	display: grid;
	grid-template-columns: 3.3rem minmax(0, 1fr) 3rem;
	align-items: center;
	gap: 0.75rem;
}

.matrix-explorer__bar-label,
.matrix-explorer__bar-value {
	font-size: 0.84rem;
	font-weight: 700;
	color: #334155;
}

.matrix-explorer__bar-track {
	height: 0.85rem;
	border-radius: 999px;
	background: rgba(148, 163, 184, 0.18);
	overflow: hidden;
}

.matrix-explorer__bar-fill {
	height: 100%;
	border-radius: 999px;
	background: linear-gradient(90deg, #38bdf8, #2563eb);
}

.matrix-explorer__value-list {
	display: flex;
	flex-wrap: wrap;
	gap: 0.55rem;
}

.matrix-explorer__value-chip {
	padding: 0.46rem 0.7rem;
	border-radius: 999px;
	background: rgba(255, 255, 255, 0.78);
	border: 1px solid rgba(148, 163, 184, 0.28);
	color: #334155;
	font-size: 0.82rem;
	font-weight: 600;
}

.matrix-explorer__value-chip--active {
	border-color: rgba(37, 99, 235, 0.35);
	background: rgba(219, 234, 254, 0.95);
	color: #1d4ed8;
}

.matrix-explorer__formula {
	margin: 0.9rem 0 0;
	padding: 0.85rem 0.9rem;
	border-radius: 14px;
	background: rgba(15, 23, 42, 0.94);
	color: #e2e8f0;
	overflow-x: auto;
}

.matrix-explorer__formula :deep(.katex) {
	color: inherit;
	font-size: 1.04em;
}

.matrix-explorer__formula :deep(.katex-display) {
	margin: 0;
}

.matrix-explorer__hint {
	margin: 0.85rem 0 0;
	line-height: 1.8;
	color: #475569;
}

@media (max-width: 960px) {
	.matrix-explorer__grid {
		grid-template-columns: 1fr;
	}
}

@media (max-width: 640px) {
	.matrix-explorer {
		padding: 0.9rem;
	}

	.matrix-explorer__bar-row {
		grid-template-columns: 3rem minmax(0, 1fr) 2.8rem;
		gap: 0.55rem;
	}
}
</style>
