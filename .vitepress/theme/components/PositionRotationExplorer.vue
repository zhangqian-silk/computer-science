<script setup lang="ts">
import { computed, ref } from 'vue'

const mode = ref<'sinusoidal' | 'rope'>('sinusoidal')
const positionA = ref(2)
const positionB = ref(5)

const center = 110
const radius = 72
const stepAngle = computed(() => (mode.value === 'sinusoidal' ? 0.55 : 0.42))

const angleA = computed(() => -Math.PI / 2 + positionA.value * stepAngle.value)
const angleB = computed(() => -Math.PI / 2 + positionB.value * stepAngle.value)

const pointA = computed(() => ({
	x: center + radius * Math.cos(angleA.value),
	y: center + radius * Math.sin(angleA.value)
}))

const pointB = computed(() => ({
	x: center + radius * Math.cos(angleB.value),
	y: center + radius * Math.sin(angleB.value)
}))

const delta = computed(() => positionB.value - positionA.value)
const deltaAngle = computed(() => ((angleB.value - angleA.value) * 180) / Math.PI)

const title = computed(() =>
	mode.value === 'sinusoidal' ? '正弦余弦位置编码：同一频率下的位置旋转' : 'RoPE：Q / K 相位差如何转成相对位移'
)

const summary = computed(() =>
	mode.value === 'sinusoidal'
		? '把某一对 sin / cos 维度看成单位圆坐标后，位置前进就等价于沿圆周继续旋转。'
		: '在 RoPE 中，query 和 key 会按各自位置旋转；内积真正敏感的是它们之间的角度差。'
)

const footnote = computed(() =>
	mode.value === 'sinusoidal'
		? `位置 ${positionA.value} 到位置 ${positionB.value} 的变化，可以理解为在同一二维子空间里额外旋转了 ${delta.value} 步。`
		: `当 query 位于位置 ${positionA.value}、key 位于位置 ${positionB.value} 时，attention 更直接感知的是两者相差 ${delta.value} 个位置。`
)
</script>

<template>
	<div class="rotation-explorer">
		<div class="rotation-explorer__tabs">
			<button
				type="button"
				class="rotation-explorer__tab"
				:class="{ 'rotation-explorer__tab--active': mode === 'sinusoidal' }"
				@click="mode = 'sinusoidal'"
			>
				Sinusoidal
			</button>
			<button
				type="button"
				class="rotation-explorer__tab"
				:class="{ 'rotation-explorer__tab--active': mode === 'rope' }"
				@click="mode = 'rope'"
			>
				RoPE
			</button>
		</div>

		<div class="rotation-explorer__body">
			<div class="rotation-explorer__copy">
				<p class="rotation-explorer__kicker">{{ mode === 'sinusoidal' ? '2D Rotation' : 'Phase Difference' }}</p>
				<h4>{{ title }}</h4>
				<p>{{ summary }}</p>

				<label class="rotation-explorer__control">
					<span>位置 A：{{ positionA }}</span>
					<input v-model="positionA" type="range" min="0" max="10" step="1" />
				</label>

				<label class="rotation-explorer__control">
					<span>{{ mode === 'sinusoidal' ? '位置 B' : '比较位置 B' }}：{{ positionB }}</span>
					<input v-model="positionB" type="range" min="0" max="10" step="1" />
				</label>

				<div class="rotation-explorer__chips">
					<span class="rotation-explorer__chip">位置差：{{ delta }}</span>
					<span class="rotation-explorer__chip">角度差：{{ deltaAngle.toFixed(1) }}°</span>
				</div>
			</div>

			<div class="rotation-explorer__viz">
				<svg viewBox="0 0 220 220" class="rotation-explorer__svg" aria-hidden="true">
					<circle cx="110" cy="110" r="72" class="rotation-explorer__ring" />
					<line x1="110" y1="110" :x2="pointA.x" :y2="pointA.y" class="rotation-explorer__ray" />
					<line x1="110" y1="110" :x2="pointB.x" :y2="pointB.y" class="rotation-explorer__ray rotation-explorer__ray--accent" />
					<circle :cx="pointA.x" :cy="pointA.y" r="7" class="rotation-explorer__point" />
					<circle :cx="pointB.x" :cy="pointB.y" r="7" class="rotation-explorer__point rotation-explorer__point--accent" />
					<text :x="pointA.x" :y="pointA.y - 12" class="rotation-explorer__label">A</text>
					<text :x="pointB.x" :y="pointB.y - 12" class="rotation-explorer__label rotation-explorer__label--accent">B</text>
					<circle cx="110" cy="110" r="4" class="rotation-explorer__center" />
				</svg>
			</div>
		</div>

		<p class="rotation-explorer__footnote">{{ footnote }}</p>
	</div>
</template>

<style scoped>
.rotation-explorer {
	margin: 1.5rem 0;
	border: 1px solid rgba(148, 163, 184, 0.35);
	border-radius: 24px;
	background:
		radial-gradient(circle at top left, rgba(191, 219, 254, 0.24), transparent 28%),
		linear-gradient(180deg, rgba(248, 250, 252, 0.98), rgba(241, 245, 249, 0.96));
	box-shadow: 0 18px 40px rgba(15, 23, 42, 0.06);
	overflow: hidden;
}

.rotation-explorer__tabs {
	display: flex;
	gap: 0.75rem;
	padding: 1rem;
	border-bottom: 1px solid rgba(148, 163, 184, 0.22);
	background: rgba(255, 255, 255, 0.8);
}

.rotation-explorer__tab {
	padding: 0.72rem 1rem;
	border-radius: 999px;
	border: 1px solid rgba(148, 163, 184, 0.45);
	background: #fff;
	color: #334155;
	font-weight: 600;
	cursor: pointer;
	transition: all 0.18s ease;
}

.rotation-explorer__tab--active {
	border-color: #0f766e;
	background: #0f766e;
	color: #fff;
	box-shadow: 0 12px 24px rgba(15, 118, 110, 0.18);
}

.rotation-explorer__body {
	display: grid;
	grid-template-columns: minmax(0, 1.1fr) minmax(260px, 0.9fr);
	gap: 1rem;
	padding: 1rem;
	align-items: center;
}

.rotation-explorer__kicker {
	margin: 0;
	font-size: 0.76rem;
	letter-spacing: 0.08em;
	text-transform: uppercase;
	color: #64748b;
}

.rotation-explorer__copy h4 {
	margin: 0.28rem 0 0.6rem;
	font-size: 1.08rem;
	color: #0f172a;
}

.rotation-explorer__copy p {
	margin: 0 0 0.85rem;
	line-height: 1.8;
	color: #334155;
}

.rotation-explorer__control {
	display: block;
	margin: 0.75rem 0;
}

.rotation-explorer__control span {
	display: block;
	margin-bottom: 0.35rem;
	font-size: 0.9rem;
	font-weight: 600;
	color: #334155;
}

.rotation-explorer__control input {
	width: 100%;
}

.rotation-explorer__chips {
	display: flex;
	flex-wrap: wrap;
	gap: 0.55rem;
	margin-top: 0.85rem;
}

.rotation-explorer__chip {
	padding: 0.42rem 0.72rem;
	border-radius: 999px;
	background: rgba(15, 118, 110, 0.1);
	color: #0f766e;
	font-size: 0.82rem;
	font-weight: 700;
}

.rotation-explorer__viz {
	display: flex;
	justify-content: center;
}

.rotation-explorer__svg {
	width: 220px;
	height: 220px;
}

.rotation-explorer__ring {
	fill: rgba(255, 255, 255, 0.8);
	stroke: rgba(148, 163, 184, 0.5);
	stroke-width: 1.5;
}

.rotation-explorer__ray {
	stroke: #2563eb;
	stroke-width: 3;
	stroke-linecap: round;
}

.rotation-explorer__ray--accent {
	stroke: #f59e0b;
}

.rotation-explorer__point {
	fill: #2563eb;
}

.rotation-explorer__point--accent {
	fill: #f59e0b;
}

.rotation-explorer__label {
	text-anchor: middle;
	font-size: 0.85rem;
	font-weight: 700;
	fill: #1d4ed8;
}

.rotation-explorer__label--accent {
	fill: #b45309;
}

.rotation-explorer__center {
	fill: #0f172a;
}

.rotation-explorer__footnote {
	margin: 0;
	padding: 0 1rem 1rem;
	line-height: 1.8;
	color: #475569;
}

@media (max-width: 860px) {
	.rotation-explorer__body {
		grid-template-columns: 1fr;
	}
}
</style>
