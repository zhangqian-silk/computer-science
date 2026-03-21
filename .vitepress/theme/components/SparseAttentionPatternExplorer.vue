<script setup lang="ts">
import { computed, ref } from 'vue'

type Pattern = {
	id: string
	title: string
	kicker: string
	description: string
	complexity: string
	properties: string[]
	rule: (query: number, key: number) => boolean
}

const positions = Array.from({ length: 12 }, (_, index) => index + 1)
const selectedPatternId = ref('local')
const selectedQuery = ref(6)

const randomNeighbors: Record<number, number[]> = {
	1: [5, 9],
	2: [6, 10],
	3: [7, 11],
	4: [8, 12],
	5: [1, 9],
	6: [2, 10],
	7: [3, 11],
	8: [4, 12],
	9: [1, 5],
	10: [2, 6],
	11: [3, 7],
	12: [4, 8]
}

const patterns: Pattern[] = [
	{
		id: 'local',
		title: '局部窗口',
		kicker: 'Sliding Window',
		description: '每个位置只与固定邻域交互，最符合“相关信息主要集中在附近”的归纳偏置。',
		complexity: '每行约 O(w) 个连接，整体常写为 O(nw)。',
		properties: ['局部性强', '实现简单', '跨层后感受野逐步扩展'],
		rule: (query, key) => Math.abs(query - key) <= 2
	},
	{
		id: 'stride',
		title: '局部 + 步长',
		kicker: 'Dilated / Strided',
		description: '在局部邻域之外，再保留按固定步长采样的远距离边，用更少连接覆盖更远范围。',
		complexity: '每行约 O(w + r) 个连接，常用于压缩长距离访问成本。',
		properties: ['远距离跳连', '覆盖更广', '适合长序列生成'],
		rule: (query, key) => Math.abs(query - key) <= 1 || (query - key) % 3 === 0
	},
	{
		id: 'block',
		title: '块稀疏',
		kicker: 'Block Sparse',
		description: '以块为单位决定可见性，把不规则边聚合为规则子矩阵，更容易映射到高效 kernel。',
		complexity: '若每个块只看常数个块，整体可近似降到 O(nbc)。',
		properties: ['硬件友好', '规则性高', '常与因果掩码结合'],
		rule: (query, key) => {
			const queryBlock = Math.floor((query - 1) / 4)
			const keyBlock = Math.floor((key - 1) / 4)
			return keyBlock === queryBlock || keyBlock === queryBlock - 1
		}
	},
	{
		id: 'hybrid',
		title: '全局 + 局部 + 随机',
		kicker: 'Global / Local / Random',
		description: '把少量全局节点、局部窗口和随机边组合起来，同时保证局部建模与跨区段连通性。',
		complexity: '每行约 O(g + w + r) 个连接，代表模式见于 BigBird 一类结构。',
		properties: ['图直径更小', '理论性质更强', '长文档任务常见'],
		rule: (query, key) => {
			if (Math.abs(query - key) <= 1) {
				return true
			}

			if (key === 1 || key === 12) {
				return true
			}

			return randomNeighbors[query]?.includes(key) ?? false
		}
	}
]

const activePattern = computed(() => patterns.find((pattern) => pattern.id === selectedPatternId.value) ?? patterns[0])

function isVisible(query: number, key: number) {
	return activePattern.value.rule(query, key)
}

function cellClass(query: number, key: number) {
	return {
		'sparse-pattern__cell--visible': isVisible(query, key),
		'sparse-pattern__cell--active': query === selectedQuery.value && isVisible(query, key),
		'sparse-pattern__cell--row': query === selectedQuery.value,
		'sparse-pattern__cell--global': key === 1 || key === 12
	}
}
</script>

<template>
	<div class="sparse-pattern">
		<div class="sparse-pattern__tabs">
			<button
				v-for="pattern in patterns"
				:key="pattern.id"
				type="button"
				class="sparse-pattern__tab"
				:class="{ 'sparse-pattern__tab--active': pattern.id === selectedPatternId }"
				@click="selectedPatternId = pattern.id"
			>
				<span class="sparse-pattern__tab-kicker">{{ pattern.kicker }}</span>
				<span class="sparse-pattern__tab-title">{{ pattern.title }}</span>
			</button>
		</div>

		<div class="sparse-pattern__body">
			<div class="sparse-pattern__summary">
				<p class="sparse-pattern__kicker">{{ activePattern.kicker }}</p>
				<h4>{{ activePattern.title }}</h4>
				<p>{{ activePattern.description }}</p>
				<p class="sparse-pattern__complexity">{{ activePattern.complexity }}</p>

				<div class="sparse-pattern__chips">
					<span
						v-for="property in activePattern.properties"
						:key="property"
						class="sparse-pattern__chip"
					>
						{{ property }}
					</span>
				</div>

				<label class="sparse-pattern__slider-label" for="sparse-query">
					观察第 {{ selectedQuery }} 行的可见范围
				</label>
				<input
					id="sparse-query"
					v-model.number="selectedQuery"
					class="sparse-pattern__slider"
					type="range"
					min="1"
					max="12"
				>

				<div class="sparse-pattern__legend">
					<span><i class="sparse-pattern__dot sparse-pattern__dot--visible" /> 可见连接</span>
					<span><i class="sparse-pattern__dot sparse-pattern__dot--active" /> 当前查询行</span>
					<span><i class="sparse-pattern__dot sparse-pattern__dot--global" /> 常作全局节点</span>
				</div>
			</div>

			<div class="sparse-pattern__matrix-wrap">
				<div class="sparse-pattern__matrix">
					<div class="sparse-pattern__axis sparse-pattern__axis--corner">Q\K</div>

					<div
						v-for="key in positions"
						:key="`head-${key}`"
						class="sparse-pattern__axis"
						:class="{ 'sparse-pattern__axis--global': key === 1 || key === 12 }"
					>
						{{ key }}
					</div>

					<template v-for="query in positions" :key="`row-${query}`">
						<div
							class="sparse-pattern__axis"
							:class="{ 'sparse-pattern__axis--active': query === selectedQuery }"
						>
							{{ query }}
						</div>

						<div
							v-for="key in positions"
							:key="`${query}-${key}`"
							class="sparse-pattern__cell"
							:class="cellClass(query, key)"
						/>
					</template>
				</div>
			</div>
		</div>
	</div>
</template>

<style scoped>
.sparse-pattern {
	margin: 1.5rem 0;
	border: 1px solid rgba(148, 163, 184, 0.35);
	border-radius: 24px;
	background: linear-gradient(180deg, rgba(248, 250, 252, 0.98), rgba(241, 245, 249, 0.96));
	box-shadow: 0 20px 44px rgba(15, 23, 42, 0.06);
	overflow: hidden;
}

.sparse-pattern__tabs {
	display: grid;
	grid-template-columns: repeat(4, minmax(0, 1fr));
	gap: 0.8rem;
	padding: 1rem;
	background: rgba(255, 255, 255, 0.78);
	border-bottom: 1px solid rgba(148, 163, 184, 0.2);
}

.sparse-pattern__tab {
	display: flex;
	flex-direction: column;
	align-items: flex-start;
	gap: 0.28rem;
	padding: 0.9rem;
	border: 1px solid rgba(148, 163, 184, 0.35);
	border-radius: 18px;
	background: linear-gradient(180deg, rgba(255, 255, 255, 0.98), rgba(248, 250, 252, 0.98));
	text-align: left;
	cursor: pointer;
	transition: all 0.18s ease;
}

.sparse-pattern__tab:hover,
.sparse-pattern__tab--active {
	transform: translateY(-1px);
	border-color: rgba(8, 145, 178, 0.45);
	box-shadow: 0 12px 28px rgba(8, 145, 178, 0.1);
}

.sparse-pattern__tab--active {
	background: linear-gradient(180deg, rgba(207, 250, 254, 0.92), rgba(236, 254, 255, 0.98));
}

.sparse-pattern__tab-kicker,
.sparse-pattern__kicker {
	margin: 0;
	font-size: 0.75rem;
	font-weight: 700;
	letter-spacing: 0.08em;
	text-transform: uppercase;
	color: #0f766e;
}

.sparse-pattern__tab-title {
	font-size: 0.96rem;
	font-weight: 700;
	color: #134e4a;
	line-height: 1.5;
}

.sparse-pattern__body {
	display: grid;
	grid-template-columns: minmax(280px, 360px) minmax(0, 1fr);
	gap: 1rem;
	padding: 1rem;
}

.sparse-pattern__summary h4 {
	margin: 0.28rem 0 0.7rem;
	font-size: 1.08rem;
	color: #134e4a;
}

.sparse-pattern__summary p {
	margin: 0;
	line-height: 1.85;
	color: #334155;
}

.sparse-pattern__complexity {
	margin-top: 0.75rem !important;
	color: #0f766e !important;
	font-weight: 600;
}

.sparse-pattern__chips {
	display: flex;
	flex-wrap: wrap;
	gap: 0.55rem;
	margin-top: 0.9rem;
}

.sparse-pattern__chip {
	padding: 0.38rem 0.72rem;
	border-radius: 999px;
	background: rgba(8, 145, 178, 0.1);
	color: #0f766e;
	font-size: 0.82rem;
	font-weight: 700;
}

.sparse-pattern__slider-label {
	display: block;
	margin-top: 1rem;
	font-size: 0.92rem;
	font-weight: 700;
	color: #134e4a;
}

.sparse-pattern__slider {
	width: 100%;
	margin-top: 0.65rem;
	accent-color: #0f766e;
}

.sparse-pattern__legend {
	display: flex;
	flex-wrap: wrap;
	gap: 0.8rem;
	margin-top: 0.95rem;
	font-size: 0.83rem;
	color: #475569;
}

.sparse-pattern__legend span {
	display: inline-flex;
	align-items: center;
	gap: 0.38rem;
}

.sparse-pattern__dot {
	display: inline-block;
	width: 0.72rem;
	height: 0.72rem;
	border-radius: 999px;
	background: rgba(148, 163, 184, 0.45);
}

.sparse-pattern__dot--visible {
	background: rgba(20, 184, 166, 0.58);
}

.sparse-pattern__dot--active {
	background: #0f766e;
}

.sparse-pattern__dot--global {
	background: rgba(245, 158, 11, 0.82);
}

.sparse-pattern__matrix-wrap {
	overflow-x: auto;
}

.sparse-pattern__matrix {
	display: grid;
	grid-template-columns: repeat(13, minmax(28px, 1fr));
	gap: 0.3rem;
	min-width: 520px;
	padding: 0.2rem;
}

.sparse-pattern__axis,
.sparse-pattern__cell {
	display: flex;
	align-items: center;
	justify-content: center;
	min-height: 28px;
	border-radius: 10px;
}

.sparse-pattern__axis {
	font-size: 0.8rem;
	font-weight: 700;
	color: #64748b;
	background: rgba(255, 255, 255, 0.8);
}

.sparse-pattern__axis--corner {
	color: #0f766e;
}

.sparse-pattern__axis--active {
	background: rgba(20, 184, 166, 0.14);
	color: #0f766e;
}

.sparse-pattern__axis--global {
	color: #b45309;
}

.sparse-pattern__cell {
	border: 1px solid rgba(203, 213, 225, 0.72);
	background: rgba(255, 255, 255, 0.72);
}

.sparse-pattern__cell--visible {
	background: rgba(20, 184, 166, 0.22);
	border-color: rgba(13, 148, 136, 0.42);
}

.sparse-pattern__cell--row {
	box-shadow: inset 0 0 0 1px rgba(15, 118, 110, 0.1);
}

.sparse-pattern__cell--active {
	background: linear-gradient(180deg, rgba(20, 184, 166, 0.78), rgba(13, 148, 136, 0.95));
	border-color: rgba(15, 118, 110, 0.9);
}

.sparse-pattern__cell--global {
	position: relative;
}

.sparse-pattern__cell--global::after {
	content: '';
	position: absolute;
	inset: 5px;
	border-radius: 999px;
	border: 1px solid rgba(245, 158, 11, 0.42);
}

@media (max-width: 960px) {
	.sparse-pattern__tabs {
		grid-template-columns: repeat(2, minmax(0, 1fr));
	}

	.sparse-pattern__body {
		grid-template-columns: 1fr;
	}
}

@media (max-width: 640px) {
	.sparse-pattern__tabs {
		grid-template-columns: 1fr;
	}
}
</style>
