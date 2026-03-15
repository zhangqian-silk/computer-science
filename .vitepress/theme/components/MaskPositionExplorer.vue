<script setup lang="ts">
import { computed, ref } from 'vue'

type Panel = {
	id: string
	title: string
	kicker: string
	leftTitle: string
	leftItems: string[]
	rightTitle: string
	rightItems: string[]
	note: string
}

const panels: Panel[] = [
	{
		id: 'mask',
		title: '掩码机制',
		kicker: 'Mask',
		leftTitle: '生成式场景',
		leftItems: ['当前位置只能看自己与历史', '未来位置被加上 -∞', 'softmax 后未来权重变为 0'],
		rightTitle: '理解式场景',
		rightItems: ['通常不做 look-ahead mask', '当前位置可同时利用左右文', '更常配合 padding mask'],
		note: '核心差别不在公式骨架，而在“当前位置是否允许访问未来信息”。'
	},
	{
		id: 'position',
		title: '位置编码',
		kicker: 'Position',
		leftTitle: '没有位置编码',
		leftItems: ['更像在无序集合上做相关性匹配', '模型只知道“谁像谁”', '难以区分前后顺序'],
		rightTitle: '加入位置编码',
		rightItems: ['输入变成 token 内容 + 位置信息', '模型既知道内容，也知道顺序', '更容易区分主语、宾语与距离'],
		note: 'Attention 本身擅长相关性匹配，但顺序感来自额外注入的位置机制。'
	}
]

const activeId = ref<Panel['id']>('mask')
const activePanel = computed(() => panels.find((panel) => panel.id === activeId.value) ?? panels[0])
</script>

<template>
	<div class="constraint-explorer">
		<div class="constraint-explorer__tabs">
			<button
				v-for="panel in panels"
				:key="panel.id"
				type="button"
				class="constraint-explorer__tab"
				:class="{ 'constraint-explorer__tab--active': panel.id === activeId }"
				@click="activeId = panel.id"
			>
				{{ panel.title }}
			</button>
		</div>

		<div class="constraint-explorer__body">
			<div class="constraint-explorer__meta">
				<p class="constraint-explorer__kicker">{{ activePanel.kicker }}</p>
				<h4>{{ activePanel.title }}</h4>
				<p>{{ activePanel.note }}</p>
			</div>

			<div class="constraint-explorer__compare">
				<section class="constraint-explorer__card">
					<h5>{{ activePanel.leftTitle }}</h5>
					<ul>
						<li v-for="item in activePanel.leftItems" :key="item">{{ item }}</li>
					</ul>
				</section>

				<section class="constraint-explorer__card constraint-explorer__card--accent">
					<h5>{{ activePanel.rightTitle }}</h5>
					<ul>
						<li v-for="item in activePanel.rightItems" :key="item">{{ item }}</li>
					</ul>
				</section>
			</div>
		</div>
	</div>
</template>

<style scoped>
.constraint-explorer {
	margin: 1.5rem 0;
	border: 1px solid rgba(148, 163, 184, 0.35);
	border-radius: 22px;
	background: linear-gradient(180deg, rgba(248, 250, 252, 0.98), rgba(241, 245, 249, 0.96));
	box-shadow: 0 18px 40px rgba(15, 23, 42, 0.06);
	overflow: hidden;
}

.constraint-explorer__tabs {
	display: flex;
	gap: 0.75rem;
	padding: 1rem;
	border-bottom: 1px solid rgba(148, 163, 184, 0.22);
	background: rgba(255, 255, 255, 0.76);
}

.constraint-explorer__tab {
	padding: 0.72rem 1rem;
	border-radius: 999px;
	border: 1px solid rgba(148, 163, 184, 0.45);
	background: #fff;
	color: #334155;
	font-weight: 600;
	cursor: pointer;
	transition: all 0.18s ease;
}

.constraint-explorer__tab--active {
	border-color: #0f766e;
	background: #0f766e;
	color: #fff;
	box-shadow: 0 12px 24px rgba(15, 118, 110, 0.18);
}

.constraint-explorer__body {
	padding: 1rem;
}

.constraint-explorer__meta {
	margin-bottom: 1rem;
}

.constraint-explorer__kicker {
	margin: 0;
	font-size: 0.76rem;
	letter-spacing: 0.08em;
	text-transform: uppercase;
	color: #64748b;
}

.constraint-explorer__meta h4 {
	margin: 0.28rem 0 0.55rem;
	font-size: 1.06rem;
	color: #0f172a;
}

.constraint-explorer__meta p {
	margin: 0;
	line-height: 1.8;
	color: #334155;
}

.constraint-explorer__compare {
	display: grid;
	grid-template-columns: repeat(2, minmax(0, 1fr));
	gap: 1rem;
}

.constraint-explorer__card {
	padding: 1rem;
	border-radius: 18px;
	border: 1px solid rgba(148, 163, 184, 0.28);
	background: rgba(255, 255, 255, 0.82);
}

.constraint-explorer__card--accent {
	background: linear-gradient(180deg, rgba(240, 253, 250, 0.98), rgba(236, 253, 245, 0.96));
	border-color: rgba(16, 185, 129, 0.3);
}

.constraint-explorer__card h5 {
	margin: 0 0 0.65rem;
	font-size: 0.98rem;
	color: #0f172a;
}

.constraint-explorer__card ul {
	margin: 0;
	padding-left: 1.15rem;
}

.constraint-explorer__card li {
	margin: 0.35rem 0;
	color: #334155;
	line-height: 1.75;
}

@media (max-width: 760px) {
	.constraint-explorer__compare {
		grid-template-columns: 1fr;
	}
}
</style>
