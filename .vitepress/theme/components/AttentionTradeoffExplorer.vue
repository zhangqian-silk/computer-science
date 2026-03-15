<script setup lang="ts">
import { computed, ref } from 'vue'

type Item = {
	id: string
	title: string
	kicker: string
	description: string
	signals: string[]
}

const items: Item[] = [
	{
		id: 'window',
		title: '更大可见范围',
		kicker: 'Step 1',
		description: '当每个位置能够看到更多 token 时，模型就更容易直接建立远距离依赖，而不需要层层转述信息。',
		signals: ['全局可见', '长上下文', '更多位置参与']
	},
	{
		id: 'ability',
		title: '更强的依赖建模',
		kicker: 'Step 2',
		description: '更大的可见范围通常会换来更强的上下文整合能力，例如更稳定的长距离指代、跨句信息对齐和主题保持。',
		signals: ['长依赖', '上下文聚合', '更完整语义']
	},
	{
		id: 'cost',
		title: '更高的计算与显存成本',
		kicker: 'Step 3',
		description: '可见范围一旦扩大，注意力分数矩阵和中间激活的规模也会迅速膨胀，训练与推理都更昂贵。',
		signals: ['O(n^2)', '显存压力', '长序列瓶颈']
	}
]

const activeId = ref(items[1].id)
const activeItem = computed(() => items.find((item) => item.id === activeId.value) ?? items[0])
const activeIndex = computed(() => items.findIndex((item) => item.id === activeId.value))
</script>

<template>
	<div class="tradeoff">
		<div class="tradeoff__rail">
			<div
				v-for="(item, index) in items"
				:key="item.id"
				class="tradeoff__step"
				:class="{ 'tradeoff__step--active': item.id === activeId }"
			>
				<button type="button" class="tradeoff__card" @click="activeId = item.id">
					<span class="tradeoff__kicker">{{ item.kicker }}</span>
					<span class="tradeoff__title">{{ item.title }}</span>
				</button>

				<div
					v-if="index < items.length - 1"
					class="tradeoff__arrow"
					:class="{ 'tradeoff__arrow--active': index < activeIndex }"
					aria-hidden="true"
				>
					<span />
				</div>
			</div>
		</div>

		<div class="tradeoff__detail">
			<div>
				<p class="tradeoff__detail-kicker">{{ activeItem.kicker }}</p>
				<h4>{{ activeItem.title }}</h4>
				<p>{{ activeItem.description }}</p>
			</div>

			<div class="tradeoff__signals">
				<span
					v-for="signal in activeItem.signals"
					:key="signal"
					class="tradeoff__signal"
				>
					{{ signal }}
				</span>
			</div>
		</div>
	</div>
</template>

<style scoped>
.tradeoff {
	margin: 1.5rem 0;
	border: 1px solid rgba(148, 163, 184, 0.35);
	border-radius: 24px;
	background: linear-gradient(180deg, rgba(248, 250, 252, 0.98), rgba(241, 245, 249, 0.96));
	box-shadow: 0 20px 44px rgba(15, 23, 42, 0.06);
	overflow: hidden;
}

.tradeoff__rail {
	display: flex;
	flex-wrap: wrap;
	align-items: center;
	gap: 0.75rem;
	padding: 1rem;
	background: rgba(255, 255, 255, 0.78);
	border-bottom: 1px solid rgba(148, 163, 184, 0.2);
}

.tradeoff__step {
	display: flex;
	align-items: center;
	gap: 0.75rem;
}

.tradeoff__card {
	display: flex;
	flex-direction: column;
	align-items: flex-start;
	gap: 0.3rem;
	min-width: 176px;
	padding: 0.95rem;
	border: 1px solid rgba(148, 163, 184, 0.36);
	border-radius: 18px;
	background: linear-gradient(180deg, rgba(255, 255, 255, 0.98), rgba(248, 250, 252, 0.98));
	text-align: left;
	cursor: pointer;
	transition: all 0.18s ease;
}

.tradeoff__card:hover,
.tradeoff__step--active .tradeoff__card {
	transform: translateY(-1px);
	border-color: rgba(234, 88, 12, 0.45);
	box-shadow: 0 12px 28px rgba(234, 88, 12, 0.1);
}

.tradeoff__step--active .tradeoff__card {
	background: linear-gradient(180deg, rgba(255, 237, 213, 0.98), rgba(255, 247, 237, 0.98));
}

.tradeoff__kicker,
.tradeoff__detail-kicker {
	margin: 0;
	font-size: 0.76rem;
	letter-spacing: 0.08em;
	text-transform: uppercase;
	color: #9a3412;
	font-weight: 700;
}

.tradeoff__title {
	font-size: 0.98rem;
	font-weight: 700;
	line-height: 1.55;
	color: #7c2d12;
}

.tradeoff__arrow {
	display: flex;
	align-items: center;
	justify-content: center;
	width: 40px;
}

.tradeoff__arrow span {
	position: relative;
	display: block;
	width: 100%;
	height: 2px;
	background: linear-gradient(90deg, rgba(251, 146, 60, 0.3), rgba(234, 88, 12, 0.7));
}

.tradeoff__arrow--active span {
	background: linear-gradient(90deg, rgba(234, 88, 12, 0.55), rgba(234, 88, 12, 1));
}

.tradeoff__arrow span::after {
	content: '';
	position: absolute;
	top: -4px;
	right: 0;
	border-top: 5px solid transparent;
	border-bottom: 5px solid transparent;
	border-left: 8px solid rgba(234, 88, 12, 0.75);
}

.tradeoff__detail {
	display: grid;
	grid-template-columns: minmax(0, 1.5fr) minmax(220px, 1fr);
	gap: 1rem;
	padding: 1rem;
}

.tradeoff__detail h4 {
	margin: 0.28rem 0 0.7rem;
	font-size: 1.08rem;
	color: #7c2d12;
}

.tradeoff__detail p {
	margin: 0;
	line-height: 1.85;
	color: #431407;
}

.tradeoff__signals {
	display: flex;
	flex-wrap: wrap;
	align-content: start;
	gap: 0.55rem;
}

.tradeoff__signal {
	padding: 0.4rem 0.72rem;
	border-radius: 999px;
	background: rgba(234, 88, 12, 0.1);
	color: #c2410c;
	font-size: 0.82rem;
	font-weight: 700;
}

@media (max-width: 960px) {
	.tradeoff__detail {
		grid-template-columns: 1fr;
	}
}

@media (max-width: 640px) {
	.tradeoff__card {
		min-width: min(100%, 220px);
	}
}
</style>
