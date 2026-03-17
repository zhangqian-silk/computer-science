<script setup lang="ts">
import { computed, ref } from 'vue'

type RouteCard = {
	id: string
	phase: string
	title: string
	question: string
	answer: string
	keywords: string[]
}

const cards: RouteCard[] = [
	{
		id: 'motivation',
		phase: '起点',
		title: '为什么必须显式加入位置信息',
		question: '如果 Transformer 可以并行处理整段序列，为什么还会“看不懂顺序”？',
		answer: '因为自注意力默认更像在无序集合上做相关性匹配。没有额外位置机制时，模型更容易知道“谁像谁”，却不知道“谁在前、谁在后”。',
		keywords: ['并行计算', '顺序缺失', '归纳偏置']
	},
	{
		id: 'absolute',
		phase: '路线一',
		title: '绝对位置编码',
		question: '能否直接告诉模型“这是第几个 token”？',
		answer: '可以。最直接的方法就是为每个位置分配一个位置向量，再与 token embedding 相加。它强调的是绝对坐标。',
		keywords: ['查表', '绝对坐标', '输入相加']
	},
	{
		id: 'sinusoidal',
		phase: '路线二',
		title: '正弦余弦位置编码',
		question: '为什么经典 Transformer 选择三角函数，而不是普通位置 ID？',
		answer: '因为不同频率的正弦与余弦既能唯一刻画位置，又让固定偏移量对应为可学习的线性变换，更利于建模距离关系。',
		keywords: ['多频率', '可外推', '线性位移']
	},
	{
		id: 'relative',
		phase: '路线三',
		title: '相对位置编码',
		question: '很多任务里，知道“相距多远”是否比知道“在第几位”更重要？',
		answer: '往往是。相对位置机制把距离直接写进注意力打分，让模型更稳定地复用“前一个词”“后两个词”这类局部模式。',
		keywords: ['距离', '平移稳定', '注意力偏置']
	},
	{
		id: 'rope',
		phase: '路线四',
		title: '旋转位置编码 RoPE',
		question: '怎样把绝对位置写进向量本身，同时又让内积自然反映相对距离？',
		answer: 'RoPE 通过对 query 和 key 做位置相关旋转，把位置信息嵌入内积结构，使注意力分数天然依赖相对位移。',
		keywords: ['旋转', '内积结构', '长上下文']
	},
	{
		id: 'practice',
		phase: '落地',
		title: '工程选型与长上下文扩展',
		question: '模型上线时，位置机制最常见的取舍是什么？',
		answer: '核心取舍通常集中在实现复杂度、长度外推、KV cache 兼容性和长上下文稳定性。现代大模型常优先采用 RoPE 或其扩展版本。',
		keywords: ['KV cache', '外推', '上下文窗口']
	}
]

const activeId = ref(cards[0].id)
const activeCard = computed(() => cards.find((card) => card.id === activeId.value) ?? cards[0])
</script>

<template>
	<div class="position-roadmap">
		<div class="position-roadmap__track">
			<button
				v-for="card in cards"
				:key="card.id"
				type="button"
				class="position-roadmap__step"
				:class="{ 'position-roadmap__step--active': card.id === activeId }"
				@click="activeId = card.id"
			>
				<span class="position-roadmap__phase">{{ card.phase }}</span>
				<span class="position-roadmap__title">{{ card.title }}</span>
			</button>
		</div>

		<div class="position-roadmap__detail">
			<div>
				<p class="position-roadmap__kicker">{{ activeCard.phase }}</p>
				<h4>{{ activeCard.title }}</h4>
				<p class="position-roadmap__question">{{ activeCard.question }}</p>
				<p>{{ activeCard.answer }}</p>
			</div>

			<div class="position-roadmap__chips">
				<span
					v-for="item in activeCard.keywords"
					:key="item"
					class="position-roadmap__chip"
				>
					{{ item }}
				</span>
			</div>
		</div>
	</div>
</template>

<style scoped>
.position-roadmap {
	margin: 1.5rem 0;
	border: 1px solid rgba(148, 163, 184, 0.35);
	border-radius: 24px;
	background:
		radial-gradient(circle at top left, rgba(254, 240, 138, 0.28), transparent 34%),
		linear-gradient(180deg, rgba(255, 251, 235, 0.98), rgba(248, 250, 252, 0.96));
	box-shadow: 0 20px 44px rgba(15, 23, 42, 0.06);
	overflow: hidden;
}

.position-roadmap__track {
	display: grid;
	grid-template-columns: repeat(3, minmax(0, 1fr));
	gap: 0.8rem;
	padding: 1rem;
	background: rgba(255, 255, 255, 0.78);
	border-bottom: 1px solid rgba(148, 163, 184, 0.22);
}

.position-roadmap__step {
	display: flex;
	flex-direction: column;
	align-items: flex-start;
	gap: 0.32rem;
	padding: 0.95rem;
	border: 1px solid rgba(148, 163, 184, 0.35);
	border-radius: 18px;
	background: linear-gradient(180deg, rgba(255, 255, 255, 0.98), rgba(248, 250, 252, 0.98));
	text-align: left;
	cursor: pointer;
	transition: all 0.18s ease;
}

.position-roadmap__step:hover,
.position-roadmap__step--active {
	transform: translateY(-1px);
	border-color: rgba(217, 119, 6, 0.45);
	box-shadow: 0 12px 28px rgba(217, 119, 6, 0.12);
}

.position-roadmap__step--active {
	background: linear-gradient(180deg, rgba(254, 243, 199, 0.98), rgba(255, 251, 235, 0.98));
}

.position-roadmap__phase {
	font-size: 0.74rem;
	font-weight: 700;
	letter-spacing: 0.08em;
	text-transform: uppercase;
	color: #78716c;
}

.position-roadmap__title {
	font-size: 0.95rem;
	font-weight: 700;
	line-height: 1.5;
	color: #111827;
}

.position-roadmap__detail {
	display: grid;
	grid-template-columns: minmax(0, 1.65fr) minmax(220px, 1fr);
	gap: 1rem;
	padding: 1rem;
	align-items: start;
}

.position-roadmap__kicker {
	margin: 0;
	font-size: 0.76rem;
	letter-spacing: 0.08em;
	text-transform: uppercase;
	color: #78716c;
}

.position-roadmap__detail h4 {
	margin: 0.3rem 0 0.7rem;
	font-size: 1.08rem;
	color: #111827;
}

.position-roadmap__question {
	margin: 0 0 0.55rem;
	font-weight: 700;
	color: #92400e;
}

.position-roadmap__detail p {
	margin: 0;
	line-height: 1.85;
	color: #374151;
}

.position-roadmap__chips {
	display: flex;
	flex-wrap: wrap;
	gap: 0.55rem;
	padding-top: 0.2rem;
}

.position-roadmap__chip {
	padding: 0.42rem 0.72rem;
	border-radius: 999px;
	background: rgba(217, 119, 6, 0.1);
	color: #b45309;
	font-size: 0.82rem;
	font-weight: 700;
}

@media (max-width: 960px) {
	.position-roadmap__track {
		grid-template-columns: repeat(2, minmax(0, 1fr));
	}

	.position-roadmap__detail {
		grid-template-columns: 1fr;
	}
}

@media (max-width: 640px) {
	.position-roadmap__track {
		grid-template-columns: 1fr;
	}
}
</style>
