<script setup lang="ts">
import { computed, ref } from 'vue'

type MechanismCard = {
	id: string
	title: string
	kicker: string
	summary: string
	focus: string
	leftEdge: string
	rightEdge: string
	takeaways: string[]
}

const cards: MechanismCard[] = [
	{
		id: 'absolute',
		title: '绝对位置编码',
		kicker: 'Absolute',
		summary: '先给每个 token 加上“你在第几位”的坐标，再由后续 attention 自己学习如何使用这些坐标。',
		focus: '「吃」看到的是带位置标签的输入表示：位置 1 的「我」、位置 2 的「吃」、位置 3 的「鱼」。',
		leftEdge: '更像“句首名词 -> 中间动词”的关系',
		rightEdge: '更像“句尾名词 -> 中间动词”的关系',
		takeaways: ['先标绝对坐标', '顺序信息在输入层注入', '模型再自己学习距离规律']
	},
	{
		id: 'relative',
		title: '相对位置编码',
		kicker: 'Relative',
		summary: '直接把“你和我相差几位”写进 attention 打分，让距离和方向成为显式结构信号。',
		focus: '「吃」更关心的是左边一格的「我」和右边一格的「鱼」，而不是它们在全句中的编号。',
		leftEdge: '距离为 -1 或 +1（取决于记号）且位于左侧',
		rightEdge: '距离为 +1 或 -1（取决于记号）且位于右侧',
		takeaways: ['直接写距离', '更容易复用局部模式', '长序列里通常更稳健']
	},
	{
		id: 'rope',
		title: 'RoPE',
		kicker: 'Rotary',
		summary: '不额外相加位置向量，而是先旋转 query / key，再让内积自动体现相对位移。',
		focus: '「吃」对应的 query 带有位置 2 的相位；「我」和「鱼」的 key 则分别带有位置 1 和 3 的相位。',
		leftEdge: '与「吃」形成一组角度差',
		rightEdge: '与「吃」形成另一组角度差',
		takeaways: ['位置写进几何相位', '内积自然显出相对位移', '和 KV cache 兼容性较好']
	}
]

const activeId = ref<MechanismCard['id']>('absolute')
const activeCard = computed(() => cards.find((card) => card.id === activeId.value) ?? cards[0])
</script>

<template>
	<div class="position-compare">
		<div class="position-compare__tabs">
			<button
				v-for="card in cards"
				:key="card.id"
				type="button"
				class="position-compare__tab"
				:class="{ 'position-compare__tab--active': card.id === activeId }"
				@click="activeId = card.id"
			>
				{{ card.title }}
			</button>
		</div>

		<div class="position-compare__body">
			<div class="position-compare__meta">
				<p class="position-compare__kicker">{{ activeCard.kicker }}</p>
				<h4>{{ activeCard.title }}</h4>
				<p>{{ activeCard.summary }}</p>
			</div>

			<div class="position-compare__scene">
				<div class="position-compare__tokens">
					<div class="position-compare__token">
						<span class="position-compare__index">1</span>
						<span class="position-compare__word">我</span>
					</div>
					<div class="position-compare__token position-compare__token--focus">
						<span class="position-compare__index">2</span>
						<span class="position-compare__word">吃</span>
					</div>
					<div class="position-compare__token">
						<span class="position-compare__index">3</span>
						<span class="position-compare__word">鱼</span>
					</div>
				</div>

				<div class="position-compare__edges">
					<section class="position-compare__edge">
						<h5>「我」 -> 「吃」</h5>
						<p>{{ activeCard.leftEdge }}</p>
					</section>
					<section class="position-compare__edge position-compare__edge--accent">
						<h5>「鱼」 -> 「吃」</h5>
						<p>{{ activeCard.rightEdge }}</p>
					</section>
				</div>
			</div>

			<div class="position-compare__explain">
				<p class="position-compare__focus">{{ activeCard.focus }}</p>
				<div class="position-compare__chips">
					<span
						v-for="item in activeCard.takeaways"
						:key="item"
						class="position-compare__chip"
					>
						{{ item }}
					</span>
				</div>
			</div>
		</div>
	</div>
</template>

<style scoped>
.position-compare {
	margin: 1.5rem 0;
	border: 1px solid rgba(148, 163, 184, 0.35);
	border-radius: 24px;
	background:
		radial-gradient(circle at top right, rgba(191, 219, 254, 0.28), transparent 30%),
		linear-gradient(180deg, rgba(248, 250, 252, 0.98), rgba(241, 245, 249, 0.96));
	box-shadow: 0 18px 40px rgba(15, 23, 42, 0.06);
	overflow: hidden;
}

.position-compare__tabs {
	display: flex;
	flex-wrap: wrap;
	gap: 0.75rem;
	padding: 1rem;
	border-bottom: 1px solid rgba(148, 163, 184, 0.22);
	background: rgba(255, 255, 255, 0.82);
}

.position-compare__tab {
	padding: 0.72rem 1rem;
	border-radius: 999px;
	border: 1px solid rgba(148, 163, 184, 0.45);
	background: #fff;
	color: #334155;
	font-weight: 600;
	cursor: pointer;
	transition: all 0.18s ease;
}

.position-compare__tab--active {
	border-color: #1d4ed8;
	background: #1d4ed8;
	color: #fff;
	box-shadow: 0 12px 24px rgba(29, 78, 216, 0.16);
}

.position-compare__body {
	padding: 1rem;
}

.position-compare__meta {
	margin-bottom: 1rem;
}

.position-compare__kicker {
	margin: 0;
	font-size: 0.76rem;
	letter-spacing: 0.08em;
	text-transform: uppercase;
	color: #64748b;
}

.position-compare__meta h4 {
	margin: 0.28rem 0 0.55rem;
	font-size: 1.08rem;
	color: #0f172a;
}

.position-compare__meta p {
	margin: 0;
	line-height: 1.8;
	color: #334155;
}

.position-compare__scene {
	display: grid;
	grid-template-columns: minmax(0, 1.1fr) minmax(0, 1.2fr);
	gap: 1rem;
	align-items: start;
	margin-bottom: 1rem;
}

.position-compare__tokens {
	display: grid;
	grid-template-columns: repeat(3, minmax(0, 1fr));
	gap: 0.8rem;
}

.position-compare__token {
	padding: 1rem 0.8rem;
	border-radius: 18px;
	border: 1px solid rgba(148, 163, 184, 0.28);
	background: rgba(255, 255, 255, 0.86);
	text-align: center;
}

.position-compare__token--focus {
	background: linear-gradient(180deg, rgba(219, 234, 254, 0.98), rgba(239, 246, 255, 0.98));
	border-color: rgba(59, 130, 246, 0.32);
	box-shadow: 0 12px 24px rgba(59, 130, 246, 0.1);
}

.position-compare__index {
	display: block;
	font-size: 0.8rem;
	font-weight: 700;
	color: #64748b;
	margin-bottom: 0.35rem;
}

.position-compare__word {
	font-size: 1.05rem;
	font-weight: 700;
	color: #0f172a;
}

.position-compare__edges {
	display: grid;
	grid-template-columns: repeat(2, minmax(0, 1fr));
	gap: 1rem;
}

.position-compare__edge {
	padding: 1rem;
	border-radius: 18px;
	border: 1px solid rgba(148, 163, 184, 0.28);
	background: rgba(255, 255, 255, 0.84);
}

.position-compare__edge--accent {
	background: linear-gradient(180deg, rgba(254, 249, 195, 0.98), rgba(254, 252, 232, 0.96));
	border-color: rgba(234, 179, 8, 0.32);
}

.position-compare__edge h5 {
	margin: 0 0 0.55rem;
	font-size: 0.95rem;
	color: #0f172a;
}

.position-compare__edge p {
	margin: 0;
	line-height: 1.75;
	color: #334155;
}

.position-compare__focus {
	margin: 0 0 0.8rem;
	line-height: 1.8;
	color: #334155;
}

.position-compare__chips {
	display: flex;
	flex-wrap: wrap;
	gap: 0.55rem;
}

.position-compare__chip {
	padding: 0.42rem 0.72rem;
	border-radius: 999px;
	background: rgba(29, 78, 216, 0.1);
	color: #1d4ed8;
	font-size: 0.82rem;
	font-weight: 700;
}

@media (max-width: 860px) {
	.position-compare__scene {
		grid-template-columns: 1fr;
	}
}

@media (max-width: 640px) {
	.position-compare__tokens,
	.position-compare__edges {
		grid-template-columns: 1fr;
	}
}
</style>
