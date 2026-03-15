<script setup lang="ts">
import { computed, ref } from 'vue'

type Section = {
	id: string
	title: string
	phase: string
	description: string
	links: string[]
}

const sections: Section[] = [
	{
		id: 'motivation',
		title: '问题动机',
		phase: '起点',
		description: '先理解为什么固定长度状态会成为瓶颈，以及 attention 为什么把序列建模改写为按需读取信息。',
		links: ['信息瓶颈', '长程依赖', '内容寻址']
	},
	{
		id: 'qkv',
		title: 'Q / K / V',
		phase: '基本形态',
		description: '建立 query、key、value 的角色分工，明确“匹配谁”和“读取谁”并不是一回事。',
		links: ['query', 'key', 'value']
	},
	{
		id: 'core',
		title: '缩放点积 Attention',
		phase: '核心公式',
		description: '从矩阵乘法视角拆开 QK^T、缩放、softmax 与 AV，看到 attention 的数学主干。',
		links: ['QK^T', 'softmax', 'AV']
	},
	{
		id: 'extensions',
		title: '掩码、位置、多头',
		phase: '关键扩展',
		description: '补上生成约束、顺序信息与多子空间并行建模，让 attention 从公式变成可落地模块。',
		links: ['Mask', 'Position', 'Multi-Head']
	},
	{
		id: 'block',
		title: 'Transformer Block',
		phase: '架构落地',
		description: '把 attention 与残差、LayerNorm、FFN 组合起来，形成编码器和解码器中的标准层结构。',
		links: ['Encoder', 'Decoder', 'FFN']
	},
	{
		id: 'practice',
		title: '训练、推理与局限',
		phase: '工程视角',
		description: '继续理解并行训练、KV cache、复杂度瓶颈，以及 attention 的解释边界和常见误区。',
		links: ['KV cache', 'O(n^2)', '误区']
	}
]

const activeId = ref(sections[0].id)
const activeSection = computed(() => sections.find((section) => section.id === activeId.value) ?? sections[0])
</script>

<template>
	<div class="attention-roadmap">
		<div class="attention-roadmap__track">
			<button
				v-for="section in sections"
				:key="section.id"
				type="button"
				class="attention-roadmap__step"
				:class="{ 'attention-roadmap__step--active': section.id === activeId }"
				@click="activeId = section.id"
			>
				<span class="attention-roadmap__phase">{{ section.phase }}</span>
				<span class="attention-roadmap__title">{{ section.title }}</span>
			</button>
		</div>

		<div class="attention-roadmap__detail">
			<div>
				<p class="attention-roadmap__kicker">{{ activeSection.phase }}</p>
				<h4>{{ activeSection.title }}</h4>
				<p>{{ activeSection.description }}</p>
			</div>

			<div class="attention-roadmap__chips">
				<span
					v-for="item in activeSection.links"
					:key="item"
					class="attention-roadmap__chip"
				>
					{{ item }}
				</span>
			</div>
		</div>
	</div>
</template>

<style scoped>
.attention-roadmap {
	margin: 1.5rem 0;
	border: 1px solid rgba(148, 163, 184, 0.35);
	border-radius: 24px;
	background: linear-gradient(180deg, rgba(248, 250, 252, 0.98), rgba(241, 245, 249, 0.96));
	box-shadow: 0 20px 44px rgba(15, 23, 42, 0.06);
	overflow: hidden;
}

.attention-roadmap__track {
	display: grid;
	grid-template-columns: repeat(6, minmax(0, 1fr));
	gap: 0.8rem;
	padding: 1rem;
	background: rgba(255, 255, 255, 0.76);
	border-bottom: 1px solid rgba(148, 163, 184, 0.2);
}

.attention-roadmap__step {
	display: flex;
	flex-direction: column;
	align-items: flex-start;
	gap: 0.3rem;
	padding: 0.9rem;
	border: 1px solid rgba(148, 163, 184, 0.35);
	border-radius: 18px;
	background: linear-gradient(180deg, rgba(255, 255, 255, 0.98), rgba(248, 250, 252, 0.98));
	text-align: left;
	cursor: pointer;
	transition: all 0.18s ease;
}

.attention-roadmap__step:hover,
.attention-roadmap__step--active {
	transform: translateY(-1px);
	border-color: rgba(37, 99, 235, 0.45);
	box-shadow: 0 12px 28px rgba(37, 99, 235, 0.1);
}

.attention-roadmap__step--active {
	background: linear-gradient(180deg, rgba(219, 234, 254, 0.9), rgba(239, 246, 255, 0.98));
}

.attention-roadmap__phase {
	font-size: 0.74rem;
	font-weight: 700;
	letter-spacing: 0.08em;
	text-transform: uppercase;
	color: #64748b;
}

.attention-roadmap__title {
	font-size: 0.95rem;
	font-weight: 700;
	color: #0f172a;
	line-height: 1.5;
}

.attention-roadmap__detail {
	display: grid;
	grid-template-columns: minmax(0, 1.6fr) minmax(220px, 1fr);
	gap: 1rem;
	padding: 1rem;
	align-items: start;
}

.attention-roadmap__kicker {
	margin: 0;
	font-size: 0.76rem;
	letter-spacing: 0.08em;
	text-transform: uppercase;
	color: #64748b;
}

.attention-roadmap__detail h4 {
	margin: 0.3rem 0 0.7rem;
	font-size: 1.08rem;
	color: #0f172a;
}

.attention-roadmap__detail p {
	margin: 0;
	line-height: 1.85;
	color: #334155;
}

.attention-roadmap__chips {
	display: flex;
	flex-wrap: wrap;
	gap: 0.55rem;
	padding: 0.25rem 0 0;
}

.attention-roadmap__chip {
	padding: 0.4rem 0.7rem;
	border-radius: 999px;
	background: rgba(37, 99, 235, 0.08);
	color: #1d4ed8;
	font-size: 0.82rem;
	font-weight: 700;
}

@media (max-width: 960px) {
	.attention-roadmap__track {
		grid-template-columns: repeat(2, minmax(0, 1fr));
	}

	.attention-roadmap__detail {
		grid-template-columns: 1fr;
	}
}

@media (max-width: 640px) {
	.attention-roadmap__track {
		grid-template-columns: 1fr;
	}
}
</style>
