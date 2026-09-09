<script setup lang="ts">
import { computed, ref, useId } from "vue"
import LearningLab from "./LearningLab.vue"
import { useLabReset } from "../use-lab-reset"
const labId = useId()
const fieldId = (name: string) => `${labId}-${name}`

const ax = ref(0.8)
const ay = ref(0.3)
const bx = ref(0.4)
const by = ref(0.9)
const dot = computed(() => ax.value * bx.value + ay.value * by.value)
const normA = computed(() => Math.hypot(ax.value, ay.value))
const normB = computed(() => Math.hypot(bx.value, by.value))
const cosine = computed(() => normA.value === 0 || normB.value === 0 ? null : dot.value / (normA.value * normB.value))
const cosineText = computed(() => cosine.value === null ? "未定义（零向量）" : cosine.value.toFixed(3))
const distance = computed(() => Math.hypot(ax.value - bx.value, ay.value - by.value))
const endpoint = (x: number, y: number) => ({ x: 110 + x * 75, y: 110 - y * 75 })
const pointA = computed(() => endpoint(ax.value, ay.value))
const pointB = computed(() => endpoint(bx.value, by.value))
const resetLab = useLabReset(ax, ay, bx, by)
</script>

<template>
	<LearningLab topic="VectorSimilarityExplorer" @reset="resetLab">
		<p class="infra-lab__hint">改变两个二维向量，比较点积、余弦和欧氏距离分别保留什么信息。</p>
		<div class="infra-controls">
			<div class="infra-control"><label :for="fieldId('vector-ax')">A.x：{{ ax.toFixed(2) }}</label><input :id="fieldId('vector-ax')" v-model.number="ax" type="range" min="-1" max="1" step="0.05"></div>
			<div class="infra-control"><label :for="fieldId('vector-ay')">A.y：{{ ay.toFixed(2) }}</label><input :id="fieldId('vector-ay')" v-model.number="ay" type="range" min="-1" max="1" step="0.05"></div>
			<div class="infra-control"><label :for="fieldId('vector-bx')">B.x：{{ bx.toFixed(2) }}</label><input :id="fieldId('vector-bx')" v-model.number="bx" type="range" min="-1" max="1" step="0.05"></div>
			<div class="infra-control"><label :for="fieldId('vector-by')">B.y：{{ by.toFixed(2) }}</label><input :id="fieldId('vector-by')" v-model.number="by" type="range" min="-1" max="1" step="0.05"></div>
		</div>
		<svg class="vector-plane" viewBox="0 0 220 220" role="img" :aria-label="`向量 A 和 B，余弦相似度 ${cosineText}`">
			<defs>
				<marker :id="fieldId('arrow-a')" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto"><path d="M0,0 L8,4 L0,8 Z" fill="var(--vp-c-brand-1)" /></marker>
				<marker :id="fieldId('arrow-b')" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto"><path d="M0,0 L8,4 L0,8 Z" fill="var(--vp-c-green-1)" /></marker>
			</defs>
			<line x1="20" y1="110" x2="200" y2="110" class="axis" />
			<line x1="110" y1="20" x2="110" y2="200" class="axis" />
			<line x1="110" y1="110" :x2="pointA.x" :y2="pointA.y" class="vector vector-a" :marker-end="`url(#${fieldId('arrow-a')})`" />
			<line x1="110" y1="110" :x2="pointB.x" :y2="pointB.y" class="vector vector-b" :marker-end="`url(#${fieldId('arrow-b')})`" />
			<text :x="pointA.x" :y="pointA.y - 8" class="label-a">A</text>
			<text :x="pointB.x" :y="pointB.y - 8" class="label-b">B</text>
		</svg>
		<div class="infra-results">
			<div class="infra-result"><span>点积</span><strong>{{ dot.toFixed(3) }}</strong></div>
			<div class="infra-result"><span>余弦相似度</span><strong>{{ cosineText }}</strong></div>
			<div class="infra-result"><span>欧氏距离</span><strong>{{ distance.toFixed(3) }}</strong></div>
		</div>
		<p class="infra-note">余弦忽略向量长度，点积同时受方向和模长影响；检索索引的度量必须与训练和归一化约定一致。</p>
	</LearningLab>
</template>

<style scoped>
.vector-plane { display: block; width: min(100%, 260px); margin: .9rem auto; border: 1px solid var(--vp-c-divider); border-radius: 10px; background: var(--vp-c-bg); }
.axis { stroke: var(--vp-c-divider); stroke-width: 1; }
.vector { stroke-width: 4; stroke-linecap: round; }
.vector-a { stroke: var(--vp-c-brand-1); }
.vector-b { stroke: var(--vp-c-green-1); }
.label-a, .label-b { font-size: 12px; font-weight: 700; text-anchor: middle; }
.label-a { fill: var(--vp-c-brand-1); }
.label-b { fill: var(--vp-c-green-1); }
@media (max-width: 640px) { .vector-plane { width: min(100%, 220px); } }
</style>
