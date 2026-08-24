<script setup lang="ts">
import { computed, ref } from 'vue'

const grades = ref([2, 0, 1, 0, 2, 0, 1, 0])
const cutoff = ref(5)
const cycleGrade = (index: number) => {
	grades.value[index] = (grades.value[index] + 1) % 3
}
const retrieved = computed(() => grades.value.slice(0, cutoff.value))
const relevantTotal = computed(() => grades.value.filter((grade) => grade > 0).length)
const relevantRetrieved = computed(() => retrieved.value.filter((grade) => grade > 0).length)
const precision = computed(() => relevantRetrieved.value / cutoff.value)
const recall = computed(() => relevantTotal.value === 0 ? 0 : relevantRetrieved.value / relevantTotal.value)
const reciprocalRank = computed(() => {
	const first = retrieved.value.findIndex((grade) => grade > 0)
	return first === -1 ? 0 : 1 / (first + 1)
})
const dcg = (values: number[]) => values.reduce((sum, grade, index) => sum + (2 ** grade - 1) / Math.log2(index + 2), 0)
const ndcg = computed(() => {
	const actual = dcg(retrieved.value)
	const ideal = dcg([...grades.value].sort((left, right) => right - left).slice(0, cutoff.value))
	return ideal === 0 ? 0 : actual / ideal
})
</script>

<template>
	<div class="infra-lab">
		<p class="infra-lab__title">检索指标实验台</p>
		<p class="infra-lab__hint">点击结果循环切换不相关、相关、高度相关，观察 K 如何改变 Precision、Recall、MRR 与 NDCG。</p>
		<div class="ranking-strip" aria-label="检索结果相关性等级">
			<button
				v-for="(grade, index) in grades"
				:key="index"
				type="button"
				:class="[`grade-${grade}`, { 'is-outside': index >= cutoff }]"
				:aria-label="`第 ${index + 1} 名，相关性 ${grade}，点击切换`"
				@click="cycleGrade(index)"
			>
				<span>#{{ index + 1 }}</span><strong>{{ grade }}</strong>
			</button>
		</div>
		<div class="infra-control"><label for="retrieval-cutoff">评估截断 K：{{ cutoff }}</label><input id="retrieval-cutoff" v-model.number="cutoff" type="range" min="1" :max="grades.length"></div>
		<div class="infra-results">
			<div class="infra-result"><span>Precision@{{ cutoff }}</span><strong>{{ precision.toFixed(3) }}</strong></div>
			<div class="infra-result"><span>Recall@{{ cutoff }}</span><strong>{{ recall.toFixed(3) }}</strong></div>
			<div class="infra-result"><span>MRR@{{ cutoff }}</span><strong>{{ reciprocalRank.toFixed(3) }}</strong></div>
			<div class="infra-result"><span>NDCG@{{ cutoff }}</span><strong>{{ ndcg.toFixed(3) }}</strong></div>
		</div>
		<p class="infra-note">Precision/Recall 把等级大于 0 视为相关；NDCG 保留 0/1/2 的分级收益。真实评估还需要冻结相关性标注、query 集合和 tie 处理。</p>
	</div>
</template>

<style scoped>
.ranking-strip { display: flex; gap: .4rem; margin: .9rem 0; overflow-x: auto; padding-bottom: .25rem; }
.ranking-strip button { min-width: 54px; min-height: 52px; border: 1px solid var(--vp-c-divider); border-radius: 8px; background: var(--vp-c-bg); color: var(--vp-c-text-1); cursor: pointer; }
.ranking-strip button span, .ranking-strip button strong { display: block; }
.ranking-strip button span { font-size: .68rem; color: var(--vp-c-text-2); }
.ranking-strip button.grade-1 { border-color: var(--vp-c-brand-1); background: var(--vp-c-brand-soft); }
.ranking-strip button.grade-2 { border-color: var(--vp-c-green-1); background: var(--vp-c-green-soft); }
.ranking-strip button.is-outside { opacity: .42; }
</style>
