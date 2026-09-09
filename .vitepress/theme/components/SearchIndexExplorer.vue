<script setup lang="ts">
import { computed, ref } from "vue"
import LearningLab from "./LearningLab.vue"
import { useLabReset } from "../use-lab-reset"

const docs: Record<string, string[]> = {
	D1: ["分布式", "系统", "设计", "缓存", "一致性"],
	D2: ["缓存", "击穿", "穿透", "雪崩", "缓存"],
	D3: ["分布式", "事务", "一致性", "补偿", "分布式"],
	D4: ["搜索", "倒排", "索引", "分布式", "系统"]
}
const docIds = Object.keys(docs)
const queryTerms = ["分布式", "缓存", "一致性", "系统"]

const mode = ref<"or" | "and">("or")
const selected = ref<string[]>(["分布式", "缓存"])

function toggle(t: string) {
	selected.value = selected.value.includes(t)
		? selected.value.filter(x => x !== t)
		: [...selected.value, t]
}

// 倒排索引：term -> 出现的文档列表
const invertedIndex = computed(() => {
	const idx: Record<string, string[]> = {}
	for (const t of queryTerms) {
		idx[t] = docIds.filter(d => docs[d].includes(t))
	}
	return idx
})

function tf(term: string, d: string): number {
	return docs[d].filter(w => w === term).length
}
function idf(term: string): number {
	const df = invertedIndex.value[term]?.length || 0
	return df ? Math.log(docIds.length / df) + 1 : 0
}

const results = computed(() => {
	if (selected.value.length === 0) return []
	const scored = docIds.map(d => {
		const matchedTerms = selected.value.filter(t => docs[d].includes(t))
		const score = matchedTerms.reduce((s, t) => s + tf(t, d) * idf(t), 0)
		return { d, score, matched: matchedTerms.length }
	})
	const filtered = scored.filter(r =>
		mode.value === "and" ? r.matched === selected.value.length : r.matched > 0
	)
	return filtered.sort((a, b) => b.score - a.score)
})

const resetLab = useLabReset(mode, selected)
</script>

<template>
	<LearningLab topic="SearchIndexExplorer" @reset="resetLab">
		<p class="infra-lab__hint">4 篇文档已建倒排索引。勾选查询词，观察命中文档与 TF-IDF 排序；切换 AND/OR 看召回与精度的取舍。</p>
		<div class="infra-tabs" role="group" aria-label="布尔模式">
			<button type="button" :aria-pressed="mode === 'or'" @click="mode = 'or'">OR 任一命中</button>
			<button type="button" :aria-pressed="mode === 'and'" @click="mode = 'and'">AND 全部命中</button>
		</div>
		<div class="si-terms">
			<button v-for="t in queryTerms" :key="t" type="button" :class="{ on: selected.includes(t) }" @click="toggle(t)">{{ t }}</button>
		</div>
		<div class="si-index">
			<div v-for="t in selected" :key="t" class="si-post"><strong>{{ t }}</strong> → [{{ invertedIndex[t].join(", ") }}] <em>idf={{ idf(t).toFixed(2) }}</em></div>
		</div>
		<ol class="si-results" aria-label="排序结果">
			<li v-for="r in results" :key="r.d">
				<span class="si-doc">{{ r.d }}</span>
				<span class="si-bar"><i :style="{ width: Math.min(100, r.score * 25) + '%' }"></i></span>
				<strong>{{ r.score.toFixed(2) }}</strong>
			</li>
			<li v-if="results.length === 0" class="si-empty">无命中文档</li>
		</ol>
		<p class="infra-note">倒排索引把「文档→词」翻转为「词→文档」，查询即取 posting list 求交/并，避免全量扫描。TF-IDF 用「词频 × 逆文档频率」打分：高频且稀有的词贡献大。AND 精度高召回低，OR 反之。</p>
	</LearningLab>
</template>

<style scoped>
.si-terms { display: flex; flex-wrap: wrap; gap: .4rem; margin: .6rem 0; }
.si-terms button { padding: .35rem .8rem; border: 1px solid var(--vp-c-divider); border-radius: 999px; background: var(--vp-c-bg); font-size: .82rem; }
.si-terms button.on { background: var(--vp-c-brand-1); color: #fff; border-color: var(--vp-c-brand-1); }
.si-index { margin: .6rem 0; font-size: .8rem; font-family: var(--vp-font-family-mono); color: var(--vp-c-text-2); display: flex; flex-direction: column; gap: .2rem; }
.si-index em { color: var(--vp-c-text-3); }
.si-results { list-style: none; padding: 0; margin: .6rem 0 0; display: flex; flex-direction: column; gap: .35rem; }
.si-results li { display: flex; align-items: center; gap: .6rem; font-size: .82rem; }
.si-doc { width: 32px; font-weight: 600; }
.si-bar { flex: 1; height: 12px; background: var(--vp-c-bg-soft); border-radius: 6px; overflow: hidden; }
.si-bar > i { display: block; height: 100%; background: var(--vp-c-brand-1); }
.si-results strong { width: 40px; text-align: right; font-variant-numeric: tabular-nums; }
.si-empty { color: var(--vp-c-text-3); }
</style>
