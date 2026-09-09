<script setup lang="ts">
import { computed, ref, useId } from "vue"
import LearningLab from "./LearningLab.vue"
import { useLabReset } from "../use-lab-reset"

const labId = useId()
const fieldId = (name: string) => `${labId}-${name}`

const players = ref([
	{ name: "Alice", score: 820 },
	{ name: "Bob", score: 640 },
	{ name: "Carol", score: 910 },
	{ name: "Dave", score: 500 },
	{ name: "Eve", score: 730 },
	{ name: "Frank", score: 680 }
])
const topN = ref(3)
const focus = ref("Dave")

const ranked = computed(() =>
	[...players.value].sort((a, b) => b.score - a.score).map((p, i) => ({ ...p, rank: i + 1 }))
)
const focusRank = computed(() => ranked.value.find(p => p.name === focus.value)?.rank ?? 0)
const neighbors = computed(() => {
	const r = focusRank.value
	return ranked.value.filter(p => p.rank >= r - 1 && p.rank <= r + 1)
})

const resetLab = useLabReset(players, topN, focus)
</script>

<template>
	<LearningLab topic="LeaderboardZSetExplorer" @reset="resetLab">
		<p class="infra-lab__hint">有序集合（ZSet）按分数排序。拖动某人的分数看排名实时变化；ZSet 的 TopN 与「查我的排名/邻居」都是 O(log N)。</p>
		<div class="lb-players">
			<div v-for="p in players" :key="p.name" class="lb-edit">
				<label :for="fieldId(p.name)">{{ p.name }}：{{ p.score }}</label>
				<input :id="fieldId(p.name)" v-model.number="p.score" type="range" min="0" max="1000" step="10">
			</div>
		</div>
		<div class="infra-control"><label :for="fieldId('topn')">TopN：{{ topN }}</label><input :id="fieldId('topn')" v-model.number="topN" type="range" min="1" max="6"></div>
		<ol class="lb-board">
			<li v-for="p in ranked" :key="p.name" :class="{ 'lb-top': p.rank <= topN, 'lb-focus': p.name === focus }">
				<span class="lb-rank">#{{ p.rank }}</span><strong>{{ p.name }}</strong><span class="lb-score">{{ p.score }}</span>
			</li>
		</ol>
		<div class="infra-control">
			<label :for="fieldId('focus')">查某人排名与邻居</label>
			<select :id="fieldId('focus')" v-model="focus"><option v-for="p in players" :key="p.name" :value="p.name">{{ p.name }}</option></select>
		</div>
		<div class="infra-result"><span>{{ focus }} 的排名</span><strong>#{{ focusRank }}（邻居：{{ neighbors.map(n => n.name).join(" / ") }}）</strong></div>
		<p class="infra-note">用 Redis ZSet：ZADD 更新分数、ZREVRANGE 取 TopN、ZREVRANK 查名次，均 O(log N)。海量榜单需分片/分段 + 冷热分层；实时榜与结算榜口径分离，写入需幂等去重防刷分。</p>
	</LearningLab>
</template>

<style scoped>
.lb-players { display: grid; grid-template-columns: repeat(auto-fit, minmax(min(100%, 180px), 1fr)); gap: .4rem; margin: .6rem 0; }
.lb-edit { display: flex; flex-direction: column; font-size: .78rem; }
.lb-board { list-style: none; padding: 0; margin: .6rem 0; display: flex; flex-direction: column; gap: .25rem; }
.lb-board li { display: flex; align-items: center; gap: .6rem; padding: .35rem .6rem; border-radius: 5px; background: var(--vp-c-bg-soft); font-size: .84rem; }
.lb-board li.lb-top { background: var(--vp-c-brand-soft); }
.lb-board li.lb-focus { outline: 2px solid var(--vp-c-warning-1); }
.lb-rank { width: 32px; color: var(--vp-c-text-3); font-variant-numeric: tabular-nums; }
.lb-board strong { flex: 1; }
.lb-score { font-variant-numeric: tabular-nums; }
</style>
