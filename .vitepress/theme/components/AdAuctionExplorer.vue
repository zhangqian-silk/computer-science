<script setup lang="ts">
import { computed, ref, useId } from "vue"
import LearningLab from "./LearningLab.vue"
import { useLabReset } from "../use-lab-reset"

const labId = useId()
const fieldId = (name: string) => `${labId}-${name}`

const ads = ref([
	{ name: "广告A", bid: 3.0, ctr: 0.05 },
	{ name: "广告B", bid: 5.0, ctr: 0.02 },
	{ name: "广告C", bid: 2.0, ctr: 0.08 }
])

// eCPM = bid × ctr × 1000（按 CPC 计费的期望千次展示收入）
const ranked = computed(() =>
	ads.value
		.map(a => ({ ...a, ecpm: a.bid * a.ctr * 1000 }))
		.sort((x, y) => y.ecpm - x.ecpm)
)

const result = computed(() => {
	const r = ranked.value
	if (r.length < 2) return null
	const winner = r[0], second = r[1]
	// 广义第二价格(GSP)：赢家实付使自己 eCPM 刚好压过第二名
	const actualCpc = (second.ecpm / (winner.ctr * 1000)) + 0.01
	return { winner, second, actualCpc, saved: winner.bid - actualCpc }
})

const resetLab = useLabReset(ads)
</script>

<template>
	<LearningLab topic="AdAuctionExplorer" @reset="resetLab">
		<p class="infra-lab__hint">广告按 eCPM = 出价 × 预估点击率 排序（而非只看出价）。赢家按「广义第二价格」计费。调整出价与点击率看谁胜出、实付多少。</p>
		<div class="ad-inputs">
			<div v-for="(a, i) in ads" :key="i" class="ad-row">
				<strong>{{ a.name }}</strong>
				<label :for="fieldId('bid' + i)">出价 ¥{{ a.bid.toFixed(1) }}</label>
				<input :id="fieldId('bid' + i)" v-model.number="a.bid" type="range" min="0.5" max="8" step="0.5">
				<label :for="fieldId('ctr' + i)">CTR {{ (a.ctr * 100).toFixed(0) }}%</label>
				<input :id="fieldId('ctr' + i)" v-model.number="a.ctr" type="range" min="0.01" max="0.12" step="0.01">
			</div>
		</div>
		<ol class="ad-rank">
			<li v-for="(a, i) in ranked" :key="a.name" :class="{ win: i === 0 }">
				<span>#{{ i + 1 }}</span><strong>{{ a.name }}</strong><span>eCPM ¥{{ a.ecpm.toFixed(1) }}</span>
			</li>
		</ol>
		<div v-if="result" class="infra-results">
			<div class="infra-result"><span>胜出</span><strong>{{ result.winner.name }}</strong></div>
			<div class="infra-result"><span>实付 CPC（GSP）</span><strong>¥{{ result.actualCpc.toFixed(2) }}</strong></div>
			<div class="infra-result"><span>相比出价省下</span><strong>¥{{ result.saved.toFixed(2) }}</strong></div>
		</div>
		<p class="infra-note">按 eCPM 排序让平台在「广告主出价」与「用户体验/点击率」间平衡：高出价但低点击的广告未必胜出。第二价格计费让广告主按真实价值出价而非博弈。</p>
	</LearningLab>
</template>

<style scoped>
.ad-inputs { display: flex; flex-direction: column; gap: .5rem; margin: .6rem 0; }
.ad-row { display: grid; grid-template-columns: 60px auto 1fr auto 1fr; gap: .5rem; align-items: center; font-size: .78rem; }
.ad-rank { list-style: none; padding: 0; margin: .6rem 0; display: flex; flex-direction: column; gap: .25rem; }
.ad-rank li { display: flex; align-items: center; gap: .6rem; padding: .35rem .6rem; border-radius: 5px; background: var(--vp-c-bg-soft); font-size: .82rem; }
.ad-rank li.win { background: var(--vp-c-brand-soft); }
.ad-rank li span:first-child { width: 26px; color: var(--vp-c-text-3); }
.ad-rank li strong { flex: 1; }
@media (max-width: 640px) { .ad-row { grid-template-columns: 1fr 1fr; } }
</style>
