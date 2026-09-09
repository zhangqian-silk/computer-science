<script setup lang="ts">
import { computed, ref, useId } from "vue"
import LearningLab from "./LearningLab.vue"
import { useLabReset } from "../use-lab-reset"

const labId = useId()
const fieldId = (name: string) => `${labId}-${name}`

const stock = ref(5)
const concurrent = ref(20)
const strategy = ref<"naive" | "atomic" | "prededuct">("naive")

const result = computed(() => {
	const S = stock.value, N = concurrent.value
	if (strategy.value === "naive") {
		// 先查后改：并发下都读到有货 -> 全部扣成功 -> 超卖
		const sold = N
		return { sold, oversold: Math.max(0, N - S), rejected: 0,
			note: "先查库存再扣减：并发请求同时读到「有货」，全部通过，卖出超过库存 —— 典型超卖。" }
	}
	if (strategy.value === "atomic") {
		const sold = Math.min(N, S)
		return { sold, oversold: 0, rejected: N - sold,
			note: "DB 原子扣减 update stock=stock-1 where stock>0：靠行锁串行化，卖完即拒，不会超卖。高并发下 DB 行锁是瓶颈。" }
	}
	const sold = Math.min(N, S)
	return { sold, oversold: 0, rejected: N - sold,
		note: "Redis 预扣减（DECR/Lua）：内存原子扣减扛住高并发，异步落库，需与 DB 对账兜底防不一致。" }
})

const resetLab = useLabReset(stock, concurrent, strategy)
</script>

<template>
	<LearningLab topic="InventoryDeductionExplorer" @reset="resetLab">
		<p class="infra-lab__hint">{{ concurrent }} 个并发请求抢 {{ stock }} 件库存。切换扣减策略，观察是否超卖。</p>
		<div class="infra-tabs" role="group" aria-label="扣减策略">
			<button type="button" :aria-pressed="strategy === 'naive'" @click="strategy = 'naive'">先查后改（有坑）</button>
			<button type="button" :aria-pressed="strategy === 'atomic'" @click="strategy = 'atomic'">DB 原子扣减</button>
			<button type="button" :aria-pressed="strategy === 'prededuct'" @click="strategy = 'prededuct'">Redis 预扣减</button>
		</div>
		<div class="infra-controls">
			<div class="infra-control"><label :for="fieldId('s')">库存：{{ stock }}</label><input :id="fieldId('s')" v-model.number="stock" type="range" min="1" max="20"></div>
			<div class="infra-control"><label :for="fieldId('c')">并发请求：{{ concurrent }}</label><input :id="fieldId('c')" v-model.number="concurrent" type="range" min="1" max="50"></div>
		</div>
		<div class="infra-results">
			<div class="infra-result"><span>成功扣减</span><strong>{{ result.sold }}</strong></div>
			<div class="infra-result"><span>被拒绝</span><strong>{{ result.rejected }}</strong></div>
			<div class="infra-result" :class="{ 'inv-bad': result.oversold > 0 }"><span>超卖</span><strong>{{ result.oversold }}</strong></div>
		</div>
		<p class="infra-note">{{ result.note }}</p>
	</LearningLab>
</template>

<style scoped>
.inv-bad strong { color: var(--vp-c-danger-1); }
</style>
