<script setup lang="ts">
import { computed, ref, useId } from "vue"
import LearningLab from "./LearningLab.vue"
import { useLabReset } from "../use-lab-reset"

const labId = useId()
const fieldId = (name: string) => `${labId}-${name}`

const algo = ref<"token" | "leaky">("token")
const rate = ref(2)      // 每秒补充令牌 / 漏出请求数
const capacity = ref(5)  // 桶容量（令牌桶=突发上限，漏桶=排队上限）

// 固定到达序列：在若干时刻各到达的请求数（模拟突发）
const arrivals = [3, 0, 4, 1, 0, 5, 0, 2, 0, 1]

type Tick = { t: number; arrive: number; pass: number; reject: number; level: number }

const sim = computed<Tick[]>(() => {
	const out: Tick[] = []
	if (algo.value === "token") {
		let tokens = capacity.value
		for (let t = 0; t < arrivals.length; t++) {
			tokens = Math.min(capacity.value, tokens + rate.value) // 每个 tick 补充
			const arrive = arrivals[t]
			const pass = Math.min(arrive, tokens)
			tokens -= pass
			out.push({ t, arrive, pass, reject: arrive - pass, level: tokens })
		}
	} else {
		let queue = 0
		for (let t = 0; t < arrivals.length; t++) {
			const arrive = arrivals[t]
			const room = capacity.value - queue
			const admitted = Math.min(arrive, room)
			queue += admitted
			const pass = Math.min(queue, rate.value) // 匀速漏出
			queue -= pass
			out.push({ t, arrive, pass, reject: arrive - admitted, level: queue })
		}
	}
	return out
})

const totals = computed(() => ({
	arrive: sim.value.reduce((a, b) => a + b.arrive, 0),
	pass: sim.value.reduce((a, b) => a + b.pass, 0),
	reject: sim.value.reduce((a, b) => a + b.reject, 0)
}))

const resetLab = useLabReset(algo, rate, capacity)
</script>

<template>
	<LearningLab topic="RateLimiterExplorer" @reset="resetLab">
		<p class="infra-lab__hint">固定到达序列（含突发）经过限流器。令牌桶允许攒令牌应对突发；漏桶匀速流出、超出排队上限则拒绝。</p>
		<div class="infra-tabs" role="group" aria-label="限流算法">
			<button type="button" :aria-pressed="algo === 'token'" @click="algo = 'token'">令牌桶（允许突发）</button>
			<button type="button" :aria-pressed="algo === 'leaky'" @click="algo = 'leaky'">漏桶（匀速整形）</button>
		</div>
		<div class="infra-controls">
			<div class="infra-control"><label :for="fieldId('rate')">速率 rate：{{ rate }}/tick</label><input :id="fieldId('rate')" v-model.number="rate" type="range" min="1" max="6"></div>
			<div class="infra-control"><label :for="fieldId('cap')">容量 capacity：{{ capacity }}</label><input :id="fieldId('cap')" v-model.number="capacity" type="range" min="1" max="10"></div>
		</div>
		<div class="rl-grid" :style="{ '--cols': arrivals.length }">
			<div class="rl-label">到达</div>
			<div v-for="c in sim" :key="'a' + c.t" class="rl-cell">{{ c.arrive }}</div>
			<div class="rl-label">通过</div>
			<div v-for="c in sim" :key="'p' + c.t" class="rl-cell rl-pass">{{ c.pass }}</div>
			<div class="rl-label">拒绝</div>
			<div v-for="c in sim" :key="'r' + c.t" class="rl-cell" :class="{ 'rl-reject': c.reject > 0 }">{{ c.reject }}</div>
			<div class="rl-label">{{ algo === 'token' ? '余量' : '排队' }}</div>
			<div v-for="c in sim" :key="'l' + c.t" class="rl-cell rl-level">{{ c.level }}</div>
		</div>
		<div class="infra-results">
			<div class="infra-result"><span>总到达</span><strong>{{ totals.arrive }}</strong></div>
			<div class="infra-result"><span>通过</span><strong>{{ totals.pass }}</strong></div>
			<div class="infra-result"><span>拒绝</span><strong>{{ totals.reject }}</strong></div>
		</div>
		<p class="infra-note">令牌桶在空闲期攒令牌，可放行一次突发；漏桶输出恒为匀速，对下游最友好但会牺牲突发吞吐。两者选择取决于下游能否承受突发。</p>
	</LearningLab>
</template>

<style scoped>
.rl-grid { display: grid; grid-template-columns: 48px repeat(var(--cols), 1fr); gap: 3px; margin: .8rem 0; overflow-x: auto; }
.rl-label { font-size: .72rem; color: var(--vp-c-text-3); display: flex; align-items: center; }
.rl-cell { text-align: center; padding: .3rem 0; font-size: .8rem; background: var(--vp-c-bg-soft); border-radius: 4px; font-variant-numeric: tabular-nums; }
.rl-pass { background: var(--vp-c-green-soft); color: var(--vp-c-green-1); }
.rl-reject { background: var(--vp-c-danger-soft); color: var(--vp-c-danger-1); }
.rl-level { background: var(--vp-c-brand-soft); color: var(--vp-c-brand-1); }
</style>
