<script setup lang="ts">
import { computed, ref, useId } from "vue"
import LearningLab from "./LearningLab.vue"
import { useLabReset } from "../use-lab-reset"

const labId = useId()
const fieldId = (name: string) => `${labId}-${name}`

const policy = ref<"lru" | "lfu" | "fifo">("lru")
const capacity = ref(3)

// 固定访问序列（含重复热点 A），便于对比不同策略的命中率
const accesses = ["A", "B", "C", "A", "D", "A", "B", "E", "A", "C"]

type Sim = { hits: number; misses: number; steps: { key: string; hit: boolean; state: string[] }[] }

const sim = computed<Sim>(() => {
	const cap = capacity.value
	let hits = 0, misses = 0
	const steps: Sim["steps"] = []
	const order: string[] = []          // 用于 LRU/FIFO 的顺序
	const freq: Record<string, number> = {}
	const inCache = new Set<string>()

	for (const key of accesses) {
		const hit = inCache.has(key)
		if (hit) {
			hits++
			if (policy.value === "lru") { // 命中提升到最近
				order.splice(order.indexOf(key), 1)
				order.push(key)
			}
			freq[key] = (freq[key] || 0) + 1
		} else {
			misses++
			if (inCache.size >= cap) {
				let victim = order[0]
				if (policy.value === "lfu") {
					// 逐出频率最低者，平局按最早进入
					victim = [...inCache].reduce((a, b) => {
						const fa = freq[a] || 0, fb = freq[b] || 0
						if (fa !== fb) return fa < fb ? a : b
						return order.indexOf(a) < order.indexOf(b) ? a : b
					})
				}
				inCache.delete(victim)
				order.splice(order.indexOf(victim), 1)
			}
			inCache.add(key)
			order.push(key)
			freq[key] = (freq[key] || 0) + 1
		}
		steps.push({ key, hit, state: [...order] })
	}
	return { hits, misses, steps }
})

const hitRate = computed(() => (sim.value.hits / accesses.length * 100).toFixed(0))
const resetLab = useLabReset(policy, capacity)
</script>

<template>
	<LearningLab topic="CacheEvictionExplorer" @reset="resetLab">
		<p class="infra-lab__hint">固定访问序列 {{ accesses.join(" ") }}（A 是热点）。切换淘汰策略与容量，观察命中率与被逐出的 key。</p>
		<div class="infra-tabs" role="group" aria-label="淘汰策略">
			<button type="button" :aria-pressed="policy === 'lru'" @click="policy = 'lru'">LRU 最近最少用</button>
			<button type="button" :aria-pressed="policy === 'lfu'" @click="policy = 'lfu'">LFU 最不经常用</button>
			<button type="button" :aria-pressed="policy === 'fifo'" @click="policy = 'fifo'">FIFO 先进先出</button>
		</div>
		<div class="infra-control"><label :for="fieldId('cap')">容量：{{ capacity }}</label><input :id="fieldId('cap')" v-model.number="capacity" type="range" min="2" max="5"></div>
		<div class="ce-steps">
			<div v-for="(s, i) in sim.steps" :key="i" class="ce-step" :class="s.hit ? 'ce-hit' : 'ce-miss'">
				<span class="ce-key">{{ s.key }}</span>
				<span class="ce-tag">{{ s.hit ? '命中' : '未命中' }}</span>
				<span class="ce-state">[{{ s.state.join(' ') }}]</span>
			</div>
		</div>
		<div class="infra-results">
			<div class="infra-result"><span>命中</span><strong>{{ sim.hits }}</strong></div>
			<div class="infra-result"><span>未命中</span><strong>{{ sim.misses }}</strong></div>
			<div class="infra-result"><span>命中率</span><strong>{{ hitRate }}%</strong></div>
		</div>
		<p class="infra-note">LRU 保近期热点、实现简单，是默认之选；LFU 保长期高频、但对突发新热点反应慢且需计数；FIFO 不看访问、命中率通常最低。缓存列表右侧为淘汰顺序（最左最先被逐）。</p>
	</LearningLab>
</template>

<style scoped>
.ce-steps { display: flex; flex-direction: column; gap: .25rem; margin: .8rem 0; font-size: .8rem; }
.ce-step { display: flex; align-items: center; gap: .6rem; padding: .3rem .5rem; border-radius: 5px; border-left: 3px solid var(--vp-c-divider); background: var(--vp-c-bg); }
.ce-hit { border-left-color: var(--vp-c-green-1); }
.ce-miss { border-left-color: var(--vp-c-warning-1); }
.ce-key { font-weight: 700; width: 18px; }
.ce-tag { width: 48px; color: var(--vp-c-text-3); font-size: .72rem; }
.ce-state { font-family: var(--vp-font-family-mono); color: var(--vp-c-text-2); }
</style>
