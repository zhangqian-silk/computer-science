<script setup lang="ts">
import { computed, ref, useId } from "vue"
import LearningLab from "./LearningLab.vue"
import { useLabReset } from "../use-lab-reset"

const labId = useId()
const fieldId = (name: string) => `${labId}-${name}`

const shards = ref(4)
const KEYS = 2000
const RING = 1 << 20

function h(n: number): number {
	let x = (n ^ 0x9e3779b9) >>> 0
	x = Math.imul(x ^ (x >>> 16), 0x85ebca6b) >>> 0
	x = Math.imul(x ^ (x >>> 13), 0xc2b2ae35) >>> 0
	return (x ^ (x >>> 16)) >>> 0
}

const ownerMap = new Map<number, number>()
function buildRingWithMap(n: number): number[] {
	ownerMap.clear()
	const arr: number[] = []
	for (let i = 0; i < n; i++) for (let v = 0; v < 60; v++) { const p = h(i * 1000 + v) % RING; arr.push(p); ownerMap.set(p, i) }
	return arr.sort((a, b) => a - b)
}

const result = computed(() => {
	const N = shards.value
	let modPlus1 = 0, modDouble = 0, consistent = 0
	const ringN = buildRingWithMap(N)
	const ownerN = new Map(ownerMap)
	const ringN1 = buildRingWithMap(N + 1)
	for (let k = 0; k < KEYS; k++) {
		const hk = h(k)
		if (hk % N !== hk % (N + 1)) modPlus1++            // mod: N -> N+1
		if (hk % N !== hk % (2 * N)) modDouble++            // mod 倍增: N -> 2N
		// 一致性哈希 N -> N+1
		const posK = hk % RING
		const o1 = ownerAt(ringN, ownerN, posK)
		const o2 = ownerAt(ringN1, ownerMap, posK)
		if (o1 !== o2) consistent++
	}
	return {
		modPlus1: (modPlus1 / KEYS) * 100,
		modDouble: (modDouble / KEYS) * 100,
		consistent: (consistent / KEYS) * 100
	}
})

function ownerAt(ring: number[], map: Map<number, number>, pos: number): number {
	let lo = 0, hi = ring.length
	while (lo < hi) { const m = (lo + hi) >> 1; if (ring[m] >= pos) hi = m; else lo = m + 1 }
	const idx = lo === ring.length ? 0 : lo
	return map.get(ring[idx]) ?? 0
}

const bars = computed(() => [
	{ name: "取模 N→N+1（加 1 台）", pct: result.value.modPlus1, tone: "danger" },
	{ name: "取模 N→2N（倍增扩容）", pct: result.value.modDouble, tone: "warning" },
	{ name: "一致性哈希 N→N+1", pct: result.value.consistent, tone: "success" }
])

const resetLab = useLabReset(shards)
</script>

<template>
	<LearningLab topic="ShardingMigrationExplorer" @reset="resetLab">
		<p class="infra-lab__hint">{{ KEYS }} 个 key 分布在 {{ shards }} 个分片上，扩容时对比三种路由策略要迁移的数据比例。</p>
		<div class="infra-control"><label :for="fieldId('n')">当前分片数 N：{{ shards }}</label><input :id="fieldId('n')" v-model.number="shards" type="range" min="2" max="16"></div>
		<div class="sm-bars">
			<div v-for="b in bars" :key="b.name" class="sm-row">
				<span class="sm-name">{{ b.name }}</span>
				<span class="sm-bar"><i :class="`t-${b.tone}`" :style="{ width: b.pct + '%' }"></i></span>
				<strong>{{ b.pct.toFixed(0) }}%</strong>
			</div>
		</div>
		<p class="infra-note">直接「取模 +1」几乎要迁移全部数据；「倍增扩容」利用 mod 性质只迁移约一半；一致性哈希只迁移约 1/N。这解释了为什么分库分表要「按 2 的幂倍增」，以及缓存分片偏爱一致性哈希。</p>
	</LearningLab>
</template>

<style scoped>
.sm-bars { display: flex; flex-direction: column; gap: .5rem; margin: .8rem 0; }
.sm-row { display: flex; align-items: center; gap: .6rem; font-size: .82rem; }
.sm-name { flex: 0 0 200px; color: var(--vp-c-text-2); }
.sm-bar { flex: 1; height: 14px; background: var(--vp-c-bg-soft); border-radius: 7px; overflow: hidden; }
.sm-bar > i { display: block; height: 100%; }
.t-danger { background: var(--vp-c-danger-1); }
.t-warning { background: var(--vp-c-warning-1); }
.t-success { background: var(--vp-c-green-1); }
.sm-row strong { width: 44px; text-align: right; font-variant-numeric: tabular-nums; }
@media (max-width: 640px) { .sm-name { flex-basis: 130px; } }
</style>
