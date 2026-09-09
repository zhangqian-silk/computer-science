<script setup lang="ts">
import { computed, ref, useId } from "vue"
import LearningLab from "./LearningLab.vue"
import { useLabReset } from "../use-lab-reset"

const labId = useId()
const fieldId = (name: string) => `${labId}-${name}`

const nodes = ref(3)
const vnodes = ref(1)
const keys = ref(120)
const removeOne = ref(false)

const RING = 65536
const palette = ["#3b82f6", "#10b981", "#f59e0b", "#ef4444", "#8b5cf6", "#14b8a6", "#ec4899", "#64748b"]

function hash(s: string): number {
	let h = 0x811c9dc5
	for (let i = 0; i < s.length; i++) {
		h ^= s.charCodeAt(i)
		h = Math.imul(h, 0x01000193)
	}
	return (h >>> 0) % RING
}

type VNode = { pos: number; node: number }

function buildRing(nodeCount: number): VNode[] {
	const ring: VNode[] = []
	for (let n = 0; n < nodeCount; n++) {
		for (let v = 0; v < vnodes.value; v++) {
			ring.push({ pos: hash(`N${n}#${v}`), node: n })
		}
	}
	return ring.sort((a, b) => a.pos - b.pos)
}

function ownerOf(ring: VNode[], kpos: number): number {
	for (const vn of ring) {
		if (vn.pos >= kpos) return vn.node
	}
	return ring[0].node // 环回
}

const analysis = computed(() => {
	const full = buildRing(nodes.value)
	const reduced = buildRing(Math.max(1, nodes.value - 1))
	const counts = new Array(nodes.value).fill(0)
	let migrated = 0
	for (let k = 0; k < keys.value; k++) {
		const kpos = hash(`key-${k}`)
		const o1 = ownerOf(full, kpos)
		counts[o1]++
		if (removeOne.value) {
			const o2 = ownerOf(reduced, kpos)
			if (o2 !== o1) migrated++
		}
	}
	const avg = keys.value / nodes.value
	const maxLoad = Math.max(...counts)
	const minLoad = Math.min(...counts)
	return {
		ring: full,
		counts,
		skew: avg ? (maxLoad / avg) : 0,
		spread: avg ? ((maxLoad - minLoad) / avg) : 0,
		migrated,
		migratePct: keys.value ? (migrated / keys.value) * 100 : 0,
		idealPct: 100 / nodes.value
	}
})

const ticks = computed(() =>
	analysis.value.ring.map(vn => {
		const a = (vn.pos / RING) * 2 * Math.PI - Math.PI / 2
		return {
			x1: 100 + 78 * Math.cos(a), y1: 100 + 78 * Math.sin(a),
			x2: 100 + 94 * Math.cos(a), y2: 100 + 94 * Math.sin(a),
			color: palette[vn.node % palette.length]
		}
	})
)

const resetLab = useLabReset(nodes, vnodes, keys, removeOne)
</script>

<template>
	<LearningLab topic="ConsistentHashRing" @reset="resetLab">
		<p class="infra-lab__hint">哈希环范围 0..{{ RING }}，key 顺时针归属第一个虚拟节点。增加虚拟节点看负载如何变均衡；勾选「移除一个节点」看迁移比例。</p>
		<div class="infra-controls">
			<div class="infra-control"><label :for="fieldId('n')">物理节点：{{ nodes }}</label><input :id="fieldId('n')" v-model.number="nodes" type="range" min="2" max="8"></div>
			<div class="infra-control"><label :for="fieldId('v')">每节点虚拟节点：{{ vnodes }}</label><input :id="fieldId('v')" v-model.number="vnodes" type="range" min="1" max="80"></div>
			<div class="infra-control"><label :for="fieldId('k')">key 数量：{{ keys }}</label><input :id="fieldId('k')" v-model.number="keys" type="range" min="30" max="400" step="10"></div>
		</div>
		<label class="ch-toggle"><input type="checkbox" v-model="removeOne"> 移除最后一个节点，统计迁移比例</label>
		<div class="ch-body">
			<svg viewBox="0 0 200 200" class="ch-ring" role="img" aria-label="一致性哈希环">
				<circle cx="100" cy="100" r="86" fill="none" stroke="var(--vp-c-divider)" stroke-width="1.5" />
				<line v-for="(t, i) in ticks" :key="i" :x1="t.x1" :y1="t.y1" :x2="t.x2" :y2="t.y2" :stroke="t.color" stroke-width="2" />
			</svg>
			<div class="ch-bars">
				<div v-for="(c, i) in analysis.counts" :key="i" class="ch-bar-row">
					<span class="ch-dot" :style="{ background: palette[i % palette.length] }"></span>
					<span class="ch-name">N{{ i }}</span>
					<span class="ch-bar"><i :style="{ width: Math.min(100, (c / keys) * 100 * 2) + '%', background: palette[i % palette.length] }"></i></span>
					<strong>{{ c }}</strong>
				</div>
			</div>
		</div>
		<div class="infra-results">
			<div class="infra-result"><span>最重节点 / 均值</span><strong>{{ analysis.skew.toFixed(2) }}×</strong></div>
			<div class="infra-result"><span>最大负载差</span><strong>{{ (analysis.spread * 100).toFixed(0) }}%</strong></div>
			<div class="infra-result"><span>移除后迁移比例</span><strong>{{ removeOne ? analysis.migratePct.toFixed(1) + '%（理想≈' + analysis.idealPct.toFixed(1) + '%）' : '—' }}</strong></div>
		</div>
		<p class="infra-note">虚拟节点越多，负载越接近均匀；移除节点时只有该节点的 key 迁移到后继，迁移比例趋近 1/N，这正是一致性哈希相对「取模」的核心优势。</p>
	</LearningLab>
</template>

<style scoped>
.ch-toggle { display: flex; align-items: center; gap: .5rem; margin: .6rem 0; font-size: .85rem; color: var(--vp-c-text-2); }
.ch-body { display: flex; gap: 1rem; flex-wrap: wrap; align-items: center; margin-top: .6rem; }
.ch-ring { width: 200px; height: 200px; flex: 0 0 auto; }
.ch-bars { flex: 1 1 220px; display: flex; flex-direction: column; gap: .35rem; }
.ch-bar-row { display: flex; align-items: center; gap: .5rem; font-size: .8rem; }
.ch-dot { width: 10px; height: 10px; border-radius: 50%; flex: 0 0 auto; }
.ch-name { width: 28px; color: var(--vp-c-text-2); }
.ch-bar { flex: 1; height: 12px; background: var(--vp-c-bg-soft); border-radius: 6px; overflow: hidden; }
.ch-bar > i { display: block; height: 100%; }
.ch-bar-row strong { width: 34px; text-align: right; font-variant-numeric: tabular-nums; }
</style>
