<script setup lang="ts">
import { computed, ref, useId } from "vue"
import LearningLab from "./LearningLab.vue"
import { useLabReset } from "../use-lab-reset"

const labId = useId()
const fieldId = (name: string) => `${labId}-${name}`

const alphabet = "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
const id = ref(1000000)

function toBase62(n: number): string {
	if (n <= 0) return "0"
	let s = ""
	while (n > 0) {
		s = alphabet[n % 62] + s
		n = Math.floor(n / 62)
	}
	return s
}

const code = computed(() => toBase62(id.value))
const length = computed(() => code.value.length)

const presets = [
	{ label: "第 1 个", v: 1 },
	{ label: "1 万", v: 10000 },
	{ label: "100 万", v: 1000000 },
	{ label: "1 亿", v: 100000000 },
	{ label: "10 亿", v: 1000000000 }
]

const capacityTable = [6, 7, 8].map(len => ({
	len,
	cap: 62 ** len
}))

function fmtCap(n: number): string {
	if (n >= 1e12) return (n / 1e12).toFixed(1) + " 万亿"
	if (n >= 1e8) return (n / 1e8).toFixed(1) + " 亿"
	return n.toLocaleString("en-US")
}

const resetLab = useLabReset(id)
</script>

<template>
	<LearningLab topic="ShortLinkBase62" @reset="resetLab">
		<p class="infra-lab__hint">发号器给每条长链一个自增 ID，再把 ID 转成 62 进制（0-9a-zA-Z）得到短码。观察发号量增长时码长如何变化。</p>
		<div class="infra-tabs" role="group" aria-label="发号量预设">
			<button v-for="p in presets" :key="p.v" type="button" :aria-pressed="id === p.v" @click="id = p.v">{{ p.label }}</button>
		</div>
		<div class="infra-control"><label :for="fieldId('id')">自增 ID：{{ id.toLocaleString("en-US") }}</label><input :id="fieldId('id')" v-model.number="id" type="range" min="1" max="1000000000" step="1"></div>
		<div class="sl-code">
			<span class="sl-domain">sho.rt/</span><strong>{{ code }}</strong>
		</div>
		<div class="infra-results">
			<div class="infra-result"><span>短码</span><strong>{{ code }}</strong></div>
			<div class="infra-result"><span>码长</span><strong>{{ length }} 位</strong></div>
		</div>
		<table class="sl-cap">
			<thead><tr><th>码长</th><th>容量 62^n</th></tr></thead>
			<tbody><tr v-for="row in capacityTable" :key="row.len"><td>{{ row.len }} 位</td><td>{{ fmtCap(row.cap) }}</td></tr></tbody>
		</table>
		<p class="infra-note">发号器 + 62 进制天然无冲突、短码随发号量单调增长；相比「哈希长链取前几位」省去了冲突处理，但连续可预测，需配合风控与不可枚举化（如加扰）。</p>
	</LearningLab>
</template>

<style scoped>
.sl-code { margin: .8rem 0; padding: .7rem; background: var(--vp-c-bg-soft); border-radius: 6px; font-family: var(--vp-font-family-mono); font-size: 1.1rem; }
.sl-domain { color: var(--vp-c-text-3); }
.sl-code strong { color: var(--vp-c-brand-1); }
.sl-cap { width: auto; margin: .6rem 0 0; font-size: .82rem; }
.sl-cap th, .sl-cap td { padding: .25rem .8rem .25rem 0; text-align: left; }
.sl-cap th { color: var(--vp-c-text-3); font-weight: 500; }
</style>
