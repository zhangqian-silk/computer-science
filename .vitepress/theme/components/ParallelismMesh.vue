<script setup lang="ts">
import { computed, ref, useId } from "vue"
import LearningLab from "./LearningLab.vue"
import { useLabReset } from "../use-lab-reset"
const labId = useId()
const fieldId = (name: string) => `${labId}-${name}`

const dp = ref(2)
const tp = ref(2)
const pp = ref(1)
const ep = ref(1)
const world = computed(() => dp.value * tp.value * pp.value * ep.value)
const cells = computed(() => Array.from({ length: world.value }, (_, rank) => {
	const tpRank = rank % tp.value
	const epRank = Math.floor(rank / tp.value) % ep.value
	const ppRank = Math.floor(rank / (tp.value * ep.value)) % pp.value
	const dpRank = Math.floor(rank / (tp.value * ep.value * pp.value))
	return { rank, label: `d${dpRank}·p${ppRank}·e${epRank}·t${tpRank}` }
}))
const resetLab = useLabReset(dp, tp, pp, ep)
</script>

<template>
	<LearningLab topic="ParallelismMesh" @reset="resetLab">
		<p class="infra-lab__hint">本教学网格把 DP、TP、PP、EP 定义为四个独立轴；不是所有训练框架都采用这一约定。</p>
		<div class="infra-controls">
			<div class="infra-control"><label :for="fieldId('mesh-dp')">DP</label><select :id="fieldId('mesh-dp')" v-model.number="dp"><option :value="1">1</option><option :value="2">2</option><option :value="4">4</option></select></div>
			<div class="infra-control"><label :for="fieldId('mesh-tp')">TP</label><select :id="fieldId('mesh-tp')" v-model.number="tp"><option :value="1">1</option><option :value="2">2</option><option :value="4">4</option></select></div>
			<div class="infra-control"><label :for="fieldId('mesh-pp')">PP</label><select :id="fieldId('mesh-pp')" v-model.number="pp"><option :value="1">1</option><option :value="2">2</option><option :value="4">4</option></select></div>
			<div class="infra-control"><label :for="fieldId('mesh-ep')">EP</label><select :id="fieldId('mesh-ep')" v-model.number="ep"><option :value="1">1</option><option :value="2">2</option><option :value="4">4</option></select></div>
		</div>
		<div class="infra-results"><div class="infra-result"><span>World Size</span><strong>{{ world }}</strong></div></div>
		<div class="mesh-grid">
			<div v-for="cell in cells" :key="cell.rank" class="mesh-cell"><strong>rank {{ cell.rank }}</strong><span>{{ cell.label }}</span></div>
		</div>
		<p class="infra-note">真实框架中 EP 可能在既有 DP 组内划分，此时不能重复乘设备数。坐标也不等于物理 GPU/NVLink/NIC 位置。</p>
	</LearningLab>
</template>

<style scoped>
.mesh-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(92px, 1fr)); gap: .45rem; margin-top: .9rem; }
.mesh-cell { padding: .5rem; border: 1px solid var(--vp-c-divider); border-radius: 7px; background: var(--vp-c-bg); text-align: center; }
.mesh-cell strong, .mesh-cell span { display: block; font-size: .78rem; }
.mesh-cell span { color: var(--vp-c-text-2); }
</style>
