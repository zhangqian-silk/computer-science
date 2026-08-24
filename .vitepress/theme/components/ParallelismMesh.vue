<script setup lang="ts">
import { computed, ref } from 'vue'

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
</script>

<template>
	<div class="infra-lab">
		<p class="infra-lab__title">并行 DeviceMesh</p>
		<p class="infra-lab__hint">每个 rank 同时属于 DP、TP、PP 与 EP 四个逻辑坐标。</p>
		<div class="infra-controls">
			<div class="infra-control"><label for="mesh-dp">DP</label><select id="mesh-dp" v-model.number="dp"><option :value="1">1</option><option :value="2">2</option><option :value="4">4</option></select></div>
			<div class="infra-control"><label for="mesh-tp">TP</label><select id="mesh-tp" v-model.number="tp"><option :value="1">1</option><option :value="2">2</option><option :value="4">4</option></select></div>
			<div class="infra-control"><label for="mesh-pp">PP</label><select id="mesh-pp" v-model.number="pp"><option :value="1">1</option><option :value="2">2</option><option :value="4">4</option></select></div>
			<div class="infra-control"><label for="mesh-ep">EP</label><select id="mesh-ep" v-model.number="ep"><option :value="1">1</option><option :value="2">2</option><option :value="4">4</option></select></div>
		</div>
		<div class="infra-results"><div class="infra-result"><span>World Size</span><strong>{{ world }}</strong></div></div>
		<div class="mesh-grid">
			<div v-for="cell in cells" :key="cell.rank" class="mesh-cell"><strong>rank {{ cell.rank }}</strong><span>{{ cell.label }}</span></div>
		</div>
		<p class="infra-note">坐标只表达逻辑分组；实际 rank 到 GPU/NVLink/NIC 的映射仍需结合物理拓扑。</p>
	</div>
</template>

<style scoped>
.mesh-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(92px, 1fr)); gap: .45rem; margin-top: .9rem; }
.mesh-cell { padding: .5rem; border: 1px solid var(--vp-c-divider); border-radius: 7px; background: var(--vp-c-bg); text-align: center; }
.mesh-cell strong, .mesh-cell span { display: block; font-size: .78rem; }
.mesh-cell span { color: var(--vp-c-text-2); }
</style>
