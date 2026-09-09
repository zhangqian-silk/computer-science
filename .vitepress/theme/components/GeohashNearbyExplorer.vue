<script setup lang="ts">
import { computed, ref, useId } from "vue"
import LearningLab from "./LearningLab.vue"
import { useLabReset } from "../use-lab-reset"

const labId = useId()
const fieldId = (name: string) => `${labId}-${name}`

const precision = ref(6)

// geohash 精度 -> 近似单元格尺寸（经验值）
const cellSize: Record<number, string> = {
	4: "≈ 39 km × 20 km",
	5: "≈ 5 km × 5 km",
	6: "≈ 1.2 km × 0.6 km",
	7: "≈ 153 m × 153 m",
	8: "≈ 38 m × 19 m"
}

const grid = computed(() => {
	const cells: { r: number; c: number; kind: string }[] = []
	for (let r = 0; r < 5; r++) {
		for (let c = 0; c < 5; c++) {
			let kind = "far"
			if (r === 2 && c === 2) kind = "center"
			else if (Math.abs(r - 2) <= 1 && Math.abs(c - 2) <= 1) kind = "neighbor"
			cells.push({ r, c, kind })
		}
	}
	return cells
})

const resetLab = useLabReset(precision)
</script>

<template>
	<LearningLab topic="GeohashNearbyExplorer" @reset="resetLab">
		<p class="infra-lab__hint">Geohash 把经纬度编码成字符串，前缀越长单元格越小。附近查询要扫描「中心格 + 周围 8 格」共 9 格，避免目标恰好落在格子边界被漏掉。</p>
		<div class="infra-control"><label :for="fieldId('p')">Geohash 精度（前缀长度）：{{ precision }}（{{ cellSize[precision] }}）</label><input :id="fieldId('p')" v-model.number="precision" type="range" min="4" max="8"></div>
		<div class="gh-grid" role="img" aria-label="Geohash 九宫格附近查询">
			<div v-for="(cell, i) in grid" :key="i" class="gh-cell" :class="`gh-${cell.kind}`">
				<span v-if="cell.kind === 'center'">★</span>
			</div>
		</div>
		<div class="infra-results">
			<div class="infra-result"><span>单元格尺寸</span><strong>{{ cellSize[precision] }}</strong></div>
			<div class="infra-result"><span>附近查询扫描</span><strong>9 个格子</strong></div>
		</div>
		<p class="infra-note">精度高：定位精准但覆盖范围小、可能要更多格子拼出查询半径；精度低：一格覆盖大、扫描少但粒度粗。查询半径与精度要匹配，并始终查九宫格以处理边界。</p>
	</LearningLab>
</template>

<style scoped>
.gh-grid { display: grid; grid-template-columns: repeat(5, 1fr); gap: 3px; max-width: 260px; margin: .8rem 0; }
.gh-cell { aspect-ratio: 1; display: flex; align-items: center; justify-content: center; border-radius: 4px; background: var(--vp-c-bg-soft); font-size: 1rem; }
.gh-neighbor { background: var(--vp-c-brand-soft); }
.gh-center { background: var(--vp-c-brand-1); color: #fff; }
</style>
