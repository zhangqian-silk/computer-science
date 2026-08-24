<script setup lang="ts">
import { computed, ref } from 'vue'

const base = [-1, 0, 1, 2]
const offset = ref(0)
const scale = ref(1)
const input = computed(() => base.map(value => value * scale.value + offset.value))
const mean = computed(() => input.value.reduce((sum, value) => sum + value, 0) / input.value.length)
const variance = computed(() => input.value.reduce((sum, value) => sum + (value - mean.value) ** 2, 0) / input.value.length)
const rms = computed(() => Math.sqrt(input.value.reduce((sum, value) => sum + value ** 2, 0) / input.value.length))
const layerNorm = computed(() => input.value.map(value => (value - mean.value) / Math.sqrt(variance.value + 1e-5)))
const rmsNorm = computed(() => input.value.map(value => value / Math.sqrt(rms.value ** 2 + 1e-5)))
const format = (values: number[]) => values.map(value => value.toFixed(2)).join(', ')
</script>

<template>
	<div class="infra-lab">
		<p class="infra-lab__title">归一化对比实验台</p>
		<p class="infra-lab__hint">对同一隐藏向量施加整体平移和缩放，比较 LayerNorm 与 RMSNorm 对输入变化的响应。</p>
		<div class="infra-controls">
			<div class="infra-control">
				<label for="normalization-offset">整体平移：{{ offset.toFixed(1) }}</label>
				<input id="normalization-offset" v-model.number="offset" type="range" min="-3" max="3" step="0.1">
			</div>
			<div class="infra-control">
				<label for="normalization-scale">整体缩放：{{ scale.toFixed(1) }}</label>
				<input id="normalization-scale" v-model.number="scale" type="range" min="0.2" max="3" step="0.1">
			</div>
		</div>
		<div class="normalization-grid" aria-live="polite">
			<span>输入</span><code>[{{ format(input) }}]</code>
			<span>LayerNorm</span><code>[{{ format(layerNorm) }}]</code>
			<span>RMSNorm</span><code>[{{ format(rmsNorm) }}]</code>
		</div>
		<div class="infra-results">
			<div class="infra-result"><span>输入均值</span><strong>{{ mean.toFixed(3) }}</strong></div>
			<div class="infra-result"><span>输入 RMS</span><strong>{{ rms.toFixed(3) }}</strong></div>
		</div>
		<p class="infra-note">为突出核心差异，这里省略可学习的缩放和偏置。LayerNorm 同时去均值和缩放，RMSNorm 只按均方根缩放，因此对整体平移并不不变。</p>
	</div>
</template>

<style scoped>
.normalization-grid { display: grid; grid-template-columns: minmax(90px, auto) minmax(0, 1fr); gap: .55rem .75rem; align-items: center; margin-top: .9rem; }
.normalization-grid span { color: var(--vp-c-text-2); font-size: .82rem; font-weight: 600; }
.normalization-grid code { overflow-x: auto; padding: .5rem .65rem; border: 1px solid var(--vp-c-divider); border-radius: 7px; background: var(--vp-c-bg); white-space: nowrap; }
@media (max-width: 520px) { .normalization-grid { grid-template-columns: 1fr; } }
</style>
