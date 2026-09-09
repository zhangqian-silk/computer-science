<script setup lang="ts">
import { computed, ref, useId } from "vue"
import LearningLab from "./LearningLab.vue"
import { useLabReset } from "../use-lab-reset"

const labId = useId()
const fieldId = (name: string) => `${labId}-${name}`

const layers = ref(3)      // 调用链层数
const retries = ref(2)     // 每层重试次数
const protectedMode = ref(false) // 是否开启退避 + 熔断 + 重试预算
const BASE = 100

const naiveBottom = computed(() => BASE * Math.pow(1 + retries.value, layers.value))
// 保护模式：熔断快速失败 + 重试预算把放大限制在很小范围（示意为最多 1.5 倍）
const protectedBottom = computed(() => Math.round(BASE * 1.5))
const bottom = computed(() => protectedMode.value ? protectedBottom.value : naiveBottom.value)
const amplification = computed(() => (bottom.value / BASE).toFixed(1))

function fmt(n: number): string {
	if (n >= 10000) return (n / 10000).toFixed(1) + " 万"
	return n.toFixed(0)
}

const resetLab = useLabReset(layers, retries, protectedMode)
</script>

<template>
	<LearningLab topic="RetryStormExplorer" @reset="resetLab">
		<p class="infra-lab__hint">{{ BASE }} 个入口请求经过多层调用，每层失败都重试。观察底层依赖被放大的请求量——这就是「重试风暴」。</p>
		<div class="infra-controls">
			<div class="infra-control"><label :for="fieldId('l')">调用链层数：{{ layers }}</label><input :id="fieldId('l')" v-model.number="layers" type="range" min="1" max="4"></div>
			<div class="infra-control"><label :for="fieldId('r')">每层重试次数：{{ retries }}</label><input :id="fieldId('r')" v-model.number="retries" type="range" min="0" max="3"></div>
		</div>
		<label class="rs-toggle"><input type="checkbox" v-model="protectedMode"> 开启退避 + 熔断 + 重试预算</label>
		<div class="infra-results">
			<div class="infra-result"><span>入口请求</span><strong>{{ BASE }}</strong></div>
			<div class="infra-result" :class="{ 'rs-bad': amplification > '3' && !protectedMode }"><span>打到底层依赖</span><strong>{{ fmt(bottom) }}</strong></div>
			<div class="infra-result"><span>放大倍数</span><strong>{{ amplification }}×</strong></div>
		</div>
		<p class="infra-note">无保护时放大是 (1+重试)^层数 的指数增长——底层故障时重试反而把它彻底压垮。对策：仅对幂等操作重试、指数退避 + 抖动、熔断快速失败、「重试预算」限制整体重试比例、只在最外层重试。</p>
	</LearningLab>
</template>

<style scoped>
.rs-toggle { display: flex; align-items: center; gap: .5rem; margin: .6rem 0; font-size: .85rem; color: var(--vp-c-text-2); }
.rs-bad strong { color: var(--vp-c-danger-1); }
</style>
