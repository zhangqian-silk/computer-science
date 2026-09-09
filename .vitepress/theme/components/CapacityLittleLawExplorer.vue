<script setup lang="ts">
import { computed, ref, useId } from "vue"
import LearningLab from "./LearningLab.vue"
import { useLabReset } from "../use-lab-reset"

const labId = useId()
const fieldId = (name: string) => `${labId}-${name}`

const qps = ref(2000)          // 到达率 λ（请求/秒）
const latencyMs = ref(50)      // 平均处理耗时 W（毫秒）
const perInstance = ref(200)   // 单实例并发处理能力
const headroom = ref(50)       // 峰值余量 %

const concurrency = computed(() => qps.value * (latencyMs.value / 1000)) // L = λ × W
const withHeadroom = computed(() => concurrency.value * (1 + headroom.value / 100))
const instances = computed(() => Math.max(1, Math.ceil(withHeadroom.value / perInstance.value)))

const resetLab = useLabReset(qps, latencyMs, perInstance, headroom)
</script>

<template>
	<LearningLab topic="CapacityLittleLawExplorer" @reset="resetLab">
		<p class="infra-lab__hint">利特尔法则 L = λ × W：稳态下系统内平均并发请求数 = 到达率 × 平均停留时间。据此推算所需并发与实例数。</p>
		<div class="infra-controls">
			<div class="infra-control"><label :for="fieldId('qps')">到达率 λ：{{ qps }} QPS</label><input :id="fieldId('qps')" v-model.number="qps" type="range" min="100" max="20000" step="100"></div>
			<div class="infra-control"><label :for="fieldId('lat')">平均耗时 W：{{ latencyMs }} ms</label><input :id="fieldId('lat')" v-model.number="latencyMs" type="range" min="5" max="500" step="5"></div>
			<div class="infra-control"><label :for="fieldId('per')">单实例并发：{{ perInstance }}</label><input :id="fieldId('per')" v-model.number="perInstance" type="range" min="20" max="1000" step="10"></div>
			<div class="infra-control"><label :for="fieldId('hr')">峰值余量：{{ headroom }}%</label><input :id="fieldId('hr')" v-model.number="headroom" type="range" min="0" max="200" step="10"></div>
		</div>
		<div class="infra-results">
			<div class="infra-result"><span>平均并发 L</span><strong>{{ concurrency.toFixed(0) }}</strong></div>
			<div class="infra-result"><span>含余量并发</span><strong>{{ withHeadroom.toFixed(0) }}</strong></div>
			<div class="infra-result"><span>所需实例数</span><strong>{{ instances }}</strong></div>
		</div>
		<p class="infra-note">降低单请求耗时 W 与提高单实例并发都能减少实例数。容量规划要按「峰值 + 余量」而非均值，并用压测校准真实的单实例并发上限（受 CPU、锁、下游依赖制约）。</p>
	</LearningLab>
</template>
