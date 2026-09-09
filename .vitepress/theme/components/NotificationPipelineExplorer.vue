<script setup lang="ts">
import { computed, ref, useId } from "vue"
import LearningLab from "./LearningLab.vue"
import { useLabReset } from "../use-lab-reset"

const labId = useId()
const fieldId = (name: string) => `${labId}-${name}`

const requested = ref(100)      // 请求发送量
const freqLimited = ref(30)     // 触发频控被拦截
const dedup = ref(15)           // 去重命中（重复请求）
const channelFail = ref(10)     // 渠道发送失败（可重试）
const retrySuccess = ref(true)  // 是否开启失败重试

const pipeline = computed(() => {
	const afterDedup = Math.max(0, requested.value - dedup.value)
	const afterFreq = Math.max(0, afterDedup - freqLimited.value)
	const failed = Math.min(channelFail.value, afterFreq)
	const recovered = retrySuccess.value ? failed : 0
	const delivered = afterFreq - failed + recovered
	return {
		afterDedup, afterFreq,
		delivered,
		droppedFail: failed - recovered,
		stages: [
			{ name: "请求发送", n: requested.value, tone: "neutral" },
			{ name: "去重后", n: afterDedup, tone: "neutral" },
			{ name: "频控后", n: afterFreq, tone: "warning" },
			{ name: "最终送达", n: delivered, tone: "success" }
		]
	}
})

const resetLab = useLabReset(requested, freqLimited, dedup, channelFail, retrySuccess)
</script>

<template>
	<LearningLab topic="NotificationPipelineExplorer" @reset="resetLab">
		<p class="infra-lab__hint">通知从「请求」到「送达」要穿过去重、频控、渠道发送与重试。调整各环节的拦截量，观察最终送达数。</p>
		<div class="infra-controls">
			<div class="infra-control"><label :for="fieldId('req')">请求量：{{ requested }}</label><input :id="fieldId('req')" v-model.number="requested" type="range" min="20" max="200" step="10"></div>
			<div class="infra-control"><label :for="fieldId('dd')">去重命中：{{ dedup }}</label><input :id="fieldId('dd')" v-model.number="dedup" type="range" min="0" max="60"></div>
			<div class="infra-control"><label :for="fieldId('fc')">频控拦截：{{ freqLimited }}</label><input :id="fieldId('fc')" v-model.number="freqLimited" type="range" min="0" max="80"></div>
			<div class="infra-control"><label :for="fieldId('cf')">渠道失败：{{ channelFail }}</label><input :id="fieldId('cf')" v-model.number="channelFail" type="range" min="0" max="40"></div>
		</div>
		<label class="np-toggle"><input type="checkbox" v-model="retrySuccess"> 失败重试（可恢复）</label>
		<div class="np-funnel">
			<div v-for="s in pipeline.stages" :key="s.name" class="np-stage" :class="`tone-${s.tone}`">
				<span class="np-n">{{ s.n }}</span><span class="np-name">{{ s.name }}</span>
			</div>
		</div>
		<div class="infra-results">
			<div class="infra-result"><span>最终送达</span><strong>{{ pipeline.delivered }}</strong></div>
			<div class="infra-result"><span>重试后仍失败</span><strong>{{ pipeline.droppedFail }}</strong></div>
			<div class="infra-result"><span>送达率</span><strong>{{ (pipeline.delivered / requested * 100).toFixed(0) }}%</strong></div>
		</div>
		<p class="infra-note">去重防重复打扰、频控防疲劳轰炸、重试提高送达。这四层缺一不可：没有去重与频控，营销触达会造成用户投诉与资损；没有重试与死信，渠道抖动会漏发关键通知。</p>
	</LearningLab>
</template>

<style scoped>
.np-toggle { display: flex; align-items: center; gap: .5rem; margin: .6rem 0; font-size: .85rem; color: var(--vp-c-text-2); }
.np-funnel { display: flex; flex-direction: column; gap: .3rem; margin: .6rem 0; }
.np-stage { display: flex; align-items: center; gap: .8rem; padding: .5rem .8rem; border-radius: 6px; border-left: 3px solid var(--vp-c-divider); background: var(--vp-c-bg-soft); }
.np-n { font-weight: 700; font-variant-numeric: tabular-nums; min-width: 42px; }
.np-name { color: var(--vp-c-text-2); font-size: .84rem; }
.tone-warning { border-left-color: var(--vp-c-warning-1); }
.tone-success { border-left-color: var(--vp-c-green-1); }
</style>
