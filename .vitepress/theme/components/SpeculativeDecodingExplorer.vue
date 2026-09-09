<script setup lang="ts">
import { computed, ref, watch, useId } from "vue"
import LearningLab from "./LearningLab.vue"
import { useLabReset } from "../use-lab-reset"
const labId = useId()
const fieldId = (name: string) => `${labId}-${name}`

const proposals = ['海', '边', '的', '风', '很', '轻']
const draftCount = ref(4)
const accepted = ref(3)
watch(draftCount, count => { accepted.value = Math.min(accepted.value, count) })
const tokenStates = computed(() => proposals.slice(0, draftCount.value).map((token, index) => ({
	token,
	state: index < accepted.value ? 'accepted' : index === accepted.value ? 'rejected' : 'unverified'
})))
const advancedTokens = computed(() => accepted.value < draftCount.value ? accepted.value + 1 : draftCount.value + 1)
const savedTargetIterations = computed(() => Math.max(0, advancedTokens.value - 1))
const resetLab = useLabReset(draftCount, accepted)
</script>

<template>
	<LearningLab topic="SpeculativeDecodingExplorer" @reset="resetLab">
		<p class="infra-lab__hint">调整草稿长度与连续接受数，观察一次目标模型验证能把序列推进多少个 token。</p>
		<div class="infra-controls">
			<div class="infra-control">
				<label :for="fieldId('speculative-draft-count')">草稿 token 数：{{ draftCount }}</label>
				<input :id="fieldId('speculative-draft-count')" v-model.number="draftCount" type="range" min="1" max="6" step="1">
			</div>
			<div class="infra-control">
				<label :for="fieldId('speculative-accepted')">连续接受数：{{ accepted }}</label>
				<input :id="fieldId('speculative-accepted')" v-model.number="accepted" type="range" min="0" :max="draftCount" step="1">
			</div>
		</div>
		<div class="token-strip" aria-live="polite">
			<span v-for="item in tokenStates" :key="item.token" class="token-chip" :class="`is-${item.state}`">{{ item.token }} · {{ item.state === 'accepted' ? '接受' : item.state === 'rejected' ? '拒绝' : '未验证' }}</span>
		</div>
		<div class="infra-results">
			<div class="infra-result"><span>一次验证推进</span><strong>{{ advancedTokens }} token</strong></div>
			<div class="infra-result"><span>相对逐 token 少用的目标迭代</span><strong>{{ savedTargetIterations }}</strong></div>
			<div class="infra-result"><span>草稿接受比例</span><strong>{{ (accepted / draftCount * 100).toFixed(1) }}%</strong></div>
		</div>
		<p class="infra-note">若全部草稿被接受，目标模型还可给出一个额外 token；若中途拒绝，则由目标分布修正该位置。实际收益还取决于草稿成本、验证批次和接受率。</p>
	</LearningLab>
</template>

<style scoped>
.token-chip.is-accepted { border-color: var(--vp-c-green-1); background: var(--vp-c-green-soft); color: var(--vp-c-green-1); }
.token-chip.is-rejected { border-color: var(--vp-c-danger-1); background: var(--vp-c-danger-soft); color: var(--vp-c-danger-1); }
.token-chip.is-unverified { border-style: dashed; color: var(--vp-c-text-3); }
</style>
