<script setup lang="ts">
import { computed, ref, useId } from "vue"
import LearningLab from "./LearningLab.vue"
import { useLabReset } from "../use-lab-reset"

const labId = useId()
const fieldId = (name: string) => `${labId}-${name}`

const slots = ref(8)   // 时间轮槽位数（每格 1 秒）
const delay = ref(20)  // 任务延迟秒数

const targetSlot = computed(() => delay.value % slots.value)
const rounds = computed(() => Math.floor(delay.value / slots.value))

const dots = computed(() =>
	Array.from({ length: slots.value }, (_, i) => {
		const a = (i / slots.value) * 2 * Math.PI - Math.PI / 2
		return {
			i,
			x: 100 + 70 * Math.cos(a),
			y: 100 + 70 * Math.sin(a),
			isTarget: i === targetSlot.value,
			isCursor: i === 0
		}
	})
)

const resetLab = useLabReset(slots, delay)
</script>

<template>
	<LearningLab topic="TimingWheelExplorer" @reset="resetLab">
		<p class="infra-lab__hint">时间轮把「延迟任务」按槽位存放，指针每秒走一格（当前指向槽 0）。每格 1 秒，共 {{ slots }} 格。</p>
		<div class="infra-controls">
			<div class="infra-control"><label :for="fieldId('slots')">槽位数：{{ slots }}</label><input :id="fieldId('slots')" v-model.number="slots" type="range" min="4" max="12"></div>
			<div class="infra-control"><label :for="fieldId('delay')">任务延迟：{{ delay }}s</label><input :id="fieldId('delay')" v-model.number="delay" type="range" min="1" max="60"></div>
		</div>
		<div class="tw-body">
			<svg viewBox="0 0 200 200" class="tw-ring" role="img" aria-label="时间轮">
				<circle cx="100" cy="100" r="70" fill="none" stroke="var(--vp-c-divider)" stroke-width="1.5" />
				<g v-for="d in dots" :key="d.i">
					<circle :cx="d.x" :cy="d.y" r="12"
						:fill="d.isTarget ? 'var(--vp-c-brand-1)' : 'var(--vp-c-bg-soft)'"
						:stroke="d.isCursor ? 'var(--vp-c-green-1)' : 'var(--vp-c-divider)'"
						:stroke-width="d.isCursor ? 3 : 1" />
					<text :x="d.x" :y="d.y + 4" text-anchor="middle" font-size="10"
						:fill="d.isTarget ? '#fff' : 'var(--vp-c-text-2)'">{{ d.i }}</text>
				</g>
				<text x="100" y="104" text-anchor="middle" font-size="11" fill="var(--vp-c-text-3)">指针→槽0</text>
			</svg>
			<div class="tw-info">
				<div class="infra-result"><span>落入槽位</span><strong>槽 {{ targetSlot }}</strong></div>
				<div class="infra-result"><span>需转圈数 rounds</span><strong>{{ rounds }} 圈</strong></div>
			</div>
		</div>
		<p class="infra-note">
			延迟 {{ delay }}s 的任务放入槽 {{ targetSlot }}，附带 rounds={{ rounds }}。指针每经过该槽就把 rounds 减 1，减到 0 才触发。
			单层时间轮用 rounds 承载「超过一圈」的延迟；多层时间轮（时/分/秒）则用进位降低空转，是 Kafka 延迟队列的做法。
		</p>
	</LearningLab>
</template>

<style scoped>
.tw-body { display: flex; gap: 1rem; flex-wrap: wrap; align-items: center; margin-top: .6rem; }
.tw-ring { width: 200px; height: 200px; flex: 0 0 auto; }
.tw-info { flex: 1 1 200px; display: flex; flex-direction: column; gap: .4rem; }
</style>
