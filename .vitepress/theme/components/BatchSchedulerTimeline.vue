<script setup lang="ts">
import { computed, ref, useId } from "vue"
import LearningLab from "./LearningLab.vue"
import { useLabReset } from "../use-lab-reset"
import { scheduleToy } from "../learning-algorithms"
const labId = useId()
const fieldId = (name: string) => `${labId}-${name}`

const mode = ref<'static' | 'continuous'>('continuous')
const shortLength = ref(2)
const schedule = computed(() => scheduleToy(mode.value, shortLength.value))
const rows = computed(() => schedule.value.jobs.map((job, i) => [job.id, ...schedule.value.rows[i]]))
const resetLab = useLabReset(mode, shortLength)
</script>

<template>
	<LearningLab topic="BatchSchedulerTimeline" @reset="resetLab">
		<p class="infra-lab__hint">A/B/C 在第 0 轮到达，新请求 D 在第 1 轮到达；并发槽位为 3。P 为 Prefill，D 为 Decode，列是调度轮次，不是等长毫秒。</p>
		<div class="infra-control"><label :for="fieldId('batch-mode')">策略</label><select :id="fieldId('batch-mode')" v-model="mode"><option value="static">Static Batch</option><option value="continuous">Continuous Batch</option></select></div>
		<div class="infra-control"><label :for="fieldId('short-length')">B 所需轮数（含 Prefill）：{{ shortLength }}</label><input :id="fieldId('short-length')" v-model.number="shortLength" type="range" min="1" max="6"></div>
		<div class="timeline" :style="{ '--rounds': rows[0].length - 1 }">
			<div v-for="row in rows" :key="row[0]" class="timeline-row">
				<strong>{{ row[0] }}</strong><span v-for="(cell, index) in row.slice(1)" :key="index" :class="`cell cell-${cell}`">{{ cell }}</span>
			</div>
		</div>
		<div class="infra-result"><span>D 首次执行轮次（从 0 开始）</span><strong>{{ schedule.started[3] }}，等待 {{ schedule.started[3] - 1 }} 轮</strong></div>
		<p class="infra-note">轨迹按剩余工作与准入规则重新计算。静态批次空位不能立即接收新人；真实调度还受 token/KV 预算及长 Prefill 干扰约束。</p>
	</LearningLab>
</template>

<style scoped>
.timeline { margin-top: .9rem; overflow-x: auto; }
.timeline-row { display: grid; grid-template-columns: 60px repeat(var(--rounds), minmax(36px, 1fr)); gap: .3rem; margin-bottom: .3rem; align-items: center; }
.timeline-row strong { font-size: .78rem; }
.cell { padding: .35rem; border-radius: 5px; text-align: center; background: var(--vp-c-default-soft); }
.cell-P { background: var(--vp-c-brand-soft); color: var(--vp-c-brand-1); }
.cell-D { background: var(--vp-c-green-soft); color: var(--vp-c-green-1); }
.cell-等 { color: var(--vp-c-text-3); }
</style>
