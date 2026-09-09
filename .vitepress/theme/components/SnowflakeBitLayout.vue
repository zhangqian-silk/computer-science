<script setup lang="ts">
import { computed, ref, useId } from "vue"
import LearningLab from "./LearningLab.vue"
import { useLabReset } from "../use-lab-reset"

const labId = useId()
const fieldId = (name: string) => `${labId}-${name}`

const dcBits = ref(5)
const workerBits = ref(5)
const seqBits = ref(12)

const tsBits = computed(() => 63 - dcBits.value - workerBits.value - seqBits.value)

const MS_PER_YEAR = 1000 * 60 * 60 * 24 * 365.25

const metrics = computed(() => {
	const years = (2 ** tsBits.value) / MS_PER_YEAR
	const machines = 2 ** (dcBits.value + workerBits.value)
	const perMsPerMachine = 2 ** seqBits.value
	return {
		years,
		machines,
		perMsPerMachine,
		qpsPerMachine: perMsPerMachine * 1000
	}
})

const segments = computed(() => [
	{ name: "符号位", bits: 1, color: "#94a3b8" },
	{ name: "时间戳", bits: tsBits.value, color: "#3b82f6" },
	{ name: "机房", bits: dcBits.value, color: "#10b981" },
	{ name: "机器", bits: workerBits.value, color: "#f59e0b" },
	{ name: "序列号", bits: seqBits.value, color: "#8b5cf6" }
])

const valid = computed(() => tsBits.value >= 28)

function fmt(n: number): string {
	if (n >= 1e8) return (n / 1e8).toFixed(2) + " 亿"
	if (n >= 1e4) return (n / 1e4).toFixed(1) + " 万"
	return n.toLocaleString("en-US")
}

const resetLab = useLabReset(dcBits, workerBits, seqBits)
</script>

<template>
	<LearningLab topic="SnowflakeBitLayout" @reset="resetLab">
		<p class="infra-lab__hint">64 位 = 1 符号位 + 时间戳 + 机房 + 机器 + 序列号。调整后三段，时间戳位自动占用剩余位，观察三者的此消彼长。</p>
		<div class="bit-bar" role="img" aria-label="Snowflake 位段布局">
			<div v-for="s in segments" :key="s.name" class="bit-seg" :style="{ flexGrow: s.bits, background: s.color }" :title="`${s.name} ${s.bits} 位`">
				<span v-if="s.bits >= 3">{{ s.name }}<br>{{ s.bits }}</span>
			</div>
		</div>
		<div class="infra-controls">
			<div class="infra-control"><label :for="fieldId('dc')">机房位：{{ dcBits }}</label><input :id="fieldId('dc')" v-model.number="dcBits" type="range" min="0" max="10"></div>
			<div class="infra-control"><label :for="fieldId('wk')">机器位：{{ workerBits }}</label><input :id="fieldId('wk')" v-model.number="workerBits" type="range" min="0" max="12"></div>
			<div class="infra-control"><label :for="fieldId('sq')">序列位：{{ seqBits }}</label><input :id="fieldId('sq')" v-model.number="seqBits" type="range" min="4" max="22"></div>
		</div>
		<div class="infra-results">
			<div class="infra-result"><span>时间戳位</span><strong :style="{ color: valid ? undefined : 'var(--vp-c-danger-1)' }">{{ tsBits }} 位（约 {{ metrics.years.toFixed(0) }} 年）</strong></div>
			<div class="infra-result"><span>可部署机器数</span><strong>{{ fmt(metrics.machines) }}</strong></div>
			<div class="infra-result"><span>单机每毫秒上限</span><strong>{{ fmt(metrics.perMsPerMachine) }}</strong></div>
			<div class="infra-result"><span>单机理论 QPS</span><strong>{{ fmt(metrics.qpsPerMachine) }}</strong></div>
		</div>
		<p class="infra-note" :class="{ 'is-warn': !valid }">
			{{ valid
				? "经典分配 41/10/12：约 69 年、1024 台机器、单机每毫秒 4096 个 ID。位段是一次容量权衡：机器多则序列少、时间短。"
				: "时间戳位过少，可用年限太短，实际不可用——把机器/序列位调小。" }}
		</p>
	</LearningLab>
</template>

<style scoped>
.bit-bar { display: flex; width: 100%; height: 54px; border-radius: 6px; overflow: hidden; margin: .6rem 0; }
.bit-seg { display: flex; align-items: center; justify-content: center; min-width: 2px; color: #fff; font-size: .68rem; line-height: 1.15; text-align: center; }
.infra-note.is-warn { color: var(--vp-c-danger-1); }
</style>
