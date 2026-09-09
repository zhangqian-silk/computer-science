<script setup lang="ts">
import { computed, ref } from "vue"
import LearningLab from "./LearningLab.vue"
import { useLabReset } from "../use-lab-reset"

const producerRetry = ref(true)   // 生产端确认 + 重试
const persistence = ref(true)     // Broker 持久化 + 多副本
const ackAfter = ref(true)        // 处理完成后再 ack
const idempotent = ref(true)      // 消费端幂等

const canLose = computed(() =>
	!producerRetry.value || !persistence.value || !ackAfter.value
)
// 重试 / 重投都会带来重复投递
const rawDuplicate = computed(() => producerRetry.value || ackAfter.value)
const netDuplicate = computed(() => rawDuplicate.value && !idempotent.value)

const semantic = computed(() => {
	if (canLose.value && !netDuplicate.value) return "At most once（至多一次）"
	if (!canLose.value && netDuplicate.value) return "At least once（至少一次）"
	if (!canLose.value && !netDuplicate.value) return "Exactly once 等价（至少一次 + 幂等）"
	return "既可能丢也可能重（配置不健全）"
})

const lossReasons = computed(() => {
	const r: string[] = []
	if (!producerRetry.value) r.push("生产发送失败未重试")
	if (!persistence.value) r.push("Broker 宕机 / 未刷盘")
	if (!ackAfter.value) r.push("处理前 ack，处理失败即丢")
	return r
})

const resetLab = useLabReset(producerRetry, persistence, ackAfter, idempotent)
</script>

<template>
	<LearningLab topic="DeliverySemanticsExplorer" @reset="resetLab">
		<p class="infra-lab__hint">一条消息经历「生产 → Broker 存储 → 消费」。开关每段的可靠性手段，观察最终投递语义。</p>
		<div class="sem-toggles">
			<label><input type="checkbox" v-model="producerRetry"> 生产端确认 + 重试</label>
			<label><input type="checkbox" v-model="persistence"> Broker 持久化 + 多副本</label>
			<label><input type="checkbox" v-model="ackAfter"> 处理完成后再 ack</label>
			<label><input type="checkbox" v-model="idempotent"> 消费端幂等去重</label>
		</div>
		<div class="infra-results">
			<div class="infra-result"><span>是否可能丢失</span><strong>{{ canLose ? "可能丢失" : "不丢" }}</strong></div>
			<div class="infra-result"><span>是否产生重复副作用</span><strong>{{ netDuplicate ? "可能重复" : (rawDuplicate ? "有重投但被幂等吸收" : "不重") }}</strong></div>
			<div class="infra-result sem-final"><span>投递语义</span><strong>{{ semantic }}</strong></div>
		</div>
		<p v-if="lossReasons.length" class="infra-note">丢失来源：{{ lossReasons.join("；") }}。</p>
		<p class="infra-note">工程主流：生产确认 + 持久化多副本 + 处理后 ack 得到「至少一次」，再叠加消费幂等等价实现「恰好一次」。</p>
	</LearningLab>
</template>

<style scoped>
.sem-toggles { display: grid; grid-template-columns: repeat(auto-fit, minmax(min(100%, 200px), 1fr)); gap: .5rem; margin: .8rem 0; }
.sem-toggles label { display: flex; align-items: center; gap: .5rem; padding: .5rem .6rem; border: 1px solid var(--vp-c-divider); border-radius: 6px; background: var(--vp-c-bg); font-size: .84rem; }
.sem-final strong { color: var(--vp-c-brand-1); }
</style>
