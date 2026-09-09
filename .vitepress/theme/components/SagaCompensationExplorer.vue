<script setup lang="ts">
import { computed, ref } from "vue"
import LearningLab from "./LearningLab.vue"
import { useLabReset } from "../use-lab-reset"

const steps = [
	{ t: "T1 创建订单", c: "C1 取消订单" },
	{ t: "T2 扣减库存", c: "C2 回补库存" },
	{ t: "T3 支付扣款", c: "C3 退款" },
	{ t: "T4 发货通知", c: "C4 撤回通知" }
]

const failStep = ref(3) // 1-based；0 表示全部成功

type Row = { forward: string; comp: string; state: "ok" | "fail" | "compensated" | "skip" }
const rows = computed<Row[]>(() =>
	steps.map((s, idx) => {
		const i = idx + 1
		if (failStep.value === 0) return { forward: s.t, comp: s.c, state: "ok" }
		if (i < failStep.value) return { forward: s.t, comp: s.c, state: "compensated" }
		if (i === failStep.value) return { forward: s.t, comp: s.c, state: "fail" }
		return { forward: s.t, comp: s.c, state: "skip" }
	})
)
const label: Record<Row["state"], string> = { ok: "已提交", fail: "失败", compensated: "已补偿回滚", skip: "未执行" }
const outcome = computed(() =>
	failStep.value === 0 ? "全部成功，事务完成" : `第 ${failStep.value} 步失败，已完成步骤按逆序补偿，系统回到一致态`
)

const resetLab = useLabReset(failStep)
</script>

<template>
	<LearningLab topic="SagaCompensationExplorer" @reset="resetLab">
		<p class="infra-lab__hint">Saga 把长事务拆成本地事务 T1..T4，每步配补偿 C。选择某步失败，观察哪些步骤会被逆序补偿。</p>
		<div class="infra-control">
			<label for="saga-fail">失败位置</label>
			<select id="saga-fail" v-model.number="failStep">
				<option :value="0">无失败（全部成功）</option>
				<option :value="1">T1 创建订单失败</option>
				<option :value="2">T2 扣减库存失败</option>
				<option :value="3">T3 支付扣款失败</option>
				<option :value="4">T4 发货通知失败</option>
			</select>
		</div>
		<ol class="saga-track" aria-label="Saga 步骤状态">
			<li v-for="(r, i) in rows" :key="i" :class="`s-${r.state}`">
				<strong>{{ r.forward }}</strong>
				<span class="saga-state">{{ label[r.state] }}</span>
				<em v-if="r.state === 'compensated'">↩ {{ r.comp }}</em>
			</li>
		</ol>
		<div class="infra-result"><span>结果</span><strong>{{ outcome }}</strong></div>
		<p class="infra-note">补偿必须幂等：某步「正向未真正生效」时补偿也要安全（与 TCC 空回滚同源）。发短信、已出库等不可补偿动作应放链路末尾或改为可对冲设计。</p>
	</LearningLab>
</template>

<style scoped>
.saga-track { list-style: none; padding: 0; margin: 1rem 0; display: flex; flex-direction: column; gap: .4rem; }
.saga-track li { display: flex; align-items: center; gap: .7rem; flex-wrap: wrap; padding: .5rem .6rem; border: 1px solid var(--vp-c-divider); border-left-width: 3px; border-radius: 6px; background: var(--vp-c-bg); }
.saga-track li strong { font-size: .85rem; font-weight: 500; }
.saga-state { font-size: .74rem; padding: .1rem .5rem; border-radius: 999px; background: var(--vp-c-default-soft); color: var(--vp-c-text-2); }
.saga-track li em { font-size: .78rem; color: var(--vp-c-warning-1); font-style: normal; }
.s-ok { border-left-color: var(--vp-c-green-1); }
.s-ok .saga-state { background: var(--vp-c-green-soft); color: var(--vp-c-green-1); }
.s-fail { border-left-color: var(--vp-c-danger-1); }
.s-fail .saga-state { background: var(--vp-c-danger-soft); color: var(--vp-c-danger-1); }
.s-compensated { border-left-color: var(--vp-c-warning-1); }
.s-skip { opacity: .5; }
</style>
