<script setup lang="ts">
import { computed, ref } from "vue"
import LearningLab from "./LearningLab.vue"
import { useLabReset } from "../use-lab-reset"

type State = "创建" | "支付中" | "已支付" | "支付失败" | "退款中" | "已退款"
const transitions: Record<State, Partial<Record<string, State>>> = {
	创建: { 发起支付: "支付中" },
	支付中: { 支付成功: "已支付", 支付失败: "支付失败" },
	已支付: { 申请退款: "退款中" },
	支付失败: { 发起支付: "支付中" },
	退款中: { 退款成功: "已退款" },
	已退款: {}
}
const events = ["发起支付", "支付成功", "支付失败", "申请退款", "退款成功"]

const state = ref<State>("创建")
const log = ref<{ ev: string; ok: boolean; msg: string }[]>([])

function fire(ev: string) {
	const next = transitions[state.value][ev]
	if (next) {
		log.value.unshift({ ev, ok: true, msg: `${state.value} → ${next}` })
		state.value = next
	} else {
		log.value.unshift({ ev, ok: false, msg: `在「${state.value}」状态非法/重复，幂等忽略` })
	}
	if (log.value.length > 6) log.value.pop()
}
function reset() { state.value = "创建"; log.value = [] }

const allStates: State[] = ["创建", "支付中", "已支付", "支付失败", "退款中", "已退款"]
const validNow = computed(() => Object.keys(transitions[state.value]))
const resetLab = useLabReset(state, log)
</script>

<template>
	<LearningLab topic="PaymentStateMachineExplorer" @reset="() => { resetLab(); reset() }">
		<p class="infra-lab__hint">支付用状态机约束流转。点击事件：合法则流转，非法或重复（如已支付再收一次成功回调）则幂等忽略。</p>
		<div class="pm-states">
			<span v-for="s in allStates" :key="s" class="pm-node" :class="{ 'pm-cur': s === state }">{{ s }}</span>
		</div>
		<div class="infra-actions">
			<button v-for="ev in events" :key="ev" type="button" :class="{ secondary: !validNow.includes(ev) }" @click="fire(ev)">{{ ev }}</button>
		</div>
		<div class="infra-result"><span>当前状态</span><strong>{{ state }}</strong></div>
		<ul class="pm-log">
			<li v-for="(l, i) in log" :key="i" :class="l.ok ? 'ok' : 'no'">{{ l.ev }}：{{ l.msg }}</li>
		</ul>
		<p class="infra-note">状态机 + 幂等是支付/订单的护城河：重复回调（渠道会重发）在终态被幂等忽略，非法跳转被拒绝，杜绝「重复扣款」「已退款又扣款」等资损。</p>
	</LearningLab>
</template>

<style scoped>
.pm-states { display: flex; flex-wrap: wrap; gap: .4rem; margin: .6rem 0; }
.pm-node { padding: .35rem .7rem; border: 1px solid var(--vp-c-divider); border-radius: 999px; font-size: .8rem; background: var(--vp-c-bg); }
.pm-node.pm-cur { background: var(--vp-c-brand-1); color: #fff; border-color: var(--vp-c-brand-1); }
.pm-log { list-style: none; padding: 0; margin: .6rem 0 0; font-size: .78rem; display: flex; flex-direction: column; gap: .2rem; }
.pm-log li.ok { color: var(--vp-c-green-1); }
.pm-log li.no { color: var(--vp-c-warning-1); }
</style>
