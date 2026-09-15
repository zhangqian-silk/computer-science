<script setup lang="ts">
import { computed, ref } from "vue"
import { seriesColor } from "../series-palette"

/**
 * 超时预算的分层，以及重试如何在预算内排布。
 *
 * 组件要回答两个常被分开讨论、实际上必须一起算的问题：
 *   1. 一次工具调用的超时不是一个数，而是一条自外向内收窄的链；
 *   2. 重试次数不能独立设定——它受制于外层剩下多少预算。
 *
 * 拖动各层预算与退避参数，可以看到重试排布何时溢出外层预算：
 * 溢出意味着外层会在内层还在重试时就已经放弃，内层的重试全部白做。
 */

/** 外层：用户可接受的整体等待（秒） */
const userBudget = ref(60)
/** 本轮 Agent 循环分给这一次工具调用的预算（秒） */
const stepBudget = ref(30)
/** 单次尝试的上限（秒） */
const attemptTimeout = ref(8)

const baseDelay = ref(1)
const factor = ref(2)
const maxAttempts = ref(4)
const jitter = ref(true)

/** 退避排布：每次尝试占 attemptTimeout，其间隔按指数增长 */
const schedule = computed(() => {
	const rows: { no: number; wait: number; start: number; end: number }[] = []
	let t = 0
	for (let i = 0; i < maxAttempts.value; i++) {
		const raw = i === 0 ? 0 : baseDelay.value * Math.pow(factor.value, i - 1)
		// 抖动按「取 [0, raw] 内的随机值」建模，此处取其期望值以保持图示稳定
		const wait = i === 0 ? 0 : jitter.value ? raw / 2 : raw
		t += wait
		const start = t
		t += attemptTimeout.value
		rows.push({ no: i + 1, wait, start, end: t })
	}
	return rows
})

const total = computed(() => (schedule.value.length ? schedule.value[schedule.value.length - 1].end : 0))
const span = computed(() => Math.max(total.value, stepBudget.value, userBudget.value, 1))

/** 在 step 预算内真正跑得完的尝试数 */
const fitted = computed(() => schedule.value.filter((r) => r.end <= stepBudget.value).length)
const overflow = computed(() => total.value > stepBudget.value)
const layerBroken = computed(() => stepBudget.value > userBudget.value || attemptTimeout.value > stepBudget.value)

const layers = computed(() => [
	{ key: "user", name: "用户可接受等待", v: userBudget.value, idx: 0, note: "最外层。超过它，任务成功与否已不重要。" },
	{ key: "step", name: "本步工具预算", v: stepBudget.value, idx: 1, note: "Agent 循环分给这一次调用的额度，含全部重试。" },
	{ key: "attempt", name: "单次尝试上限", v: attemptTimeout.value, idx: 2, note: "最内层。它只约束一次尝试，不约束重试总量。" }
])

const fmt = (n: number) => (Number.isInteger(n) ? `${n}` : n.toFixed(1))
</script>

<template>
	<div class="trb">
		<div class="trb__controls">
			<label class="trb__ctl">
				<span>用户等待</span>
				<input v-model.number="userBudget" type="range" min="5" max="180" step="5" />
				<b>{{ userBudget }}s</b>
			</label>
			<label class="trb__ctl">
				<span>本步预算</span>
				<input v-model.number="stepBudget" type="range" min="5" max="180" step="5" />
				<b>{{ stepBudget }}s</b>
			</label>
			<label class="trb__ctl">
				<span>单次上限</span>
				<input v-model.number="attemptTimeout" type="range" min="1" max="60" step="1" />
				<b>{{ attemptTimeout }}s</b>
			</label>
			<label class="trb__ctl">
				<span>基础退避</span>
				<input v-model.number="baseDelay" type="range" min="0.5" max="10" step="0.5" />
				<b>{{ fmt(baseDelay) }}s</b>
			</label>
			<label class="trb__ctl">
				<span>退避倍数</span>
				<input v-model.number="factor" type="range" min="1" max="4" step="0.5" />
				<b>×{{ fmt(factor) }}</b>
			</label>
			<label class="trb__ctl">
				<span>最大尝试</span>
				<input v-model.number="maxAttempts" type="range" min="1" max="8" step="1" />
				<b>{{ maxAttempts }}</b>
			</label>
		</div>

		<label class="cs-toggle trb__jitter">
			<input v-model="jitter" type="checkbox" />
			<span>加入抖动（图示取其期望值，实际每次随机）</span>
		</label>

		<div class="trb__layers">
			<div v-for="l in layers" :key="l.key" class="trb__layer">
				<span class="trb__layer-name">{{ l.name }}</span>
				<div class="trb__track">
					<span class="trb__fill" :style="{ width: `${(l.v / span) * 100}%`, background: seriesColor(l.idx) }"></span>
				</div>
				<b class="cs-mono trb__layer-val">{{ l.v }}s</b>
			</div>
		</div>

		<div class="trb__timeline">
			<div
				v-for="r in schedule"
				:key="r.no"
				class="trb__attempt"
				:class="{ 'is-cut': r.end > stepBudget }"
				:style="{ left: `${(r.start / span) * 100}%`, width: `${(attemptTimeout / span) * 100}%` }"
				:title="`第 ${r.no} 次：${fmt(r.start)}s → ${fmt(r.end)}s`"
			>{{ r.no }}</div>
			<div class="trb__mark trb__mark--step" :style="{ left: `${(stepBudget / span) * 100}%` }"><i>本步预算</i></div>
		</div>

		<p class="trb__read" :class="{ 'is-bad': overflow || layerBroken }">
			<template v-if="layerBroken">
				<b>分层被打破</b>：内层预算大于外层。外层会先放弃，内层的等待与重试不会产生任何效果——
				这不是调参问题，是三条约束的包含关系搞反了。
			</template>
			<template v-else-if="overflow">
				配置的 <b>{{ maxAttempts }}</b> 次尝试共需 <b>{{ fmt(total) }}s</b>，超过本步预算 <b>{{ stepBudget }}s</b>；
				实际只跑得完 <b>{{ fitted }}</b> 次，其余的重试永远不会发生。
				<b>重试次数不是独立参数</b>，它由「预算 ÷ （单次上限 + 退避）」反推而来。
			</template>
			<template v-else>
				<b>{{ maxAttempts }}</b> 次尝试共需 <b>{{ fmt(total) }}s</b>，落在本步预算 <b>{{ stepBudget }}s</b> 之内，
				且本步预算不超过用户可接受的 <b>{{ userBudget }}s</b>。三层是包含关系，这一组取值成立。
			</template>
		</p>

		<p class="pc-note">
			时间数值为便于对照的示意量级，不是任何产品的推荐值。两条可迁移的结论：其一，
			<b>超时是一条自外向内收窄的链，内层必须真包含于外层</b>，否则内层配置不产生效果；其二，
			<b>退避的作用是错开重试而不是减少重试</b>——抖动尤其如此，它解决的是「同一批调用同时醒来再次压垮下游」的同步问题。
			还有一条不在图里但同样重要：<b>只有确认幂等的调用才允许自动重试</b>，超时并不意味着下游没有执行。
		</p>
	</div>
</template>

<style scoped>
.trb { margin: 1rem 0; background: var(--cs-color-bg); border: 1px solid var(--cs-color-border); border-radius: var(--cs-radius-lg); padding: var(--cs-space-5) var(--cs-space-6); }
.trb__controls { display: grid; grid-template-columns: repeat(3, 1fr); gap: var(--cs-space-1) var(--cs-space-6); }
.trb__ctl { display: grid; grid-template-columns: 4.5rem 1fr 2.8rem; align-items: center; gap: var(--cs-space-2); min-height: var(--cs-tap-target); font-size: var(--cs-text-sm); }
.trb__ctl span { color: var(--cs-color-text-muted); }
.trb__ctl b { font-family: var(--cs-font-mono); text-align: right; color: var(--cs-color-brand); }
.trb__ctl input { width: 100%; accent-color: var(--cs-color-brand); }
.trb__jitter { margin: var(--cs-space-2) 0 var(--cs-space-5); font-size: var(--cs-text-xs); }
.trb__layers { display: grid; gap: var(--cs-space-2); }
.trb__layer { display: grid; grid-template-columns: 7rem minmax(0, 1fr) 3rem; align-items: center; gap: var(--cs-space-3); }
.trb__layer-name { font-size: var(--cs-text-xs); color: var(--cs-color-text-muted); }
.trb__layer-val { font-size: var(--cs-text-sm); text-align: right; color: var(--cs-color-text); }
.trb__track { height: 14px; border-radius: var(--cs-radius-sm); background: var(--cs-color-neutral-soft); overflow: hidden; }
.trb__fill { display: block; height: 100%; transition: width var(--cs-duration-base) var(--cs-ease-standard); }
.trb__timeline { position: relative; height: 40px; margin: var(--cs-space-5) 0 var(--cs-space-4) 7rem; border-bottom: 1px solid var(--cs-color-border); }
.trb__attempt { position: absolute; top: 6px; height: 22px; min-width: 10px; display: flex; align-items: center; justify-content: center; font-family: var(--cs-font-mono); font-size: var(--cs-text-3xs); color: var(--cs-color-on-brand); background: var(--cs-color-brand); border-radius: var(--cs-radius-xs); transition: var(--cs-transition-colors); }
.trb__attempt.is-cut { background: var(--cs-color-neutral-soft); color: var(--cs-color-text-subtle); }
.trb__mark { position: absolute; top: 0; bottom: -1px; border-left: 2px dashed var(--cs-color-warning); }
.trb__mark i { position: absolute; bottom: -1.3rem; left: 2px; font-style: normal; font-family: var(--cs-font-mono); font-size: var(--cs-text-3xs); color: var(--cs-color-warning); white-space: nowrap; }
.trb__read { margin: var(--cs-space-6) 0 var(--cs-space-4); font-size: var(--cs-text-base); line-height: var(--cs-leading-relaxed); color: var(--cs-color-text-muted); }
.trb__read b { color: var(--cs-color-text); }
.trb__read.is-bad b { color: var(--cs-color-danger); }
@media (max-width: 860px) {
	.trb__controls { grid-template-columns: repeat(2, 1fr); }
}
@media (max-width: 520px) {
	.trb__controls { grid-template-columns: 1fr; }
	.trb__layer { grid-template-columns: 5.5rem minmax(0, 1fr) 2.6rem; }
	.trb__timeline { margin-left: 0; }
}
</style>
