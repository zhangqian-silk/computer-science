<script setup lang="ts">
import { computed, ref } from "vue"

/**
 * 三种人在环机制的横向对照，重点在**恢复语义**。
 *
 * 三者都能做到「执行前停下来等人点头」，差别在于点头之后从哪里继续：
 *   - 异常式中断：整个节点从头重跑，中断之前的副作用会再发生一次；
 *   - 可序列化状态：把待批准的调用与状态一起落盘，恢复时从调用本身继续；
 *   - 回调式放行：调用在运行时内被同步拦住，批准即原地继续，没有重跑。
 *
 * 下方的时间线演示同一段「先扣款、再等批准」的代码在三种机制下的行为，
 * 可以直接看到哪一种会产生重复扣款。这不是实现质量问题，是机制语义差异。
 */

type Mech = {
	key: string
	name: string
	how: string
	resume: string
	/** 恢复时是否重跑中断点之前的代码 */
	replays: boolean
	need: string
	fit: string
	idx: number
}

const mechs: Mech[] = [
	{
		key: "interrupt", name: "异常式中断", idx: 0,
		how: "在节点内调用中断函数，它抛出一个特殊异常，运行时捕获后保存状态快照并暂停。",
		resume: "带着答复重新执行「整个节点」，中断之前的代码全部再跑一遍。",
		replays: true,
		need: "需要检查点存储与线程标识；答复按出现次序与中断点做下标匹配。",
		fit: "节点内在中断之前没有副作用，或副作用天然幂等。"
	},
	{
		key: "state", name: "可序列化状态", idx: 1,
		how: "把「待批准的调用」连同全部执行状态一起落盘，进程可以完全退出。",
		resume: "从被挂起的那次调用本身继续，之前的步骤不重放。",
		replays: false,
		need: "状态必须可完整序列化，这反过来约束了代码里能持有什么。",
		fit: "审批周期长到跨越进程生命周期，例如等人第二天上班再看。"
	},
	{
		key: "callback", name: "回调式放行", idx: 2,
		how: "运行时在执行每个工具前同步询问一个回调，由它返回允许、拒绝或改写后的参数。",
		resume: "谈不上恢复——调用一直卡在原地等回调返回。",
		replays: false,
		need: "调用方进程必须一直在线等待；回调本身要能处理超时。",
		fit: "交互式会话中的逐次确认，人就在旁边。"
	}
]

/** 演示脚本：一段「先扣款、再请求批准、最后发货」的节点代码 */
const steps = [
	{ label: "读取订单", effect: false },
	{ label: "扣款", effect: true },
	{ label: "请求批准", effect: false, pause: true },
	{ label: "发货", effect: true }
]

const sel = ref(0)
const cur = computed(() => mechs[sel.value])

/** 暂停点所在的下标：初次执行跑到这一步为止 */
const pauseAt = steps.findIndex((s) => s.pause)

/** 初次执行到暂停点为止，恢复之后再按各机制的语义继续 */
const trace = computed(() => {
	const first = steps.slice(0, pauseAt + 1).map((s, i) => ({ ...s, i, phase: "first" as const }))
	// 重跑语义：整个节点从头再来；续跑语义：从暂停点之后继续
	const after = cur.value.replays
		? steps.map((s, i) => ({ ...s, i, phase: "resume" as const }))
		: steps.slice(pauseAt + 1).map((s, i) => ({ ...s, i: i + pauseAt + 1, phase: "resume" as const }))
	return [...first, ...after]
})

const dupEffects = computed(() => {
	const seen = new Set<number>()
	const dup = new Set<number>()
	for (const t of trace.value) {
		if (!t.effect) continue
		if (seen.has(t.i)) dup.add(t.i)
		else seen.add(t.i)
	}
	return dup
})
</script>

<template>
	<div class="hil">
		<div class="infra-tabs hil__tabs" role="group" aria-label="人在环机制">
			<button
				v-for="(m, i) in mechs"
				:key="m.key"
				type="button"
				:aria-pressed="sel === i"
				@click="sel = i"
			>{{ m.name }}</button>
		</div>

		<p class="hil__script">
			演示节点：<code>读取订单 → 扣款 → 请求批准 → 发货</code>。扣款被放在了批准之前——
			这是实践中很常见的写法，因为「先占住额度再请人确认」看起来更稳妥。
		</p>

		<ol class="hil__trace">
			<li
				v-for="(t, n) in trace"
				:key="`${t.phase}-${n}`"
				class="hil__step"
				:class="{
					'is-resume': t.phase === 'resume',
					'is-effect': t.effect,
					'is-dup': t.effect && t.phase === 'resume' && dupEffects.has(t.i),
					'is-pause': t.pause
				}"
			>
				<span class="hil__step-no">{{ t.phase === "first" ? "初次" : "恢复" }}</span>
				<span class="hil__step-label">{{ t.label }}</span>
				<span v-if="t.pause" class="hil__step-tag">在此暂停</span>
				<span v-else-if="t.effect && t.phase === 'resume' && dupEffects.has(t.i)" class="hil__step-tag is-bad">重复副作用</span>
				<span v-else-if="t.effect" class="hil__step-tag">有副作用</span>
			</li>
		</ol>

		<p class="hil__read" :class="{ 'is-bad': dupEffects.size > 0 }">
			<template v-if="dupEffects.size > 0">
				<b>这一种会重复扣款。</b>恢复时整个节点从头再跑，而「扣款」写在中断之前，于是它执行了两次。
				对策不是加一层 try，而是<b>把副作用挪到中断之后</b>，或者让它幂等，或者干脆拆成独立的节点。
			</template>
			<template v-else>
				<b>这一种不会重复扣款</b>，因为恢复不从节点开头开始。代价在别处：
				{{ cur.key === "state" ? "全部状态必须可序列化，这会反过来限制代码里能持有哪些对象。" : "调用方进程必须全程在线等待，审批一长就撑不住。" }}
			</template>
		</p>

		<dl class="hil__kv">
			<dt>怎么停</dt>
			<dd>{{ cur.how }}</dd>
			<dt>怎么继续</dt>
			<dd>{{ cur.resume }}</dd>
			<dt>前提</dt>
			<dd>{{ cur.need }}</dd>
			<dt>适用</dt>
			<dd>{{ cur.fit }}</dd>
		</dl>

		<p class="pc-note">
			三种机制的行为依据各自框架的公开文档，属一方实现语义而非通用规范，且随版本变化。
			可迁移的结论只有一条，但它足够硬：<b>选人在环机制时，真正要问的不是「能不能停下来」，而是「恢复之后从哪里继续」</b>。
			这个问题决定了副作用该写在暂停点的哪一侧，而这件事写错了不会报错，只会多扣一笔钱。
		</p>
	</div>
</template>

<style scoped>
.hil { margin: 1rem 0; background: var(--cs-color-bg); border: 1px solid var(--cs-color-border); border-radius: var(--cs-radius-lg); padding: var(--cs-space-5) var(--cs-space-6); }
.hil__tabs { margin-top: 0; }
.hil__script { margin: 0 0 var(--cs-space-4); font-size: var(--cs-text-sm); line-height: var(--cs-leading-relaxed); color: var(--cs-color-text-muted); }
.hil__trace { list-style: none; display: flex; flex-wrap: wrap; gap: var(--cs-space-2); margin: 0 0 var(--cs-space-4); padding: 0; }
.hil__step { display: grid; gap: 2px; padding: var(--cs-space-2) var(--cs-space-3); border: 1px solid var(--cs-color-border); border-radius: var(--cs-radius-md); background: var(--cs-color-bg); min-width: 5.5rem; }
.hil__step.is-resume { border-style: dashed; }
.hil__step.is-effect { border-color: var(--cs-color-warning); background: var(--cs-color-warning-soft); }
.hil__step.is-dup { border-color: var(--cs-color-danger); background: var(--cs-color-danger-soft); }
.hil__step.is-pause { border-color: var(--cs-color-info); background: var(--cs-color-info-soft); }
.hil__step-no { font-family: var(--cs-font-mono); font-size: var(--cs-text-3xs); color: var(--cs-color-text-subtle); }
.hil__step-label { font-size: var(--cs-text-sm); color: var(--cs-color-text); }
.hil__step-tag { font-size: var(--cs-text-3xs); color: var(--cs-color-text-muted); }
.hil__step-tag.is-bad { color: var(--cs-color-danger); }
.hil__read { margin: 0 0 var(--cs-space-5); padding: var(--cs-space-3) var(--cs-space-4); border-left: 3px solid var(--cs-color-success); border-radius: var(--cs-radius-sm); background: var(--cs-color-bg-soft); font-size: var(--cs-text-base); line-height: var(--cs-leading-relaxed); color: var(--cs-color-text-muted); }
.hil__read.is-bad { border-left-color: var(--cs-color-danger); }
.hil__read b { color: var(--cs-color-text); }
.hil__kv { display: grid; grid-template-columns: 4.5rem minmax(0, 1fr); gap: var(--cs-space-2) var(--cs-space-3); margin: 0 0 var(--cs-space-4); }
.hil__kv dt { font-family: var(--cs-font-mono); font-size: var(--cs-text-3xs); color: var(--cs-color-text-subtle); }
.hil__kv dd { margin: 0; font-size: var(--cs-text-sm); line-height: var(--cs-leading-normal); color: var(--cs-color-text-muted); }
@media (max-width: 520px) {
	.hil__trace { flex-direction: column; }
	.hil__step { min-width: 0; }
	.hil__kv { grid-template-columns: 1fr; gap: var(--cs-space-1) 0; }
}
</style>
