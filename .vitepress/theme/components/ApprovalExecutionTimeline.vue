<script setup lang="ts">
import { computed, ref } from "vue"

/**
 * 工具调用 → 审批 → 沙箱执行 → 回写 → 检查点 的关键链路。
 *
 * 这条链路上每一步之间都有一个「如果此刻进程死了会怎样」的缝隙，
 * 而这些缝隙正是恢复语义的全部难点所在。组件用逐步推进的方式呈现：
 * 每一步除了「发生了什么」，还标出它是否可重放、以及崩在这一步之后会留下什么。
 *
 * 「可重放」的定义很窄：重放后系统的最终状态与只执行一次相同。
 * 一旦某步不可重放，恢复就不能简单地「从上一个检查点再跑一遍」。
 *
 * 各步的行为描述综合自四个实现的公开文档与源码（核验于 2026-09-14），
 * 是对共同结构的归纳，不逐字对应任何单一产品的实现。
 */

type Step = {
	n: number
	name: string
	what: string
	/** 崩溃发生在这一步「完成之后、下一步之前」时的后果 */
	crash: string
	replay: "safe" | "risky" | "unsafe"
}

const steps: Step[] = [
	{
		n: 1,
		name: "模型产出调用提议",
		what: "模型输出一个「调用哪个工具、传什么参数」的结构化请求。此刻什么都还没发生——它是一份提议，不是一次执行。",
		crash: "没有副作用。整轮可以原样重来，代价只是这次采样的费用。",
		replay: "safe"
	},
	{
		n: 2,
		name: "策略判定",
		what: "宿主按规则决定放行、询问还是拒绝。判定依据是命令字符串与目标路径——也就是「这条命令看起来要做什么」。",
		crash: "没有副作用。但要注意判定结果本身通常不落盘，恢复后会重新判定一次，而规则可能已经被改过了。",
		replay: "safe"
	},
	{
		n: 3,
		name: "等待人工确认",
		what: "需要人拍板的动作停在这里。这一步的时长不由系统决定，可能是三秒，也可能是第二天早上。",
		crash: "待批请求悬空。恢复时必须能区分「没批过」与「批过但没记下来」——把这两者混同，要么重复问人，要么把未批准的动作当成已批准。",
		replay: "risky"
	},
	{
		n: 4,
		name: "沙箱内执行",
		what: "命令在隔离边界内运行。边界由内核施加，因此它约束的是进程实际访问了什么，而不是命令名暗示了什么。",
		crash: "这里是最危险的缝隙。文件可能已经改了、请求可能已经发出去了，而系统没有任何记录。恢复时无从判断这个动作是否已经生效。",
		replay: "unsafe"
	},
	{
		n: 5,
		name: "结果回填",
		what: "退出码、输出与沙箱拒绝原因被整理成一条观察。超量输出在此处落盘并只回填摘要与路径。",
		crash: "动作已生效但观察丢失。重放会重复执行一次已经成功的动作——除非该动作本身是幂等的。",
		replay: "unsafe"
	},
	{
		n: 6,
		name: "写入检查点",
		what: "把这一步的状态变更追加到持久记录里。写完之后，这一步才算「系统知道它发生过」。",
		crash: "写入本身要么完整要么不算——追加式记录在这里比就地覆盖更安全，因为半条记录可以被识别并丢弃。",
		replay: "safe"
	}
]

const cursor = ref(4)
const cur = computed(() => steps.find((s) => s.n === cursor.value)!)

const replayLabel: Record<Step["replay"], string> = {
	safe: "可安全重放",
	risky: "重放前须先判定状态",
	unsafe: "重放不安全"
}
const replayState: Record<Step["replay"], string> = {
	safe: "pass",
	risky: "warn",
	unsafe: "fail"
}
</script>

<template>
	<div class="aet">
		<ol class="aet__track">
			<li v-for="s in steps" :key="s.n">
				<button
					type="button"
					class="aet__step"
					:class="[`is-${s.replay}`, { 'is-on': cursor === s.n }]"
					:aria-pressed="cursor === s.n"
					@click="cursor = s.n"
				>
					<span class="aet__n cs-mono">{{ s.n }}</span>
					<span class="aet__label">{{ s.name }}</span>
				</button>
			</li>
		</ol>

		<div class="aet__legend">
			<span><em class="is-safe"></em>可安全重放</span>
			<span><em class="is-risky"></em>须先判定状态</span>
			<span><em class="is-unsafe"></em>重放不安全</span>
		</div>

		<div class="aet__detail">
			<div class="aet__head">
				<b>{{ cur.n }} · {{ cur.name }}</b>
				<span class="cs-state" :class="`cs-state--${replayState[cur.replay]}`">
					{{ replayLabel[cur.replay] }}
				</span>
			</div>
			<p class="aet__what">{{ cur.what }}</p>
			<p class="aet__crash"><i>此刻崩溃会留下什么</i>{{ cur.crash }}</p>
		</div>

		<p class="pc-note">
			六步里真正棘手的只有第 4 步与第 5 步之间那道缝：<b>动作已经生效，而系统还不知道。</b>
			没有任何检查点机制能消除它——检查点只能记录「已知发生过的事」，
			而这道缝的定义就是「发生了但尚未被记录」。
			工程上能做的有三件：把有副作用的动作尽量做成幂等的，
			让重放不改变最终状态；把副作用<b>推迟到确认之后</b>，让缝隙落在无副作用的一侧；
			以及在恢复时不假设「没记录就是没发生」，而是去环境里核对当前的真实状态。
			第三条最容易被略过，也最常导致「恢复之后又执行了一遍」。
		</p>
	</div>
</template>

<style scoped>
.aet { margin: 1rem 0; }
.aet__track { list-style: none; margin: 0; padding: 0; display: grid; gap: var(--cs-space-2); }
.aet__step { display: flex; align-items: center; gap: var(--cs-space-3); width: 100%; min-height: var(--cs-tap-target); padding: var(--cs-space-3) var(--cs-space-4); cursor: pointer; text-align: left; background: var(--cs-color-bg); border: 1px solid var(--cs-color-border); border-left: 3px solid var(--cs-color-success); border-radius: var(--cs-radius-md); transition: var(--cs-transition-colors); }
.aet__step.is-risky { border-left-color: var(--cs-color-warning); }
.aet__step.is-unsafe { border-left-color: var(--cs-color-danger); }
.aet__step:hover { background: var(--cs-color-bg-soft); }
.aet__step.is-on { background: var(--cs-color-brand-soft); }
.aet__n { flex: none; width: 1.5rem; font-size: var(--cs-text-sm); color: var(--cs-color-text-subtle); }
.aet__step.is-on .aet__n { color: var(--cs-color-brand); font-weight: 700; }
.aet__label { font-size: var(--cs-text-base); }
.aet__legend { display: flex; flex-wrap: wrap; gap: var(--cs-space-5); margin: var(--cs-space-3) 0 var(--cs-space-5); font-family: var(--cs-font-mono); font-size: var(--cs-text-3xs); color: var(--cs-color-text-subtle); }
.aet__legend span { display: flex; align-items: center; gap: var(--cs-space-2); }
.aet__legend em { width: 10px; height: 3px; border-radius: var(--cs-radius-pill); }
.aet__legend em.is-safe { background: var(--cs-color-success); }
.aet__legend em.is-risky { background: var(--cs-color-warning); }
.aet__legend em.is-unsafe { background: var(--cs-color-danger); }
.aet__detail { display: grid; gap: var(--cs-space-3); }
.aet__head { display: flex; flex-wrap: wrap; align-items: center; justify-content: space-between; gap: var(--cs-space-3); }
.aet__head b { font-size: var(--cs-text-md); color: var(--cs-color-brand); }
.aet__what { margin: 0; font-size: var(--cs-text-base); line-height: var(--cs-leading-relaxed); }
.aet__crash { margin: 0; padding: var(--cs-space-4); border-radius: var(--cs-radius-sm); background: var(--cs-color-neutral-soft); font-size: var(--cs-text-base); line-height: var(--cs-leading-relaxed); }
.aet__crash i { display: block; font-family: var(--cs-font-mono); font-size: var(--cs-text-3xs); letter-spacing: .08em; font-style: normal; margin-bottom: var(--cs-space-1); color: var(--cs-color-text-subtle); }
</style>
