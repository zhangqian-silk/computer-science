<script setup lang="ts">
import { computed, ref } from "vue"

/**
 * 组合语义对照：多个扩展同时命中一个位置时，它们之间是什么关系。
 *
 * 三种范式在单个扩展上看起来差不多，差别只在「有第二个扩展」时才显现：
 * 观察者互不影响、责任链有短路与优先级、中间件层层嵌套且返回路径逆序。
 * 组件按范式展示三个扩展 A/B/C 的实际执行轨迹。
 */

type Paradigm = {
	key: string
	name: string
	sub: string
	color: string
	/** 执行轨迹：每一步一行 */
	trace: { depth: number; text: string; kind: "in" | "core" | "out" | "skip" }[]
	rule: string
	/** 这种范式下「第二个扩展」带来的关键后果 */
	consequence: string
	failure: string
}

const paradigms: Paradigm[] = [
	{
		key: "observer",
		name: "观察者",
		sub: "生命周期回调 / 纯通知",
		color: "var(--cs-color-info)",
		trace: [
			{ depth: 0, text: "事件发生", kind: "core" },
			{ depth: 1, text: "A 收到通知 → 记一条日志", kind: "in" },
			{ depth: 1, text: "B 收到通知 → 上报一个指标", kind: "in" },
			{ depth: 1, text: "C 收到通知 → 写一条追踪 span", kind: "in" },
			{ depth: 0, text: "流程继续，不受任何一方影响", kind: "core" }
		],
		rule: "全部都会被调用，返回值不参与控制流。顺序通常不保证，也不应当被依赖。",
		consequence: "加第二个扩展是安全的：它们之间不存在互相覆盖的可能，因为谁都改不了结果。",
		failure: "想在这里拦截。这类接口的返回类型往往就是空——设计上它只负责「知道」，不负责「决定」。有些框架把这条分工做得很直白：生命周期回调只观察，阻断与审批交给另一套专门的机制。"
	},
	{
		key: "chain",
		name: "责任链",
		sub: "hooks / 权限规则",
		color: "var(--cs-color-warning)",
		trace: [
			{ depth: 0, text: "工具调用待决", kind: "core" },
			{ depth: 1, text: "A 判定：放行", kind: "in" },
			{ depth: 1, text: "B 判定：拒绝", kind: "in" },
			{ depth: 1, text: "C 判定：询问用户", kind: "in" },
			{ depth: 0, text: "汇总：取最严的一条 → 拒绝", kind: "core" },
			{ depth: 1, text: "A 的「放行」不生效", kind: "skip" }
		],
		rule: "多个判定同时存在时按严格程度排序，拒绝优先于询问用户、询问用户优先于放行。多个扩展常常并发启动，因此谁都不能阻止别人开始。",
		consequence: "加第二个扩展会改变结论：一个新装的包只要投出一张拒绝票，原本能跑的动作就跑不了了，而它甚至可能不知道你的那条规则存在。",
		failure: "写一个「放行」的扩展来豁免某个动作，然后发现它被另一处的拒绝规则压过去了。放行在这种语义下从来不是覆盖，只是弃权。"
	},
	{
		key: "middleware",
		name: "中间件",
		sub: "洋葱模型 / 包裹式",
		color: "var(--cs-color-brand)",
		trace: [
			{ depth: 0, text: "请求进入", kind: "core" },
			{ depth: 1, text: "A 前段：改写请求", kind: "in" },
			{ depth: 2, text: "B 前段：挑选本轮工具集", kind: "in" },
			{ depth: 3, text: "C 前段：开始计时", kind: "in" },
			{ depth: 4, text: "真正的模型调用", kind: "core" },
			{ depth: 3, text: "C 后段：记录耗时", kind: "out" },
			{ depth: 2, text: "B 后段：校验输出", kind: "out" },
			{ depth: 1, text: "A 后段：判定需重试 → 再走一遍内层", kind: "out" }
		],
		rule: "每一层都拿到「继续往下走」的句柄，自己决定调用它零次、一次或多次。前段按注册顺序，后段逆序，最外层最后收尾。",
		consequence: "加第二个扩展会改变嵌套深度：它被夹在哪两层之间，决定了它看到的是原始请求还是已被改写的请求。顺序在这里是语义的一部分，不是风格问题。",
		failure: "在外层写重试，而内层带着副作用。句柄被调用了三次，内层的副作用也就发生了三次。"
	}
]

const idx = ref(1)
const cur = computed(() => paradigms[idx.value])
</script>

<template>
	<div class="ecs">
		<div class="ecs__tabs">
			<button
				v-for="(p, i) in paradigms"
				:key="p.key"
				type="button"
				class="ecs__tab"
				:class="{ 'is-on': idx === i }"
				:style="{ '--c': p.color }"
				:aria-pressed="idx === i"
				@click="idx = i"
			>
				<strong>{{ p.name }}</strong>
				<span>{{ p.sub }}</span>
			</button>
		</div>

		<div class="ecs__body" :style="{ '--c': cur.color }">
			<div class="ecs__trace">
				<span class="ecs__lab">三个扩展同时命中时的轨迹</span>
				<ol>
					<li
						v-for="(t, i) in cur.trace"
						:key="i"
						:class="`is-${t.kind}`"
						:style="{ '--d': t.depth }"
					>{{ t.text }}</li>
				</ol>
			</div>

			<div class="ecs__facts">
				<p><i>组合规则</i>{{ cur.rule }}</p>
				<p><i>加第二个扩展会怎样</i>{{ cur.consequence }}</p>
				<p class="ecs__fail"><i>典型翻车</i>{{ cur.failure }}</p>
			</div>
		</div>

		<p class="pc-note">
			三种范式的差别在只有一个扩展时完全看不出来，<b>装上第二个才显形</b>。
			这解释了一类难以复现的现象：某个扩展「单独测都好好的，装在一起就不对」。
			排查这类问题的第一步不是读某一个扩展的代码，而是先确认<b>这个位置采用的是哪种组合语义</b>——
			是三方各行其是，是投票取最严，还是层层包裹并且顺序即语义。
		</p>
	</div>
</template>

<style scoped>
.ecs { margin: 1rem 0; }
.ecs__tabs { display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: var(--cs-space-2); }
.ecs__tab { display: grid; gap: var(--cs-space-1); text-align: left; cursor: pointer; min-height: var(--cs-tap-target); padding: var(--cs-space-3); background: var(--cs-color-bg); border: 1px solid var(--cs-color-border); border-top: 3px solid var(--c); border-radius: var(--cs-radius-lg); transition: var(--cs-transition-colors); }
.ecs__tab:hover { background: var(--cs-color-bg-soft); }
.ecs__tab.is-on { background: color-mix(in srgb, var(--c) 10%, transparent); border-color: var(--c); }
.ecs__tab strong { font-size: var(--cs-text-md); line-height: var(--cs-leading-tight); }
.ecs__tab span { font-size: var(--cs-text-3xs); color: var(--cs-color-text-muted); font-family: var(--cs-font-mono); }
.ecs__body { display: grid; grid-template-columns: minmax(0, 1fr) minmax(0, 1fr); gap: var(--cs-space-4); margin-top: var(--cs-space-4); }
.ecs__trace { background: var(--cs-color-bg-soft); border-radius: var(--cs-radius-lg); padding: var(--cs-space-4); }
.ecs__lab { font-family: var(--cs-font-mono); font-size: var(--cs-text-3xs); letter-spacing: .08em; color: var(--cs-color-text-subtle); }
.ecs__trace ol { list-style: none; margin: var(--cs-space-3) 0 0; padding: 0; display: grid; gap: var(--cs-space-1); }
.ecs__trace li { position: relative; font-family: var(--cs-font-mono); font-size: var(--cs-text-3xs); line-height: var(--cs-leading-normal); color: var(--cs-color-text-muted); padding: var(--cs-space-1) var(--cs-space-2); padding-left: calc(var(--cs-space-2) + var(--d) * var(--cs-space-4)); border-left: 2px solid transparent; }
.ecs__trace li.is-core { color: var(--cs-color-text); background: var(--cs-color-bg); border-radius: var(--cs-radius-xs); border-left-color: var(--c); }
.ecs__trace li.is-in { border-left-color: var(--cs-color-border-strong); }
.ecs__trace li.is-out { border-left-color: var(--cs-color-border-strong); color: var(--cs-color-text-subtle); }
.ecs__trace li.is-skip { color: var(--cs-color-danger); text-decoration: line-through; }
.ecs__facts { display: grid; align-content: start; gap: var(--cs-space-3); }
.ecs__facts p { margin: 0; font-size: var(--cs-text-sm); line-height: var(--cs-leading-relaxed); color: var(--cs-color-text); }
.ecs__facts i { display: block; font-family: var(--cs-font-mono); font-size: var(--cs-text-3xs); letter-spacing: .08em; font-style: normal; color: var(--cs-color-text-subtle); margin-bottom: var(--cs-space-1); }
.ecs__fail { padding-top: var(--cs-space-3); border-top: 1px dashed var(--cs-color-border); color: var(--cs-color-text-muted); }
@media (max-width: 860px) {
	.ecs__body { grid-template-columns: 1fr; }
}
@media (max-width: 520px) {
	.ecs__tabs { grid-template-columns: 1fr; }
}
</style>
