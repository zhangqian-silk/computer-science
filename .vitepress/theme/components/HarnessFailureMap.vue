<script setup lang="ts">
import { computed, ref } from "vue"

/**
 * 故障域分层与故障注入点。
 *
 * 承载的是一件文字很难讲清的事：Agent 的故障并非平铺的一张清单，
 * 而是按「谁能判定这次失败」分层的——传输层能凭状态码判定，
 * 执行层要看退出码与副作用，认知层则根本没有可判定的错误信号。
 * 层级越往下，自动恢复越不可靠，越需要把判定权交还给人。
 *
 * 每层挂着若干故障注入点（F 编号），点选后给出「期望的系统行为」，
 * 因为一条注入点若不写明期望行为，就无法判断演练结果是通过还是失败。
 *
 * 结构与分层取自公开文档与源码的行为描述（核验于 2026-09-14）；
 * 编号 F1–F12 是本文为便于引用而设的编排，不对应任何产品的内部编号。
 */

type Injection = {
	id: string
	name: string
	trigger: string
	expect: string
}

type Layer = {
	key: string
	name: string
	verdict: string
	signal: string
	/** 自动恢复的可靠度：由信号的确定性决定 */
	autonomy: "high" | "medium" | "low" | "none"
	injections: Injection[]
}

const layers: Layer[] = [
	{
		key: "l1",
		name: "L1 传输与网关",
		verdict: "状态码与连接状态",
		signal: "确定：429 就是 429，连接重置就是连接重置",
		autonomy: "high",
		injections: [
			{
				id: "F1",
				name: "上游返回 429",
				trigger: "网关对第 N 次请求返回限流",
				expect: "按退避重试且带抖动；重试次数有上限；耗尽后返回可解释的失败而不是静默截断。若响应带重试延迟建议，超过本地上限时应立刻失败并说明，而不是长时间静默等待"
			},
			{
				id: "F2",
				name: "流中途断开",
				trigger: "SSE 已输出部分内容后连接中断",
				expect: "区分「尚未产出」与「已产出部分」：前者可整请求重放，后者重放会产生重复内容，需要按已接收位置续接或显式丢弃本次输出"
			},
			{
				id: "F3",
				name: "流空闲但不断开",
				trigger: "连接保持但长时间无字节到达",
				expect: "由空闲看门狗在阈值处主动关闭并计入重试，而不是挂到请求总超时。阈值要容纳长思考停顿与代理缓冲"
			},
			{
				id: "F4",
				name: "配额耗尽",
				trigger: "返回额度/计费类错误，HTTP 状态可能同为 429",
				expect: "识别为「不可重试」并立即失败。把它当作瞬时限流去退避重试，只会在配额恢复前空耗时间与预算"
			}
		]
	},
	{
		key: "l2",
		name: "L2 工具与执行",
		verdict: "退出码、超时与沙箱拒绝",
		signal: "较确定，但伴随副作用：失败前可能已经改了东西",
		autonomy: "medium",
		injections: [
			{
				id: "F5",
				name: "命令超时",
				trigger: "工具执行超过时限被终止",
				expect: "终止整个进程组而非仅父进程；把超时作为一次观察回填，并保留已产生的部分输出与其被截断的事实"
			},
			{
				id: "F6",
				name: "输出超量",
				trigger: "单次工具返回远超预算的文本",
				expect: "落盘并只回填摘要与路径，而不是把全文塞回上下文。截断必须让模型看得见，否则它会把残缺输出当成完整结果"
			},
			{
				id: "F7",
				name: "沙箱拒绝",
				trigger: "写入越界路径或访问未放行域名",
				expect: "返回可归因的拒绝原因（被拒的是哪个路径或主机），而不是一个泛化的失败。原因不具体，模型只会盲目重试同一个动作"
			},
			{
				id: "F8",
				name: "工具连接失败",
				trigger: "外部工具进程启动超时或中途退出",
				expect: "启动阶段的超时与调用阶段的超时分开计量；单个工具不可用不应阻塞整个会话启动，除非它被显式标记为必需"
			}
		]
	},
	{
		key: "l3",
		name: "L3 状态与持久化",
		verdict: "落盘记录与进程状态的差集",
		signal: "弱：进程死了才暴露，而那时已经无人记录",
		autonomy: "low",
		injections: [
			{
				id: "F9",
				name: "执行中崩溃",
				trigger: "在工具已执行、结果尚未写入时杀掉进程",
				expect: "重启后能恢复到最后一个完整检查点；已发生副作用但未记录的动作要能被识别出来，而不是悄悄重放一遍"
			},
			{
				id: "F10",
				name: "并发写同一会话",
				trigger: "两个进程同时写同一份会话记录",
				expect: "由写者锁裁决归属，落败方明确失败而不是交错写入。交错写入产生的损坏记录往往要到下一次恢复时才被发现"
			}
		]
	},
	{
		key: "l4",
		name: "L4 认知与目标",
		verdict: "没有错误信号",
		signal: "无：系统一切正常，只是做错了事",
		autonomy: "none",
		injections: [
			{
				id: "F11",
				name: "重复同一失败动作",
				trigger: "连续多轮以相同参数调用同一失败工具",
				expect: "由外部循环检测并打断，而不是等预算耗尽。检测依据是动作与观察的重复，不是模型的自述"
			},
			{
				id: "F12",
				name: "压缩后行为回退",
				trigger: "触发一次上下文压缩，再观察后续行为",
				expect: "压缩前后的关键约束、待办与已达成结论保持一致。这条只能靠对比断言，因为系统不会为此报错"
			}
		]
	}
]

const activeLayer = ref("l2")
const activeInjection = ref("F5")

const cur = computed(() => layers.find((l) => l.key === activeLayer.value)!)
const curInjection = computed(
	() => layers.flatMap((l) => l.injections).find((i) => i.id === activeInjection.value)!
)

function pickLayer(key: string) {
	activeLayer.value = key
	activeInjection.value = layers.find((l) => l.key === key)!.injections[0].id
}

const autonomyLabel: Record<Layer["autonomy"], string> = {
	high: "可自动恢复",
	medium: "可自动恢复，但需处理副作用",
	low: "只能恢复到检查点",
	none: "无法自动恢复"
}

const autonomyState: Record<Layer["autonomy"], string> = {
	high: "pass",
	medium: "warn",
	low: "warn",
	none: "fail"
}
</script>

<template>
	<div class="hfm">
		<div class="hfm__tabs">
			<button
				v-for="l in layers"
				:key="l.key"
				type="button"
				class="hfm__tab"
				:class="{ 'is-on': activeLayer === l.key }"
				:aria-pressed="activeLayer === l.key"
				@click="pickLayer(l.key)"
			>{{ l.name }}</button>
		</div>

		<div class="hfm__layer">
			<div class="hfm__meta">
				<p><i>判定依据</i>{{ cur.verdict }}</p>
				<p><i>信号强度</i>{{ cur.signal }}</p>
			</div>
			<div class="cs-state" :class="`cs-state--${autonomyState[cur.autonomy]}`">
				{{ autonomyLabel[cur.autonomy] }}
			</div>
		</div>

		<div class="hfm__chips">
			<button
				v-for="inj in cur.injections"
				:key="inj.id"
				type="button"
				class="token-chip hfm__chip"
				:class="{ 'is-selected': activeInjection === inj.id }"
				:aria-pressed="activeInjection === inj.id"
				@click="activeInjection = inj.id"
			>{{ inj.id }} · {{ inj.name }}</button>
		</div>

		<div class="hfm__detail">
			<p class="hfm__trigger"><i>注入方式</i>{{ curInjection.trigger }}</p>
			<p class="hfm__expect"><i>期望的系统行为</i>{{ curInjection.expect }}</p>
		</div>

		<p class="pc-note">
			四层的差别不在严重程度，而在<b>谁能判定这次失败</b>。L1 的信号是确定的，所以退避重试足以覆盖；
			L2 的信号仍然明确，但失败前可能已经产生副作用，重试要先回答「这个动作可不可以再做一次」；
			L3 的失败在进程活着时不可见；到了 L4，系统根本不会报错——它只是做错了事。
			<b>自动恢复的能力沿着这四层单调递减，把人接回来的必要性则单调递增。</b>
		</p>
	</div>
</template>

<style scoped>
.hfm { margin: 1rem 0; }
.hfm__tabs { display: flex; flex-wrap: wrap; gap: var(--cs-space-1); }
.hfm__tab { min-height: var(--cs-tap-target); padding: var(--cs-space-1) var(--cs-space-4); cursor: pointer; font-size: var(--cs-text-xs); color: var(--cs-color-text-muted); background: var(--cs-color-bg); border: 1px solid var(--cs-color-border); border-radius: var(--cs-radius-pill); transition: var(--cs-transition-colors); }
.hfm__tab:hover { background: var(--cs-color-bg-soft); }
.hfm__tab.is-on { background: var(--cs-color-brand-soft); border-color: var(--cs-color-brand); color: var(--cs-color-brand); font-weight: 600; }
.hfm__layer { display: grid; gap: var(--cs-space-3); margin-top: var(--cs-space-5); }
.hfm__meta { display: grid; gap: var(--cs-space-2); padding: var(--cs-space-4); border: 1px solid var(--cs-color-border); border-radius: var(--cs-radius-md); background: var(--cs-color-bg-soft); }
.hfm__meta p { margin: 0; font-size: var(--cs-text-base); line-height: var(--cs-leading-normal); }
.hfm__meta i, .hfm__detail i { display: block; font-family: var(--cs-font-mono); font-size: var(--cs-text-3xs); letter-spacing: .08em; font-style: normal; color: var(--cs-color-text-subtle); margin-bottom: var(--cs-space-1); }
.hfm__chips { display: flex; flex-wrap: wrap; gap: var(--cs-space-2); margin-top: var(--cs-space-5); }
.hfm__chip { min-height: var(--cs-tap-target); cursor: pointer; font-size: var(--cs-text-xs); text-align: left; }
.hfm__detail { display: grid; gap: var(--cs-space-3); margin-top: var(--cs-space-4); }
.hfm__trigger, .hfm__expect { margin: 0; padding: var(--cs-space-4); border-radius: var(--cs-radius-sm); font-size: var(--cs-text-base); line-height: var(--cs-leading-relaxed); }
.hfm__trigger { background: var(--cs-color-neutral-soft); }
.hfm__expect { background: var(--cs-color-info-soft); }
.hfm__expect i { color: var(--cs-color-info); }
@media (min-width: 861px) {
	.hfm__layer { grid-template-columns: minmax(0, 1fr) auto; align-items: center; }
	.hfm__detail { grid-template-columns: 1fr 1fr; }
}
</style>
