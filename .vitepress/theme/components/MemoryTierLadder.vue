<script setup lang="ts">
import { computed, ref } from "vue"

/**
 * 记忆的三层分层与层间固化路径。
 *
 * 组件要说明两件事：
 *   1. 三层的载体、寿命、容量与失效方式各不相同，因此不能用同一套手段治理；
 *   2. 层与层之间靠「固化」而非「自动延续」相连——每一道固化都是一次显式的写操作，
 *      任何一道缺失，信息就在该层的寿命到期时消失。
 * 点击层可查看细节，点击箭头可查看那一道固化路径的触发条件与失败后果。
 */

type Tier = {
	key: string
	name: string
	en: string
	carrier: string
	life: string
	capacity: string
	color: string
	holds: string[]
	loss: string
	govern: string
}

type Path = {
	key: string
	from: string
	to: string
	name: string
	trigger: string
	actor: string
	what: string
	ifMissing: string
}

const tiers: Tier[] = [
	{
		key: "t0",
		name: "工作记忆",
		en: "working memory",
		carrier: "本次请求载荷",
		life: "一次模型调用",
		capacity: "受上下文窗口硬约束",
		color: "var(--cs-color-danger)",
		holds: ["系统提示与工具定义", "最近若干轮原文", "本轮召回进来的内容", "刚产生的工具结果"],
		loss: "调用结束即不存在。它不是「存储」，而是「本次可见范围」——下一次调用要重新组装。",
		govern: "靠挑选与压缩治理，不靠扩容。"
	},
	{
		key: "t1",
		name: "会话内记忆",
		en: "session state",
		carrier: "会话状态存储（检查点 / 事件日志）",
		life: "一个会话或线程",
		capacity: "远大于窗口，但按会话隔离",
		color: "var(--cs-color-warning)",
		holds: ["完整消息历史", "待办与子任务状态", "本次任务的中间产物", "压缩后的摘要"],
		loss: "会话关闭或过期即失效。换一个会话读不到，即使是同一个用户。",
		govern: "靠持久化与可恢复性治理：能恢复、能回放、能审计。"
	},
	{
		key: "t2",
		name: "跨会话长期记忆",
		en: "long-term memory",
		carrier: "记忆存储（文件 / 向量库 / 图 / 关系表）",
		life: "跨会话，直到被改写或删除",
		capacity: "实际无上限，但召回预算有上限",
		color: "var(--cs-color-success)",
		holds: ["稳定偏好与身份事实", "过往经历的结论", "工作流与操作规程", "实体及其关系"],
		loss: "不会自然失效，反而会「过期而不自知」——旧结论一直被召回，这是它特有的失效形态。",
		govern: "靠有效性判定治理：时效、来源、置信度、矛盾检测。"
	}
]

const paths: Path[] = [
	{
		key: "p0",
		from: "工作记忆",
		to: "会话内记忆",
		name: "落盘",
		trigger: "每一步执行完成的那一刻",
		actor: "框架（自动）",
		what: "把这一步产生的消息、工具结果、状态变更追加写入会话存储。这一道通常由框架无条件执行，所以最容易被当成「本来就会发生」。",
		ifMissing: "进程重启即全部丢失，任务无法恢复，也无从回放排障。"
	},
	{
		key: "p1",
		from: "会话内记忆",
		to: "跨会话长期记忆",
		name: "抽取与固化",
		trigger: "会话结束、达到轮次阈值，或模型判定「这条值得记住」",
		actor: "模型判定 + 显式写入",
		what: "从会话里挑出跨会话仍然成立的部分，改写成独立于原对话也能读懂的条目，写入长期存储。这一道必须显式设计——没有任何机制会自动完成它。",
		ifMissing: "每个新会话都从零开始：用户重复交代同样的偏好，Agent 重复踩同样的坑。"
	},
	{
		key: "p2",
		from: "跨会话长期记忆",
		to: "工作记忆",
		name: "召回",
		trigger: "组装下一次请求之前",
		actor: "检索 + 打分 + 预算裁剪",
		what: "按当前任务把长期存储里相关的少数条目取回，放进本次载荷。存了不等于被看到——只有走完这一道，记忆才真正影响输出。",
		ifMissing: "存储里躺着正确的记忆，模型却表现得一无所知。这是记忆系统最常见的故障，而且排障方向在召回侧，不在存储侧。"
	}
]

const activeTier = ref(2)
const activePath = ref<string | null>(null)
const cur = computed(() => tiers[activeTier.value])
const curPath = computed(() => paths.find((p) => p.key === activePath.value) ?? null)

function pickTier(i: number) {
	activeTier.value = i
	activePath.value = null
}
function pickPath(key: string) {
	activePath.value = activePath.value === key ? null : key
}
</script>

<template>
	<div class="mtl">
		<div class="mtl__stack">
			<template v-for="(t, i) in tiers" :key="t.key">
				<button
					type="button"
					class="mtl__tier"
					:class="{ 'is-on': activeTier === i && !activePath }"
					:style="{ '--c': t.color }"
					:aria-pressed="activeTier === i && !activePath"
					@click="pickTier(i)"
				>
					<span class="mtl__tier-head">
						<strong>{{ t.name }}</strong>
						<i class="cs-mono">{{ t.en }}</i>
					</span>
					<span class="mtl__tier-meta">
						<em>载体</em>{{ t.carrier }}
					</span>
					<span class="mtl__tier-meta">
						<em>寿命</em>{{ t.life }}
					</span>
				</button>

				<div v-if="i < paths.length" class="mtl__arrow-row">
					<button
						type="button"
						class="mtl__arrow"
						:class="{ 'is-on': activePath === paths[i].key }"
						:aria-pressed="activePath === paths[i].key"
						@click="pickPath(paths[i].key)"
					>
						<span class="mtl__arrow-mark" aria-hidden="true">↕</span>
						<span class="mtl__arrow-name">{{ paths[i].name }}</span>
						<span class="mtl__arrow-actor">{{ paths[i].actor }}</span>
					</button>
				</div>
			</template>
		</div>

		<article v-if="curPath" class="mtl__panel mtl__panel--path">
			<header>
				<b class="pc-serif">{{ curPath.from }} → {{ curPath.to }}：{{ curPath.name }}</b>
				<span class="mtl__badge">{{ curPath.trigger }}</span>
			</header>
			<p class="mtl__body">{{ curPath.what }}</p>
			<p class="mtl__foot"><i>这一道缺失会怎样</i>{{ curPath.ifMissing }}</p>
		</article>

		<article v-else class="mtl__panel" :style="{ '--c': cur.color }">
			<header>
				<b class="pc-serif">{{ cur.name }}</b>
				<span class="mtl__badge">{{ cur.capacity }}</span>
			</header>
			<ul class="mtl__holds">
				<li v-for="h in cur.holds" :key="h">{{ h }}</li>
			</ul>
			<p class="mtl__body"><i>怎么失效</i>{{ cur.loss }}</p>
			<p class="mtl__foot"><i>治理手段</i>{{ cur.govern }}</p>
		</article>

		<p class="pc-note">
			三层的关系不是「同一份数据的三种精度」，而是三套独立的机制：载体不同、寿命不同、失效方式不同。
			真正决定一个记忆系统好坏的往往不是某一层做得多精细，而是<b>三道固化路径是否都被显式实现</b>——
			中间任何一道缺失，信息都会在上一层的寿命到期时安静地消失，而且现象各不相同：缺落盘表现为「重启就没了」，
			缺抽取表现为「每次都要重新交代」，缺召回表现为「明明存了却不用」。
		</p>
	</div>
</template>

<style scoped>
.mtl { margin: 1rem 0; }
.mtl__stack { display: grid; gap: 0; }
.mtl__tier { display: grid; gap: var(--cs-space-1); text-align: left; cursor: pointer; background: var(--cs-color-bg); border: 1px solid var(--cs-color-border); border-left: 3px solid var(--c); border-radius: var(--cs-radius-lg); padding: var(--cs-space-4) var(--cs-space-5); min-height: var(--cs-tap-target); transition: var(--cs-transition-colors); }
.mtl__tier:hover { background: var(--cs-color-bg-soft); }
.mtl__tier.is-on { background: color-mix(in srgb, var(--c) 10%, transparent); border-color: var(--c); }
.mtl__tier-head { display: flex; flex-wrap: wrap; align-items: baseline; gap: var(--cs-space-3); }
.mtl__tier-head strong { font-size: var(--cs-text-lg); }
.mtl__tier-head i { font-size: var(--cs-text-3xs); font-style: normal; color: var(--cs-color-text-subtle); }
.mtl__tier-meta { font-size: var(--cs-text-xs); color: var(--cs-color-text-muted); }
.mtl__tier-meta em { display: inline-block; min-width: 2.6rem; font-family: var(--cs-font-mono); font-size: var(--cs-text-3xs); font-style: normal; color: var(--cs-color-text-subtle); }
.mtl__arrow-row { display: flex; justify-content: center; padding: var(--cs-space-1) 0; }
.mtl__arrow { display: inline-flex; align-items: center; gap: var(--cs-space-3); min-height: var(--cs-tap-target); padding: var(--cs-space-1) var(--cs-space-4); cursor: pointer; background: var(--cs-color-bg-soft); border: 1px dashed var(--cs-color-border-strong); border-radius: var(--cs-radius-pill); transition: var(--cs-transition-colors); }
.mtl__arrow:hover { background: var(--cs-color-neutral-soft); }
.mtl__arrow.is-on { background: var(--cs-color-brand-soft); border-color: var(--cs-color-brand); border-style: solid; }
.mtl__arrow-mark { font-family: var(--cs-font-mono); color: var(--cs-color-text-subtle); }
.mtl__arrow-name { font-size: var(--cs-text-sm); font-weight: 600; color: var(--cs-color-text); }
.mtl__arrow-actor { font-family: var(--cs-font-mono); font-size: var(--cs-text-3xs); color: var(--cs-color-text-subtle); }
.mtl__panel { margin-top: var(--cs-space-5); background: var(--cs-color-bg); border: 1px solid var(--cs-color-border); border-left: 3px solid var(--c); border-radius: var(--cs-radius-lg); padding: var(--cs-space-5) var(--cs-space-6); }
.mtl__panel--path { --c: var(--cs-color-brand); }
.mtl__panel header { display: flex; flex-wrap: wrap; align-items: baseline; justify-content: space-between; gap: var(--cs-space-3); }
.mtl__panel header b { font-size: var(--cs-text-lg); }
.mtl__badge { font-family: var(--cs-font-mono); font-size: var(--cs-text-3xs); color: var(--c); border: 1px solid var(--c); border-radius: var(--cs-radius-pill); padding: var(--cs-space-1) var(--cs-space-3); }
.mtl__holds { display: grid; grid-template-columns: repeat(2, 1fr); gap: var(--cs-space-1) var(--cs-space-4); margin: var(--cs-space-3) 0; padding-left: var(--cs-space-5); }
.mtl__holds li { font-size: var(--cs-text-sm); color: var(--cs-color-text-muted); }
.mtl__body { margin: var(--cs-space-3) 0; font-size: var(--cs-text-base); line-height: var(--cs-leading-relaxed); color: var(--cs-color-text); }
.mtl__foot { margin: 0; padding-top: var(--cs-space-3); border-top: 1px dashed var(--cs-color-border); font-size: var(--cs-text-sm); line-height: var(--cs-leading-normal); color: var(--cs-color-text-muted); }
.mtl__body i, .mtl__foot i { display: block; font-family: var(--cs-font-mono); font-size: var(--cs-text-3xs); letter-spacing: .08em; font-style: normal; color: var(--cs-color-text-subtle); margin-bottom: var(--cs-space-1); }
@media (max-width: 640px) {
	.mtl__holds { grid-template-columns: 1fr; }
	.mtl__arrow { flex-wrap: wrap; gap: var(--cs-space-2); }
}
</style>
