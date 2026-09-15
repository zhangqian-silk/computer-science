<script setup lang="ts">
import { computed, ref } from "vue"
import { seriesColor } from "../series-palette"

/**
 * 接入相关名词的分层归属。
 *
 * 这张图要解决的是一个纯粹的分类问题：stdio、PTY、JSON-RPC、ACP、MCP、沙箱、SDK
 * 这些名词经常被并列讨论，但它们回答的不是同一个问题。把它们按「回答哪个问题」
 * 归到五层之后，「A 和 B 怎么选」这类提法本身就会显出问题——不同层之间不是选项关系，
 * 而是叠加关系。
 *
 * 第三层另有一个正交维度：方向。谁向谁发起调用，决定了一个协议解决的是什么问题。
 */

type Layer = {
	key: string
	idx: number
	name: string
	question: string
	members: string[]
	note: string
	swap: string
}

const layers: Layer[] = [
	{
		key: "model",
		idx: 0,
		name: "模型层",
		question: "一次生成里输入什么、输出什么？",
		members: ["消息与工具定义", "结构化输出约束", "前缀缓存标记"],
		note: "这一层只有一次无状态的请求与响应。工具调用在这里仅仅是一种输出格式——模型给出的是「想调用哪个工具、传什么参数」，不是执行结果。多轮、循环、状态全都是上面几层加进去的。",
		swap: "把上面几层的职责压到这一层，就得到「自己写循环」：可控性最高，但压缩、编辑应用、权限校验、记账全部要自己实现一遍。"
	},
	{
		key: "runtime",
		idx: 1,
		name: "运行时层",
		question: "循环怎么转、上下文怎么管、动作要不要放行？",
		members: ["主循环与终止条件", "工具调度与结果回填", "上下文压缩", "权限与审批策略", "会话持久化与恢复"],
		note: "这一层是把「一次生成」变成「一个能干完活的程序」的全部工程。它的价值几乎全在细节里：读文件要能分页、搜索结果要能截断、编辑要能校验、失败要能归因。这些既不属于模型，也不属于任何协议。",
		swap: "换掉这一层意味着换掉 Agent 本身。上层协议做得好的标志，恰恰是换掉运行时而客户端不用改。"
	},
	{
		key: "protocol",
		idx: 2,
		name: "协议层",
		question: "会话、事件、审批的语义怎么约定？",
		members: ["初始化与能力协商", "会话建立与恢复", "流式更新通知", "权限请求与回应", "取消与停止原因"],
		note: "语义约定的核心不是消息长什么样，而是「谁能向谁发起请求」。审批这件事天然是运行时向界面发问，所以只要协议不允许服务端主动发起请求，审批就无法表达——这一条决定了协议能不能承载有人在旁边看着的场景。",
		swap: "同一个运行时可以同时说多种协议；同一个协议也可以接多个运行时。这一层做标准化，边际接入成本才会随对接数量下降。"
	},
	{
		key: "transport",
		idx: 3,
		name: "传输层",
		question: "字节怎么流动、消息怎么分帧？",
		members: ["stdio（按行分隔）", "HTTP 与 SSE", "WebSocket", "伪终端 PTY"],
		note: "传输只管字节和分帧，不管语义。同一套协议语义可以跑在 stdio 上，也可以跑在 WebSocket 上；反过来，同一个 stdio 通道上可以跑完全不同的协议。把两者混为一谈，就会问出「该用 stdio 还是用 ACP」这样无法回答的问题。",
		swap: "stdio 换成 WebSocket 只影响连通性与生命周期绑定，不改变任何一条方法的含义——前提是协议本身没有假设单连接。"
	},
	{
		key: "isolation",
		idx: 4,
		name: "隔离层",
		question: "能写哪里、能不能联网、越界要不要问人？",
		members: ["文件系统可写范围", "网络出站策略", "系统调用过滤", "受保护路径"],
		note: "隔离是与上面四层完全正交的一维：它不改变任何消息的含义，只改变一个动作能不能真的发生。因此它既不能替代审批，也不能被审批替代——一个管边界，一个管时机。",
		swap: "隔离可以整层叠加或整层去掉，而上面四层的代码一行都不用改。这正是它应当被单独讨论的理由。"
	}
]

type Dir = {
	key: string
	label: string
	arrow: string
	solves: string
	examples: string
	confuse: string
}

const dirs: Dir[] = [
	{
		key: "c2a",
		label: "界面 → 运行时",
		arrow: "客户端驱动 Agent",
		solves: "建立会话、发出请求、接收流式进展、回答审批",
		examples: "编辑器接一个外部 Agent；自建面板驱动一个运行时；命令行的非交互模式向脚本吐事件流",
		confuse: "这一向的协议必须允许反向发起请求，否则审批与代读文件都表达不出来。"
	},
	{
		key: "a2t",
		label: "运行时 → 工具",
		arrow: "Agent 获取能力",
		solves: "发现有哪些工具、参数长什么样、怎么鉴权、结果怎么回填",
		examples: "内置工具；按协议连接的外部工具服务；直接调用的 HTTP 接口",
		confuse: "它与上一向不冲突也不重叠：一个让界面能驱动 Agent，一个让 Agent 能触达外部系统，实践中通常同时存在。"
	},
	{
		key: "a2a",
		label: "运行时 → 运行时",
		arrow: "Agent 之间委派",
		solves: "把一个完整的 Agent 当作一个可被调用的能力暴露出去",
		examples: "把某个运行时包装成工具服务，供另一个运行时调用；主流程派生受限的子执行者",
		confuse: "方向决定归属：同一个运行时既可以是被驱动的一方，也可以是发起调用的一方，看的是这一次连接里谁在等谁。"
	}
]

const cur = ref("protocol")
const dir = ref<string | null>(null)
const curLayer = computed(() => layers.find((l) => l.key === cur.value)!)
const curDir = computed(() => dirs.find((d) => d.key === dir.value) ?? null)

function pickLayer(key: string) {
	cur.value = key
	dir.value = null
}
</script>

<template>
	<div class="ils">
		<p class="ils__hint">
			自上而下是从「一次生成」到「能不能真的动手」。
			<b>不同层之间不是选项关系，而是叠加关系</b>——任何一次真实接入都同时用到了每一层。
		</p>

		<div class="ils__stack">
			<button
				v-for="l in layers"
				:key="l.key"
				type="button"
				class="ils__layer"
				:class="{ 'is-on': cur === l.key && !dir }"
				:style="{ '--c': seriesColor(l.idx) }"
				:aria-pressed="cur === l.key && !dir"
				@click="pickLayer(l.key)"
			>
				<span class="ils__layer-head">
					<strong>{{ l.name }}</strong>
					<i>{{ l.question }}</i>
				</span>
				<span class="ils__chips">
					<span v-for="m in l.members" :key="m" class="ils__chip">{{ m }}</span>
				</span>
			</button>
		</div>

		<div class="ils__dirs">
			<span class="ils__dirs-label cs-mono">协议层的正交维度 · 方向</span>
			<button
				v-for="d in dirs"
				:key="d.key"
				type="button"
				class="ils__dir"
				:class="{ 'is-on': dir === d.key }"
				:aria-pressed="dir === d.key"
				@click="dir = dir === d.key ? null : d.key"
			>{{ d.label }}</button>
		</div>

		<article v-if="curDir" class="ils__panel ils__panel--dir">
			<header>
				<b class="pc-serif">{{ curDir.label }}</b>
				<span class="ils__badge">{{ curDir.arrow }}</span>
			</header>
			<p class="ils__row"><i>解决什么</i>{{ curDir.solves }}</p>
			<p class="ils__row"><i>典型形态</i>{{ curDir.examples }}</p>
			<p class="ils__foot"><i>容易弄反的地方</i>{{ curDir.confuse }}</p>
		</article>

		<article v-else class="ils__panel" :style="{ '--c': seriesColor(curLayer.idx) }">
			<header>
				<b class="pc-serif">{{ curLayer.name }}</b>
				<span class="ils__badge">{{ curLayer.question }}</span>
			</header>
			<p class="ils__row"><i>这一层在做什么</i>{{ curLayer.note }}</p>
			<p class="ils__foot"><i>换掉这一层会怎样</i>{{ curLayer.swap }}</p>
		</article>

		<p class="pc-note">
			分层的用处不在于「记住有五层」，而在于它把一类无法回答的问题消解掉了。
			「伪终端和标准输入输出该选哪个」是传输层内部的选择，
			「会话语义怎么约定」是协议层的选择，两者不构成同一道取舍；
			而<b>隔离层与上面四层完全正交</b>——它可以整层叠加或整层去掉，上面的代码一行不改。
			由此得到一条可以带走的判断：<b>先确认一个名词回答的是哪一层的问题，再谈它和谁并列。</b>
		</p>
	</div>
</template>

<style scoped>
.ils { margin: 1rem 0; }
.ils__hint { margin: 0 0 var(--cs-space-4); font-size: var(--cs-text-sm); line-height: var(--cs-leading-normal); color: var(--cs-color-text-muted); }
.ils__stack { display: grid; gap: var(--cs-space-2); }
.ils__layer { display: grid; gap: var(--cs-space-2); text-align: left; cursor: pointer; min-height: var(--cs-tap-target); padding: var(--cs-space-3) var(--cs-space-5); background: var(--cs-color-bg); border: 1px solid var(--cs-color-border); border-left: 3px solid var(--c); border-radius: var(--cs-radius-lg); transition: var(--cs-transition-colors); }
.ils__layer:hover { background: var(--cs-color-bg-soft); }
.ils__layer.is-on { background: color-mix(in srgb, var(--c) 10%, transparent); border-color: var(--c); }
.ils__layer-head { display: flex; flex-wrap: wrap; align-items: baseline; gap: var(--cs-space-3); }
.ils__layer-head strong { font-size: var(--cs-text-lg); color: var(--cs-color-text); }
.ils__layer-head i { font-size: var(--cs-text-2xs); font-style: normal; color: var(--cs-color-text-subtle); }
.ils__chips { display: flex; flex-wrap: wrap; gap: var(--cs-space-1); }
.ils__chip { font-family: var(--cs-font-mono); font-size: var(--cs-text-3xs); color: var(--cs-color-text-muted); background: var(--cs-color-bg-soft); border: 1px solid var(--cs-color-border); border-radius: var(--cs-radius-xs); padding: 0 var(--cs-space-2); }
.ils__dirs { display: flex; flex-wrap: wrap; align-items: center; gap: var(--cs-space-1); margin-top: var(--cs-space-5); }
.ils__dirs-label { font-size: var(--cs-text-3xs); letter-spacing: .08em; color: var(--cs-color-text-subtle); margin-right: var(--cs-space-2); }
.ils__dir { min-height: var(--cs-tap-target); padding: var(--cs-space-1) var(--cs-space-4); cursor: pointer; font-size: var(--cs-text-xs); color: var(--cs-color-text-muted); background: var(--cs-color-bg); border: 1px solid var(--cs-color-border); border-radius: var(--cs-radius-pill); transition: var(--cs-transition-colors); }
.ils__dir:hover { background: var(--cs-color-bg-soft); }
.ils__dir.is-on { background: var(--cs-color-brand-soft); border-color: var(--cs-color-brand); color: var(--cs-color-brand); font-weight: 600; }
.ils__panel { margin-top: var(--cs-space-5); background: var(--cs-color-bg); border: 1px solid var(--cs-color-border); border-top: 3px solid var(--c); border-radius: var(--cs-radius-lg); padding: var(--cs-space-5) var(--cs-space-6); }
.ils__panel--dir { --c: var(--cs-color-brand); }
.ils__panel header { display: flex; flex-wrap: wrap; align-items: baseline; justify-content: space-between; gap: var(--cs-space-3); margin-bottom: var(--cs-space-3); }
.ils__panel header b { font-size: var(--cs-text-lg); color: var(--c); }
.ils__badge { font-family: var(--cs-font-mono); font-size: var(--cs-text-3xs); color: var(--cs-color-text-subtle); border: 1px solid var(--cs-color-border); border-radius: var(--cs-radius-pill); padding: var(--cs-space-1) var(--cs-space-3); }
.ils__row { margin: 0 0 var(--cs-space-3); font-size: var(--cs-text-base); line-height: var(--cs-leading-relaxed); color: var(--cs-color-text); }
.ils__foot { margin: 0; padding-top: var(--cs-space-3); border-top: 1px dashed var(--cs-color-border); font-size: var(--cs-text-sm); line-height: var(--cs-leading-normal); color: var(--cs-color-text-muted); }
.ils__row i, .ils__foot i { display: block; font-family: var(--cs-font-mono); font-size: var(--cs-text-3xs); letter-spacing: .08em; font-style: normal; color: var(--cs-color-text-subtle); margin-bottom: var(--cs-space-1); }
@media (max-width: 640px) {
	.ils__dirs-label { flex: 1 0 100%; margin-bottom: var(--cs-space-1); }
}
</style>
