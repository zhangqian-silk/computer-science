<script setup lang="ts">
import { computed, ref } from "vue"
import { seriesColor } from "../series-palette"

/**
 * Skill 与相邻机制的边界矩阵。
 *
 * 这张表的正确读法是**纵向**：把同一个维度在七种机制之间拉平对照，
 * 就能看出「该写成什么」并不是偏好问题，而是由几个客观维度共同决定的。
 * 横向读只能得到各机制的定义，得不出判据。
 *
 * 七个对象覆盖了实践中最容易互相顶替的一组：
 * 系统提示、Skill、tool、MCP server、slash command、subagent、RAG 检索。
 */

type Dim = {
	key: string
	label: string
	question: string
}

const dims: Dim[] = [
	{ key: "carrier", label: "复用载体", question: "这份能力以什么形态存在？" },
	{ key: "decider", label: "谁决定调用", question: "本次要不要用它，由谁判断？" },
	{ key: "inject", label: "知识注入方式", question: "内容如何进入模型可见范围？" },
	{ key: "loading", label: "上下文加载策略", question: "什么时候占上下文，占多少？" },
	{ key: "state", label: "状态管理", question: "跨轮次、跨会话是否留下状态？" },
	{ key: "perm", label: "权限与审批", question: "它本身是否授予新的执行能力？" },
	{ key: "port", label: "可移植性", question: "换一个宿主还能不能用？" },
	{ key: "determ", label: "确定性取舍", question: "结果的稳定性由什么保证？" }
]

type Mech = {
	key: string
	name: string
	idx: number
	one: string
	carrier: string
	decider: string
	inject: string
	loading: string
	state: string
	perm: string
	port: string
	determ: string
	/** 什么时候该选它 */
	pick: string
	/** 什么时候不该选它 */
	avoid: string
}

const mechs: Mech[] = [
	{
		key: "sysprompt",
		name: "系统提示",
		idx: 0,
		one: "每一轮都必须生效的全局约定",
		carrier: "请求里的一段固定文本，或磁盘上被每轮重新读取的指令文件",
		decider: "没有人判断——它无条件生效",
		inject: "直接写入，不经检索也不经判断",
		loading: "常驻。占的是每一次调用的固定预算",
		state: "无状态；改动即刻对所有后续轮次生效",
		perm: "不授予任何执行能力，只改变倾向",
		port: "最高：纯文本，任何宿主都能接受",
		determ: "靠「每次都在」换取零漏召，但遵守率随长度下降",
		pick: "命中率接近 100% 的硬约束：安全边界、输出语言、提交前必跑的检查。漏一次就出错的东西不能交给检索。",
		avoid: "只在特定话题下才相关的长材料。堆进系统提示会挤占每一次调用的预算，且越长遵守率越低。"
	},
	{
		key: "skill",
		name: "Skill",
		idx: 1,
		one: "按需加载的操作知识包",
		carrier: "文件系统上的一个目录：SKILL.md 加可选的脚本、参考文档、模板",
		decider: "模型按 description 匹配，或用户显式点名",
		inject: "元信息常驻、正文按需读入、附件再按需读入（渐进披露）",
		loading: "分级：元信息每个约百 token 常驻，正文命中才载入，未读附件为零",
		state: "正文一旦读入即留在历史中；宿主通常不会在后续轮次重读该文件",
		perm: "本身不授予执行能力，所需工具须由环境另行提供",
		port: "较高：同一份 SKILL.md 在多个遵循同一格式的宿主之间基本可用，宿主扩展字段除外",
		determ: "指令部分是概率性的；需要确定性时靠包内脚本，由代码而非生成来保证",
		pick: "一类任务的做法反复被重新交代、且只在这类任务出现时才需要：领域规程、格式规范、多步流程与其检查项。",
		avoid: "需要访问模型环境里根本没有的系统时——那是连通性缺口，写多少指令都补不上。"
	},
	{
		key: "tool",
		name: "tool（函数）",
		idx: 2,
		one: "一次可校验的具体动作",
		carrier: "宿主注册的一份参数模式加一个执行实现",
		decider: "模型提议调用，程序校验并决定是否执行",
		inject: "工具定义常驻在请求里；结果以观察形式回填",
		loading: "定义常驻且通常位于上下文最前部，增删会连带击穿其后的前缀缓存",
		state: "单次调用无状态，副作用留在外部世界",
		perm: "这是权限的实际入口：授权与副作用分级都落在这一层",
		port: "低：模式与调用协议随宿主而异",
		determ: "执行部分完全确定，不确定性只在「选哪个工具、传什么参数」",
		pick: "需要真正做一件事、且这件事的边界能写成参数模式：读文件、发请求、跑命令。",
		avoid: "把一整套多步流程压成一个巨型工具。步骤之间的判断需要观察参与，压进一次调用就失去了纠偏机会。"
	},
	{
		key: "mcp",
		name: "MCP server",
		idx: 3,
		one: "对外部系统的标准化连通层",
		carrier: "一个按协议暴露工具、资源与提示的进程或服务",
		decider: "模型调用其中的工具；是否连接由用户与配置决定",
		inject: "以工具定义与资源的形式进入上下文",
		loading: "常驻。服务器一多，工具定义本身就成为可观的固定开销",
		state: "连接是长期的；服务端可持有自己的会话与凭据",
		perm: "授予新的执行能力，因此需要认证、审批与工具级策略",
		port: "较高：协议是公开的，多个客户端可连同一个服务器",
		determ: "服务端行为确定，模型侧仍需判断何时调用、参数是否正确",
		pick: "模型此刻根本触达不到那个系统：数据库、工单系统、第三方 API、浏览器状态。",
		avoid: "只是想规范「已有工具该怎么用得更好」。那是判断力缺口而非连通性缺口，架一个服务器解决不了。"
	},
	{
		key: "command",
		name: "slash command",
		idx: 4,
		one: "由人在特定时刻手动触发的动作",
		carrier: "一个可被命令名引用的提示片段或技能条目",
		decider: "用户。存在只允许人触发、对模型隐藏的配置",
		inject: "触发时把对应内容展开进上下文",
		loading: "未触发时通常不占（对模型隐藏时连描述也不进上下文）",
		state: "一次触发一次展开；参数不同则重复展开",
		perm: "不授予新能力，但常用于封装有副作用的操作以便控制时机",
		port: "低到中：命名与触发机制随宿主而异",
		determ: "触发时刻完全确定——这正是它相对隐式匹配的全部价值",
		pick: "时机必须由人决定的动作：部署、发布、对外发消息。不希望模型「看代码写好了就自己发」。",
		avoid: "希望模型在合适的时候自己想起来用。只允许人触发意味着模型不会主动使用它。"
	},
	{
		key: "subagent",
		name: "subagent",
		idx: 5,
		one: "一个独立上下文里的受限执行者",
		carrier: "一份独立的系统提示加工具白名单，运行在自己的窗口里",
		decider: "主流程委派，或模型按其描述选择",
		inject: "任务与结论跨窗口传递；过程细节不回流主上下文",
		loading: "隔离：过程不占主窗口，只有回传的结论占",
		state: "自带一次完整的循环状态，结束即回收",
		perm: "可以比主流程更严：常用于把工具集收窄成只读",
		port: "低：定义格式与委派协议随宿主而异",
		determ: "隔离降低了相互干扰，但增加了协调开销与结论合并的不确定性",
		pick: "需要真正隔离的执行边界：大量探索性读取、需要不同权限、可并行的独立调查。",
		avoid: "只是想复用一套做法。做法应当写成可被任何执行者加载的知识包，而不是绑死在某个执行者的提示里。"
	},
	{
		key: "rag",
		name: "RAG 检索",
		idx: 6,
		one: "对外部语料的按需召回",
		carrier: "一份索引加一条检索链路，语料本体在库里",
		decider: "程序或模型发起查询；召回哪几条由排序决定",
		inject: "召回片段作为资料进入本次上下文，用完即弃",
		loading: "按查询计费；命中几条占几条",
		state: "无状态；语料更新即全体生效，不需逐用户改写",
		perm: "不授予执行能力，只提供事实",
		port: "高：与宿主无关，是应用侧的一段链路",
		determ: "召回是概率性的，漏召是静默失败",
		pick: "参数里没有、且对所有使用者都一样的客观事实：产品文档、法规条文、接口手册。",
		avoid: "承载必须每次都生效的规程。检索总有漏召概率，而规程漏一次就错，且错得看起来很合理。"
	}
]

const dim = ref<string>("loading")
const focus = ref<string | null>("skill")
const curDim = computed(() => dims.find((d) => d.key === dim.value)!)
const curMech = computed(() => (focus.value ? mechs.find((m) => m.key === focus.value)! : null))
</script>

<template>
	<div class="sbm">
		<p class="sbm__hint">
			选一个维度，横向比较七种机制在这个维度上的取值——<b>纵向对照才是这张表的用处</b>，
			逐个读定义得不出「该写成什么」的判据。
		</p>

		<div class="sbm__dims">
			<button
				v-for="d in dims"
				:key="d.key"
				type="button"
				class="sbm__dim"
				:class="{ 'is-on': dim === d.key }"
				:aria-pressed="dim === d.key"
				@click="dim = d.key"
			>{{ d.label }}</button>
		</div>

		<p class="sbm__question"><i>这个维度问的是</i>{{ curDim.question }}</p>

		<ul class="sbm__list">
			<li
				v-for="m in mechs"
				:key="m.key"
				:class="{ 'is-focus': focus === m.key }"
				:style="{ '--c': seriesColor(m.idx) }"
			>
				<button type="button" class="sbm__name" :aria-pressed="focus === m.key" @click="focus = focus === m.key ? null : m.key">
					<span class="sbm__dot" aria-hidden="true"></span>
					{{ m.name }}
				</button>
				<span class="sbm__val">{{ m[(curDim.key as keyof Mech)] }}</span>
			</li>
		</ul>

		<article v-if="curMech" class="sbm__card" :style="{ '--c': seriesColor(curMech.idx) }">
			<header>
				<b class="pc-serif">{{ curMech.name }}</b>
				<i>{{ curMech.one }}</i>
			</header>
			<div class="sbm__judge">
				<p class="sbm__pick"><i>该选它</i>{{ curMech.pick }}</p>
				<p class="sbm__avoid"><i>不该选它</i>{{ curMech.avoid }}</p>
			</div>
		</article>

		<p class="pc-note">
			七种机制不是互斥的选项，实践中常常同时存在——真正的问题从来不是「选哪一个」，而是
			<b>同一份能力该落在哪一层</b>。三条最常被弄反的对应关系：连通性缺口归 MCP 或 tool，
			判断力缺口归 Skill，时机控制归命令与审批。把它们互相顶替的代价不是效率损失，而是各自的失败形态会安静地出现：
			该常驻的被检索（漏召一次就错）、该外置的被塞进系统提示（每轮付费且遵守率下降）、
			该隔离的留在主窗口（无关细节挤占预算）。
		</p>
	</div>
</template>

<style scoped>
.sbm { margin: 1rem 0; }
.sbm__hint { margin: 0 0 var(--cs-space-4); font-size: var(--cs-text-sm); line-height: var(--cs-leading-normal); color: var(--cs-color-text-muted); }
.sbm__dims { display: flex; flex-wrap: wrap; gap: var(--cs-space-1); }
.sbm__dim { min-height: var(--cs-tap-target); padding: var(--cs-space-1) var(--cs-space-4); cursor: pointer; font-size: var(--cs-text-xs); color: var(--cs-color-text-muted); background: var(--cs-color-bg); border: 1px solid var(--cs-color-border); border-radius: var(--cs-radius-pill); transition: var(--cs-transition-colors); }
.sbm__dim:hover { background: var(--cs-color-bg-soft); }
.sbm__dim.is-on { background: var(--cs-color-brand-soft); border-color: var(--cs-color-brand); color: var(--cs-color-brand); font-weight: 600; }
.sbm__question { margin: var(--cs-space-4) 0 var(--cs-space-3); font-size: var(--cs-text-base); color: var(--cs-color-text); }
.sbm__question i { display: block; font-family: var(--cs-font-mono); font-size: var(--cs-text-3xs); letter-spacing: .08em; font-style: normal; color: var(--cs-color-text-subtle); margin-bottom: var(--cs-space-1); }
.sbm__list { list-style: none; margin: 0; padding: 0; }
.sbm__list li { display: grid; grid-template-columns: 9rem minmax(0, 1fr); gap: var(--cs-space-3); align-items: start; padding: 0 0 0 var(--cs-space-3); border-left: 3px solid var(--c); border-bottom: 1px dashed var(--cs-color-border); }
.sbm__list li:last-child { border-bottom: 0; }
.sbm__list li.is-focus { background: color-mix(in srgb, var(--c) 8%, transparent); }
.sbm__name { display: inline-flex; align-items: center; gap: var(--cs-space-2); min-height: var(--cs-tap-target); cursor: pointer; background: none; border: 0; padding: 0; text-align: left; font-size: var(--cs-text-sm); font-weight: 600; color: var(--cs-color-text); }
.sbm__dot { width: 8px; height: 8px; border-radius: var(--cs-radius-circle); background: var(--c); flex: none; }
.sbm__val { font-size: var(--cs-text-sm); line-height: var(--cs-leading-normal); color: var(--cs-color-text-muted); padding: var(--cs-space-3) 0; }
.sbm__card { margin-top: var(--cs-space-5); background: var(--cs-color-bg); border: 1px solid var(--cs-color-border); border-top: 3px solid var(--c); border-radius: var(--cs-radius-lg); padding: var(--cs-space-5) var(--cs-space-6); }
.sbm__card header { display: flex; flex-wrap: wrap; align-items: baseline; gap: var(--cs-space-3); margin-bottom: var(--cs-space-4); }
.sbm__card header b { font-size: var(--cs-text-xl); color: var(--c); }
.sbm__card header i { font-size: var(--cs-text-2xs); font-style: normal; color: var(--cs-color-text-subtle); }
.sbm__judge { display: grid; gap: var(--cs-space-4); }
.sbm__pick, .sbm__avoid { margin: 0; padding: var(--cs-space-3) var(--cs-space-4); border-radius: var(--cs-radius-sm); font-size: var(--cs-text-base); line-height: var(--cs-leading-relaxed); }
.sbm__pick { background: var(--cs-color-success-soft); }
.sbm__avoid { background: var(--cs-color-danger-soft); }
.sbm__pick i, .sbm__avoid i { display: block; font-family: var(--cs-font-mono); font-size: var(--cs-text-3xs); letter-spacing: .08em; font-style: normal; margin-bottom: var(--cs-space-1); }
.sbm__pick i { color: var(--cs-color-success); }
.sbm__avoid i { color: var(--cs-color-danger); }
@media (min-width: 861px) {
	.sbm__judge { grid-template-columns: 1fr 1fr; }
}
@media (max-width: 640px) {
	.sbm__list li { grid-template-columns: 1fr; gap: 0; }
	.sbm__val { padding-top: 0; padding-bottom: var(--cs-space-3); }
}
</style>
