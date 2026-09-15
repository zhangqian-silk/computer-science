<script setup lang="ts">
import { computed, ref } from "vue"
import { seriesColor } from "../series-palette"

/**
 * 六种接入面的能力对照。
 *
 * 读法是纵向的：选一个维度，把六种接入面在这个维度上的取值拉平比较，
 * 就能看出「该选哪种」并非偏好问题，而是由几个客观维度共同决定的。
 * 逐个读定义只能得到各自的说明，得不出判据。
 *
 * 六个对象覆盖了实践中最容易互相顶替的一组接入形态；
 * 隔离不在其列，因为它与这六种全部正交，不构成同一道选择。
 */

type Dim = {
	key: string
	label: string
	question: string
}

const dims: Dim[] = [
	{ key: "shape", label: "事件形态", question: "拿到的是什么粒度的信息？" },
	{ key: "approve", label: "审批能力", question: "运行时需要问人时，这条通道能表达吗？" },
	{ key: "state", label: "状态精确度", question: "「现在进行到哪一步」是读来的还是猜的？" },
	{ key: "session", label: "会话与续跑", question: "跨调用怎么接上一次的进度？" },
	{ key: "coupling", label: "耦合面", question: "对方改动什么会把你打断？" },
	{ key: "cost", label: "接入代价", question: "第一次跑通要写多少东西？" }
]

type Surface = {
	key: string
	name: string
	idx: number
	one: string
	shape: string
	approve: string
	state: string
	session: string
	coupling: string
	cost: string
	pick: string
	avoid: string
}

const surfaces: Surface[] = [
	{
		key: "own",
		name: "自建循环",
		idx: 0,
		one: "直接对着模型接口写自己的运行时",
		shape: "由自己定义。每一个中间态都是自己产生的，想暴露什么就暴露什么",
		approve: "完全自定义，因为放行判断本来就写在自己代码里",
		state: "最高：状态就是自己的变量，不存在解释歧义",
		session: "自己设计存储与恢复点，代价是要自己保证可恢复",
		coupling: "只绑模型接口。模型可替换，但接口本身变更要自己跟",
		cost: "最高。压缩、编辑应用、路径校验、重试、记账都要从零写一遍",
		pick: "产品形态需要深度定制、权限体系要贴自己的组织结构、或者模型必须可替换。",
		avoid: "只是想让某类任务自动跑起来。成熟运行时里最值钱的恰恰是那些细节，重写一遍很难更好。"
	},
	{
		key: "sdk",
		name: "官方 SDK",
		idx: 1,
		one: "厂商对自家运行时的语言级封装",
		shape: "类型化的事件与生命周期回调，字段含义由 SDK 保证",
		approve: "通常提供回调点，可在动作执行前介入",
		state: "较高：状态由 SDK 维护并暴露，不必自己拼",
		session: "多由 SDK 托管，恢复接口通常现成",
		coupling: "最紧。绑语言、绑厂商、绑版本，升级可能带来不兼容改动",
		cost: "低。但排障时往往要能落到下一层去验证",
		pick: "在后端服务里嵌一个 Agent 能力，团队要快速上线且接受厂商绑定。",
		avoid: "需要跨多个运行时统一接入。SDK 天然是一家一套，接第二家就要再写一遍。"
	},
	{
		key: "headless",
		name: "命令行非交互",
		idx: 2,
		one: "以子进程方式跑命令行，读结构化事件流",
		shape: "按行分隔的结构化事件，粒度取决于开关是否打开",
		approve: "弱。多依赖预授权或非交互策略，中途问人这件事很难表达",
		state: "中：事件是真实的，但事件集合与含义随版本变化",
		session: "进程级。多轮要靠会话标识续跑，或每次重新开始",
		coupling: "绑参数与事件字段。两者都属于弱契约，升级会变",
		cost: "最低。一条命令加一个按行解析就能跑通",
		pick: "持续集成、批量任务、定时作业——没有人在旁边看着，成败用退出码表达就够。",
		avoid: "需要中途问人的交互场景。把审批塞进一条单向事件流会得到一个假的审批。"
	},
	{
		key: "pty",
		name: "伪终端驱动",
		idx: 3,
		one: "分配伪终端，像人一样喂按键、读屏幕",
		shape: "字节流。要先渲染成屏幕网格，才谈得上「读到了什么」",
		approve: "只能靠识别画面再喂按键，可靠性取决于界面是否改版",
		state: "最低：状态是从画面推断出来的，不是读来的",
		session: "由被驱动的程序自己管，外部只能观察",
		coupling: "不绑任何契约，但绑界面外观——改版即失效",
		cost: "中。终端模拟、尺寸同步、完成判定各自都是一摊工程",
		pick: "对方没有任何结构化入口，或者本来就是要给人看一个真实的终端画面。",
		avoid: "有结构化入口时还用它。用推断替换可读的事实，是自己给自己制造不确定性。"
	},
	{
		key: "rpc",
		name: "长连接双向调用",
		idx: 4,
		one: "有状态的长连接，两端都能发起请求",
		shape: "结构化的请求、响应与通知，覆盖运行时的完整能力面",
		approve: "原生。运行时向界面发起请求并等待回答，这正是双向的用处",
		state: "最高：状态由协议显式表达，客户端只负责渲染",
		session: "会话是一等概念，通常连恢复与列举都在协议里",
		coupling: "绑某一家的方法集。能力最全，但换运行时要重写绑定",
		cost: "较高。要实现客户端侧的调用绑定，包括处理入站请求",
		pick: "自建界面且需要审批与精确状态：编辑器插件、团队面板、桌面端。",
		avoid: "以为它像无状态接口那样可以随意扩容。会话是有状态的，多租户要另做一层。"
	},
	{
		key: "standard",
		name: "标准化编辑器协议",
		idx: 5,
		one: "面向「编辑器接 Agent」的公开协议",
		shape: "结构化通知承载全部流式内容：消息分片、思考、工具调用、计划",
		approve: "原生，且属于客户端必须实现的基线能力",
		state: "较高：由协议规定，跨实现一致",
		session: "新建是基线，恢复属于要声明的可选能力",
		coupling: "最低。一次实现对多家生效，能力差异靠协商而非分支代码",
		cost: "中。协议本身要学，但接第二家的边际成本接近零",
		pick: "要让一个运行时接入多个界面，或让一个界面接入多个运行时——边际成本随对接数量下降。",
		avoid: "只对接一家且要用到对方全部特有能力。标准协议覆盖公共面，独有能力得靠扩展机制自己接。"
	}
]

const dim = ref("approve")
const focus = ref<string | null>("standard")
const curDim = computed(() => dims.find((d) => d.key === dim.value)!)
const curSurface = computed(() => (focus.value ? surfaces.find((s) => s.key === focus.value)! : null))
</script>

<template>
	<div class="ism">
		<p class="ism__hint">
			选一个维度，横向比较六种接入面的取值——<b>纵向对照才是这张表的用处</b>。
			隔离不在这六项之内：它与全部六种正交，不构成同一道选择。
		</p>

		<div class="ism__dims">
			<button
				v-for="d in dims"
				:key="d.key"
				type="button"
				class="ism__dim"
				:class="{ 'is-on': dim === d.key }"
				:aria-pressed="dim === d.key"
				@click="dim = d.key"
			>{{ d.label }}</button>
		</div>

		<p class="ism__question"><i>这个维度问的是</i>{{ curDim.question }}</p>

		<ul class="ism__list">
			<li
				v-for="s in surfaces"
				:key="s.key"
				:class="{ 'is-focus': focus === s.key }"
				:style="{ '--c': seriesColor(s.idx) }"
			>
				<button
					type="button"
					class="ism__name"
					:aria-pressed="focus === s.key"
					@click="focus = focus === s.key ? null : s.key"
				>
					<span class="ism__dot" aria-hidden="true"></span>
					{{ s.name }}
				</button>
				<span class="ism__val">{{ s[(curDim.key as keyof Surface)] }}</span>
			</li>
		</ul>

		<article v-if="curSurface" class="ism__card" :style="{ '--c': seriesColor(curSurface.idx) }">
			<header>
				<b class="pc-serif">{{ curSurface.name }}</b>
				<i>{{ curSurface.one }}</i>
			</header>
			<div class="ism__judge">
				<p class="ism__pick"><i>该选它</i>{{ curSurface.pick }}</p>
				<p class="ism__avoid"><i>不该选它</i>{{ curSurface.avoid }}</p>
			</div>
		</article>

		<p class="pc-note">
			把「审批能力」与「状态精确度」两个维度连起来看，会得到本表最有用的一条观察：
			<b>这两列的排序几乎完全一致</b>，而且都由同一件事决定——通道是不是双向的。
			单向事件流拿不到审批，是因为它没有让运行时向界面发问的语法；
			读屏幕拿不到精确状态，是因为它读的是渲染结果而不是状态本身。
			于是选型的第一问不是「哪个更好」，而是<b>这次接入需不需要中途问人</b>：
			需要，就只能在双向那一侧挑；不需要，最省的那一种通常就够。
		</p>
	</div>
</template>

<style scoped>
.ism { margin: 1rem 0; }
.ism__hint { margin: 0 0 var(--cs-space-4); font-size: var(--cs-text-sm); line-height: var(--cs-leading-normal); color: var(--cs-color-text-muted); }
.ism__dims { display: flex; flex-wrap: wrap; gap: var(--cs-space-1); }
.ism__dim { min-height: var(--cs-tap-target); padding: var(--cs-space-1) var(--cs-space-4); cursor: pointer; font-size: var(--cs-text-xs); color: var(--cs-color-text-muted); background: var(--cs-color-bg); border: 1px solid var(--cs-color-border); border-radius: var(--cs-radius-pill); transition: var(--cs-transition-colors); }
.ism__dim:hover { background: var(--cs-color-bg-soft); }
.ism__dim.is-on { background: var(--cs-color-brand-soft); border-color: var(--cs-color-brand); color: var(--cs-color-brand); font-weight: 600; }
.ism__question { margin: var(--cs-space-4) 0 var(--cs-space-3); font-size: var(--cs-text-base); color: var(--cs-color-text); }
.ism__question i { display: block; font-family: var(--cs-font-mono); font-size: var(--cs-text-3xs); letter-spacing: .08em; font-style: normal; color: var(--cs-color-text-subtle); margin-bottom: var(--cs-space-1); }
.ism__list { list-style: none; margin: 0; padding: 0; }
.ism__list li { display: grid; grid-template-columns: 9rem minmax(0, 1fr); gap: var(--cs-space-3); align-items: start; padding: 0 0 0 var(--cs-space-3); border-left: 3px solid var(--c); border-bottom: 1px dashed var(--cs-color-border); }
.ism__list li:last-child { border-bottom: 0; }
.ism__list li.is-focus { background: color-mix(in srgb, var(--c) 8%, transparent); }
.ism__name { display: inline-flex; align-items: center; gap: var(--cs-space-2); min-height: var(--cs-tap-target); cursor: pointer; background: none; border: 0; padding: 0; text-align: left; font-size: var(--cs-text-sm); font-weight: 600; color: var(--cs-color-text); }
.ism__dot { width: 8px; height: 8px; border-radius: var(--cs-radius-circle); background: var(--c); flex: none; }
.ism__val { font-size: var(--cs-text-sm); line-height: var(--cs-leading-normal); color: var(--cs-color-text-muted); padding: var(--cs-space-3) 0; }
.ism__card { margin-top: var(--cs-space-5); background: var(--cs-color-bg); border: 1px solid var(--cs-color-border); border-top: 3px solid var(--c); border-radius: var(--cs-radius-lg); padding: var(--cs-space-5) var(--cs-space-6); }
.ism__card header { display: flex; flex-wrap: wrap; align-items: baseline; gap: var(--cs-space-3); margin-bottom: var(--cs-space-4); }
.ism__card header b { font-size: var(--cs-text-xl); color: var(--c); }
.ism__card header i { font-size: var(--cs-text-2xs); font-style: normal; color: var(--cs-color-text-subtle); }
.ism__judge { display: grid; gap: var(--cs-space-4); }
.ism__pick, .ism__avoid { margin: 0; padding: var(--cs-space-3) var(--cs-space-4); border-radius: var(--cs-radius-sm); font-size: var(--cs-text-base); line-height: var(--cs-leading-relaxed); }
.ism__pick { background: var(--cs-color-success-soft); }
.ism__avoid { background: var(--cs-color-danger-soft); }
.ism__pick i, .ism__avoid i { display: block; font-family: var(--cs-font-mono); font-size: var(--cs-text-3xs); letter-spacing: .08em; font-style: normal; margin-bottom: var(--cs-space-1); }
.ism__pick i { color: var(--cs-color-success); }
.ism__avoid i { color: var(--cs-color-danger); }
@media (min-width: 861px) {
	.ism__judge { grid-template-columns: 1fr 1fr; }
}
@media (max-width: 640px) {
	.ism__list li { grid-template-columns: 1fr; gap: 0; }
	.ism__val { padding-top: 0; padding-bottom: var(--cs-space-3); }
}
</style>
