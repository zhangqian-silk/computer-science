<script setup lang="ts">
import { computed, ref } from "vue"

/**
 * 六组容易混为一谈的对象，逐组给出「它们各是什么 → 区分依据 → 混淆之后的具体代价」。
 *
 * 设计上刻意不做成一张对照表：每组的区分依据都不同，
 * 表格会把这种差异压平成同一种句式，反而看不出各组的要点在哪。
 */

type Pair = {
	key: string
	a: string
	b: string
	aIs: string
	bIs: string
	basis: string
	cost: string
	tell: string
}

const pairs: Pair[] = [
	{
		key: "wm-window",
		a: "工作记忆",
		b: "上下文窗口",
		aIs: "本次调用中模型实际可用的那些信息——一个内容概念。",
		bIs: "模型一次能接收的 token 上限——一个容量参数。",
		basis: "一个说「装了什么」，一个说「能装多少」。前者是内容，后者是容量。",
		cost: "混淆之后会得出「窗口够大就不用治理」这个结论。窗口大只意味着装得下，不意味着模型能同等地用好其中每一部分。",
		tell: "问一句「它有单位吗」：能用 token 计量的是窗口，不能的是工作记忆。"
	},
	{
		key: "ltm-kb",
		a: "长期记忆",
		b: "知识库",
		aIs: "这个用户 / 这个项目在交互中积累下来的东西，来源是经历。",
		bIs: "对所有用户都一样的领域内容，来源是外部权威文档。",
		basis: "看内容因人而异还是对所有人一样，以及来源是交互还是外部文档。",
		cost: "把领域文档抽成每个用户各存一份的记忆，会导致文档更新时记忆不跟着更新；反过来把用户偏好塞进公共知识库，会让一个人的偏好影响到所有人。",
		tell: "问「换一个用户，这条还成立吗」：成立的是知识，不成立的是记忆。"
	},
	{
		key: "rag-recall",
		a: "检索增强",
		b: "记忆召回",
		aIs: "为了补足模型参数里没有的客观知识而去外部检索。",
		bIs: "为了恢复此前交互中形成的状态与结论而去记忆存储检索。",
		basis: "两者的读路径几乎一样（都是查询、排序、截断），区别在「检索目标」：一个找「关于世界的」，一个找「关于你的」。",
		cost: "实现上把两者合成一路，会出现两种典型故障：用户偏好与领域文档在同一个排序里竞争名额，偏好因为「相关度不够高」被挤掉；以及知识库文档被误判为可覆盖写，被记忆更新逻辑改掉。",
		tell: "问「这条内容有没有一个外部的、可追溯的源」：有的走检索增强，没有的走记忆召回。"
	},
	{
		key: "prompt-proc",
		a: "系统提示里的规则",
		b: "程序性记忆",
		aIs: "在请求里手工写死的行为约束，每次调用原样带上。",
		bIs: "「这件事该怎么做」这类知识本身，可以来自人工编写，也可以由系统在犯错后自行沉淀。",
		basis: "系统提示是「载体」，程序性记忆是「内容类型」。同一条规程既可以放在系统提示里，也可以放在磁盘文件里、放在可检索的规则库里。",
		cost: "把两者当成一回事，会得出「程序性记忆就是写系统提示」的结论，于是所有规程都往系统提示里堆，固定预算越来越大而遵守率反而下降；也会想不到「规程可以按文件路径作用域按需加载」这类做法。",
		tell: "载体与内容是两个问题：先问「这是什么类型的知识」，再问「放在哪个载体上」。"
	},
	{
		key: "session-store",
		a: "会话状态持久化",
		b: "跨会话记忆存储",
		aIs: "把一个会话的完整状态存下来，以便中断后恢复、回放、审计。按会话隔离。",
		bIs: "把跨会话仍然成立的结论存下来，以便换一个会话也能用。按用户 / 项目隔离。",
		basis: "隔离维度不同：一个按会话，一个按主体。前者要求「完整可恢复」，后者要求「精炼可召回」。",
		cost: "只做前者，换个会话就一无所知；只做后者，进程一崩当前任务就无法恢复。更隐蔽的错误是用同一套设施承担两件事：为了可恢复而保存完整历史，结果跨会话召回时把整段旧对话都捞回来。",
		tell: "问「换一个会话还要不要它」：要的进记忆存储，只为恢复本次的留在会话状态里。"
	},
	{
		key: "doc-episodic",
		a: "知识库文档",
		b: "情景记忆",
		aIs: "对事物的一般性描述：这个 API 怎么用、这个组件做什么。",
		bIs: "一次具体经历的记录：那天试了什么、结果如何、最后怎么定的。",
		basis: "看有没有绑定到一个具体的时间点与一次具体的事件。文档说「通常如此」，情景记忆说「那次这样」。",
		cost: "把情景记忆当文档用会过度泛化——「上次这个方案不行」变成「这个方案不行」，而它当时不行可能只是因为当时的特定条件；反过来把文档存成情景记忆，会让本来长期有效的内容背上一个无谓的时间戳，被时效衰减误伤。",
		tell: "问「它需要带时间戳吗」：需要的是情景，不需要的是文档。"
	}
]

const idx = ref(2)
const cur = computed(() => pairs[idx.value])
</script>

<template>
	<div class="mdp">
		<div class="mdp__tabs" role="tablist" aria-label="选择一组容易混淆的对象">
			<button
				v-for="(p, i) in pairs"
				:key="p.key"
				type="button"
				role="tab"
				:aria-selected="idx === i"
				class="mdp__tab"
				:class="{ 'is-on': idx === i }"
				@click="idx = i"
			>
				<span>{{ p.a }}</span>
				<em aria-hidden="true">vs</em>
				<span>{{ p.b }}</span>
			</button>
		</div>

		<div class="cs-split mdp__body">
			<div class="cs-split__figure mdp__sides">
				<section class="mdp__side mdp__side--a">
					<b>{{ cur.a }}</b>
					<p>{{ cur.aIs }}</p>
				</section>
				<section class="mdp__side mdp__side--b">
					<b>{{ cur.b }}</b>
					<p>{{ cur.bIs }}</p>
				</section>
			</div>
			<div class="cs-split__data mdp__notes">
				<p class="mdp__basis"><i>区分依据</i>{{ cur.basis }}</p>
				<p class="mdp__cost"><i>混淆的代价</i>{{ cur.cost }}</p>
				<p class="mdp__tell"><i>一句话判别</i>{{ cur.tell }}</p>
			</div>
		</div>
	</div>
</template>

<style scoped>
.mdp { margin: 1rem 0; }
.mdp__tabs { display: flex; flex-wrap: wrap; gap: var(--cs-space-1); margin-bottom: var(--cs-space-4); }
.mdp__tab { display: inline-flex; align-items: center; gap: var(--cs-space-2); min-height: var(--cs-tap-target); padding: var(--cs-space-1) var(--cs-space-3); cursor: pointer; background: var(--cs-color-bg); border: 1px solid var(--cs-color-border); border-radius: var(--cs-radius-sm); transition: var(--cs-transition-colors); }
.mdp__tab:hover { background: var(--cs-color-bg-soft); }
.mdp__tab.is-on { background: var(--cs-color-brand-soft); border-color: var(--cs-color-brand); }
.mdp__tab span { font-size: var(--cs-text-xs); color: var(--cs-color-text); }
.mdp__tab em { font-family: var(--cs-font-mono); font-size: var(--cs-text-3xs); font-style: normal; color: var(--cs-color-text-subtle); }
.mdp__tab.is-on span { color: var(--cs-color-brand); font-weight: 600; }
.mdp__sides { display: grid; gap: var(--cs-space-3); }
.mdp__side { padding: var(--cs-space-4) var(--cs-space-5); border-radius: var(--cs-radius-lg); border: 1px solid var(--cs-color-border); border-left: 3px solid var(--c); background: var(--cs-color-bg); }
.mdp__side--a { --c: var(--cs-color-success); }
.mdp__side--b { --c: var(--cs-color-info); }
.mdp__side b { display: block; font-size: var(--cs-text-md); color: var(--c); margin-bottom: var(--cs-space-2); }
.mdp__side p { margin: 0; font-size: var(--cs-text-sm); line-height: var(--cs-leading-relaxed); color: var(--cs-color-text); }
.mdp__notes { display: grid; gap: var(--cs-space-3); align-content: start; }
.mdp__notes p { margin: 0; font-size: var(--cs-text-sm); line-height: var(--cs-leading-relaxed); }
.mdp__notes i { display: block; font-family: var(--cs-font-mono); font-size: var(--cs-text-3xs); letter-spacing: .08em; font-style: normal; color: var(--cs-color-text-subtle); margin-bottom: var(--cs-space-1); }
.mdp__basis { color: var(--cs-color-text); }
.mdp__cost { padding: var(--cs-space-3) var(--cs-space-4); background: var(--cs-color-warning-soft); border-radius: var(--cs-radius-sm); color: var(--cs-color-text); }
.mdp__tell { padding-top: var(--cs-space-3); border-top: 1px dashed var(--cs-color-border); color: var(--cs-color-text-muted); }
</style>
