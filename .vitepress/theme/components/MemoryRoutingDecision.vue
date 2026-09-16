<script setup lang="ts">
import { computed, ref } from "vue"

/**
 * 归属判断器：拿到一条信息，它该进 memory / context / knowledge 的哪一套。
 *
 * 判断链刻意设计成四个可回答的问题，而不是一张分类表——因为实际困难从来不是
 * 「知道有三类」，而是「这条到底算哪类」。每个样例给出正确归属、判断路径，
 * 以及放错地方的具体后果（后果比归属本身更有说服力）。
 */

type Home = "context" | "memory" | "knowledge" | "prompt"

type Item = {
	id: string
	text: string
	home: Home
	path: string[]
	why: string
	ifWrong: string
}

const homeMeta: Record<Home, { label: string; sub: string; color: string }> = {
	context: { label: "只放本次 context", sub: "用完即弃，不固化", color: "var(--cs-color-brand)" },
	memory: { label: "写入长期 memory", sub: "跨会话存续", color: "var(--cs-color-success)" },
	knowledge: { label: "进 knowledge 库", sub: "外部权威内容", color: "var(--cs-color-info)" },
	prompt: { label: "写进系统提示或规程文件", sub: "每次常驻，不靠召回", color: "var(--cs-color-warning)" }
}

const items: Item[] = [
	{
		id: "i1",
		text: "「把这个函数的返回值改成 error」",
		home: "context",
		path: ["作用域只限本次任务", "→ 不必跨会话存续"],
		why: "一次性任务指令。它的全部价值在本次任务完成之前，完成之后再保留只会造成干扰。",
		ifWrong: "写进 memory 会变成一条永久规程：以后每次改函数它都想加 error 返回值。"
	},
	{
		id: "i2",
		text: "「我们团队所有服务都用 error 返回值，不用 panic」",
		home: "prompt",
		path: ["跨会话有效", "→ 几乎每次都相关", "→ 常驻比召回可靠"],
		why: "这是团队级的程序性约定，命中率接近 100%。对于几乎每次都要生效的规则，常驻注入比检索召回可靠——检索总有漏召的概率，而这类规则漏一次就出错。",
		ifWrong: "放进检索式 memory，会在某次没被召回时静默失效，而且很难发现——输出看起来是合理的，只是不符合团队约定。"
	},
	{
		id: "i3",
		text: "「用户上周说过更喜欢先看结论」",
		home: "memory",
		path: ["跨会话有效", "→ 因人而异", "→ 内容源自与这个人的交互"],
		why: "稳定偏好：跨会话有效，但只对这个用户成立，且它的来源就是交互本身——没有任何外部文档能提供它。",
		ifWrong: "只留在 context 里，下个会话就没了，用户得反复交代同一件事。"
	},
	{
		id: "i4",
		text: "某框架 v3 的 API 变更说明文档",
		home: "knowledge",
		path: ["跨会话有效", "→ 不因人而异", "→ 有外部权威源"],
		why: "客观领域事实，对所有用户都一样，且有可信的外部来源与版本。它属于知识库，不属于任何用户的记忆。",
		ifWrong: "抽成 memory 条目会带来两个问题：文档更新时记忆不会跟着更新，而且每个用户各存一份完全相同的内容。"
	},
	{
		id: "i5",
		text: "「上次试过在网关层做重试，压测下超时反而上升，回滚了」",
		home: "memory",
		path: ["跨会话有效", "→ 因项目而异", "→ 记录的是一次发生过的经历"],
		why: "情景经历。它不是普适知识（在别的架构下重试可能是对的），也不是当前状态，而是「这个项目里试过、结果如此」。这类信息只有经历过的一方才有。",
		ifWrong: "不记就会重复踩坑，而且每次都花同样的时间才发现结论；记成普适规则则会过度泛化成「重试有害」。"
	},
	{
		id: "i6",
		text: "刚 grep 出来的 40 行调用点清单",
		home: "context",
		path: ["作用域只限本次任务", "→ 可以随时重新取得"],
		why: "派生物：随时能用同一条命令重新生成，而且代码一改它就过期。这类内容的正确形态是保留一个能重取它的句柄（那条命令本身），而不是保留结果。",
		ifWrong: "写进 memory 会成为一份很快过期的清单，而且它「看起来很具体」，所以特别容易被后续推理当成当前事实使用。"
	},
	{
		id: "i7",
		text: "「生产库密码是 …」",
		home: "context",
		path: ["跨会话可能有用", "→ 但属于凭据"],
		why: "凭据不进任何持久化记忆。判据不是「有没有用」而是「泄露代价」：记忆是设计上可被检索出来的，凡是写进去的都要假定会被读回。",
		ifWrong: "写进 memory 就等于把凭据存进一个以「便于召回」为目标的系统里，且可能出现在此后任意一次响应中。"
	},
	{
		id: "i8",
		text: "「本次会话临时把超时调成 60 秒试一下」",
		home: "context",
		path: ["作用域只限本次任务", "→ 用户已声明是临时"],
		why: "用户自己给出了作用域标记（「本次」「临时」「试一下」）。这类措辞是判断作用域最可靠的信号，比任何启发式规则都准。",
		ifWrong: "记成长期配置事实，之后 Agent 会以为超时就是 60 秒，并据此推理——这是「把意图当成现状」的一种。"
	}
]

const questions = [
	"这条信息在本次任务结束之后还有用吗？",
	"它是因人／因项目而异，还是对所有人都一样？",
	"移除之后能不能从别处重新取得？",
	"它几乎每次都相关，还是只在特定话题下相关？"
]

const idx = ref(0)
const cur = computed(() => items[idx.value])
const guess = ref<Home | null>(null)
const homes: Home[] = ["context", "memory", "knowledge", "prompt"]

function pick(h: Home) {
	guess.value = h
}
function go(i: number) {
	idx.value = i
	guess.value = null
}
</script>

<template>
	<div class="mrt">
		<div class="mrt__questions">
			<span class="mrt__q-lab">判断链</span>
			<ol>
				<li v-for="q in questions" :key="q">{{ q }}</li>
			</ol>
		</div>

		<div class="mrt__nav">
			<button
				v-for="(it, i) in items"
				:key="it.id"
				type="button"
				class="mrt__nav-btn"
				:class="{ 'is-on': idx === i }"
				:aria-label="`第 ${i + 1} 条信息`"
				:aria-pressed="idx === i"
				@click="go(i)"
			>{{ i + 1 }}</button>
		</div>

		<div class="mrt__card">
			<p class="mrt__item"><i>拿到这条信息</i>{{ cur.text }}</p>

			<div class="mrt__homes">
				<button
					v-for="h in homes"
					:key="h"
					type="button"
					class="mrt__home"
					:class="{
						'is-picked': guess === h,
						'is-right': guess !== null && h === cur.home,
						'is-wrong': guess === h && h !== cur.home
					}"
					:style="{ '--c': homeMeta[h].color }"
					@click="pick(h)"
				>
					<strong>{{ homeMeta[h].label }}</strong>
					<span>{{ homeMeta[h].sub }}</span>
				</button>
			</div>

			<div v-if="guess !== null" class="mrt__answer">
				<p class="mrt__verdict" :class="guess === cur.home ? 'cs-state--pass' : 'cs-state--warn'">
					归属是<b>{{ homeMeta[cur.home].label }}</b>
					<span v-if="guess !== cur.home">，你选的是「{{ homeMeta[guess].label }}」</span>
				</p>
				<ul class="mrt__path">
					<li v-for="p in cur.path" :key="p">{{ p }}</li>
				</ul>
				<p class="mrt__why"><i>判据</i>{{ cur.why }}</p>
				<p class="mrt__wrong"><i>放错地方会怎样</i>{{ cur.ifWrong }}</p>
			</div>
			<p v-else class="mrt__hint">先按上面的判断链自己走一遍，再选一个归属。</p>
		</div>

		<p class="pc-note">
			八条样例里出现了四个归属而不是三个：<b>「写进系统提示或规程文件」是长期记忆的一种落地形态，
			但它不走召回路径</b>。这个区分很实用——判定一条信息「该跨会话存续」只是第一步，
			接着还要判断它是「几乎每次都相关」还是「只在特定话题下相关」：前者常驻更可靠，后者才值得进检索。
			把高命中率的硬约束放进检索式记忆，是一个安静的错误：它平时都对，只在漏召的那一次出错。
		</p>
	</div>
</template>

<style scoped>
.mrt { margin: 1rem 0; }
.mrt__questions { padding: var(--cs-space-3) var(--cs-space-4); background: var(--cs-color-bg-soft); border-radius: var(--cs-radius-lg); margin-bottom: var(--cs-space-4); }
.mrt__q-lab { font-family: var(--cs-font-mono); font-size: var(--cs-text-3xs); letter-spacing: .08em; color: var(--cs-color-text-subtle); }
.mrt__questions ol { margin: var(--cs-space-2) 0 0; padding-left: var(--cs-space-6); }
.mrt__questions li { font-size: var(--cs-text-sm); line-height: var(--cs-leading-normal); color: var(--cs-color-text-muted); }
.mrt__nav { display: flex; flex-wrap: wrap; gap: var(--cs-space-1); margin-bottom: var(--cs-space-3); }
.mrt__nav-btn { min-width: var(--cs-tap-target); min-height: var(--cs-tap-target); cursor: pointer; font-family: var(--cs-font-mono); font-size: var(--cs-text-xs); color: var(--cs-color-text-muted); background: var(--cs-color-bg); border: 1px solid var(--cs-color-border); border-radius: var(--cs-radius-sm); transition: var(--cs-transition-colors); }
.mrt__nav-btn:hover { background: var(--cs-color-bg-soft); }
.mrt__nav-btn.is-on { background: var(--cs-color-brand); border-color: var(--cs-color-brand); color: var(--cs-color-on-brand); font-weight: 600; }
.mrt__card { background: var(--cs-color-bg); border: 1px solid var(--cs-color-border); border-radius: var(--cs-radius-lg); padding: var(--cs-space-5) var(--cs-space-6); }
.mrt__item { margin: 0 0 var(--cs-space-4); padding: var(--cs-space-3) var(--cs-space-4); background: var(--cs-color-brand-soft); border-radius: var(--cs-radius-sm); font-size: var(--cs-text-md); line-height: var(--cs-leading-normal); color: var(--cs-color-text); }
.mrt__item i { display: block; font-family: var(--cs-font-mono); font-size: var(--cs-text-3xs); letter-spacing: .08em; font-style: normal; color: var(--cs-color-text-subtle); margin-bottom: var(--cs-space-1); }
.mrt__homes { display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: var(--cs-space-2); }
.mrt__home { display: grid; gap: var(--cs-space-1); text-align: left; cursor: pointer; min-height: var(--cs-tap-target); padding: var(--cs-space-3); background: var(--cs-color-bg); border: 1px solid var(--cs-color-border); border-top: 3px solid var(--c); border-radius: var(--cs-radius-sm); transition: var(--cs-transition-colors); }
.mrt__home:hover { background: var(--cs-color-bg-soft); }
.mrt__home.is-picked { background: var(--cs-color-bg-soft); border-color: var(--cs-color-border-strong); }
.mrt__home.is-right { border-color: var(--cs-color-success); background: var(--cs-color-success-soft); }
.mrt__home.is-wrong { border-color: var(--cs-color-danger); background: var(--cs-color-danger-soft); }
.mrt__home strong { font-size: var(--cs-text-sm); line-height: var(--cs-leading-tight); }
.mrt__home span { font-size: var(--cs-text-3xs); color: var(--cs-color-text-muted); }
.mrt__answer { margin-top: var(--cs-space-4); }
.mrt__verdict { margin: 0; padding: var(--cs-space-3) var(--cs-space-4); border-radius: var(--cs-radius-sm); font-size: var(--cs-text-md); }
.mrt__verdict span { font-size: var(--cs-text-sm); color: var(--cs-color-text-muted); }
.mrt__path { list-style: none; display: flex; flex-wrap: wrap; gap: var(--cs-space-2); margin: var(--cs-space-3) 0; padding: 0; }
.mrt__path li { font-family: var(--cs-font-mono); font-size: var(--cs-text-3xs); color: var(--cs-color-text-muted); background: var(--cs-color-neutral-soft); border-radius: var(--cs-radius-pill); padding: var(--cs-space-1) var(--cs-space-3); }
.mrt__why, .mrt__wrong { margin: 0 0 var(--cs-space-2); font-size: var(--cs-text-sm); line-height: var(--cs-leading-relaxed); color: var(--cs-color-text); }
.mrt__wrong { margin-bottom: 0; padding-top: var(--cs-space-2); border-top: 1px dashed var(--cs-color-border); color: var(--cs-color-text-muted); }
.mrt__why i, .mrt__wrong i { display: block; font-family: var(--cs-font-mono); font-size: var(--cs-text-3xs); letter-spacing: .08em; font-style: normal; color: var(--cs-color-text-subtle); }
.mrt__hint { margin: var(--cs-space-4) 0 0; font-size: var(--cs-text-sm); color: var(--cs-color-text-subtle); }
@media (max-width: 720px) {
	.mrt__homes { grid-template-columns: repeat(2, minmax(0, 1fr)); }
}
</style>
