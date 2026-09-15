<script setup lang="ts">
import { computed, ref } from "vue"

/**
 * 返回值判读器：同一个 hook 脚本，不同的退出方式会被宿主解释成完全不同的指令。
 *
 * 这个组件要纠正的是一个具体且常见的误解——「脚本失败了，动作就不会执行」。
 * 实际上只有约定的那一个退出码才表示阻断；习惯上表示失败的 exit 1 被当作
 * 非阻断错误，动作照常发生。用户挑一种返回方式，组件给出宿主的实际解释。
 */

type Verdict = "proceed" | "blocked" | "rewritten" | "noted"

type Exit = {
	key: string
	label: string
	form: string
	verdict: Verdict
	/** 宿主如何解释 */
	reading: string
	/** stdout / stderr 的去向 */
	channel: string
	note?: string
}

const verdictMeta: Record<Verdict, { label: string; cls: string }> = {
	proceed: { label: "动作照常执行", cls: "cs-state--fail" },
	blocked: { label: "动作被阻断", cls: "cs-state--pass" },
	rewritten: { label: "入参被改写后执行", cls: "cs-state--warn" },
	noted: { label: "执行，但模型收到反馈", cls: "cs-state--warn" }
}

const exits: Exit[] = [
	{
		key: "e0",
		label: "exit 0，无输出",
		form: "exit 0",
		verdict: "proceed",
		reading: "成功，无意见。宿主继续原本的流程。这是绝大多数 hook 的正常出口。",
		channel: "标准输出写进调试日志，模型看不到。"
	},
	{
		key: "e1",
		label: "exit 1（习惯上的失败码）",
		form: "exit 1",
		verdict: "proceed",
		reading: "被当作非阻断错误：宿主记下这个 hook 出错了，然后照常执行工具调用。",
		channel: "错误输出写进日志，界面上通常只显示第一行。",
		note: "这是本组件最值得记住的一格。Unix 习惯里非零即失败，但在这套协议下只有约定的那个码才表示阻断。一个「校验失败就 exit 1」的脚本看起来在工作，实际上一次也没拦住过。"
	},
	{
		key: "e2",
		label: "约定的阻断码",
		form: "exit 2 + stderr",
		verdict: "blocked",
		reading: "阻断。此时标准输出连同其中的 JSON 一并被忽略，改为把错误输出的文本回喂给模型当作理由。",
		channel: "错误输出成为模型可见的反馈，因此它要写给模型看，而不是写给人看。",
		note: "同一个阻断码在不同事件上后果不同：工具调用前是不执行，收尾事件上是不许停、继续干。有些事件根本不可阻断，这时它退化成一条日志。"
	},
	{
		key: "e3",
		label: "结构化拒绝",
		form: 'exit 0 + {"permissionDecision":"deny"}',
		verdict: "blocked",
		reading: "同样是拒绝，但走结构化字段而不是退出码。多数实现要求二选一：打印 JSON 就必须以 0 退出，否则 JSON 会被丢弃。",
		channel: "理由字段按事件约定分别给人或给模型看，两者并不总是同一份文本。"
	},
	{
		key: "e4",
		label: "放行并改写入参",
		form: 'exit 0 + {"permissionDecision":"allow","updatedInput":{…}}',
		verdict: "rewritten",
		reading: "不拦截，而是替换掉这次调用的入参再执行。常见约定是整体替换，没改的字段也要一并带上。",
		channel: "改写通常对模型不可见，它以为自己调用的还是原来那份参数。",
		note: "这条路径的表达力最强，也最容易埋下难查的问题：日志里记的是模型的意图，实际发生的是改写后的动作，两者对不上。"
	},
	{
		key: "e5",
		label: "事后拒绝",
		form: 'exit 0 + {"decision":"block"}（工具调用后）',
		verdict: "noted",
		reading: "工具已经跑完了。这里的「拒绝」不回滚任何东西，而是把反馈替换掉原本的工具结果，让模型从这段反馈继续。",
		channel: "反馈文本成为模型看到的那份结果。",
		note: "把它当成事务回滚是一类真实的设计错误：文件已经写了、消息已经发了，改变的只是模型接下来读到什么。"
	}
]

const idx = ref(1)
const cur = computed(() => exits[idx.value])
</script>

<template>
	<div class="hrc">
		<div class="hrc__list">
			<button
				v-for="(e, i) in exits"
				:key="e.key"
				type="button"
				class="hrc__item"
				:class="{ 'is-on': idx === i }"
				:aria-pressed="idx === i"
				@click="idx = i"
			>
				<code>{{ e.form }}</code>
				<span>{{ e.label }}</span>
			</button>
		</div>

		<article class="hrc__card">
			<p class="hrc__verdict" :class="verdictMeta[cur.verdict].cls">
				宿主的判读：<b>{{ verdictMeta[cur.verdict].label }}</b>
			</p>
			<p class="hrc__reading">{{ cur.reading }}</p>
			<p class="hrc__channel"><i>输出去哪儿</i>{{ cur.channel }}</p>
			<p v-if="cur.note" class="hrc__note"><i>值得留意</i>{{ cur.note }}</p>
		</article>

		<p class="pc-note">
			六种返回方式里有<b>两种表示阻断、三种表示继续</b>，而其中一种「继续」长得最像失败。
			这就是为什么护栏类扩展必须<b>验证它真的拦住了</b>，而不是看脚本有没有报错——
			脚本报错与动作被阻断在这套协议下是两件不相干的事。具体的码值与字段名各宿主不同，
			本图呈现的是判读方式的结构：<b>退出码表达粗粒度意图，结构化输出表达细粒度决策，两者不能混用。</b>
		</p>
	</div>
</template>

<style scoped>
.hrc { margin: 1rem 0; }
.hrc__list { display: grid; gap: var(--cs-space-2); }
.hrc__item { display: grid; gap: var(--cs-space-1); text-align: left; cursor: pointer; min-height: var(--cs-tap-target); padding: var(--cs-space-3) var(--cs-space-4); background: var(--cs-color-bg); border: 1px solid var(--cs-color-border); border-radius: var(--cs-radius-sm); transition: var(--cs-transition-colors); }
.hrc__item:hover { background: var(--cs-color-bg-soft); }
.hrc__item.is-on { border-color: var(--cs-color-brand); background: var(--cs-color-brand-soft); }
.hrc__item code { font-family: var(--cs-font-mono); font-size: var(--cs-text-2xs); color: var(--cs-color-text); background: none; padding: 0; word-break: break-all; }
.hrc__item span { font-size: var(--cs-text-3xs); color: var(--cs-color-text-muted); }
.hrc__card { margin-top: var(--cs-space-4); background: var(--cs-color-bg); border: 1px solid var(--cs-color-border); border-radius: var(--cs-radius-lg); padding: var(--cs-space-5) var(--cs-space-6); }
.hrc__verdict { margin: 0 0 var(--cs-space-3); padding: var(--cs-space-3) var(--cs-space-4); border-radius: var(--cs-radius-sm); font-size: var(--cs-text-md); }
.hrc__reading { margin: 0 0 var(--cs-space-3); font-size: var(--cs-text-base); line-height: var(--cs-leading-relaxed); color: var(--cs-color-text); }
.hrc__channel, .hrc__note { margin: 0 0 var(--cs-space-2); font-size: var(--cs-text-sm); line-height: var(--cs-leading-relaxed); color: var(--cs-color-text-muted); }
.hrc__note { margin-bottom: 0; padding-top: var(--cs-space-3); border-top: 1px dashed var(--cs-color-border); }
.hrc__channel i, .hrc__note i { display: block; font-family: var(--cs-font-mono); font-size: var(--cs-text-3xs); letter-spacing: .08em; font-style: normal; color: var(--cs-color-text-subtle); margin-bottom: var(--cs-space-1); }
@media (min-width: 861px) {
	.hrc__list { grid-template-columns: repeat(2, minmax(0, 1fr)); }
}
@media (max-width: 520px) {
	.hrc__card { padding: var(--cs-space-4); }
}
</style>
