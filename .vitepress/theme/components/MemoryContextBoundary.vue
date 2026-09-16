<script setup lang="ts">
import { computed, ref } from "vue"

/**
 * memory / context / knowledge 三套机制的正面对照。
 *
 * 排布上刻意把 context 放在中间：另两者的本体都在 context 之外，
 * 都必须经由「进入 context」这一步才能影响输出。这个结构本身就是主要判据的来源。
 * 每一行对照一个维度，切换维度即可看出三者在哪里真正不同。
 */

type Dim = {
	key: string
	label: string
	memory: string
	context: string
	knowledge: string
	takeaway: string
}

const dims: Dim[] = [
	{
		key: "problem", label: "解决什么问题",
		memory: "让跨越多次调用的经历不必重新交代",
		context: "让这一次生成看得见它需要的信息",
		knowledge: "让模型参数里没有的领域事实可被查到",
		takeaway: "三个问题彼此独立：任一个解决了，另两个依然存在。"
	},
	{
		key: "entity", label: "本体存放在哪",
		memory: "context 之外：记忆存储、文件、图、键值表",
		context: "就是 context 本身——它没有「本体在别处」",
		knowledge: "context 之外：文档库、索引、数据库",
		takeaway: "这是最锋利的一条判据。memory 与 knowledge 的本体都在外面，context 的本体就是它自己。"
	},
	{
		key: "write", label: "谁来写、什么时候写",
		memory: "系统或模型在交互过程中写入，内容源自这个用户的经历",
		context: "每次调用前由组装逻辑现场拼装，不存在「写入」",
		knowledge: "由内容生产方离线维护，与任何单个用户的交互无关",
		takeaway: "写入主体不同：memory 的内容来自交互，knowledge 的内容来自外部权威。"
	},
	{
		key: "read", label: "怎么进入模型视野",
		memory: "按当前任务召回少数条目，或作为常驻块始终可见",
		context: "无需进入——它就是模型的视野",
		knowledge: "按查询检索片段，通常一次性用完不留存",
		takeaway: "memory 与 knowledge 的读路径形似（都是检索），但召回的目标不同：一个找「关于你的」，一个找「关于世界的」。"
	},
	{
		key: "life", label: "生命周期",
		memory: "跨会话存续，直到被改写、失效或删除",
		context: "一次模型调用。下一次要重新组装",
		knowledge: "随文档版本演进，与会话完全无关",
		takeaway: "三种寿命差了几个数量级，所以三者的「过期」形态也完全不同。"
	},
	{
		key: "consistency", label: "一致性要求",
		memory: "同一事实只应有一个当前有效值；矛盾必须被消解",
		context: "本次内部不应自相矛盾，但跨次无需一致",
		knowledge: "以文档源为准，一致性由内容治理保证，不由 Agent 保证",
		takeaway: "memory 需要 Agent 自己维护一致性，knowledge 不需要——它的一致性是别人的责任。"
	},
	{
		key: "removed", label: "移除之后能否重建",
		memory: "不能。它记录的是一次性发生过的经历，删了就没有第二个来源",
		context: "能，而且每一轮都在重建",
		knowledge: "能。文档还在，重新检索即可得到同样的片段",
		takeaway: "第二条判据：可重建的是派生物，不可重建的才是本体。"
	},
	{
		key: "fail", label: "典型失效形态",
		memory: "过期而不自知：旧结论一直被召回，没人告诉系统它已经假了",
		context: "该看见的没看见：存了、检索到了，但没进入本次载荷",
		knowledge: "召回错片段或知识覆盖不全：答案不在库里，或在库里却排不到前面",
		takeaway: "三种失效的排障方向完全不同，混淆之后会一直在错误的层里找原因。"
	},
	{
		key: "debug", label: "出问题先查哪里",
		memory: "查这条记忆的写入时间、来源与有效期，再查它是否该被改写",
		context: "查本次请求的实际载荷——它到底有没有出现在里面",
		knowledge: "查检索排序与切分：文档在不在库里，在库里为什么没排进来",
		takeaway: "「先看本次载荷」几乎总是第一步，因为它能一次性区分「没召回」与「召回了但没用对」。"
	}
]

const idx = ref(1)
const cur = computed(() => dims[idx.value])
const cols = [
	{ key: "memory" as const, name: "memory", zh: "记忆", color: "var(--cs-color-success)", where: "本体在 context 之外" },
	{ key: "context" as const, name: "context", zh: "上下文", color: "var(--cs-color-brand)", where: "本体就是它自己" },
	{ key: "knowledge" as const, name: "knowledge", zh: "知识", color: "var(--cs-color-info)", where: "本体在 context 之外" }
]
</script>

<template>
	<div class="mcb">
		<div class="mcb__dims">
			<button
				v-for="(d, i) in dims"
				:key="d.key"
				type="button"
				class="mcb__dim"
				:class="{ 'is-on': idx === i }"
				:aria-pressed="idx === i"
				@click="idx = i"
			>{{ d.label }}</button>
		</div>

		<div class="mcb__cols">
			<section v-for="c in cols" :key="c.key" class="mcb__col" :style="{ '--c': c.color }">
				<header>
					<b class="cs-mono">{{ c.name }}</b>
					<span>{{ c.zh }}</span>
					<i>{{ c.where }}</i>
				</header>
				<p class="mcb__val">{{ cur[c.key] }}</p>
			</section>
		</div>

		<p class="mcb__takeaway"><i>这一行说明什么</i>{{ cur.takeaway }}</p>

		<p class="pc-note">
			注意中间那一列的特殊之处：其余两者都能回答「本体存放在哪」，只有 context 的答案是「就是它自己」。
			由此得到两条可以直接用的判据——<b>看本体在 context 之内还是之外</b>，以及<b>移除之后能不能重建</b>。
			需要强调的是，「能不能被丢弃」并不是判据：context 每一轮都被丢弃重建，memory 里的过期条目也应当被丢弃，
			knowledge 检索到的片段用完即弃。可丢弃性是三者共有的现象，用它区分不了任何东西。
		</p>
	</div>
</template>

<style scoped>
.mcb { margin: 1rem 0; }
.mcb__dims { display: flex; flex-wrap: wrap; gap: var(--cs-space-1); margin-bottom: var(--cs-space-4); }
.mcb__dim { min-height: var(--cs-tap-target); padding: 0 var(--cs-space-3); cursor: pointer; font-size: var(--cs-text-xs); color: var(--cs-color-text-muted); background: var(--cs-color-bg); border: 1px solid var(--cs-color-border); border-radius: var(--cs-radius-pill); transition: var(--cs-transition-colors); }
.mcb__dim:hover { background: var(--cs-color-bg-soft); }
.mcb__dim.is-on { background: var(--cs-color-brand-soft); border-color: var(--cs-color-brand); color: var(--cs-color-brand); font-weight: 600; }
.mcb__cols { display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: var(--cs-space-3); align-items: stretch; }
.mcb__col { display: flex; flex-direction: column; background: var(--cs-color-bg); border: 1px solid var(--cs-color-border); border-top: 3px solid var(--c); border-radius: var(--cs-radius-lg); overflow: hidden; }
.mcb__col > header { padding: var(--cs-space-3) var(--cs-space-4); background: var(--cs-color-bg-soft); border-bottom: 1px solid var(--cs-color-border); }
.mcb__col > header b { font-size: var(--cs-text-md); color: var(--c); }
.mcb__col > header span { font-size: var(--cs-text-3xs); color: var(--cs-color-text-muted); margin-left: var(--cs-space-2); }
.mcb__col > header i { display: block; margin-top: var(--cs-space-1); font-family: var(--cs-font-mono); font-size: var(--cs-text-3xs); font-style: normal; color: var(--cs-color-text-subtle); }
.mcb__val { flex: 1; margin: 0; padding: var(--cs-space-4); font-size: var(--cs-text-sm); line-height: var(--cs-leading-relaxed); color: var(--cs-color-text); }
.mcb__takeaway { margin: var(--cs-space-4) 0 0; padding: var(--cs-space-3) var(--cs-space-4); border-left: 3px solid var(--cs-color-border-strong); background: var(--cs-color-bg-soft); border-radius: var(--cs-radius-sm); font-size: var(--cs-text-base); line-height: var(--cs-leading-relaxed); color: var(--cs-color-text); }
.mcb__takeaway i { display: block; font-family: var(--cs-font-mono); font-size: var(--cs-text-3xs); letter-spacing: .08em; font-style: normal; color: var(--cs-color-text-subtle); margin-bottom: var(--cs-space-1); }
@media (max-width: 760px) {
	.mcb__cols { grid-template-columns: 1fr; }
}
</style>
