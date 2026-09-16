<script setup lang="ts">
import { computed, ref } from "vue"

/**
 * 从「每次调用都是无状态的」出发，逐步加机制，看跨会话连续性如何被一步步补齐。
 *
 * 五个阶段，每一步只加一件事，并明确指出：加了它解决什么、还剩什么没解决。
 * 目的是让「为什么会分化出三套机制」成为一个推导结果，而不是一个需要记住的分类。
 */

type Stage = {
	key: string
	name: string
	adds: string
	solves: string
	remains: string
	/** 三种能力的达成度，0–3 */
	within: number
	across: number
	domain: number
	scene: { turn: string; sees: string; ok: boolean }[]
}

const stages: Stage[] = [
	{
		key: "s0",
		name: "裸调用",
		adds: "什么都没加：一次请求进，一次响应出",
		solves: "能回答有自足信息的单个问题。",
		remains: "第二次调用对第一次一无所知。模型不是「忘了」——它从来没有存放上一次的地方。",
		within: 0, across: 0, domain: 0,
		scene: [
			{ turn: "第 1 次调用", sees: "「我叫小张」", ok: true },
			{ turn: "第 2 次调用", sees: "「我叫什么？」——只有这一句", ok: false }
		]
	},
	{
		key: "s1",
		name: "把历史一起发过去",
		adds: "每次把之前的消息拼在前面重新发送",
		solves: "同一段对话里的连续性有了。这就是「多轮对话」的全部机制——它不是模型记住了，而是每次都重发一遍。",
		remains: "历史随轮次单调增长，迟早撞上窗口上限；而且换一个会话仍然从零开始。",
		within: 2, across: 0, domain: 0,
		scene: [
			{ turn: "第 2 次调用", sees: "「我叫小张」+「我叫什么？」", ok: true },
			{ turn: "第 200 次调用", sees: "前 199 轮全文——已经装不下", ok: false }
		]
	},
	{
		key: "s2",
		name: "治理这一次要发什么",
		adds: "挑选、压缩、隔离：每一轮重新决定哪些内容值得占用本次的窗口",
		solves: "长会话不再撞墙。到这一步，context 作为一套独立机制成形了——它管的是「本次让模型看见什么」。",
		remains: "被压缩掉的细节没有别处可取；换一个会话依然一无所知。",
		within: 3, across: 0, domain: 0,
		scene: [
			{ turn: "第 200 次调用", sees: "摘要 + 最近若干轮 + 本次召回", ok: true },
			{ turn: "新开一个会话", sees: "空的——上一个会话的一切都不在", ok: false }
		]
	},
	{
		key: "s3",
		name: "把该留的写到窗口之外",
		adds: "跨会话的存储：写入、判定有效性、按需召回",
		solves: "换会话也能接上。到这一步，memory 作为一套独立机制成形了——它管的是「跨越多次调用的经历如何存续」。",
		remains: "模型参数里没有的客观领域事实，仍然没有来源——它既不在历史里，也不属于任何用户的经历。",
		within: 3, across: 3, domain: 0,
		scene: [
			{ turn: "新会话第 1 次调用", sees: "召回的偏好与项目事实", ok: true },
			{ turn: "问某框架 v3 的新 API", sees: "记忆里没有，历史里也没有", ok: false }
		]
	},
	{
		key: "s4",
		name: "把外部权威内容接进来",
		adds: "文档索引与检索：按查询取回外部内容的片段",
		solves: "领域事实有了来源，且能随文档版本更新。knowledge 作为第三套机制成形。",
		remains: "三套机制都在，但它们都要经过同一个瓶颈——本次载荷的容量，于是「该让哪些进来」重新成为核心问题。",
		within: 3, across: 3, domain: 3,
		scene: [
			{ turn: "问 v3 新 API", sees: "检索到的文档片段 + 项目记忆", ok: true },
			{ turn: "三路都想塞进来", sees: "召回预算需要在三者之间分配", ok: false }
		]
	}
]

const at = ref(2)
const cur = computed(() => stages[at.value])
const caps = [
	{ key: "within" as const, label: "会话内连续性" },
	{ key: "across" as const, label: "跨会话连续性" },
	{ key: "domain" as const, label: "领域事实覆盖" }
]
const levelText = (v: number) => ["无", "勉强", "可用", "成立"][v]
</script>

<template>
	<div class="mgs">
		<div class="mgs__track">
			<button
				v-for="(s, i) in stages"
				:key="s.key"
				type="button"
				class="mgs__step"
				:class="{ 'is-on': at === i, 'is-done': i < at }"
				:aria-pressed="at === i"
				@click="at = i"
			>
				<span class="mgs__step-idx cs-mono">{{ i }}</span>
				<span class="mgs__step-name">{{ s.name }}</span>
			</button>
		</div>

		<article class="mgs__panel">
			<p class="mgs__adds"><i>这一步加了什么</i>{{ cur.adds }}</p>

			<div class="mgs__caps">
				<div v-for="c in caps" :key="c.key" class="mgs__cap">
					<div class="mgs__cap-top">
						<span>{{ c.label }}</span>
						<b class="cs-mono">{{ levelText(cur[c.key]) }}</b>
					</div>
					<div class="infra-bar">
						<i
							:style="{
								width: `${(cur[c.key] / 3) * 100}%`,
								background: cur[c.key] === 3 ? 'var(--cs-color-success)' : cur[c.key] === 0 ? 'var(--cs-color-danger)' : 'var(--cs-color-warning)'
							}"
						></i>
					</div>
				</div>
			</div>

			<ul class="mgs__scene">
				<li v-for="sc in cur.scene" :key="sc.turn" :class="sc.ok ? 'cs-state--pass' : 'cs-state--fail'">
					<span class="mgs__scene-turn">{{ sc.turn }}</span>
					<span class="mgs__scene-sees">模型看到：{{ sc.sees }}</span>
				</li>
			</ul>

			<p class="mgs__solves"><i>解决了什么</i>{{ cur.solves }}</p>
			<p class="mgs__remains"><i>还剩什么没解决</i>{{ cur.remains }}</p>
		</article>
	</div>
</template>

<style scoped>
.mgs { margin: 1rem 0; }
.mgs__track { display: grid; grid-template-columns: repeat(5, minmax(0, 1fr)); gap: var(--cs-space-2); }
.mgs__step { display: grid; gap: var(--cs-space-1); text-align: left; cursor: pointer; min-height: var(--cs-tap-target); padding: var(--cs-space-3); background: var(--cs-color-bg); border: 1px solid var(--cs-color-border); border-top: 3px solid var(--cs-color-border); border-radius: var(--cs-radius-sm); transition: var(--cs-transition-colors); }
.mgs__step:hover { background: var(--cs-color-bg-soft); }
.mgs__step.is-done { border-top-color: var(--cs-color-success); }
.mgs__step.is-on { border-color: var(--cs-color-brand); border-top-color: var(--cs-color-brand); background: var(--cs-color-brand-soft); }
.mgs__step-idx { font-size: var(--cs-text-3xs); color: var(--cs-color-text-subtle); }
.mgs__step-name { font-size: var(--cs-text-xs); line-height: var(--cs-leading-tight); color: var(--cs-color-text); }
.mgs__step.is-on .mgs__step-name { color: var(--cs-color-brand); font-weight: 600; }
.mgs__panel { margin-top: var(--cs-space-3); background: var(--cs-color-bg); border: 1px solid var(--cs-color-border); border-radius: var(--cs-radius-lg); padding: var(--cs-space-4) var(--cs-space-5); }
.mgs__panel p { margin: 0 0 var(--cs-space-3); font-size: var(--cs-text-base); line-height: var(--cs-leading-relaxed); color: var(--cs-color-text); }
.mgs__panel i { display: block; font-family: var(--cs-font-mono); font-size: var(--cs-text-3xs); letter-spacing: .08em; font-style: normal; color: var(--cs-color-text-subtle); margin-bottom: var(--cs-space-1); }
.mgs__caps { display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: var(--cs-space-3); margin-bottom: var(--cs-space-3); }
.mgs__cap-top { display: flex; align-items: baseline; justify-content: space-between; gap: var(--cs-space-2); margin-bottom: var(--cs-space-1); }
.mgs__cap-top span { font-size: var(--cs-text-2xs); color: var(--cs-color-text-muted); }
.mgs__cap-top b { font-size: var(--cs-text-3xs); color: var(--cs-color-text); }
.mgs__scene { list-style: none; margin: 0 0 var(--cs-space-3); padding: 0; display: grid; gap: var(--cs-space-2); }
.mgs__scene li { display: grid; grid-template-columns: 9rem minmax(0, 1fr); gap: var(--cs-space-3); padding: var(--cs-space-2) var(--cs-space-3); border-radius: var(--cs-radius-sm); }
.mgs__scene-turn { font-family: var(--cs-font-mono); font-size: var(--cs-text-3xs); color: var(--cs-color-text-muted); }
.mgs__scene-sees { font-size: var(--cs-text-sm); line-height: var(--cs-leading-normal); color: var(--cs-color-text); }
.mgs__remains { margin-bottom: 0 !important; padding-top: var(--cs-space-2); border-top: 1px dashed var(--cs-color-border); color: var(--cs-color-text-muted); }
@media (max-width: 760px) {
	.mgs__track { grid-template-columns: repeat(2, minmax(0, 1fr)); }
	.mgs__caps { grid-template-columns: 1fr; gap: var(--cs-space-2); }
	.mgs__scene li { grid-template-columns: 1fr; gap: var(--cs-space-1); }
}
</style>
