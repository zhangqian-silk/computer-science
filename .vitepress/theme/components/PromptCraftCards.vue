<script setup lang="ts">
const cards = [
	{ t: "结构化分节", d: "把背景、指令、工具说明、输出描述拆成独立区块，用 XML 标签或 Markdown 标题分隔。边界清晰的 prompt 更易被稳定解析，也更好维护。", icon: "M4 5h16M4 10h11M4 15h16M4 20h8" },
	{ t: "少样本示例", d: "示例是模型的「图片」，一图胜千言。关键不在多，而在典型且多样——覆盖主干形态，而不是堆砌所有边界情况。", icon: "M3 4h8v6H3zM13 4h8v6h-8zM3 14h8v6H3zM15 17h6M18 14v6" },
	{ t: "思维链", d: "让模型先写推理再给结论，对数学、逻辑、多步判断提升明显；代价是 token 与延迟。用标签把「思考」与「答案」隔开，便于程序只取后者。", icon: "M12 3v6M12 15v6M5 7l4 3M15 14l4 3M19 7l-4 3M9 14l-4 3" },
	{ t: "输出契约", d: "明确字段、类型、枚举值与「不知道时怎么填」。能用结构化输出约束的，不要靠自然语言描述格式。", icon: "M8 4H6a2 2 0 0 0-2 2v3a2 2 0 0 1-2 2 2 2 0 0 1 2 2v3a2 2 0 0 0 2 2h2M16 4h2a2 2 0 0 1 2 2v3a2 2 0 0 0 2 2 2 2 0 0 0-2 2v3a2 2 0 0 1-2 2h-2" },
	{ t: "校准「高度」", d: "系统提示要落在两种失败之间——一端是写死 if-else 的脆弱逻辑，一端是含糊到给不出信号的空话。足够具体以指导行为，又足够灵活以留出启发式。", icon: "M4 6h16M4 12h16M4 18h16M9 4v4M15 10v4M7 16v4" },
	{ t: "最小起步，按失败迭代", d: "先用最小 prompt 在最强模型上跑基线，再针对实测暴露的失败补指令和示例，而不是一上来就写三千字。", icon: "M3 20l4-4M7 16l4 1 3-6 3 3 4-8" }
]
</script>

<template>
	<div class="pc-craft">
		<div class="pc-craft__grid">
			<article v-for="(c, i) in cards" :key="c.t" class="pc-craft__card">
				<div class="pc-craft__head">
					<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path :d="c.icon" /></svg>
					<span class="pc-craft__idx">0{{ i + 1 }}</span>
				</div>
				<h4 class="pc-serif">{{ c.t }}</h4>
				<p>{{ c.d }}</p>
			</article>
		</div>
		<pre class="pc-craft__snippet">&lt;instructions&gt;判断评论情感，仅输出 POSITIVE 或 NEGATIVE。&lt;/instructions&gt;

输入：这家店服务很快，菜也好吃。
输出：POSITIVE

输入：等了四十分钟，还上错菜。
输出：NEGATIVE</pre>
	</div>
</template>

<style scoped>
.pc-craft { margin: 1rem 0; }
.pc-craft__grid { display: grid; grid-template-columns: repeat(3, 1fr); gap: .8rem; }
.pc-craft__card { position: relative; background: var(--cs-color-bg); border: 1px solid var(--cs-color-border); border-top: 3px solid var(--cs-color-warning); border-radius: 10px; padding: 1rem 1.05rem; }
.pc-craft__head { display: flex; align-items: center; justify-content: space-between; margin-bottom: .5rem; }
.pc-craft__head svg { width: 26px; height: 26px; color: var(--cs-color-warning); }
.pc-craft__idx { font-family: var(--cs-font-mono); font-size: .78rem; color: var(--cs-color-text-subtle); }
.pc-craft__card h4 { margin: 0 0 .35rem; font-size: 1.02rem; font-weight: 700; line-height: 1.3; }
.pc-craft__card p { margin: 0; font-size: .86rem; line-height: 1.7; color: var(--cs-color-text-muted); }
.pc-craft__snippet { margin: .8rem 0 0; background: var(--cs-color-bg-soft); border: 1px solid var(--cs-color-border); border-left: 3px solid var(--cs-color-warning); border-radius: 8px; padding: .8rem .9rem; font-family: var(--cs-font-mono); font-size: .78rem; line-height: 1.7; color: var(--cs-color-text-muted); white-space: pre-wrap; overflow-x: auto; }
@media (max-width: 860px) { .pc-craft__grid { grid-template-columns: 1fr 1fr; } }
@media (max-width: 560px) { .pc-craft__grid { grid-template-columns: 1fr; } }
</style>
