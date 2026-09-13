<script setup lang="ts">
const rows = [
	{ dim: "要回答的问题", pe: "这句话怎么说，模型才照做", ce: "此刻该让模型看见什么、不该看见什么" },
	{ dim: "操作对象", pe: "系统提示 + 用户消息", ce: "系统提示、工具、示例、检索、记忆、历史、工具返回" },
	{ dim: "时间维度", pe: "一次性：写好即固定", ce: "迭代循环：每轮推理前重新决定传什么" },
	{ dim: "主要约束", pe: "表达是否清晰、格式是否可解析", ce: "注意力预算有限，长上下文存在召回衰减" },
	{ dim: "典型场景", pe: "单轮分类、抽取、改写、生成", ce: "多轮 Agent、长程任务、代码库迁移、深度研究" },
	{ dim: "典型失败", pe: "格式跑偏、答非所问、风格不对", ce: "越跑越偏、重复踩坑、忘掉早期约束、工具选错" },
	{ dim: "优化手段", pe: "改措辞、加示例、拆步骤、定输出结构", ce: "压缩、做笔记、按需检索、裁剪工具集、拆子代理" },
	{ dim: "衡量指标", pe: "单次输出的准确率 / 合格率", ce: "长任务完成率、步数、token 成本、连贯性" }
]
</script>

<template>
	<section class="pc-hero">
		<p class="pc-eyebrow">工程笔记 · LLM 应用构建</p>
		<p class="pc-hero__title pc-serif">从「怎么问」到「让模型看见什么」</p>
		<p class="pc-hero__sub">Prompt 与 Context 不是新旧替代关系，而是包含关系：提示词工程解决单次指令的表达，上下文工程决定模型在每一步推理时处于什么信息环境。当应用从「一问一答」走向「自主循环的 Agent」，后者成为决定成败的主要变量。</p>

		<div class="pc-eq">
			<div class="pc-eq__cell pe">
				<span class="pc-eq__lab">Prompt Engineering</span>
				<span class="pc-eq__txt pc-serif">为一次调用，写出并组织出最有效的指令。</span>
			</div>
			<div class="pc-eq__mid" aria-label="子集"><span>⊂</span></div>
			<div class="pc-eq__cell ce">
				<span class="pc-eq__lab">Context Engineering</span>
				<span class="pc-eq__txt pc-serif">在推理过程中，持续策展并维护那组最优的 token。</span>
			</div>
		</div>

		<table class="pc-compare">
			<thead>
				<tr><th>对照维度</th><th class="pe">Prompt</th><th class="ce">Context</th></tr>
			</thead>
			<tbody>
				<tr v-for="row in rows" :key="row.dim">
					<td class="dim">{{ row.dim }}</td>
					<td class="pe">{{ row.pe }}</td>
					<td class="ce">{{ row.ce }}</td>
				</tr>
			</tbody>
		</table>
	</section>
</template>

<style scoped>
.pc-hero { border: 1px solid var(--cs-color-border); border-radius: 14px; padding: 1.5rem 1.6rem; background: var(--cs-color-bg-soft); margin: 1.2rem 0 0; }
.pc-eyebrow { color: var(--cs-color-warning); }
.pc-hero__title { font-size: clamp(1.5rem, 3.4vw, 2.1rem); font-weight: 800; line-height: 1.25; letter-spacing: -.01em; margin: 0; }
.pc-hero__sub { color: var(--cs-color-text-muted); font-size: 1rem; line-height: 1.85; margin: 1rem 0 0; max-width: 62ch; }
.pc-eq { display: grid; grid-template-columns: 1fr auto 1fr; gap: 0; margin-top: 1.4rem; border: 1px solid var(--cs-color-border); border-radius: 10px; overflow: hidden; background: var(--cs-color-bg); }
.pc-eq__cell { padding: 1.1rem 1.15rem; }
.pc-eq__cell.pe { border-top: 3px solid var(--cs-color-warning); }
.pc-eq__cell.ce { border-top: 3px solid var(--cs-color-success); }
.pc-eq__lab { display: block; font-family: var(--cs-font-mono); font-size: .68rem; letter-spacing: .1em; text-transform: uppercase; margin-bottom: .5rem; }
.pc-eq__cell.pe .pc-eq__lab { color: var(--cs-color-warning); }
.pc-eq__cell.ce .pc-eq__lab { color: var(--cs-color-success); }
.pc-eq__txt { display: block; font-size: 1.02rem; font-weight: 600; line-height: 1.5; color: var(--cs-color-text); }
.pc-eq__mid { display: flex; align-items: center; justify-content: center; padding: 0 1rem; background: var(--cs-color-bg-soft); border-left: 1px solid var(--cs-color-border); border-right: 1px solid var(--cs-color-border); }
.pc-eq__mid span { font-family: var(--cs-font-mono); font-size: 1.4rem; color: var(--cs-color-text-subtle); }
.pc-compare { width: 100%; border-collapse: collapse; margin: 1.3rem 0 0; font-size: .9rem; display: table; }
.pc-compare th, .pc-compare td { text-align: left; padding: .6rem .7rem; border-bottom: 1px solid var(--cs-color-border); vertical-align: top; }
.pc-compare thead th { font-family: var(--cs-font-mono); font-size: .68rem; letter-spacing: .08em; text-transform: uppercase; color: var(--cs-color-text-subtle); border-bottom: 1.5px solid var(--cs-color-text); background: var(--cs-color-bg-soft); }
.pc-compare thead th.pe { color: var(--cs-color-warning); }
.pc-compare thead th.ce { color: var(--cs-color-success); }
.pc-compare td.dim { font-weight: 700; color: var(--cs-color-text); width: 22%; }
.pc-compare td.pe { background: color-mix(in srgb, var(--cs-color-warning) 7%, transparent); }
.pc-compare td.ce { background: color-mix(in srgb, var(--cs-color-success) 7%, transparent); }
.pc-compare tbody tr:last-child td { border-bottom: 0; }
.pc-compare tbody tr { background: transparent; }
@media (max-width: 720px) {
	.pc-eq { grid-template-columns: 1fr; }
	.pc-eq__mid { border-left: 0; border-right: 0; border-top: 1px solid var(--cs-color-border); border-bottom: 1px solid var(--cs-color-border); padding: .35rem 0; }
	.pc-eq__mid span { transform: rotate(90deg); display: inline-block; }
	.pc-compare { font-size: .84rem; }
}
</style>
