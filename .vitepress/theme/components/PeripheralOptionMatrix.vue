<script setup lang="ts">
import { computed, ref } from "vue"

/**
 * 边缘方案速览：每个只回答三个问题——补什么短板、代价是什么、什么条件下才轮到它。
 *
 * 之所以按「前置条件」而不是按能力排列：这些方案的问题从来不是「好不好」，
 * 而是「适用范围窄」。把前置条件放在最显眼的位置，
 * 是为了让读者先判断自己是否落在范围内，再决定要不要往下看机制。
 */

type Option = {
	key: string
	name: string
	/** 它补的是主流链路的哪一处短板 */
	fixes: string
	/** 换来这个能力付出了什么 */
	costs: string
	/** 不满足这一条就不必考虑它 */
	precondition: string
	/** 落在哪一类问题上 */
	family: "预计算" | "检索器" | "权重" | "前置层"
}

const options: Option[] = [
	{
		key: "cag", name: "整体预载与缓存复用", family: "预计算",
		fixes: "省掉整条检索链路，也就不存在召回漏掉与切分切错——模型直接看到全文。",
		costs: "首次要把全文过一遍模型；内容一改，缓存就得重建。",
		precondition: "语料能整体放进一次调用的窗口，且更新不频繁。规模再大一个量级就不成立。"
	},
	{
		key: "raptor", name: "递归摘要树", family: "预计算",
		fixes: "补「全局主题」类问题：普通切分只能召回局部片段，答不了跨章节的综合。",
		costs: "构建阶段要反复聚类与摘要，成本高；新内容进来可能触发级联重建。",
		precondition: "问题确实是综合型的。单跳事实查询走这条路只会引入噪声。"
	},
	{
		key: "graph", name: "实体关系图与社区摘要", family: "预计算",
		fixes: "把散在多篇文档里的线索连起来，回答需要跨文档综合才能得出的结论。",
		costs: "抽取阶段依赖模型判断，带概率性；增量更新困难，社区摘要需重算。",
		precondition: "问题需要「连点成线」。精确事实查询用它是纯粹的浪费。"
	},
	{
		key: "sparse", name: "学习式稀疏检索", family: "检索器",
		fixes: "补词面匹配：稠密向量对错误码、版本号这类精确串不敏感，而这类查询占比不低。",
		costs: "要额外训练与维护一套索引，虽然仍可复用倒排结构。",
		precondition: "查询里确实高频出现专有标识符，且纯词面匹配又不够。"
	},
	{
		key: "late", name: "延迟交互多向量", family: "检索器",
		fixes: "保留词级信号：把查询与内容各自编码成一组向量，逐词找最匹配的对应位置。",
		costs: "存储成本成倍上升，每个单元不再是一个向量而是一组。",
		precondition: "需要比单向量更细的粒度，同时又不能承受逐对打分的延迟。"
	},
	{
		key: "visual", name: "视觉文档检索", family: "检索器",
		fixes: "跳过版面解析：直接把页面当图像编码，图表与复杂排版不必先转成文字。",
		costs: "每页要存大量向量，且检索侧需要视觉语言模型参与。",
		precondition: "语料里扫描件与图表占比高，传统解析这一步本身就是主要误差来源。"
	},
	{
		key: "params", name: "写进权重", family: "权重",
		fixes: "把风格、格式、领域用语这类「模型行为」固化下来，推理时零检索开销。",
		costs: "更新以训练周期计；无法回答依据来自哪里；有遗忘风险。",
		precondition: "要固化的是行为而非事实。事实类知识放进权重会同时失去时效与溯源。"
	},
	{
		key: "cache", name: "语义缓存", family: "前置层",
		fixes: "挡掉重复与近重复的提问，在高并发场景下省掉绝大部分下游开销。",
		costs: "判定阈值定不好就会误命中，把两个不同问题当成同一个。",
		precondition: "流量里确实有大量重复提问，且答案可以在内容更新时被准确失效。"
	}
]

const families = ["预计算", "检索器", "权重", "前置层"] as const
const active = ref<string>("全部")
const tabs = ["全部", ...families]

const shown = computed(() =>
	active.value === "全部" ? options : options.filter((o) => o.family === active.value)
)
</script>

<template>
	<div class="pom">
		<div class="infra-tabs pom__tabs">
			<button
				v-for="t in tabs"
				:key="t"
				type="button"
				:aria-pressed="active === t"
				@click="active = t"
			>{{ t }}</button>
		</div>

		<div class="pom__grid">
			<article v-for="o in shown" :key="o.key" class="pom__card">
				<header>
					<b>{{ o.name }}</b>
					<span class="cs-badge">{{ o.family }}</span>
				</header>
				<p class="pom__pre"><i>不满足这一条就不必考虑</i>{{ o.precondition }}</p>
				<p class="pom__fix"><i>补什么短板</i>{{ o.fixes }}</p>
				<p class="pom__cost"><i>代价</i>{{ o.costs }}</p>
			</article>
		</div>

		<p class="pc-note">
			这些方案的共同特点是<b>适用范围窄</b>，而不是水平低：落在范围内它们往往是最省的解，
			落在范围外则纯属负担。所以判断顺序应当是先读前置条件，
			确认自己在范围内，再去看它的机制值不值得引入——反过来做，
			很容易为了用上某个方案而把问题重新定义一遍。
		</p>
	</div>
</template>

<style scoped>
.pom { margin: 1rem 0; }
.pom__tabs { margin-bottom: var(--cs-space-4); }
.pom__tabs button { min-height: var(--cs-tap-target); font-size: var(--cs-text-3xs); }
.pom__grid { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: var(--cs-space-3); }
.pom__card { background: var(--cs-color-bg); border: 1px solid var(--cs-color-border); border-radius: var(--cs-radius-lg); padding: var(--cs-space-4); }
.pom__card > header { display: flex; flex-wrap: wrap; align-items: baseline; justify-content: space-between; gap: var(--cs-space-2); margin-bottom: var(--cs-space-3); }
.pom__card > header b { font-size: var(--cs-text-md); color: var(--cs-color-text); }
.pom__card p { margin: 0 0 var(--cs-space-2); font-size: var(--cs-text-sm); line-height: var(--cs-leading-relaxed); }
.pom__card p:last-child { margin-bottom: 0; }
.pom__card i { display: block; font-family: var(--cs-font-mono); font-size: var(--cs-text-3xs); letter-spacing: .08em; font-style: normal; color: var(--cs-color-text-subtle); margin-bottom: var(--cs-space-1); }
.pom__pre { padding: var(--cs-space-2) var(--cs-space-3); border-left: 3px solid var(--cs-color-warning); background: var(--cs-color-bg-soft); border-radius: var(--cs-radius-sm); color: var(--cs-color-text); }
.pom__fix { color: var(--cs-color-text); }
.pom__cost { color: var(--cs-color-text-muted); }
@media (max-width: 760px) {
	.pom__grid { grid-template-columns: 1fr; }
}
</style>
