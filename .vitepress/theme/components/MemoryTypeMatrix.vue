<script setup lang="ts">
import { computed, ref } from "vue"
import { seriesColor } from "../series-palette"

/**
 * 记忆的类型学：每一类给出「内容 → 存储形态 → 检索方式 → 更新策略 → 易错点」五段。
 *
 * 分类的用处不在于命名，而在于**每一类的工程决策不同**：
 * 偏好类要覆盖写、情景类要追加写；事实类适合键值或向量、关系类只能靠图。
 * 所以这张表的正确读法是纵向对照同一行在不同类型下的取值，而不是横向记住某一类的定义。
 */

type Kind = {
	key: string
	name: string
	en: string
	question: string
	example: string
	store: string
	retrieve: string
	update: string
	pitfall: string
	idx: number
}

const kinds: Kind[] = [
	{
		key: "semantic",
		name: "语义事实",
		en: "semantic",
		question: "关于世界与用户，什么是真的？",
		example: "「用户在做一个 Go 写的支付网关」「团队用 pnpm 而不是 npm」",
		store: "键值条目或短句向量；每条自带来源与时间戳",
		retrieve: "向量相似度为主，配合关键词兜底专有名词",
		update: "同一事实覆盖写：新值替换旧值，旧值转入历史而非直接抹掉",
		pitfall: "把一次性的当前状态写成长期事实。「这个分支现在有冲突」明天就假了。",
		idx: 0
	},
	{
		key: "episodic",
		name: "情景经历",
		en: "episodic",
		question: "什么时候发生过什么，结果如何？",
		example: "「上周三试过把重试放在网关层，压测下超时反而上升，回滚了」",
		store: "按时间排序的追加写记录，带时间戳与结果标签",
		retrieve: "相似情境检索 + 时间范围过滤；「上次是怎么做的」是典型查询",
		update: "只追加，不修改。结论变了就追加新条目，旧条目仍是「当时确实这样」",
		pitfall: "拿它当事实用。情景记忆说的是「那次这样」，不是「一直这样」。",
		idx: 1
	},
	{
		key: "procedural",
		name: "程序性规程",
		en: "procedural",
		question: "这件事在这里应该怎么做？",
		example: "「提交前跑 npm run check 与 docs:build」「迁移要拆成加列、回填、删列三步」",
		store: "指令文件或规则条目，多数实现直接放进每次请求的固定部分",
		retrieve: "通常不检索——常驻注入，或按文件路径作用域自动附着",
		update: "人工编辑为主；有实现让模型在反复犯同一个错之后自行改写",
		pitfall: "越写越长。它占的是每一次调用的固定预算，膨胀之后遵守率反而下降。",
		idx: 2
	},
	{
		key: "preference",
		name: "偏好与人设",
		en: "preference",
		question: "该以什么方式对待这个人？",
		example: "「回答要简短，先给结论」「代码注释用中文」",
		store: "小而稳定的常驻块，通常与语义事实同库但单独作用域",
		retrieve: "多数常驻不检索——它几乎每一轮都相关，检索反而漏召",
		update: "覆盖写，且要能被用户直接查看与修改",
		pitfall: "把一次性指令当成长期偏好。「这次简短点」不等于「以后都简短」。",
		idx: 3
	},
	{
		key: "entity",
		name: "实体与关系",
		en: "entity graph",
		question: "谁与谁、什么与什么之间是什么关系？",
		example: "「A 服务依赖 B 服务」「这位同事负责结算模块」",
		store: "图：节点是实体，边是带时间的关系事实",
		retrieve: "图遍历 + 向量 + 全文的混合；多跳问题只有图能答",
		update: "新事实与旧事实矛盾时，给旧边标注失效时间而不是删除",
		pitfall: "抽取质量决定一切。实体没对齐，同一个人会变成三个节点。",
		idx: 4
	},
	{
		key: "shared",
		name: "共享记忆",
		en: "shared",
		question: "多个执行者之间，哪些认知必须一致？",
		example: "「组织政策」「主 Agent 与子 Agent 共看的任务看板」",
		store: "被多方挂载的同一个块或同一条命名空间",
		retrieve: "挂载即可见，通常无须检索",
		update: "整块替换的语义下后写覆盖先写；需要只读位与并发约定",
		pitfall: "并发覆盖。两个执行者同时改一块，最后一次写入吞掉前面的全部修改。",
		idx: 5
	}
]

const rows = [
	{ key: "question", label: "回答什么问题" },
	{ key: "example", label: "内容示例" },
	{ key: "store", label: "存储形态" },
	{ key: "retrieve", label: "检索方式" },
	{ key: "update", label: "更新策略" },
	{ key: "pitfall", label: "易错点" }
] as const

const active = ref("procedural")
const cur = computed(() => kinds.find((k) => k.key === active.value)!)
const compare = ref<(typeof rows)[number]["key"] | null>(null)
</script>

<template>
	<div class="mkt">
		<div class="mkt__chips">
			<button
				v-for="k in kinds"
				:key="k.key"
				type="button"
				class="token-chip mkt__chip"
				:class="{ 'is-selected': active === k.key }"
				:style="{ '--c': seriesColor(k.idx) }"
				:aria-pressed="active === k.key"
				@click="active = k.key"
			>
				<span class="mkt__dot" aria-hidden="true"></span>
				{{ k.name }}
			</button>
		</div>

		<article class="mkt__card" :style="{ '--c': seriesColor(cur.idx) }">
			<header>
				<b class="pc-serif">{{ cur.name }}</b>
				<i class="cs-mono">{{ cur.en }}</i>
			</header>
			<dl class="mkt__dl">
				<template v-for="r in rows" :key="r.key">
					<dt>{{ r.label }}</dt>
					<dd :class="{ 'is-pitfall': r.key === 'pitfall' }">{{ cur[r.key] }}</dd>
				</template>
			</dl>
		</article>

		<div class="mkt__compare">
			<p class="mkt__compare-hint">
				横向对照同一维度在六类之间的差异——这才是分类的用处所在：
			</p>
			<div class="mkt__compare-tabs">
				<button
					v-for="r in rows"
					:key="`c-${r.key}`"
					type="button"
					class="mkt__compare-tab"
					:class="{ 'is-on': compare === r.key }"
					:aria-pressed="compare === r.key"
					@click="compare = compare === r.key ? null : r.key"
				>{{ r.label }}</button>
			</div>
			<ul v-if="compare" class="mkt__compare-list">
				<li v-for="k in kinds" :key="`cl-${k.key}`" :style="{ '--c': seriesColor(k.idx) }">
					<span class="mkt__compare-name">{{ k.name }}</span>
					<span class="mkt__compare-val">{{ k[compare] }}</span>
				</li>
			</ul>
		</div>

		<p class="pc-note">
			这六类不是互斥的分区，而是六种不同的工程决策组合：同一句话可能既是偏好又是程序性规程，
			该按哪一类处理取决于你打算怎么存、怎么召、怎么改。最值得注意的是「更新策略」一行——
			<b>覆盖写与追加写混用是记忆系统最常见的设计错误</b>：把情景经历覆盖掉就失去了「当时为什么这么判断」的依据，
			把偏好追加写则会攒出一堆互相矛盾的条目，谁也不知道哪条现在有效。
		</p>
	</div>
</template>

<style scoped>
.mkt { margin: 1rem 0; }
.mkt__chips { display: flex; flex-wrap: wrap; gap: var(--cs-space-2); }
.mkt__chip { display: inline-flex; align-items: center; gap: var(--cs-space-2); min-height: var(--cs-tap-target); cursor: pointer; }
.mkt__chip.is-selected { border-color: var(--c); background: color-mix(in srgb, var(--c) 14%, transparent); }
.mkt__dot { width: 8px; height: 8px; border-radius: var(--cs-radius-circle); background: var(--c); flex: none; }
.mkt__card { margin-top: var(--cs-space-4); background: var(--cs-color-bg); border: 1px solid var(--cs-color-border); border-top: 3px solid var(--c); border-radius: var(--cs-radius-lg); padding: var(--cs-space-5) var(--cs-space-6); }
.mkt__card header { display: flex; flex-wrap: wrap; align-items: baseline; gap: var(--cs-space-3); margin-bottom: var(--cs-space-4); }
.mkt__card header b { font-size: var(--cs-text-xl); color: var(--c); }
.mkt__card header i { font-size: var(--cs-text-3xs); font-style: normal; color: var(--cs-color-text-subtle); }
.mkt__dl { display: grid; grid-template-columns: 7.5rem minmax(0, 1fr); gap: var(--cs-space-2) var(--cs-space-4); margin: 0; }
.mkt__dl dt { font-family: var(--cs-font-mono); font-size: var(--cs-text-3xs); color: var(--cs-color-text-subtle); padding-top: 2px; }
.mkt__dl dd { margin: 0; font-size: var(--cs-text-base); line-height: var(--cs-leading-normal); color: var(--cs-color-text); }
.mkt__dl dd.is-pitfall { color: var(--cs-color-text-muted); border-top: 1px dashed var(--cs-color-border); padding-top: var(--cs-space-2); }
.mkt__dl dt:last-of-type { border-top: 1px dashed var(--cs-color-border); padding-top: calc(var(--cs-space-2) + 2px); }
.mkt__compare { margin-top: var(--cs-space-5); }
.mkt__compare-hint { margin: 0 0 var(--cs-space-2); font-size: var(--cs-text-sm); color: var(--cs-color-text-muted); }
.mkt__compare-tabs { display: flex; flex-wrap: wrap; gap: var(--cs-space-1); }
.mkt__compare-tab { min-height: var(--cs-tap-target); padding: var(--cs-space-1) var(--cs-space-4); cursor: pointer; font-size: var(--cs-text-xs); color: var(--cs-color-text-muted); background: var(--cs-color-bg); border: 1px solid var(--cs-color-border); border-radius: var(--cs-radius-pill); transition: var(--cs-transition-colors); }
.mkt__compare-tab:hover { background: var(--cs-color-bg-soft); }
.mkt__compare-tab.is-on { background: var(--cs-color-brand-soft); border-color: var(--cs-color-brand); color: var(--cs-color-brand); font-weight: 600; }
.mkt__compare-list { list-style: none; margin: var(--cs-space-3) 0 0; padding: 0; }
.mkt__compare-list li { display: grid; grid-template-columns: 6.5rem minmax(0, 1fr); gap: var(--cs-space-3); padding: var(--cs-space-2) 0 var(--cs-space-2) var(--cs-space-3); border-left: 3px solid var(--c); border-bottom: 1px dashed var(--cs-color-border); }
.mkt__compare-list li:last-child { border-bottom: 0; }
.mkt__compare-name { font-size: var(--cs-text-sm); font-weight: 600; color: var(--cs-color-text); }
.mkt__compare-val { font-size: var(--cs-text-sm); line-height: var(--cs-leading-normal); color: var(--cs-color-text-muted); }
@media (max-width: 640px) {
	.mkt__dl { grid-template-columns: 1fr; gap: 0; }
	.mkt__dl dt { margin-top: var(--cs-space-3); }
	.mkt__dl dt:last-of-type { border-top: 0; }
	.mkt__dl dd.is-pitfall { border-top: 0; padding-top: 0; }
	.mkt__compare-list li { grid-template-columns: 1fr; gap: var(--cs-space-1); }
}
</style>
