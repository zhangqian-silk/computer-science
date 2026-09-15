<script setup lang="ts">
import { computed, ref } from "vue"
import { seriesColor } from "../series-palette"

/**
 * 知识进入一次请求的五条路径，按「综合发生在什么时候」排列。
 *
 * 排列次序本身是结论：越靠左，内容越接近它的原件，维护越轻、越不会过期；
 * 越靠右，越多工作被提前做掉，单次请求越省，但副本与真相之间的距离越远。
 * 每条路径给出四个维度的相对位置——它们不是评分，而是用来说明「代价换到了什么」。
 */

type Axis = "fresh" | "upkeep" | "trace" | "scale"

type Route = {
	key: string
	step: string
	name: string
	when: string
	how: string
	/** 该路径把「综合知识」这件事放在什么时刻做 */
	moment: string
	cost: string
	/** 四个维度的相对档位，1 最低、4 最高 */
	axes: Record<Axis, number>
	color: string
}

const axisMeta: { key: Axis; label: string; high: string; low: string }[] = [
	{ key: "fresh", label: "新鲜度", high: "读到的就是当前状态", low: "副本落后于原件" },
	{ key: "upkeep", label: "维护成本", high: "需要持续构建与对账", low: "几乎不需要维护" },
	{ key: "trace", label: "可溯源", high: "能指回具体位置", low: "说不出依据在哪" },
	{ key: "scale", label: "可承载规模", high: "可扩到海量语料", low: "受单次窗口限制" }
]

const routes: Route[] = [
	{
		key: "live", step: "①", name: "现场读取", color: seriesColor(0),
		when: "请求进行中", moment: "完全不提前综合，需要什么当场去看",
		how: "把列目录、按模式匹配、读取指定片段这类原语交给模型，由它自己决定看哪里、看几次。",
		cost: "多轮往返带来延迟与 token；模型不会规划就会在目录里打转。",
		axes: { fresh: 4, upkeep: 1, trace: 4, scale: 2 }
	},
	{
		key: "convention", step: "②", name: "约定文件", color: seriesColor(1),
		when: "会话开始", moment: "由人提前写好，随源码一起版本化",
		how: "把必须默认知道的规则与导航写进仓库内的说明文件，按目录层级逐级加载。",
		cost: "占每一次调用的固定预算，写长了成本上升而遵守率下降。",
		axes: { fresh: 3, upkeep: 2, trace: 3, scale: 1 }
	},
	{
		key: "retrieval", step: "③", name: "检索召回", color: seriesColor(2),
		when: "查询到达时", moment: "离线切分与建索引，查询时现场排序拼装",
		how: "把语料切成片段、建立稠密与稀疏两路索引，按查询召回、重排，再把少数片段拼进载荷。",
		cost: "整条离线链路都要维护；召回不确定，且副本与原文之间存在延迟。",
		axes: { fresh: 2, upkeep: 4, trace: 3, scale: 4 }
	},
	{
		key: "tool", step: "④", name: "接口直连", color: seriesColor(3),
		when: "请求进行中", moment: "不做副本，把问题翻译成对权威系统的一次调用",
		how: "把知识以工具或接口的形式暴露，由模型在需要时发起调用，读到的是当前值。",
		cost: "要求对象本身有结构与契约；权限与危险操作必须在服务端拦住。",
		axes: { fresh: 4, upkeep: 2, trace: 4, scale: 3 }
	},
	{
		key: "params", step: "⑤", name: "参数内化", color: seriesColor(4),
		when: "训练时", moment: "在训练阶段一次性写进权重",
		how: "用继续预训练或微调把风格、格式与领域用语固化进模型，推理时无需任何检索。",
		cost: "更新以训练周期计，且无法回答「这句话的依据是什么」。",
		axes: { fresh: 1, upkeep: 3, trace: 1, scale: 2 }
	}
]

const openKey = ref(routes[2].key)
const cur = computed(() => routes.find((r) => r.key === openKey.value) ?? routes[0])
const axisWidth = (level: number) => `${level * 25}%`
</script>

<template>
	<div class="ksr">
		<ol class="ksr__track">
			<li v-for="r in routes" :key="r.key" :style="{ '--c': r.color }">
				<button
					type="button"
					class="ksr__node"
					:class="{ 'is-open': openKey === r.key }"
					:aria-pressed="openKey === r.key"
					@click="openKey = r.key"
				>
					<span class="ksr__step">{{ r.step }}</span>
					<strong>{{ r.name }}</strong>
					<span class="ksr__when">{{ r.when }}</span>
				</button>
			</li>
		</ol>

		<p class="ksr__axisnote">
			从左到右，「把知识综合成可用形态」这件事发生得越来越早：
			<b>①④ 完全不提前做</b>，<b>② 由人提前写好</b>，<b>③ 离线建好索引、查询时现场拼装</b>，<b>⑤ 在训练时就写进权重</b>。
		</p>

		<section class="ksr__panel" :style="{ '--c': cur.color }">
			<header>
				<b>{{ cur.step }} {{ cur.name }}</b>
				<span class="ksr__badge">{{ cur.moment }}</span>
			</header>

			<p class="ksr__how">{{ cur.how }}</p>

			<div class="ksr__axes">
				<div v-for="a in axisMeta" :key="a.key" class="ksr__axis">
					<span class="ksr__axis-label">{{ a.label }}</span>
					<span class="infra-bar"><i :style="{ width: axisWidth(cur.axes[a.key]), background: cur.color }" /></span>
					<span class="ksr__axis-hint">{{ cur.axes[a.key] >= 3 ? a.high : a.low }}</span>
				</div>
			</div>

			<p class="ksr__cost"><i>代价</i>{{ cur.cost }}</p>
		</section>

		<p class="pc-note">
			这五条路径不是竞品清单，真实系统几乎总是组合使用：规则与导航走②，大体量语料走③，
			强时效与聚合类问题走④，需要看真实源码时回落到①，而⑤只承担风格与用语这类「模型行为」。
			四个维度的档位用于呈现相对关系，不构成对任何方案的评分。
		</p>
	</div>
</template>

<style scoped>
.ksr { margin: 1rem 0; }
.ksr__track { display: grid; grid-template-columns: repeat(5, minmax(0, 1fr)); gap: var(--cs-space-2); margin: 0 0 var(--cs-space-3); padding: 0; list-style: none; }
.ksr__track li { margin: 0; }
.ksr__node { display: grid; gap: var(--cs-space-1); width: 100%; min-height: var(--cs-tap-target); padding: var(--cs-space-3) var(--cs-space-2); text-align: left; cursor: pointer; background: var(--cs-color-bg); border: 1px solid var(--cs-color-border); border-top: 3px solid var(--c); border-radius: var(--cs-radius-lg); transition: var(--cs-transition-colors); }
.ksr__node:hover { background: var(--cs-color-bg-soft); }
.ksr__node.is-open { background: color-mix(in srgb, var(--c) 10%, transparent); border-color: var(--c); }
.ksr__step { font-family: var(--cs-font-mono); font-size: var(--cs-text-3xs); color: var(--c); }
.ksr__node strong { font-size: var(--cs-text-md); line-height: var(--cs-leading-tight); color: var(--cs-color-text); }
.ksr__when { font-size: var(--cs-text-3xs); color: var(--cs-color-text-subtle); }
.ksr__axisnote { margin: 0 0 var(--cs-space-4); font-size: var(--cs-text-sm); line-height: var(--cs-leading-relaxed); color: var(--cs-color-text-muted); }
.ksr__axisnote b { color: var(--cs-color-text); font-weight: 600; }
.ksr__panel { background: var(--cs-color-bg); border: 1px solid var(--cs-color-border); border-left: 3px solid var(--c); border-radius: var(--cs-radius-lg); padding: var(--cs-space-5) var(--cs-space-6); }
.ksr__panel header { display: flex; flex-wrap: wrap; align-items: baseline; justify-content: space-between; gap: var(--cs-space-3); }
.ksr__panel header b { font-size: var(--cs-text-lg); color: var(--cs-color-text); }
.ksr__badge { font-size: var(--cs-text-3xs); color: var(--c); border: 1px solid var(--c); border-radius: var(--cs-radius-pill); padding: var(--cs-space-1) var(--cs-space-3); }
.ksr__how { margin: var(--cs-space-3) 0 var(--cs-space-4); font-size: var(--cs-text-base); line-height: var(--cs-leading-relaxed); color: var(--cs-color-text); }
.ksr__axes { display: grid; gap: var(--cs-space-2); }
.ksr__axis { display: grid; grid-template-columns: 5.5rem 1fr minmax(0, 11rem); align-items: center; gap: var(--cs-space-3); }
.ksr__axis-label { font-size: var(--cs-text-3xs); color: var(--cs-color-text-muted); }
.ksr__axis-hint { font-size: var(--cs-text-3xs); color: var(--cs-color-text-subtle); }
.ksr__cost { margin: var(--cs-space-4) 0 0; padding-top: var(--cs-space-3); border-top: 1px dashed var(--cs-color-border); font-size: var(--cs-text-sm); line-height: var(--cs-leading-normal); color: var(--cs-color-text-muted); }
.ksr__cost i { display: block; font-family: var(--cs-font-mono); font-size: var(--cs-text-3xs); letter-spacing: .08em; font-style: normal; color: var(--cs-color-text-subtle); margin-bottom: var(--cs-space-1); }
@media (max-width: 860px) {
	.ksr__track { grid-template-columns: repeat(2, minmax(0, 1fr)); }
	.ksr__axis { grid-template-columns: 5rem 1fr; }
	.ksr__axis-hint { grid-column: 1 / -1; }
}
@media (max-width: 520px) {
	.ksr__track { grid-template-columns: 1fr; }
	.ksr__panel { padding: var(--cs-space-4); }
}
</style>
