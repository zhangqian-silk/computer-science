<script setup lang="ts">
import { computed, ref } from "vue"

/**
 * 预建索引 vs 现场检索：同一个问题的两种答法，以及决定选哪个的那些条件。
 *
 * 这不是优劣对比。两条路线的分野来自一个具体事实：
 * 索引是源码在某一时刻的快照，而源码一直在变——重命名一次，快照就开始漂移。
 * 组件让读者按自己场景的条件去看结论怎么翻转，而不是背下某一方的胜负。
 */

type Cond = {
	key: string
	q: string
	/** 选左侧（现场读取）时该条件的取值描述 */
	live: string
	/** 选右侧（预建索引）时该条件的取值描述 */
	index: string
	/** 该条件在多大程度上把天平推向索引侧；正值偏索引，负值偏现场 */
	weight: number
}

const conds: Cond[] = [
	{ key: "size", q: "语料规模", live: "单仓或中小目录，逐层展开看得完", index: "跨仓、跨团队、几十万文档以上", weight: 3 },
	{ key: "churn", q: "变更频率", live: "改动频繁，改完就要立刻生效", index: "相对稳定，可接受分钟级滞后", weight: -3 },
	{ key: "shape", q: "关键关系的性质", live: "定义与引用这类可精确解析的确定性关系", index: "「业务名词落到哪个模块」这类语义关联", weight: 2 },
	{ key: "latency", q: "延迟容忍", live: "可以等，多轮往返换准确", index: "要求毫秒级返回，不能多轮试探", weight: 2 },
	{ key: "exact", q: "查询形态", live: "已知确切符号或错误码，字面匹配即可", index: "只能描述意图，说不出确切的词", weight: 2 },
	{ key: "ops", q: "可投入的维护", live: "不想维护任何离线链路", index: "有人负责索引构建、对账与重建", weight: -2 }
]

/** 每个条件当前选在哪一侧：true 表示选了「索引侧」的取值 */
const picks = ref<Record<string, boolean>>({
	size: false, churn: false, shape: false, latency: false, exact: false, ops: false
})

const score = computed(() =>
	conds.reduce((sum, c) => sum + (picks.value[c.key] ? Math.abs(c.weight) : -Math.abs(c.weight)), 0)
)

const verdict = computed(() => {
	if (score.value >= 4) return { tone: "index", text: "这些条件下预建索引更合适：规模与查询形态已经超出逐层展开能覆盖的范围。" }
	if (score.value <= -4) return { tone: "live", text: "这些条件下先不建索引：现场读取拿到的是当前状态，且省掉整条离线链路。" }
	return { tone: "both", text: "条件互相拉扯，说明这是典型的组合场景：用轻量结构做高层导航，把精确验证交给现场读取。" }
})

/** 天平位置，0 最偏现场、100 最偏索引 */
const tilt = computed(() => {
	const max = conds.reduce((s, c) => s + Math.abs(c.weight), 0)
	return Math.round(((score.value + max) / (2 * max)) * 100)
})

const sides = [
	{
		key: "live", name: "现场读取", claim: "不建副本，需要时直接看原件",
		wins: ["读到的永远是当前状态，不存在快照漂移", "结果是真实内容，不经过任何有损环节", "改完立即可见，零索引维护", "能指出确切位置，天然可溯源"],
		loses: ["依赖模型自己会规划，弱一些的模型会在目录里打转", "多轮往返，延迟与 token 都高", "语料一大就走不完，需要更高层的导航兜底"]
	},
	{
		key: "index", name: "预建索引", claim: "离线做成可查副本，查询时召回",
		wins: ["一次构建长期复用，单次查询开销低且稳定", "能承载远超单次窗口的语料规模", "支持「说不出确切词」的语义定位", "可加过滤与权限，按元数据裁剪候选"],
		loses: ["索引是某一刻的快照，源头一改就开始漂移", "整条离线链路都要维护，还要处理删除与重建", "召回是概率性的，漏召时表面上看不出来"]
	}
]
</script>

<template>
	<div class="rrc">
		<div class="rrc__sides">
			<section v-for="s in sides" :key="s.key" class="rrc__side" :class="`is-${s.key}`">
				<header>
					<b>{{ s.name }}</b>
					<span>{{ s.claim }}</span>
				</header>
				<ul class="rrc__wins">
					<li v-for="w in s.wins" :key="w">{{ w }}</li>
				</ul>
				<ul class="rrc__loses">
					<li v-for="l in s.loses" :key="l">{{ l }}</li>
				</ul>
			</section>
		</div>

		<div class="rrc__picker">
			<p class="rrc__hint">逐条选出你所在场景的实际取值，看结论往哪一侧偏：</p>
			<div v-for="c in conds" :key="c.key" class="rrc__cond">
				<span class="rrc__q">{{ c.q }}</span>
				<div class="infra-tabs rrc__opts">
					<button
						type="button"
						:aria-pressed="!picks[c.key]"
						@click="picks[c.key] = false"
					>{{ c.live }}</button>
					<button
						type="button"
						:aria-pressed="picks[c.key]"
						@click="picks[c.key] = true"
					>{{ c.index }}</button>
				</div>
			</div>
		</div>

		<div class="rrc__scale">
			<div class="rrc__track">
				<i :style="{ left: `${tilt}%` }" />
			</div>
			<div class="rrc__ends">
				<span>偏现场读取</span>
				<span>偏预建索引</span>
			</div>
		</div>

		<p class="rrc__verdict" :class="`is-${verdict.tone}`">{{ verdict.text }}</p>

		<p class="pc-note">
			天平的权重只用于呈现「哪些条件更能改变结论」，不是可套用的公式。
			真正需要记住的是分野的来源：<b>索引是某一时刻的快照，而源头一直在变</b>——
			这条决定了变更频率与维护投入这两项，往往比语料规模更早把结论定下来。
		</p>
	</div>
</template>

<style scoped>
.rrc { margin: 1rem 0; }
.rrc__sides { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: var(--cs-space-3); }
.rrc__side { background: var(--cs-color-bg); border: 1px solid var(--cs-color-border); border-top: 3px solid var(--cs-color-success); border-radius: var(--cs-radius-lg); padding: var(--cs-space-4); }
.rrc__side.is-index { border-top-color: var(--cs-color-brand); }
.rrc__side > header { margin-bottom: var(--cs-space-3); }
.rrc__side > header b { display: block; font-size: var(--cs-text-md); color: var(--cs-color-text); }
.rrc__side > header span { font-size: var(--cs-text-3xs); color: var(--cs-color-text-subtle); }
.rrc__side ul { margin: 0; padding: 0; list-style: none; display: grid; gap: var(--cs-space-1); }
.rrc__side li { position: relative; padding-left: var(--cs-space-5); font-size: var(--cs-text-sm); line-height: var(--cs-leading-normal); }
.rrc__wins li { color: var(--cs-color-text); }
.rrc__wins li::before { content: "＋"; position: absolute; left: 0; color: var(--cs-color-success); font-family: var(--cs-font-mono); font-size: var(--cs-text-3xs); }
.rrc__loses { margin-top: var(--cs-space-3) !important; padding-top: var(--cs-space-3) !important; border-top: 1px dashed var(--cs-color-border); }
.rrc__loses li { color: var(--cs-color-text-muted); }
.rrc__loses li::before { content: "－"; position: absolute; left: 0; color: var(--cs-color-danger); font-family: var(--cs-font-mono); font-size: var(--cs-text-3xs); }
.rrc__picker { margin-top: var(--cs-space-5); display: grid; gap: var(--cs-space-2); }
.rrc__hint { margin: 0 0 var(--cs-space-1); font-size: var(--cs-text-sm); color: var(--cs-color-text-muted); }
.rrc__cond { display: grid; grid-template-columns: 7rem 1fr; align-items: center; gap: var(--cs-space-3); }
.rrc__q { font-size: var(--cs-text-3xs); color: var(--cs-color-text-muted); }
.rrc__opts { flex-wrap: nowrap; }
.rrc__opts button { flex: 1; min-height: var(--cs-tap-target); font-size: var(--cs-text-3xs); text-align: left; }
.rrc__scale { margin-top: var(--cs-space-5); }
.rrc__track { position: relative; height: 6px; border-radius: var(--cs-radius-pill); background: linear-gradient(to right, var(--cs-color-success-soft), var(--cs-color-neutral-soft), var(--cs-color-brand-soft)); border: 1px solid var(--cs-color-border); }
.rrc__track i { position: absolute; top: 50%; width: 14px; height: 14px; margin-left: -7px; border-radius: var(--cs-radius-circle); background: var(--cs-color-brand); border: 2px solid var(--cs-color-bg); transform: translateY(-50%); transition: left var(--cs-duration-base) var(--cs-ease-standard); }
.rrc__ends { display: flex; justify-content: space-between; margin-top: var(--cs-space-1); font-size: var(--cs-text-3xs); color: var(--cs-color-text-subtle); }
.rrc__verdict { margin: var(--cs-space-4) 0 0; padding: var(--cs-space-3) var(--cs-space-4); border-left: 3px solid var(--cs-color-border-strong); background: var(--cs-color-bg-soft); border-radius: var(--cs-radius-sm); font-size: var(--cs-text-base); line-height: var(--cs-leading-relaxed); color: var(--cs-color-text); }
.rrc__verdict.is-live { border-left-color: var(--cs-color-success); }
.rrc__verdict.is-index { border-left-color: var(--cs-color-brand); }
.rrc__verdict.is-both { border-left-color: var(--cs-color-warning); }
@media (max-width: 760px) {
	.rrc__sides { grid-template-columns: 1fr; }
	.rrc__cond { grid-template-columns: 1fr; gap: var(--cs-space-1); }
	.rrc__opts { flex-wrap: wrap; }
}
</style>
