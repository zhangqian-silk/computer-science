<script setup lang="ts">
import { computed, ref } from "vue"
import { seriesColor } from "../series-palette"

/**
 * 工具规模变大之后的三条出路：全量注入、按需检索、代码编排。
 *
 * 三者省的是**不同的东西**，这是选型的关键：
 *   - 全量注入：什么都不省，定义与中间结果全进上下文；
 *   - 按需检索：省的是**工具定义**（只有命中的那几个进上下文）；
 *   - 代码编排：省的是**中间结果**（多步结果留在执行环境里，只有最终输出回到模型）。
 * 两者正交，可以叠加；把它们当成同一件事的两种做法，会在选错时得不到预期收益。
 *
 * 数值为便于对照的量级估算，口径写在常量注释里。
 */

/** 单个工具定义的平均占用（token，量级估算） */
const DEF_TOKENS = 900
/** 检索工具自身的常驻开销（token，量级估算） */
const SEARCH_OVERHEAD = 500
/** 单条中间结果的平均占用（token，量级估算） */
const RESULT_TOKENS = 1_800

const toolCount = ref(60)
const hitCount = ref(4)
const stepCount = ref(12)
/** 最终需要模型看到的汇总结果占用 */
const FINAL_TOKENS = 600

const rows = computed(() => {
	const defsAll = toolCount.value * DEF_TOKENS
	const defsHit = SEARCH_OVERHEAD + hitCount.value * DEF_TOKENS
	const resAll = stepCount.value * RESULT_TOKENS
	return [
		{
			key: "eager", name: "全量注入", idx: 0,
			defs: defsAll, res: resAll,
			one: "启动时把全部工具定义写进请求，每一步的结果也逐条回到上下文。",
			saves: "不省任何一项。",
			risk: "定义本身就可能吃掉大半窗口；名称相近的工具还会互相干扰选择。"
		},
		{
			key: "search", name: "按需检索", idx: 1,
			defs: defsHit, res: resAll,
			one: "只常驻一个检索工具，其余定义标记为延迟加载，命中之后再展开为完整定义。",
			saves: "省的是「工具定义」这一项。",
			risk: "多一次检索往返；工具很少或几乎全都会用到时，这次往返换不回什么。"
		},
		{
			key: "code", name: "代码编排", idx: 2,
			defs: defsAll, res: FINAL_TOKENS,
			one: "模型写一段代码在沙箱里调用多个工具，循环与筛选都在代码里完成，只有最终输出回到上下文。",
			saves: "省的是「中间结果」这一项。",
			risk: "需要代码执行环境；单步简单查询用它反而更慢。"
		},
		{
			key: "both", name: "两者叠加", idx: 3,
			defs: defsHit, res: FINAL_TOKENS,
			one: "延迟加载解决定义膨胀，代码编排解决结果膨胀。两项正交，可同时启用。",
			saves: "两项都省。",
			risk: "复杂度最高，只有在两个瓶颈同时存在时才值得。"
		}
	].map((r) => ({ ...r, total: r.defs + r.res }))
})

const maxTotal = computed(() => Math.max(...rows.value.map((r) => r.total), 1))
const detail = ref(1)
const cur = computed(() => rows.value[detail.value])
const base = computed(() => rows.value[0].total)

const fmt = (n: number) => (n >= 1000 ? `${(n / 1000).toFixed(n >= 10_000 ? 0 : 1)}K` : `${Math.round(n)}`)
</script>

<template>
	<div class="tsr">
		<div class="tsr__controls">
			<label class="tsr__ctl">
				<span>可用工具数</span>
				<input v-model.number="toolCount" type="range" min="5" max="300" step="5" />
				<b>{{ toolCount }}</b>
			</label>
			<label class="tsr__ctl">
				<span>本次命中数</span>
				<input v-model.number="hitCount" type="range" min="1" :max="Math.min(20, toolCount)" step="1" />
				<b>{{ hitCount }}</b>
			</label>
			<label class="tsr__ctl">
				<span>调用步数</span>
				<input v-model.number="stepCount" type="range" min="1" max="60" step="1" />
				<b>{{ stepCount }}</b>
			</label>
		</div>

		<div class="tsr__chart">
			<div v-for="(r, i) in rows" :key="r.key" class="tsr__row" :class="{ 'is-open': detail === i }" @click="detail = i">
				<span class="tsr__row-name">{{ r.name }}</span>
				<div class="tsr__track">
					<span class="tsr__seg" :style="{ width: `${(r.defs / maxTotal) * 100}%`, background: seriesColor(0) }" :title="`工具定义 ${fmt(r.defs)}`"></span>
					<span class="tsr__seg" :style="{ width: `${(r.res / maxTotal) * 100}%`, background: seriesColor(2) }" :title="`中间结果 ${fmt(r.res)}`"></span>
				</div>
				<b class="cs-mono tsr__row-val">{{ fmt(r.total) }}</b>
			</div>
		</div>

		<p class="tsr__legend">
			<span><i :style="{ background: seriesColor(0) }"></i>工具定义</span>
			<span><i :style="{ background: seriesColor(2) }"></i>中间结果</span>
		</p>

		<article class="tsr__panel">
			<header>
				<b class="pc-serif">{{ cur.name }}</b>
				<span class="tsr__badge">
					{{ fmt(cur.total) }} token
					<template v-if="detail > 0">· 相对全量注入 {{ ((1 - cur.total / base) * 100).toFixed(0) }}%</template>
				</span>
			</header>
			<p class="tsr__one">{{ cur.one }}</p>
			<dl class="tsr__kv">
				<dt>省的是</dt>
				<dd>{{ cur.saves }}</dd>
				<dt>代价</dt>
				<dd>{{ cur.risk }}</dd>
			</dl>
		</article>

		<p class="pc-note">
			单项占用为便于对照的量级估算，真实值取决于模式复杂度与返回体积。要点不在具体数字，而在
			<b>两条路径省的不是同一项</b>：延迟加载把「工具定义」的乘数从可用数降到命中数，
			代码编排把「中间结果」从每步一条降到最终一条。因此排查时应当先看占用构成——
			定义占大头就上延迟加载，结果占大头就上代码编排，两者都大才需要叠加。
		</p>
	</div>
</template>

<style scoped>
.tsr { margin: 1rem 0; background: var(--cs-color-bg); border: 1px solid var(--cs-color-border); border-radius: var(--cs-radius-lg); padding: var(--cs-space-5) var(--cs-space-6); }
.tsr__controls { display: grid; grid-template-columns: repeat(3, 1fr); gap: var(--cs-space-2) var(--cs-space-6); }
.tsr__ctl { display: grid; grid-template-columns: 5.5rem 1fr 2.6rem; align-items: center; gap: var(--cs-space-2); min-height: var(--cs-tap-target); font-size: var(--cs-text-sm); }
.tsr__ctl span { color: var(--cs-color-text-muted); }
.tsr__ctl b { font-family: var(--cs-font-mono); text-align: right; color: var(--cs-color-brand); }
.tsr__ctl input { width: 100%; accent-color: var(--cs-color-brand); }
.tsr__chart { margin: var(--cs-space-5) 0 var(--cs-space-2); display: grid; gap: var(--cs-space-2); }
.tsr__row { display: grid; grid-template-columns: 5.5rem minmax(0, 1fr) 3.2rem; align-items: center; gap: var(--cs-space-3); cursor: pointer; padding: var(--cs-space-1) var(--cs-space-2); border-radius: var(--cs-radius-sm); min-height: var(--cs-tap-target); transition: var(--cs-transition-colors); }
.tsr__row:hover { background: var(--cs-color-bg-soft); }
.tsr__row.is-open { background: var(--cs-color-brand-soft); }
.tsr__row-name { font-size: var(--cs-text-xs); color: var(--cs-color-text-muted); }
.tsr__row-val { font-size: var(--cs-text-sm); text-align: right; color: var(--cs-color-text); }
.tsr__track { display: flex; height: 18px; border-radius: var(--cs-radius-sm); background: var(--cs-color-neutral-soft); overflow: hidden; }
.tsr__seg { transition: width var(--cs-duration-base) var(--cs-ease-standard); }
.tsr__legend { display: flex; gap: var(--cs-space-5); margin: 0 0 var(--cs-space-4); padding-left: 6rem; font-size: var(--cs-text-3xs); color: var(--cs-color-text-subtle); }
.tsr__legend span { display: inline-flex; align-items: center; gap: var(--cs-space-2); }
.tsr__legend i { width: 10px; height: 10px; border-radius: var(--cs-radius-xs); }
.tsr__panel { background: var(--cs-color-bg); border: 1px solid var(--cs-color-border); border-left: 3px solid var(--cs-color-brand); border-radius: var(--cs-radius-lg); padding: var(--cs-space-5) var(--cs-space-6); }
.tsr__panel header { display: flex; flex-wrap: wrap; align-items: baseline; justify-content: space-between; gap: var(--cs-space-3); }
.tsr__panel header b { font-size: var(--cs-text-lg); }
.tsr__badge { font-family: var(--cs-font-mono); font-size: var(--cs-text-3xs); color: var(--cs-color-brand); border: 1px solid var(--cs-color-brand); border-radius: var(--cs-radius-pill); padding: var(--cs-space-1) var(--cs-space-3); }
.tsr__one { margin: var(--cs-space-3) 0; font-size: var(--cs-text-base); line-height: var(--cs-leading-relaxed); color: var(--cs-color-text); }
.tsr__kv { display: grid; grid-template-columns: 4rem minmax(0, 1fr); gap: var(--cs-space-2) var(--cs-space-3); margin: 0; padding-top: var(--cs-space-3); border-top: 1px dashed var(--cs-color-border); }
.tsr__kv dt { font-family: var(--cs-font-mono); font-size: var(--cs-text-3xs); color: var(--cs-color-text-subtle); }
.tsr__kv dd { margin: 0; font-size: var(--cs-text-sm); line-height: var(--cs-leading-normal); color: var(--cs-color-text-muted); }
@media (max-width: 720px) {
	.tsr__controls { grid-template-columns: 1fr; }
	.tsr__row { grid-template-columns: 4.6rem minmax(0, 1fr) 3rem; }
	.tsr__legend { padding-left: 0; }
}
</style>
