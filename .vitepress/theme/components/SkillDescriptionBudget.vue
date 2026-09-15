<script setup lang="ts">
import { computed, ref } from "vue"

/**
 * 描述预算与命中率：为什么「技能装得越多，描述反而越要写得准」。
 *
 * 两个机制叠在一起决定了隐式命中是否可靠：
 *   1. 描述清单有**字符预算**（公开实现按上下文窗口的一小比例折算）；
 *      超出后宿主会先缩短描述，再整条丢弃，被丢的往往是最少用到的那些；
 *   2. 描述一旦被截短，用于匹配的关键词可能正好落在被砍掉的那一段里。
 *
 * 于是「装了却不触发」这类故障通常不在模型判断力上，而在这条预算链上。
 * 组件让这条链可以被算出来，而不是靠感觉。
 */

/** 描述清单预算：按窗口的百分比折算（公开实现的量级为 1%–2%） */
const BUDGET_PCT = 1.5
/** 名称与分隔符等固定开销的估算字符数 */
const NAME_OVERHEAD = 28

const windowK = ref(200)
const skillCount = ref(60)
/** 作者写的完整描述平均字符数；规范给出的上限是 1024 */
const descLen = ref(180)
/** 关键触发词落在描述前多少字符内 */
const keywordAt = ref(90)

/**
 * 关键词不可能落在描述之外。滑块的 max 随描述长度收窄，但先调大关键词位置再调小描述长度时
 * 已写入的 ref 值不会自动回落，于是会显示「描述只有 40 字符，关键词在第 1020 字符处」
 * 这种自相矛盾的结论。这里显式钳制。
 */
const kwAt = computed(() => Math.min(keywordAt.value, descLen.value))

const budget = computed(() => Math.round(windowK.value * 1000 * (BUDGET_PCT / 100)))
const rawNeed = computed(() => skillCount.value * (descLen.value + NAME_OVERHEAD))
const overflow = computed(() => rawNeed.value > budget.value)

/** 溢出时每条描述被压到的可用长度 */
const perSkill = computed(() => {
	if (!overflow.value) return descLen.value
	const avail = budget.value / skillCount.value - NAME_OVERHEAD
	return Math.max(0, Math.floor(avail))
})

/** 名称一定保留，描述可被压到零；压到零意味着只剩名字可供匹配 */
const state = computed(() => {
	if (!overflow.value) return { tone: "pass", label: "预算充足" }
	if (perSkill.value >= kwAt.value) return { tone: "warn", label: "描述被压缩，关键词仍在" }
	if (perSkill.value > 0) return { tone: "fail", label: "关键词已被截掉" }
	return { tone: "fail", label: "描述归零，只剩名称" }
})

const verdict = computed(() => {
	if (!overflow.value) {
		return `${skillCount.value} 条描述共需约 ${fmt(rawNeed.value)} 字符，预算 ${fmt(budget.value)} 字符够用，每条描述完整进入清单，关键词不受影响。`
	}
	if (perSkill.value >= kwAt.value) {
		return `清单溢出，每条描述被压到约 ${perSkill.value} 字符。关键触发词位于第 ${kwAt.value} 字符之前，因此仍然保留——这正是「关键词前置」的价值所在。`
	}
	if (perSkill.value > 0) {
		return `清单溢出，每条描述只剩约 ${perSkill.value} 字符，而关键触发词写在第 ${kwAt.value} 字符处，已被截掉。此时技能仍然装着，但模型再也匹配不上它——故障现象是「明明装了却从不触发」，而根因不在模型判断力。`
	}
	return `预算已被条目数本身吃光，描述被压到零，只剩名称可供匹配。这种情况下隐式命中基本不可用，只能靠显式点名。`
})

/** 同一组参数在不同安装规模下的对照 */
const table = computed(() =>
	[10, 30, 60, 120, 250].map((n) => {
		const need = n * (descLen.value + NAME_OVERHEAD)
		const over = need > budget.value
		const per = over ? Math.max(0, Math.floor(budget.value / n - NAME_OVERHEAD)) : descLen.value
		return {
			n,
			need,
			per,
			kept: per >= kwAt.value,
			tone: per >= kwAt.value ? "pass" : "fail"
		}
	})
)

function fmt(n: number) {
	return n >= 1000 ? `${(n / 1000).toFixed(1)}K` : `${Math.round(n)}`
}

const fillPct = computed(() => Math.min(100, (rawNeed.value / budget.value) * 100))
</script>

<template>
	<div class="sdb">
		<div class="sdb__controls">
			<label class="sdb__ctl">
				<span>窗口容量</span>
				<input v-model.number="windowK" type="range" min="32" max="1000" step="8" />
				<b>{{ windowK }}K</b>
			</label>
			<label class="sdb__ctl">
				<span>已安装技能</span>
				<input v-model.number="skillCount" type="range" min="5" max="250" step="5" />
				<b>{{ skillCount }}</b>
			</label>
			<label class="sdb__ctl">
				<span>描述字符数</span>
				<input v-model.number="descLen" type="range" min="40" max="1024" step="20" />
				<b>{{ descLen }}</b>
			</label>
			<label class="sdb__ctl">
				<span>关键词位置</span>
				<input v-model.number="keywordAt" type="range" min="10" :max="descLen" step="10" />
				<b>{{ kwAt }}</b>
			</label>
		</div>

		<div class="sdb__gauge">
			<div class="sdb__track">
				<span class="sdb__fill" :class="`is-${state.tone}`" :style="{ width: `${fillPct}%` }"></span>
			</div>
			<div class="sdb__gauge-meta">
				<span class="cs-mono">需求 {{ fmt(rawNeed) }} / 预算 {{ fmt(budget) }} 字符</span>
				<span class="sdb__chip" :class="`is-${state.tone}`">{{ state.label }}</span>
			</div>
		</div>

		<p class="sdb__verdict" :class="`is-${state.tone}`">{{ verdict }}</p>

		<table class="sdb__table">
			<caption>
				固定描述长度 {{ descLen }} 字符、关键词在第 {{ kwAt }} 字符、窗口 {{ windowK }}K
				时，安装规模如何改变每条描述的实际可用长度
			</caption>
			<thead>
				<tr>
					<th>已安装</th>
					<th>清单需求</th>
					<th>每条实际长度</th>
					<th>关键词</th>
				</tr>
			</thead>
			<tbody>
				<tr v-for="r in table" :key="r.n" :class="{ 'is-cur': r.n === skillCount }">
					<td class="cs-mono">{{ r.n }}</td>
					<td class="cs-mono">{{ fmt(r.need) }}</td>
					<td class="cs-mono">{{ r.per }}</td>
					<td>
						<span class="sdb__chip" :class="`is-${r.tone}`">{{ r.kept ? "保留" : "被截掉" }}</span>
					</td>
				</tr>
			</tbody>
		</table>

		<p class="pc-note">
			预算按窗口的 {{ BUDGET_PCT }}% 折算，是对公开实现所述量级（约 1%–2%，窗口未知时退化为固定字符数）的教学化取值，
			不等于任何产品的实测参数。要点在结构而不在数值：<b>清单预算随窗口走，需求随安装数走，两者的增长速度并不匹配</b>。
			由此得到两条可迁移的写法约束——<b>关键触发词前置</b>，因为截断从尾部开始；
			<b>描述要写清适用与不适用的边界</b>，因为它是隐式匹配唯一的依据。
			「适用于所有工作」这类描述在预算充足时也没有选择价值。
		</p>
	</div>
</template>

<style scoped>
.sdb { margin: 1rem 0; background: var(--cs-color-bg); border: 1px solid var(--cs-color-border); border-radius: var(--cs-radius-lg); padding: var(--cs-space-5) var(--cs-space-6); }
.sdb__controls { display: grid; grid-template-columns: repeat(2, 1fr); gap: var(--cs-space-2) var(--cs-space-6); }
.sdb__ctl { display: grid; grid-template-columns: 5.6rem 1fr 2.8rem; align-items: center; gap: var(--cs-space-2); min-height: var(--cs-tap-target); font-size: var(--cs-text-sm); }
.sdb__ctl span { color: var(--cs-color-text-muted); }
.sdb__ctl b { font-family: var(--cs-font-mono); text-align: right; color: var(--cs-color-brand); }
.sdb__ctl input { width: 100%; min-height: var(--cs-tap-target); accent-color: var(--cs-color-brand); }
.sdb__gauge { margin: var(--cs-space-6) 0 var(--cs-space-4); }
.sdb__track { height: 20px; border-radius: var(--cs-radius-pill); background: var(--cs-color-neutral-soft); overflow: hidden; }
.sdb__fill { display: block; height: 100%; transition: width var(--cs-duration-base) var(--cs-ease-standard); }
.sdb__fill.is-pass { background: var(--cs-color-success); }
.sdb__fill.is-warn { background: var(--cs-color-warning); }
.sdb__fill.is-fail { background: var(--cs-color-danger); }
.sdb__gauge-meta { display: flex; flex-wrap: wrap; justify-content: space-between; align-items: center; gap: var(--cs-space-3); margin-top: var(--cs-space-2); font-size: var(--cs-text-3xs); color: var(--cs-color-text-subtle); }
.sdb__chip { display: inline-block; white-space: nowrap; font-family: var(--cs-font-mono); font-size: var(--cs-text-3xs); border-radius: var(--cs-radius-pill); padding: var(--cs-space-1) var(--cs-space-3); }
.sdb__chip.is-pass { background: var(--cs-color-success-soft); color: var(--cs-color-success); }
.sdb__chip.is-warn { background: var(--cs-color-warning-soft); color: var(--cs-color-warning); }
.sdb__chip.is-fail { background: var(--cs-color-danger-soft); color: var(--cs-color-danger); }
.sdb__verdict { margin: 0 0 var(--cs-space-6); padding: var(--cs-space-3) var(--cs-space-4); border-radius: var(--cs-radius-sm); font-size: var(--cs-text-base); line-height: var(--cs-leading-relaxed); }
.sdb__verdict.is-pass { background: var(--cs-color-success-soft); }
.sdb__verdict.is-warn { background: var(--cs-color-warning-soft); }
.sdb__verdict.is-fail { background: var(--cs-color-danger-soft); }
.sdb__table { width: 100%; margin: 0; font-size: var(--cs-text-sm); }
.sdb__table caption { caption-side: top; text-align: left; font-size: var(--cs-text-2xs); line-height: var(--cs-leading-normal); color: var(--cs-color-text-subtle); padding-bottom: var(--cs-space-2); }
.sdb__table th { text-align: left; font-size: var(--cs-text-2xs); color: var(--cs-color-text-muted); font-weight: 600; }
.sdb__table tr.is-cur { background: var(--cs-color-brand-soft); }
@media (max-width: 720px) {
	.sdb__controls { grid-template-columns: 1fr; }
	.sdb__table { font-size: var(--cs-text-xs); }
	/* 窄屏下表头会被压成两行、状态芯片也会换行；缩表头字号并让首列不折行 */
	.sdb__table th { font-size: var(--cs-text-3xs); white-space: nowrap; }
	.sdb__table td:first-child { white-space: nowrap; }
	.sdb__chip { padding: var(--cs-space-1) var(--cs-space-2); }
}
</style>
