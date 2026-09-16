<script setup lang="ts">
import { computed, ref } from "vue"

/**
 * 五级处置阶梯：从源头收窄到摘要压缩，按「信息损失风险」由低到高排列。
 *
 * 组件同时呈现两件事：
 *   1. 每一级的触发者、成本与可逆性——说明为什么次序不能颠倒；
 *   2. 逐级叠加后的窗口占用——说明前四级做足之后，留给摘要的压力有多小。
 * 条形图里的 token 数为便于对照的量级估算，口径写在下面的常量里。
 */

type Level = {
	key: string
	tier: string
	name: string
	trigger: string
	actor: string
	cost: string
	reversible: boolean
	color: string
	what: string
	guard: string
	/** 该级施加后从峰值中回收的 token（估算量级，仅用于说明相对关系） */
	freed: number
}

/** 未加任何治理时的窗口占用（估算量级） */
const PEAK = 620_000

const levels: Level[] = [
	{
		key: "l0", tier: "L0", name: "源头收窄", trigger: "工具构造返回值的那一刻",
		actor: "工具实现自身", cost: "零", reversible: true, color: "var(--cs-color-success)", freed: 210_000,
		what: "分页、字段裁剪、默认摘要模式、只取匹配行；单条超阈值直接写文件并返回路径与前几行预览。",
		guard: "预览必须包含足以判断「要不要展开」的信息，否则模型只会立刻把全文读回来。"
	},
	{
		key: "l1", tier: "L1", name: "轮内清理", trigger: "同一轮内，模型判定某条结果不相关之后",
		actor: "模型判定 + 框架执行", cost: "零 API 调用", reversible: true, color: "var(--cs-color-success)", freed: 96_000,
		what: "把已判定无关的只读结果正文换成占位符，不等这一轮结束——这些 token 会参与其间每一次推理。",
		guard: "豁免最近若干条结果（当前推理链可能正在引用）；单次回收量不达门槛就整批放弃。"
	},
	{
		key: "l2", tier: "L2", name: "轮次边界", trigger: "上一轮闭合、组装下一次载荷之前",
		actor: "框架", cost: "零 API 调用", reversible: true, color: "var(--cs-color-info)", freed: 84_000,
		what: "上一轮的推理块使命已尽，可整块不投影：它不参与工具调用配对，也无需留占位符。",
		guard: "推理可以丢，结论不行。只留在推理里的判断，下一轮就不存在了。"
	},
	{
		key: "l3", tier: "L3", name: "决策点闭合", trigger: "任务状态跃迁：定稿、提交、落盘、方案被否",
		actor: "任务状态", cost: "零 API 调用", reversible: true, color: "var(--cs-color-info)", freed: 122_000,
		what: "一批内容的价值随状态跃迁瞬间归零：方案定稿则支撑它的检索正文可清，代码已提交则原文可清。",
		guard: "判据是「信息已经转移到别处」，不是「它变旧了」。必须先确认写入成功，再清原文。"
	},
	{
		key: "l4", tier: "L4", name: "摘要压缩", trigger: "输入 token 越过触发线",
		actor: "模型（一次额外采样）", cost: "一次采样 + 细节损失", reversible: false, color: "var(--cs-color-danger)", freed: 62_000,
		what: "把切点之前的历史交给模型改写成一段结构化摘要，只保留最近消息与摘要继续运行。",
		guard: "唯一不可逆的一级。落到这里之前，前四级应当已经把能回收的都回收了。"
	}
]

/** 选中前 n 级：默认全开，用户可以逐级取消，观察峰值如何回涨 */
const enabled = ref<Set<string>>(new Set(levels.map((l) => l.key)))
const detail = ref(0)

function toggle(key: string) {
	const next = new Set(enabled.value)
	if (next.has(key)) next.delete(key)
	else next.add(key)
	enabled.value = next
}

const remaining = computed(() => {
	let v = PEAK
	for (const l of levels) if (enabled.value.has(l.key)) v -= l.freed
	return v
})
const barMax = PEAK
const segments = computed(() => {
	const on = levels.filter((l) => enabled.value.has(l.key))
	return on.map((l) => ({ key: l.key, color: l.color, pct: (l.freed / barMax) * 100, freed: l.freed, tier: l.tier }))
})
const remainPct = computed(() => (remaining.value / barMax) * 100)
const cheapFreed = computed(() =>
	levels.filter((l) => l.key !== "l4" && enabled.value.has(l.key)).reduce((s, l) => s + l.freed, 0)
)
const fmt = (n: number) => `${Math.round(n / 1000)}K`
const cur = computed(() => levels[detail.value])
</script>

<template>
	<div class="cld">
		<div class="cld__bar" role="img"
			:aria-label="`峰值 ${fmt(PEAK)}，已启用的处置级别共回收 ${fmt(PEAK - remaining)}，剩余 ${fmt(remaining)}`">
			<span
				v-for="s in segments"
				:key="s.key"
				class="cld__seg"
				:style="{ width: `${s.pct}%`, background: s.color }"
				:title="`${s.tier} 回收 ${fmt(s.freed)}`"
			>{{ s.pct > 9 ? `${s.tier} −${fmt(s.freed)}` : "" }}</span>
			<span class="cld__seg cld__seg--rest" :style="{ width: `${remainPct}%` }">{{ remainPct > 12 ? `剩余 ${fmt(remaining)}` : "" }}</span>
		</div>
		<p class="cld__read">
			未加治理的峰值 <b>{{ fmt(PEAK) }}</b>；当前启用的级别共回收 <b>{{ fmt(PEAK - remaining) }}</b>，
			其中 <b>{{ fmt(cheapFreed) }}</b> 来自 L0–L3 这四级零 API 开销、可回退的操作。
			取消勾选任意一级，可以看到压力如何转移到后面的级别。
		</p>

		<div class="cld__grid">
			<button
				v-for="(l, i) in levels"
				:key="l.key"
				type="button"
				class="cld__card"
				:class="{ 'is-off': !enabled.has(l.key), 'is-open': detail === i }"
				:style="{ '--c': l.color }"
				:aria-pressed="enabled.has(l.key)"
				@click="detail = i"
			>
				<span class="cld__title"><i class="cld__tier">{{ l.tier }}</i><strong>{{ l.name }}</strong></span>
				<span class="cld__meta">{{ l.actor }} · {{ l.cost }}</span>
				<span class="cld__foot">
					<span class="cld__rev" :class="l.reversible ? 'is-ok' : 'is-no'">{{ l.reversible ? "可回退" : "不可逆" }}</span>
					<label class="cld__sw" @click.stop>
						<input type="checkbox" :checked="enabled.has(l.key)" @change="toggle(l.key)" />
						<span>计入</span>
					</label>
				</span>
			</button>
		</div>

		<article class="cld__panel" :style="{ '--c': cur.color }">
			<header>
				<b>{{ cur.tier }} · {{ cur.name }}</b>
				<span class="cld__badge">{{ cur.trigger }}</span>
			</header>
			<p class="cld__what">{{ cur.what }}</p>
			<p class="cld__guard"><b>护栏</b>{{ cur.guard }}</p>
		</article>

	</div>
</template>

<style scoped>
.cld { margin: 1rem 0; }
.cld__bar { display: flex; height: 30px; border-radius: var(--cs-radius-sm); overflow: hidden; border: 1px solid var(--cs-color-border); }
.cld__seg { display: flex; align-items: center; justify-content: center; font-family: var(--cs-font-mono); font-size: var(--cs-text-3xs); color: var(--cs-color-on-brand); white-space: nowrap; transition: width var(--cs-duration-base) var(--cs-ease-standard); }
.cld__seg--rest { background: var(--cs-color-neutral-soft); color: var(--cs-color-text-muted); }
.cld__read { margin: var(--cs-space-2) 0 var(--cs-space-4); font-size: var(--cs-text-base); line-height: var(--cs-leading-relaxed); color: var(--cs-color-text-muted); }
.cld__read b { color: var(--cs-color-text); font-family: var(--cs-font-mono); }
.cld__grid { display: grid; grid-template-columns: repeat(5, 1fr); gap: var(--cs-space-2); }
.cld__card { position: relative; display: grid; gap: 1px; text-align: left; cursor: pointer; background: var(--cs-color-bg); border: 1px solid var(--cs-color-border); border-top: 3px solid var(--c); border-radius: var(--cs-radius-lg); padding: var(--cs-space-2) var(--cs-space-3); transition: var(--cs-transition-colors); }
.cld__card:hover { background: var(--cs-color-bg-soft); }
.cld__card.is-open { background: color-mix(in srgb, var(--c) 10%, transparent); border-color: var(--c); }
.cld__card.is-off { opacity: .55; }
.cld__title { display: flex; align-items: baseline; gap: var(--cs-space-2); }
.cld__tier { font-family: var(--cs-font-mono); font-size: var(--cs-text-3xs); letter-spacing: .1em; color: var(--c); font-style: normal; }
.cld__card strong { font-size: var(--cs-text-sm); line-height: var(--cs-leading-tight); }
.cld__meta { font-size: var(--cs-text-3xs); color: var(--cs-color-text-muted); }
.cld__foot { display: flex; align-items: center; gap: var(--cs-space-2); }
.cld__rev { font-size: var(--cs-text-3xs); }
.cld__rev.is-ok { color: var(--cs-color-success); }
.cld__rev.is-no { color: var(--cs-color-danger); }
.cld__sw { display: inline-flex; align-items: center; gap: var(--cs-space-1); margin-left: auto; min-height: var(--cs-tap-target); font-size: var(--cs-text-3xs); color: var(--cs-color-text-subtle); cursor: pointer; }
.cld__panel { margin-top: var(--cs-space-2); background: var(--cs-color-bg); border: 1px solid var(--cs-color-border); border-left: 3px solid var(--c); border-radius: var(--cs-radius-lg); padding: var(--cs-space-2) var(--cs-space-4); }
.cld__panel header { display: flex; flex-wrap: wrap; align-items: baseline; justify-content: space-between; gap: var(--cs-space-3); }
.cld__panel header b { font-size: var(--cs-text-sm); }
.cld__badge { font-family: var(--cs-font-mono); font-size: var(--cs-text-3xs); color: var(--c); border: 1px solid var(--c); border-radius: var(--cs-radius-pill); padding: var(--cs-space-1) var(--cs-space-3); }
.cld__what { margin: var(--cs-space-1) 0; font-size: var(--cs-text-sm); line-height: var(--cs-leading-normal); color: var(--cs-color-text); }
.cld__guard { margin: 0; padding-top: var(--cs-space-1); border-top: 1px dashed var(--cs-color-border); font-size: var(--cs-text-xs); line-height: var(--cs-leading-normal); color: var(--cs-color-text-muted); }
.cld__guard b { font-weight: 600; color: var(--cs-color-text-subtle); margin-right: var(--cs-space-1); }
@media (max-width: 860px) {
	.cld__grid { grid-template-columns: repeat(2, 1fr); }
}
@media (max-width: 520px) {
	.cld__grid { grid-template-columns: 1fr; gap: var(--cs-space-1); }
	.cld__card { display: flex; flex-wrap: wrap; align-items: center; gap: var(--cs-space-2); padding: var(--cs-space-2) var(--cs-space-3); }
	.cld__tier { width: auto; }
	.cld__card strong { font-size: var(--cs-text-sm); }
	.cld__meta { display: none; }
	.cld__cost { margin-left: auto; }
	.cld__sw { min-height: auto; }
	/* 窄屏各段太窄放不下文字，一律隐去；数值由条形图下方的说明文字承担 */
	.cld__seg { font-size: 0; }
}
</style>
