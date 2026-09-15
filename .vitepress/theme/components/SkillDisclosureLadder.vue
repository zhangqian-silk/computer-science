<script setup lang="ts">
import { computed, ref } from "vue"

/**
 * 渐进披露的三级加载与上下文占用对照。
 *
 * 组件要说明的不是「有三层」这件事本身，而是三层各自的**计费时机**：
 * 元信息按已安装技能数计费且常驻，正文按本次真正命中的技能数计费，
 * 附件按实际读取的文件数计费。三者的乘数完全不同，
 * 这才是「装很多技能不贵、装很多正文很贵」这条结论的来源。
 *
 * 拖动「已安装技能数」可以看到：全量注入随安装数线性上涨，
 * 渐进披露只有元信息那一项随安装数上涨，而它的单项成本小两个量级。
 */

/** 每个技能的元信息（name + description）估算占用，量级取自公开文档给出的「约 100 token」 */
const METADATA_PER_SKILL = 100

const installed = ref(40)
/** 本次任务真正命中并读入正文的技能数 */
const activated = ref(1)
/** SKILL.md 正文估算占用；公开建议是控制在 5000 token 以内 */
const bodyTokens = ref(3_000)
/** 被读取的附属文件数与单文件估算占用 */
const resourceFiles = ref(1)
const resourceTokens = ref(2_000)

/**
 * 命中数不可能超过已安装数。滑块的 max 会随安装数收窄，但先调大命中数再调小安装数时
 * 已写入的 ref 值不会自动回落，因此这里显式钳制——否则会算出「渐进披露比全量注入还贵」
 * 这种不成立的结论。
 */
const hit = computed(() => Math.min(activated.value, installed.value))

const levels = computed(() => [
	{
		key: "meta",
		tier: "第一级",
		name: "元信息",
		when: "启动时，全部技能一次性载入",
		multiplier: `× ${installed.value} 个已安装技能`,
		cost: installed.value * METADATA_PER_SKILL,
		unit: `${METADATA_PER_SKILL} token / 个`,
		what: "name 与 description 进入系统提示。它是唯一常驻的一层，决定模型「知道有哪些技能可用」。",
		note: "常驻意味着它随安装数线性增长——这是渐进披露唯一无法回避的固定支出。",
		idx: 0
	},
	{
		key: "body",
		tier: "第二级",
		name: "正文指令",
		when: "命中之后，读取该技能的 SKILL.md",
		multiplier: `× ${hit.value} 个本次命中`,
		cost: hit.value * bodyTokens.value,
		unit: `${fmt(bodyTokens.value)} / 个`,
		what: "SKILL.md 的 Markdown 正文整体进入上下文：步骤、边界条件、判据、对附属文件的索引。",
		note: "整体加载，没有「只读一半」这回事——所以正文该短，长材料该外移。",
		idx: 1
	},
	{
		key: "res",
		tier: "第三级",
		name: "附属资源",
		when: "正文指向之后，按需逐个读取",
		multiplier: `× ${resourceFiles.value} 个实际读取`,
		cost: resourceFiles.value * resourceTokens.value,
		unit: `${fmt(resourceTokens.value)} / 个`,
		what: "参考文档、模板、schema 被读进上下文；脚本走执行路径，只有输出计费，代码本身不进上下文。",
		note: "未被读取的文件占用为零，因此可打包的资料量在实践中没有上限。",
		idx: 2
	}
])

const progressive = computed(() => levels.value.reduce((s, l) => s + l.cost, 0))
/**
 * 对照口径：把全部已安装技能的正文都塞进系统提示。
 *
 * 附件那一项两种口径下都要付——它是本次任务真的读了那几个文件，与「正文怎么加载」无关，
 * 所以必须一并计入基线，否则会算出「渐进披露比全量注入还贵」这种伪结论。
 * 把附件对齐之后，两者之差就干净地只剩正文那一项的乘数差异，正是本图要说明的那一点。
 */
const eager = computed(
	() => installed.value * (METADATA_PER_SKILL + bodyTokens.value) + resourceFiles.value * resourceTokens.value
)
const ratio = computed(() => (progressive.value > 0 ? eager.value / progressive.value : 0))
const maxBar = computed(() => Math.max(eager.value, progressive.value, 1))

function fmt(n: number) {
	return n >= 1000 ? `${(n / 1000).toFixed(n >= 10_000 ? 0 : 1)}K` : `${Math.round(n)}`
}

const detail = ref(0)
const cur = computed(() => levels.value[detail.value])
</script>

<template>
	<div class="sdl">
		<div class="sdl__controls">
			<label class="sdl__ctl">
				<span>已安装技能数</span>
				<input v-model.number="installed" type="range" min="1" max="200" step="1" />
				<b>{{ installed }}</b>
			</label>
			<label class="sdl__ctl">
				<span>本次命中数</span>
				<input v-model.number="activated" type="range" min="0" :max="Math.min(8, installed)" step="1" />
				<b>{{ hit }}</b>
			</label>
			<label class="sdl__ctl">
				<span>正文长度</span>
				<input v-model.number="bodyTokens" type="range" min="500" max="12000" step="500" />
				<b>{{ fmt(bodyTokens) }}</b>
			</label>
			<label class="sdl__ctl">
				<span>读取附件数</span>
				<input v-model.number="resourceFiles" type="range" min="0" max="6" step="1" />
				<b>{{ resourceFiles }}</b>
			</label>
		</div>

		<div class="sdl__compare">
			<div class="sdl__row">
				<span class="sdl__row-label">渐进披露</span>
				<div class="sdl__track">
					<span
						v-for="l in levels"
						:key="l.key"
						class="sdl__seg"
						:style="{ width: `${(l.cost / maxBar) * 100}%`, background: `var(--cs-series-${l.idx + 1})` }"
						:title="`${l.name} ${fmt(l.cost)}`"
					></span>
				</div>
				<b class="cs-mono sdl__row-val">{{ fmt(progressive) }}</b>
			</div>
			<div class="sdl__row">
				<span class="sdl__row-label">全量注入</span>
				<div class="sdl__track">
					<span class="sdl__seg sdl__seg--eager" :style="{ width: `${(eager / maxBar) * 100}%` }"></span>
				</div>
				<b class="cs-mono sdl__row-val">{{ fmt(eager) }}</b>
			</div>
		</div>

		<p class="sdl__read">
			同样一批 <b>{{ installed }}</b> 个技能：把全部正文写进系统提示要花
			<b>{{ fmt(eager) }}</b>，而渐进披露花 <b>{{ fmt(progressive) }}</b>，
			<template v-if="ratio >= 1.05">相差约 <b>{{ ratio.toFixed(1) }} 倍</b>。</template>
			<template v-else>
				两者相当——因为命中数已经接近安装数。渐进披露的收益来自「装得多、用得少」这一前提，前提不成立时它并不省。
			</template>
			差距的来源只有一处：<b>正文那一项的乘数从「已安装数」降成了「本次命中数」</b>。
		</p>

		<div class="sdl__tiers">
			<button
				v-for="(l, i) in levels"
				:key="l.key"
				type="button"
				class="sdl__tier"
				:class="{ 'is-open': detail === i, 'is-zero': l.cost === 0 }"
				:style="{ '--c': `var(--cs-series-${l.idx + 1})` }"
				:aria-pressed="detail === i"
				@click="detail = i"
			>
				<span class="sdl__tier-tag">{{ l.tier }}</span>
				<strong>{{ l.name }}</strong>
				<span class="sdl__tier-mul">{{ l.multiplier }}</span>
				<span class="sdl__tier-cost cs-mono">{{ fmt(l.cost) }} token</span>
			</button>
		</div>

		<article class="sdl__panel" :style="{ '--c': `var(--cs-series-${cur.idx + 1})` }">
			<header>
				<b class="pc-serif">{{ cur.tier }} · {{ cur.name }}</b>
				<span class="sdl__badge">{{ cur.when }}</span>
			</header>
			<p class="sdl__what">{{ cur.what }}</p>
			<p class="sdl__note"><i>代价形态</i>{{ cur.note }}（单项 {{ cur.unit }}）</p>
		</article>

		<p class="pc-note">
			token 数为便于对照的量级估算：元信息按公开文档给出的「每个技能约 100 token」取值，正文与附件由上方控件给定。
			真实占用取决于描述长度与技能内容，此处只呈现三级之间的相对关系。可迁移的结论是
			<b>三级的乘数不同</b>——元信息乘已安装数，正文乘命中数，附件乘实际读取数。所以
			<b>多装技能的边际成本很低，写长正文的边际成本很高</b>。
		</p>
	</div>
</template>

<style scoped>
.sdl { margin: 1rem 0; background: var(--cs-color-bg); border: 1px solid var(--cs-color-border); border-radius: var(--cs-radius-lg); padding: var(--cs-space-5) var(--cs-space-6); }
.sdl__controls { display: grid; grid-template-columns: repeat(2, 1fr); gap: var(--cs-space-2) var(--cs-space-6); }
.sdl__ctl { display: grid; grid-template-columns: 6rem 1fr 2.8rem; align-items: center; gap: var(--cs-space-2); min-height: var(--cs-tap-target); font-size: var(--cs-text-sm); }
.sdl__ctl span { color: var(--cs-color-text-muted); }
.sdl__ctl b { font-family: var(--cs-font-mono); text-align: right; color: var(--cs-color-brand); }
.sdl__ctl input { width: 100%; min-height: var(--cs-tap-target); accent-color: var(--cs-color-brand); }
.sdl__compare { margin: var(--cs-space-6) 0 var(--cs-space-4); display: grid; gap: var(--cs-space-3); }
.sdl__row { display: grid; grid-template-columns: 4.5rem minmax(0, 1fr) 3.4rem; align-items: center; gap: var(--cs-space-3); }
.sdl__row-label { font-size: var(--cs-text-xs); color: var(--cs-color-text-muted); }
.sdl__row-val { font-size: var(--cs-text-sm); text-align: right; color: var(--cs-color-text); }
.sdl__track { display: flex; height: 20px; border-radius: var(--cs-radius-sm); background: var(--cs-color-neutral-soft); overflow: hidden; }
.sdl__seg { transition: width var(--cs-duration-base) var(--cs-ease-standard); }
.sdl__seg--eager { background: var(--cs-color-danger); }
.sdl__read { margin: 0 0 var(--cs-space-6); font-size: var(--cs-text-base); line-height: var(--cs-leading-relaxed); color: var(--cs-color-text-muted); }
.sdl__read b { color: var(--cs-color-text); }
.sdl__tiers { display: grid; grid-template-columns: repeat(3, 1fr); gap: var(--cs-space-2); }
.sdl__tier { display: grid; gap: var(--cs-space-1); text-align: left; cursor: pointer; background: var(--cs-color-bg); border: 1px solid var(--cs-color-border); border-top: 3px solid var(--c); border-radius: var(--cs-radius-lg); padding: var(--cs-space-3); transition: var(--cs-transition-colors); }
.sdl__tier:hover { background: var(--cs-color-bg-soft); }
.sdl__tier.is-open { background: color-mix(in srgb, var(--c) 10%, transparent); border-color: var(--c); }
.sdl__tier.is-zero { opacity: .6; }
.sdl__tier-tag { font-family: var(--cs-font-mono); font-size: var(--cs-text-3xs); letter-spacing: .08em; color: var(--c); }
.sdl__tier strong { font-size: var(--cs-text-md); line-height: var(--cs-leading-tight); }
.sdl__tier-mul { font-size: var(--cs-text-3xs); color: var(--cs-color-text-muted); }
.sdl__tier-cost { font-size: var(--cs-text-3xs); color: var(--cs-color-text-subtle); }
.sdl__panel { margin-top: var(--cs-space-4); background: var(--cs-color-bg); border: 1px solid var(--cs-color-border); border-left: 3px solid var(--c); border-radius: var(--cs-radius-lg); padding: var(--cs-space-5) var(--cs-space-6); }
.sdl__panel header { display: flex; flex-wrap: wrap; align-items: baseline; justify-content: space-between; gap: var(--cs-space-3); }
.sdl__panel header b { font-size: var(--cs-text-lg); }
.sdl__badge { font-family: var(--cs-font-mono); font-size: var(--cs-text-3xs); color: var(--c); border: 1px solid var(--c); border-radius: var(--cs-radius-pill); padding: var(--cs-space-1) var(--cs-space-3); }
.sdl__what { margin: var(--cs-space-3) 0; font-size: var(--cs-text-base); line-height: var(--cs-leading-relaxed); color: var(--cs-color-text); }
.sdl__note { margin: 0; padding-top: var(--cs-space-3); border-top: 1px dashed var(--cs-color-border); font-size: var(--cs-text-sm); line-height: var(--cs-leading-normal); color: var(--cs-color-text-muted); }
.sdl__note i { display: block; font-family: var(--cs-font-mono); font-size: var(--cs-text-3xs); letter-spacing: .08em; font-style: normal; color: var(--cs-color-text-subtle); margin-bottom: var(--cs-space-1); }
@media (max-width: 720px) {
	.sdl__controls { grid-template-columns: 1fr; }
	.sdl__tiers { grid-template-columns: 1fr; }
	.sdl__row { grid-template-columns: 4rem minmax(0, 1fr) 3rem; }
}
</style>
