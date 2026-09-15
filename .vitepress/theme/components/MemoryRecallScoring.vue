<script setup lang="ts">
import { computed, ref } from "vue"

/**
 * 召回打分：relevance × recency × importance 三个信号如何决定谁进入本次上下文。
 *
 * 要说明的核心不是公式本身，而是三件事：
 *   1. 只用相关度会怎样——过期的旧结论会稳定压过新事实，因为它「更像」查询；
 *   2. 时效衰减的强度是个真实的取舍，不是常数：调得太狠会丢掉稳定偏好；
 *   3. 召回预算是硬约束，排序的意义在于「谁被截断」，而不是「谁分数高」。
 * 数值为便于对照的示意，用于呈现相对关系。
 */

type Mem = {
	id: string
	text: string
	kind: string
	/** 与当前查询的语义相似度，0–1 */
	rel: number
	/** 距今天数 */
	ageDays: number
	/** 写入时由模型给出的重要性，1–10 */
	imp: number
	/** 这条现在是否仍然成立——用于判断排序结果好坏 */
	stillTrue: boolean
	note: string
}

/** 当前查询：「部署要注意什么」 */
const QUERY = "这个项目部署的时候要注意什么"

const mems: Mem[] = [
	{
		id: "m1", text: "部署流程：先跑 make check，再走蓝绿切换，切换后观察 10 分钟",
		kind: "程序性", rel: 0.90, ageDays: 45, imp: 9, stillTrue: true,
		note: "规程类，写下来之后长期有效，任何权重下都该在预算内"
	},
	{
		id: "m2", text: "部署脚本目前有 bug，临时手动执行 step 3",
		kind: "情景", rel: 0.80, ageDays: 180, imp: 5, stillTrue: false,
		note: "当时为真、现在已修复。相关度很高，是最容易误召的一类"
	},
	{
		id: "m3", text: "上周把发布窗口从周五改到周二，避开周末值班",
		kind: "语义事实", rel: 0.62, ageDays: 5, imp: 7, stillTrue: true,
		note: "新事实，相关度不算最高，但恰恰是本次最该看到的"
	},
	{
		id: "m4", text: "用户偏好：部署结论先给一句话结论，再给细节",
		kind: "偏好", rel: 0.58, ageDays: 240, imp: 10, stillTrue: true,
		note: "很旧但一直有效。把时效权重调高，它会第一批被挤出去"
	},
	{
		id: "m5", text: "生产库连接串放在 vault 里，不要写进配置文件",
		kind: "语义事实", rel: 0.48, ageDays: 150, imp: 9, stillTrue: true,
		note: "重要性极高的硬约束，却因为字面相关度低而长期挤不进预算——这类条目更适合常驻而非检索"
	},
	{
		id: "m6", text: "上个月换了新的日志采集器，部署时多一步配置同步",
		kind: "情景", rel: 0.44, ageDays: 25, imp: 4, stillTrue: true,
		note: "较新但价值一般。时效权重一高，它就会顶掉更有价值的旧条目"
	},
	{
		id: "m7", text: "上周三那次部署耗时 12 分钟，比平时略慢",
		kind: "情景", rel: 0.38, ageDays: 10, imp: 4, stillTrue: true,
		note: "最新但几乎没有决策价值。它挤进预算是时效权重过高的直接症状"
	},
	{
		id: "m8", text: "早期部署在裸机上，用 supervisor 托管进程",
		kind: "语义事实", rel: 0.70, ageDays: 420, imp: 3, stillTrue: false,
		note: "架构已改，早该失效。相关度不低，是典型的过期记忆"
	}
]

const wRel = ref(1)
const wRec = ref(0.6)
const wImp = ref(0.5)
/** 时效衰减半衰期（天）：越小衰减越狠 */
const halfLife = ref(90)
/** 召回预算：本次最多带几条进上下文 */
const topK = ref(4)

const recencyOf = (m: Mem) => Math.pow(0.5, m.ageDays / halfLife.value)

const scored = computed(() => {
	const list = mems.map((m) => {
		const rec = recencyOf(m)
		const imp = m.imp / 10
		const score = wRel.value * m.rel + wRec.value * rec + wImp.value * imp
		return { ...m, rec, impNorm: imp, score }
	})
	list.sort((a, b) => b.score - a.score)
	return list.map((m, i) => ({ ...m, rank: i + 1, inBudget: i < topK.value }))
})

const maxScore = computed(() => Math.max(...scored.value.map((s) => s.score), 0.001))

/** 只用相关度排序时的结果，用于对照 */
const relOnly = computed(() => {
	const list = [...mems].sort((a, b) => b.rel - a.rel)
	return list.slice(0, topK.value).map((m) => m.id)
})

const picked = computed(() => scored.value.filter((s) => s.inBudget))
const staleIn = computed(() => picked.value.filter((s) => !s.stillTrue))
const trueOut = computed(() => scored.value.filter((s) => !s.inBudget && s.stillTrue && s.rel >= 0.5))
const staleInRelOnly = computed(() => mems.filter((m) => relOnly.value.includes(m.id) && !m.stillTrue))
/** 「很旧但一直有效」的偏好是否被挤出预算——用于验证时效衰减的取舍 */
const prefOut = computed(() => scored.value.some((s) => s.kind === "偏好" && !s.inBudget))
/** 挤进预算但价值很低的新条目数：时效权重过高的直接症状 */
const trivialIn = computed(() => picked.value.filter((s) => s.stillTrue && s.imp <= 4).length)

const verdict = computed(() => {
	const parts: string[] = []
	if (staleIn.value.length === 0) {
		parts.push(`预算内 ${topK.value} 条全部仍然成立`)
	} else {
		const kinds = [...new Set(staleIn.value.map((s) => s.kind))].join("、")
		parts.push(`预算内有 ${staleIn.value.length} 条已经不成立（${kinds}类）`)
	}
	if (prefOut.value) parts.push("那条「很旧却一直有效」的偏好被挤出了预算")
	if (trivialIn.value > 0) parts.push(`同时有 ${trivialIn.value} 条较新但价值很低的条目占了名额`)
	const bad = staleIn.value.length + (prefOut.value ? 1 : 0) + trivialIn.value
	return {
		tone: bad >= 2 ? "fail" : bad === 1 ? "warn" : "pass",
		text: `${parts.join("；")}。`
	}
})

const pct = (v: number) => `${Math.round(v * 100)}%`
</script>

<template>
	<div class="mrs">
		<p class="mrs__query">
			<i>本次查询</i>{{ QUERY }}
		</p>

		<div class="infra-controls mrs__controls">
			<label class="infra-control mrs__ctl">
				<span>相关度权重</span>
				<input v-model.number="wRel" type="range" min="0" max="2" step="0.1" />
				<b>{{ wRel.toFixed(1) }}</b>
			</label>
			<label class="infra-control mrs__ctl">
				<span>时效权重</span>
				<input v-model.number="wRec" type="range" min="0" max="2" step="0.1" />
				<b>{{ wRec.toFixed(1) }}</b>
			</label>
			<label class="infra-control mrs__ctl">
				<span>重要性权重</span>
				<input v-model.number="wImp" type="range" min="0" max="2" step="0.1" />
				<b>{{ wImp.toFixed(1) }}</b>
			</label>
			<label class="infra-control mrs__ctl">
				<span>衰减半衰期</span>
				<input v-model.number="halfLife" type="range" min="15" max="365" step="15" />
				<b>{{ halfLife }}天</b>
			</label>
			<label class="infra-control mrs__ctl">
				<span>召回预算</span>
				<input v-model.number="topK" type="range" min="1" max="8" step="1" />
				<b>{{ topK }} 条</b>
			</label>
		</div>

		<ol class="mrs__list">
			<li
				v-for="s in scored"
				:key="s.id"
				class="mrs__item"
				:class="{ 'is-in': s.inBudget, 'is-out': !s.inBudget, 'is-stale': !s.stillTrue }"
			>
				<div class="mrs__item-head">
					<span class="mrs__rank cs-mono">#{{ s.rank }}</span>
					<span class="mrs__kind">{{ s.kind }}</span>
					<span class="mrs__text">{{ s.text }}</span>
					<span class="mrs__score cs-mono">{{ s.score.toFixed(2) }}</span>
				</div>
				<div class="infra-bar mrs__bar">
					<i :style="{ width: `${(s.score / maxScore) * 100}%` }"></i>
				</div>
				<div class="mrs__signals">
					<span>相关 {{ pct(s.rel) }}</span>
					<span>时效 {{ pct(s.rec) }}（{{ s.ageDays }} 天前）</span>
					<span>重要 {{ s.imp }}/10</span>
					<span class="mrs__flag" :class="s.stillTrue ? 'is-ok' : 'is-no'">
						{{ s.stillTrue ? "仍然成立" : "已经失效" }}
					</span>
					<span class="mrs__state">{{ s.inBudget ? "进入上下文" : "被预算截断" }}</span>
				</div>
				<p class="mrs__note">{{ s.note }}</p>
			</li>
		</ol>

		<div class="mrs__verdicts">
			<p class="mrs__verdict" :class="`is-${verdict.tone}`">
				<b>当前打分结果</b>：{{ verdict.text }}
				<template v-if="trueOut.length">
					另有 {{ trueOut.length }} 条仍然成立、相关度不低的条目落在预算之外。
				</template>
			</p>
			<p class="mrs__verdict is-fail">
				<b>只按相关度排序</b>：前 {{ topK }} 条里有
				<b class="cs-mono">{{ staleInRelOnly.length }}</b> 条已经失效<template v-if="staleInRelOnly.length">——
					包括「{{ staleInRelOnly[0].text.slice(0, 18) }}…」这类当时为真、如今已假的条目</template>。
				过期记忆往往与查询高度相似，所以相似度单独用永远治不了它。
			</p>
		</div>

		<p class="pc-note">
			八条记忆与三个信号的数值为便于对照的示意，非实测。把时效权重从默认值往上推到 1 以上，可以看到一次完整的取舍：
			那条已经失效的条目确实被压出了预算，<b>但代价是那条「很旧却一直有效」的偏好也一起被挤掉，
			换进来两条更新、却几乎没有决策价值的记录</b>。时效衰减对稳定偏好和过期结论一视同仁，所以它不是越强越好。
			把时效与重要性权重都调到 0，则会看到另一端：纯相似度排序下预算内同时出现两条失效条目。
			这正是为什么成熟的做法不止于调权重：还要按类型分别设定衰减（规程与偏好几乎不衰减，情景经历衰减快），
			并且在写入时就记录有效期，让「失效」成为一个可判定的字段，而不是一个靠打分猜出来的结果。
			另外留意那条重要性很高、却因字面相关度低而长期挤不进预算的硬约束——<b>这类内容更适合常驻注入，而不是交给检索</b>。
		</p>
	</div>
</template>

<style scoped>
.mrs { margin: 1rem 0; background: var(--cs-color-bg); border: 1px solid var(--cs-color-border); border-radius: var(--cs-radius-lg); padding: var(--cs-space-5) var(--cs-space-6); }
.mrs__query { margin: 0 0 var(--cs-space-4); padding-left: var(--cs-space-4); border-left: 3px solid var(--cs-color-brand); font-size: var(--cs-text-md); color: var(--cs-color-text); }
.mrs__query i { display: block; font-family: var(--cs-font-mono); font-size: var(--cs-text-3xs); font-style: normal; letter-spacing: .08em; color: var(--cs-color-text-subtle); }
.mrs__controls { margin-bottom: var(--cs-space-5); }
.mrs__ctl { display: grid; grid-template-columns: 5.6rem 1fr 3.4rem; align-items: center; gap: var(--cs-space-2); min-height: var(--cs-tap-target); font-size: var(--cs-text-sm); }
.mrs__ctl span { color: var(--cs-color-text-muted); }
.mrs__ctl input { width: 100%; accent-color: var(--cs-color-brand); }
.mrs__ctl b { font-family: var(--cs-font-mono); text-align: right; color: var(--cs-color-brand); }
.mrs__list { list-style: none; margin: 0; padding: 0; }
.mrs__item { padding: var(--cs-space-3) var(--cs-space-4); margin-bottom: var(--cs-space-2); border-radius: var(--cs-radius-sm); border-left: 3px solid var(--cs-color-border); background: var(--cs-color-bg-soft); transition: var(--cs-transition-colors); }
.mrs__item.is-in { border-left-color: var(--cs-color-success); background: var(--cs-color-success-soft); }
.mrs__item.is-out { opacity: .62; }
.mrs__item.is-in.is-stale { border-left-color: var(--cs-color-danger); background: var(--cs-color-danger-soft); }
.mrs__item-head { display: grid; grid-template-columns: auto auto minmax(0, 1fr) auto; align-items: baseline; gap: var(--cs-space-2) var(--cs-space-3); }
.mrs__rank { font-size: var(--cs-text-3xs); color: var(--cs-color-text-subtle); }
.mrs__kind { font-size: var(--cs-text-3xs); color: var(--cs-color-text-muted); border: 1px solid var(--cs-color-border); border-radius: var(--cs-radius-pill); padding: 0 var(--cs-space-2); white-space: nowrap; }
.mrs__text { font-size: var(--cs-text-sm); line-height: var(--cs-leading-normal); color: var(--cs-color-text); }
.mrs__score { font-size: var(--cs-text-sm); color: var(--cs-color-text); }
.mrs__bar { margin: var(--cs-space-2) 0; }
.mrs__signals { display: flex; flex-wrap: wrap; gap: var(--cs-space-2) var(--cs-space-4); font-family: var(--cs-font-mono); font-size: var(--cs-text-2xs); line-height: var(--cs-leading-normal); color: var(--cs-color-text-subtle); }
.mrs__flag.is-ok { color: var(--cs-color-success); }
.mrs__flag.is-no { color: var(--cs-color-danger); }
.mrs__state { color: var(--cs-color-text-muted); }
.mrs__note { margin: var(--cs-space-2) 0 0; font-size: var(--cs-text-2xs); line-height: var(--cs-leading-relaxed); color: var(--cs-color-text-subtle); }
.mrs__verdicts { margin-top: var(--cs-space-5); display: grid; gap: var(--cs-space-2); }
.mrs__verdict { margin: 0; padding: var(--cs-space-3) var(--cs-space-4); border-radius: var(--cs-radius-sm); font-size: var(--cs-text-sm); line-height: var(--cs-leading-relaxed); }
.mrs__verdict.is-pass { background: var(--cs-color-success-soft); }
.mrs__verdict.is-warn { background: var(--cs-color-warning-soft); }
.mrs__verdict.is-fail { background: var(--cs-color-danger-soft); }
@media (max-width: 720px) {
	.mrs__ctl { grid-template-columns: 5rem 1fr 3.2rem; }
	.mrs__item-head { grid-template-columns: auto auto 1fr; }
	.mrs__score { grid-column: 1 / -1; }
}
</style>
