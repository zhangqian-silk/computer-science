<script setup lang="ts">
import { computed, ref } from "vue"

/**
 * 原文、片段、向量、图这四层的写入时序，以及它们之间怎么变脏。
 *
 * 四层是同一份内容的四种形态，只有第一层是本体，其余三层都是派生副本。
 * 关键在于：四条写入路径的延迟并不相同，因此存在一段窗口，
 * 各层对「当前有效内容是什么」的回答互相矛盾。组件按时间轴推进，
 * 让读者看到某一步失败时，究竟是哪几层留在了旧状态。
 */

type Layer = {
	key: string
	name: string
	role: string
	derived: boolean
}

const layers: Layer[] = [
	{ key: "raw", name: "原文", role: "唯一事实源，其余三层都从它推导", derived: false },
	{ key: "chunk", name: "片段与元数据", role: "切分产物，携带定位信息与版本号", derived: true },
	{ key: "vector", name: "向量", role: "片段的编码，供近邻检索使用", derived: true },
	{ key: "graph", name: "结构关系", role: "符号或实体之间的边，供多跳扩展使用", derived: true }
]

type Step = {
	key: string
	label: string
	action: string
	/** 该步之后，各层处于什么状态：new 已是新版、old 仍是旧版、gone 已删除 */
	state: Record<string, "new" | "old" | "gone">
	/** 如果链路在这一步之后中断，用户会看到什么 */
	ifStops: string
}

const steps: Step[] = [
	{
		key: "s0", label: "起点", action: "文档尚未更新，四层一致，检索结果与原文相符。",
		state: { raw: "old", chunk: "old", vector: "old", graph: "old" },
		ifStops: "一致状态，没有异常。"
	},
	{
		key: "s1", label: "写原文", action: "新版本落到对象存储，得到一个新的版本标识；派生三层还没动。",
		state: { raw: "new", chunk: "old", vector: "old", graph: "old" },
		ifStops: "打开文档看到的是新内容，检索却仍然返回旧内容——最常被报为「更新了却搜不到」。"
	},
	{
		key: "s2", label: "换片段", action: "按文档标识删除旧片段，写入新切分产物，并标记为新版本。",
		state: { raw: "new", chunk: "new", vector: "old", graph: "old" },
		ifStops: "片段是新的、向量还是旧的，两者对不上：召回按旧向量命中，取回的正文却已换成新片段。"
	},
	{
		key: "s3", label: "换向量", action: "为新片段计算向量写入索引，旧向量置删除标记。",
		state: { raw: "new", chunk: "new", vector: "new", graph: "old" },
		ifStops: "文档问答已经正常，但沿关系多跳扩展时仍会走到已经不存在的节点上。"
	},
	{
		key: "s4", label: "换关系", action: "受影响的边按反向传播重算：先定位该文件定义的符号，再找引用它们的上游。",
		state: { raw: "new", chunk: "new", vector: "new", graph: "new" },
		ifStops: "四层重新一致。此时才可以把文档状态标记为可用。"
	}
]

const at = ref(0)
const cur = computed(() => steps[at.value])

const toneOf = (s: "new" | "old" | "gone") =>
	s === "new" ? "cs-state--pass" : s === "old" ? "cs-state--warn" : "cs-state--fail"
const textOf = (s: "new" | "old" | "gone") =>
	s === "new" ? "新版本" : s === "old" ? "仍是旧版本" : "已删除"

/** 是否叠加版本号过滤这道兜底 */
const guarded = ref(false)
const inconsistent = computed(() =>
	layers.filter((l) => l.derived && cur.value.state[l.key] === "old").length
)
</script>

<template>
	<div class="fcl">
		<div class="infra-tabs fcl__steps">
			<button
				v-for="(s, i) in steps"
				:key="s.key"
				type="button"
				:aria-pressed="at === i"
				@click="at = i"
			>{{ s.label }}</button>
		</div>

		<p class="fcl__action">{{ cur.action }}</p>

		<div class="fcl__layers">
			<div
				v-for="l in layers"
				:key="l.key"
				class="fcl__layer cs-state"
				:class="toneOf(cur.state[l.key])"
			>
				<b>{{ l.name }}</b>
				<span class="fcl__badge">{{ l.derived ? "派生副本" : "本体" }}</span>
				<span class="fcl__state">{{ textOf(cur.state[l.key]) }}</span>
				<span class="fcl__role">{{ l.role }}</span>
			</div>
		</div>

		<label class="cs-toggle fcl__sw">
			<input v-model="guarded" type="checkbox">
			<span>在查询侧强制按「当前有效版本」过滤</span>
		</label>

		<p class="fcl__verdict" :class="guarded ? 'is-ok' : inconsistent > 0 ? 'is-risk' : 'is-ok'">
			<template v-if="guarded">
				即使派生层还停在旧版本，查询侧按版本过滤后旧内容不会进入结果：
				此时的表现从「返回旧答案」退化为「暂时召不回」——<b>后者是可接受的，前者不是</b>。
			</template>
			<template v-else-if="inconsistent > 0">
				当前有 {{ inconsistent }} 层派生副本停留在旧版本，而查询侧没有任何兜底，
				旧内容会照常被召回并当作依据。
			</template>
			<template v-else>
				四层一致，任何一路召回都指向同一版本。
			</template>
		</p>

		<p v-if="!guarded" class="fcl__ifstops"><i>如果链路停在这一步</i>{{ cur.ifStops }}</p>

		<p class="pc-note">
			这条时序说明了两件事。其一，<b>「删了为什么还能搜到」几乎总是派生层的问题</b>，
			而不是删除操作本身失败——只删本体不删副本，副本会继续被召回。
			其二，既然四条路径的延迟无法拉平，就不要指望靠时序对齐解决，
			而应当在查询侧留一道版本过滤：让最坏情况落在「暂时查不到」，而不是「拿旧内容当依据」。
		</p>
	</div>
</template>

<style scoped>
.fcl { margin: 1rem 0; }
.fcl__steps { margin-bottom: var(--cs-space-3); }
.fcl__steps button { min-height: var(--cs-tap-target); font-size: var(--cs-text-3xs); }
.fcl__action { margin: 0 0 var(--cs-space-4); font-size: var(--cs-text-base); line-height: var(--cs-leading-relaxed); color: var(--cs-color-text); }
.fcl__layers { display: grid; gap: var(--cs-space-2); }
.fcl__layer { display: grid; grid-template-columns: 8rem 5rem 6.5rem 1fr; align-items: center; gap: var(--cs-space-3); margin: 0; }
.fcl__layer b { font-size: var(--cs-text-sm); }
.fcl__badge { font-family: var(--cs-font-mono); font-size: var(--cs-text-3xs); color: var(--cs-color-text-subtle); }
.fcl__state { font-size: var(--cs-text-3xs); font-weight: 600; }
.fcl__role { font-size: var(--cs-text-3xs); color: var(--cs-color-text-muted); }
.fcl__sw { display: inline-flex; align-items: center; gap: var(--cs-space-1); min-height: var(--cs-tap-target); margin: var(--cs-space-4) 0 var(--cs-space-3); font-size: var(--cs-text-3xs); color: var(--cs-color-text-subtle); cursor: pointer; }
.fcl__verdict { margin: 0; padding: var(--cs-space-3) var(--cs-space-4); border-left: 3px solid var(--cs-color-border-strong); background: var(--cs-color-bg-soft); border-radius: var(--cs-radius-sm); font-size: var(--cs-text-base); line-height: var(--cs-leading-relaxed); color: var(--cs-color-text); }
.fcl__verdict.is-ok { border-left-color: var(--cs-color-success); }
.fcl__verdict.is-risk { border-left-color: var(--cs-color-danger); }
.fcl__ifstops { margin: var(--cs-space-3) 0 0; font-size: var(--cs-text-sm); line-height: var(--cs-leading-normal); color: var(--cs-color-text-muted); }
.fcl__ifstops i { display: block; font-family: var(--cs-font-mono); font-size: var(--cs-text-3xs); letter-spacing: .08em; font-style: normal; color: var(--cs-color-text-subtle); margin-bottom: var(--cs-space-1); }
@media (max-width: 860px) {
	.fcl__layer { grid-template-columns: 1fr 1fr; }
	.fcl__role { grid-column: 1 / -1; }
}
</style>
