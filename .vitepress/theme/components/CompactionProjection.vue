<script setup lang="ts">
import { computed, ref } from "vue"

/**
 * 三层数据模型：事件日志（磁盘）→ 工作集（内存标记）→ 请求载荷（每轮投影）。
 *
 * 这个组件要说明的唯一一件事：「把某条内容从上下文里移除」在实现上不是删除元素，
 * 而是在组装下一次请求时不投影它。切换下面的处置动作，观察同一条工作集条目
 * 在三层里的不同去向——磁盘那一层永远不变。
 */

type Kind = "system" | "user" | "assistant" | "thinking" | "tool_use" | "tool_result"
type Disposal = "keep" | "clear" | "drop" | "summarize"

type Entry = {
	id: string
	kind: Kind
	label: string
	tokens: number
	/** 该条目在四种处置动作下的归属；用于演示同一条内容的不同去向 */
	fate: Record<Disposal, "same" | "placeholder" | "gone" | "folded">
}

const KIND_LABEL: Record<Kind, string> = {
	system: "system",
	user: "user",
	assistant: "assistant",
	thinking: "thinking",
	tool_use: "tool_use",
	tool_result: "tool_result"
}

/** 一段典型的轨迹片段：读了两份文件、跑了一次测试，中间夹着推理与回复 */
const entries: Entry[] = [
	{ id: "e1", kind: "system", label: "系统提示与工具定义", tokens: 3800, fate: { keep: "same", clear: "same", drop: "same", summarize: "same" } },
	{ id: "e2", kind: "user", label: "用户：改一下超时逻辑", tokens: 120, fate: { keep: "same", clear: "same", drop: "same", summarize: "same" } },
	{ id: "e3", kind: "thinking", label: "推理：定位调用链", tokens: 2600, fate: { keep: "same", clear: "same", drop: "gone", summarize: "folded" } },
	{ id: "e4", kind: "tool_use", label: "read(config.go)", tokens: 40, fate: { keep: "same", clear: "same", drop: "same", summarize: "folded" } },
	{ id: "e5", kind: "tool_result", label: "config.go 全文", tokens: 5200, fate: { keep: "same", clear: "placeholder", drop: "placeholder", summarize: "folded" } },
	{ id: "e6", kind: "tool_use", label: "bash(go test ./...)", tokens: 30, fate: { keep: "same", clear: "same", drop: "same", summarize: "folded" } },
	{ id: "e7", kind: "tool_result", label: "测试日志 1,900 行", tokens: 7400, fate: { keep: "same", clear: "placeholder", drop: "placeholder", summarize: "folded" } },
	{ id: "e8", kind: "assistant", label: "回复：定位到 A、B 两处", tokens: 180, fate: { keep: "same", clear: "same", drop: "same", summarize: "folded" } },
	{ id: "e9", kind: "user", label: "用户：先改 A", tokens: 30, fate: { keep: "same", clear: "same", drop: "same", summarize: "same" } }
]

const disposals: { key: Disposal; label: string; desc: string; api: string; color: string }[] = [
	{ key: "keep", label: "原样投影", desc: "工作集里没有任何标记，三层内容一致。", api: "无操作", color: "var(--cs-color-info)" },
	{ key: "clear", label: "清理正文", desc: "tool_result 正文换成占位符，tool_use 与消息条数都不动。", api: "打 cleared 标记", color: "var(--cs-color-success)" },
	{ key: "drop", label: "清理正文 + 丢弃推理", desc: "推理块不参与配对，可整块不投影，且无需留占位符。", api: "打 cleared / 不投影", color: "var(--cs-color-warning)" },
	{ key: "summarize", label: "摘要压缩", desc: "切点之前的条目折叠成一段摘要，仅保留最近消息。", api: "生成摘要 + 记录切点", color: "var(--cs-color-danger)" }
]

const active = ref<Disposal>("clear")
const PLACEHOLDER_TOKENS = 24
const SUMMARY_TOKENS = 900

/** 请求载荷层：按当前处置动作投影出模型真正看到的内容 */
const payload = computed(() => {
	const rows: { id: string; kind: Kind; label: string; tokens: number; state: "same" | "placeholder" }[] = []
	let folded = 0
	for (const e of entries) {
		const fate = e.fate[active.value]
		if (fate === "gone") continue
		if (fate === "folded") {
			folded += e.tokens
			continue
		}
		rows.push({
			id: e.id,
			kind: e.kind,
			label: fate === "placeholder" ? placeholderText(e) : e.label,
			tokens: fate === "placeholder" ? PLACEHOLDER_TOKENS : e.tokens,
			state: fate
		})
	}
	return { rows, folded }
})

function placeholderText(e: Entry): string {
	return e.label.includes("日志") ? "[已清理：退出码 1，日志见 ./logs/test-3.txt]" : "[已清理：可重读 config.go]"
}

const diskTotal = entries.reduce((s, e) => s + e.tokens, 0)
const worksetTotal = diskTotal
const payloadTotal = computed(() => {
	const body = payload.value.rows.reduce((s, r) => s + r.tokens, 0)
	return active.value === "summarize" ? body + SUMMARY_TOKENS : body
})
const saved = computed(() => diskTotal - payloadTotal.value)
const savedPct = computed(() => (saved.value / diskTotal) * 100)
const current = computed(() => disposals.find((d) => d.key === active.value)!)
const fmt = (n: number) => n.toLocaleString()

/** 工作集层的标记文案：说明「移除」到底改了什么 */
const marks = computed(() =>
	entries.map((e) => {
		const fate = e.fate[active.value]
		return {
			...e,
			mark: fate === "same" ? "" : fate === "placeholder" ? "cleared" : fate === "gone" ? "not projected" : "folded"
		}
	})
)
</script>

<template>
	<div class="cpj">
		<div class="cpj__tabs" role="tablist" aria-label="选择一种处置动作，查看三层数据的差异">
			<button
				v-for="d in disposals"
				:key="d.key"
				type="button"
				role="tab"
				:aria-selected="active === d.key"
				class="cpj__tab"
				:class="{ 'is-active': active === d.key }"
				:style="{ '--c': d.color }"
				@click="active = d.key"
			>
				<strong>{{ d.label }}</strong>
				<span>{{ d.api }}</span>
			</button>
		</div>

		<p class="cpj__desc" :style="{ '--c': current.color }">{{ current.desc }}</p>

		<div class="cpj__cols">
			<section class="cpj__col">
				<header>
					<b>① 事件日志</b>
					<span>磁盘 · append-only</span>
				</header>
				<ul class="cpj__list">
					<li v-for="e in entries" :key="e.id">
						<i class="cpj__kind">{{ KIND_LABEL[e.kind] }}</i>
						<span class="cpj__lab">{{ e.label }}</span>
						<em class="cpj__tok">{{ fmt(e.tokens) }}</em>
					</li>
				</ul>
				<footer>
					合计 <b>{{ fmt(diskTotal) }}</b> token · <span class="cpj__inv">四种动作下完全一致</span>
				</footer>
			</section>

			<section class="cpj__col">
				<header>
					<b>② 工作集</b>
					<span>内存 · 带标记的数组</span>
				</header>
				<ul class="cpj__list">
					<li v-for="e in marks" :key="e.id" :class="{ 'is-marked': e.mark }">
						<i class="cpj__kind">{{ KIND_LABEL[e.kind] }}</i>
						<span class="cpj__lab">{{ e.label }}</span>
						<em v-if="e.mark" class="cpj__mark" :style="{ '--c': current.color }">{{ e.mark }}</em>
						<em v-else class="cpj__tok">{{ fmt(e.tokens) }}</em>
					</li>
				</ul>
				<footer>
					条目数 <b>{{ entries.length }}</b> 条不变 · 元素数量 <b>{{ fmt(worksetTotal) }}</b> token 仍在
				</footer>
			</section>

			<section class="cpj__col cpj__col--out">
				<header>
					<b>③ 请求载荷</b>
					<span>本次 API 调用实际发出</span>
				</header>
				<ul class="cpj__list">
					<li v-if="payload.folded" class="is-summary">
						<i class="cpj__kind">summary</i>
						<span class="cpj__lab">摘要：目标、已完成、待办、下一步</span>
						<em class="cpj__tok">{{ fmt(SUMMARY_TOKENS) }}</em>
					</li>
					<li v-for="r in payload.rows" :key="r.id" :class="{ 'is-ph': r.state === 'placeholder' }">
						<i class="cpj__kind">{{ KIND_LABEL[r.kind] }}</i>
						<span class="cpj__lab">{{ r.label }}</span>
						<em class="cpj__tok">{{ fmt(r.tokens) }}</em>
					</li>
				</ul>
				<footer>
					本轮输入 <b class="cpj__hi">{{ fmt(payloadTotal) }}</b> token · 较日志少
					<b class="cpj__hi">{{ savedPct.toFixed(0) }}%</b>
				</footer>
			</section>
		</div>

		<p class="pc-note">
			示意用的 token 数为便于对照的量级估算，非实测。三列同时呈现同一段轨迹：磁盘那一列在四种动作下逐字不变，
			工作集的条目数也始终是 {{ entries.length }} 条——变的只有第三列。所以「把内容从上下文中移除」的准确含义是
			<b>组装下一次请求时不投影它</b>；只有摘要压缩才真正生成了新内容，也只有它不可逆。
		</p>
	</div>
</template>

<style scoped>
.cpj { margin: 1rem 0; }
.cpj__tabs { display: flex; flex-wrap: wrap; gap: .35rem; }
.cpj__tab { flex: 1 1 150px; min-width: 140px; min-height: var(--cs-tap-target); text-align: left; cursor: pointer; background: var(--cs-color-bg); border: 1px solid var(--cs-color-border); border-top: 3px solid var(--c); border-radius: var(--cs-radius-lg); padding: var(--cs-space-3) var(--cs-space-4); transition: var(--cs-transition-colors); }
.cpj__tab:hover { background: var(--cs-color-bg-soft); }
.cpj__tab.is-active { background: color-mix(in srgb, var(--c) 10%, transparent); border-color: var(--c); }
.cpj__tab strong { display: block; font-size: var(--cs-text-base); color: var(--c); }
.cpj__tab span { display: block; font-family: var(--cs-font-mono); font-size: var(--cs-text-3xs); color: var(--cs-color-text-subtle); margin-top: var(--cs-space-1); }
.cpj__desc { margin: var(--cs-space-4) 0 var(--cs-space-4); padding-left: var(--cs-space-4); border-left: 3px solid var(--c); font-size: var(--cs-text-base); line-height: var(--cs-leading-relaxed); color: var(--cs-color-text); }
.cpj__cols { display: grid; grid-template-columns: repeat(3, 1fr); gap: var(--cs-space-3); align-items: start; }
.cpj__col { background: var(--cs-color-bg); border: 1px solid var(--cs-color-border); border-radius: var(--cs-radius-lg); overflow: hidden; }
.cpj__col--out { border-color: var(--cs-color-border-strong); }
.cpj__col > header { padding: var(--cs-space-3) var(--cs-space-4); background: var(--cs-color-bg-soft); border-bottom: 1px solid var(--cs-color-border); }
.cpj__col > header b { display: block; font-size: var(--cs-text-md); }
.cpj__col > header span { display: block; font-family: var(--cs-font-mono); font-size: var(--cs-text-3xs); color: var(--cs-color-text-subtle); }
.cpj__list { list-style: none; margin: 0; padding: var(--cs-space-2) 0; }
.cpj__list li { display: grid; grid-template-columns: 1fr auto; gap: 0 var(--cs-space-2); padding: var(--cs-space-2) var(--cs-space-4); border-bottom: 1px dashed var(--cs-color-border); }
.cpj__list li:last-child { border-bottom: 0; }
.cpj__list li.is-marked { background: var(--cs-color-neutral-soft); }
.cpj__list li.is-ph { background: var(--cs-color-success-soft); }
.cpj__list li.is-summary { background: var(--cs-color-danger-soft); }
.cpj__kind { grid-column: 1 / -1; font-family: var(--cs-font-mono); font-size: var(--cs-text-3xs); letter-spacing: .04em; color: var(--cs-color-text-subtle); font-style: normal; }
.cpj__lab { font-size: var(--cs-text-xs); line-height: var(--cs-leading-tight); color: var(--cs-color-text); }
.cpj__tok { font-family: var(--cs-font-mono); font-size: var(--cs-text-3xs); color: var(--cs-color-text-muted); font-style: normal; align-self: end; }
.cpj__mark { font-family: var(--cs-font-mono); font-size: var(--cs-text-3xs); color: var(--c); font-style: normal; align-self: end; }
.cpj__col > footer { padding: var(--cs-space-3) var(--cs-space-4); border-top: 1px solid var(--cs-color-border); font-size: var(--cs-text-2xs); line-height: var(--cs-leading-normal); color: var(--cs-color-text-muted); }
.cpj__inv { color: var(--cs-color-text-subtle); }
.cpj__hi { color: var(--cs-color-brand); font-family: var(--cs-font-mono); }
@media (max-width: 860px) {
	.cpj__cols { grid-template-columns: 1fr; }
	.cpj__tab { flex-basis: 46%; }
}
</style>
