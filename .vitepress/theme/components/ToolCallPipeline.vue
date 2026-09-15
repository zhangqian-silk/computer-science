<script setup lang="ts">
import { computed, ref } from "vue"

/**
 * 一次工具调用从「模型说要调」到「结果回到模型」之间的七道关卡。
 *
 * 组件要说明的不是「有七步」，而是每一步**失败时结果去哪里**：
 * 有的关卡失败应当回灌给模型让它自纠，有的应当直接中止本轮。
 * 这条区分决定了错误处理代码写在哪一层，也是实现里最容易混淆的地方。
 *
 * 点选任一关卡查看它的判据；顶部开关可以切换「只看会回灌给模型的关卡」。
 */

type Stage = {
	key: string
	no: string
	name: string
	actor: string
	/** 失败后的去向 */
	fate: "refeed" | "abort" | "retry"
	what: string
	fail: string
	guard: string
	idx: number
}

const stages: Stage[] = [
	{
		key: "emit", no: "1", name: "意图产生", actor: "模型", fate: "refeed", idx: 0,
		what: "模型在本轮输出里给出一或多个调用块：工具名、参数、以及一个用于配对结果的调用 ID。",
		fail: "工具名不存在，或参数是一段无法解析的 JSON 片段——后者在流式场景里常因截断产生。",
		guard: "调用 ID 必须原样带回。少一条配对，多数接口会整体拒绝下一次请求，而不是忽略那一条。"
	},
	{
		key: "parse", no: "2", name: "参数校验", actor: "运行时", fate: "refeed", idx: 1,
		what: "按声明的 JSON Schema 校验参数：类型、枚举、必填、取值范围。",
		fail: "日期格式不对、数值越界、必填项缺失、把 ID 填成了人名。",
		guard: "校验失败应当作为「工具执行错误」回灌，而不是抛成协议层错误——前者模型能据此改参数重试，后者它通常只能放弃。"
	},
	{
		key: "authz", no: "3", name: "授权判定", actor: "策略层", fate: "abort", idx: 2,
		what: "按规则链判断这次调用允许、拒绝，还是需要人确认。判定依据是工具名加实际参数，不是工具名本身。",
		fail: "命中拒绝规则；或需要审批而当前是无人值守模式。",
		guard: "这是唯一不应当让模型自纠的一关。把「你没有权限」的细节回灌给模型，等于邀请它换个说法再试一次。"
	},
	{
		key: "dedupe", no: "4", name: "幂等与去重", actor: "运行时", fate: "retry", idx: 3,
		what: "为有副作用的调用生成幂等键，查询此键是否已经执行过；已执行则直接返回上次结果。",
		fail: "同一批并行调用里出现两条语义相同的写操作；或恢复执行时重放了已完成的调用。",
		guard: "幂等键应当由「调用内容」派生而非随机生成，否则重试时算出的是一个新键，去重形同虚设。"
	},
	{
		key: "exec", no: "5", name: "执行", actor: "工具实现", fate: "refeed", idx: 4,
		what: "真正跑起来：发 HTTP、读文件、起子进程。这一层承担超时、取消与并发额度。",
		fail: "下游 5xx、连接被重置、超时、进程被杀。",
		guard: "客户端超时不等于下游没执行：断开连接只是停止等待，对端的写操作可能已经落库。"
	},
	{
		key: "shape", no: "6", name: "结果整形", actor: "工具实现", fate: "refeed", idx: 5,
		what: "把原始返回值裁成模型用得上的形状：分页、字段裁剪、超阈值转存文件并返回路径。",
		fail: "一次返回三十万行日志，一口气顶穿窗口；或只返回一串 UUID，模型无从判断下一步。",
		guard: "被截断时要顺带告诉模型「怎么取下一段」。裸截断会让它重跑整个调用。"
	},
	{
		key: "feed", no: "7", name: "回灌", actor: "运行时", fate: "abort", idx: 6,
		what: "把结果按调用 ID 配对写回消息历史，进入下一次推理。",
		fail: "配对错位、结果丢失、或把错误信息写成了空字符串。",
		guard: "空结果比错误信息更糟：模型看不出发生了什么，最省事的选择就是把这个工具再调一遍。"
	}
]

const fateMeta: Record<Stage["fate"], { label: string; hint: string; color: string }> = {
	refeed: { label: "回灌模型", hint: "作为工具结果写回，模型可据此自行纠正", color: "var(--cs-color-success)" },
	retry: { label: "就地处理", hint: "运行时自行解决，模型通常不需要知道", color: "var(--cs-color-info)" },
	abort: { label: "中止本轮", hint: "不交给模型判断，由策略或运行时直接决定", color: "var(--cs-color-warning)" }
}

const onlyRefeed = ref(false)
const detail = ref(1)

const shown = computed(() => (onlyRefeed.value ? stages.filter((s) => s.fate === "refeed") : stages))
const cur = computed(() => stages[detail.value])
const curFate = computed(() => fateMeta[cur.value.fate])
const refeedCount = computed(() => stages.filter((s) => s.fate === "refeed").length)

function pick(s: Stage) {
	detail.value = s.idx
}
</script>

<template>
	<div class="tcp">
		<div class="tcp__head">
			<p class="tcp__lead">
				一次调用要穿过七道关卡。真正决定实现复杂度的不是关卡数量，而是
				<b>每一关失败之后结果去哪里</b>——其中 <b>{{ refeedCount }}</b> 关应当把失败回灌给模型让它自纠，
				其余的不应当。
			</p>
			<label class="cs-toggle tcp__filter">
				<input v-model="onlyRefeed" type="checkbox" />
				<span>只看回灌给模型的关卡</span>
			</label>
		</div>

		<ol class="tcp__flow">
			<li v-for="s in shown" :key="s.key">
				<button
					type="button"
					class="tcp__node"
					:class="{ 'is-open': detail === s.idx }"
					:style="{ '--c': fateMeta[s.fate].color }"
					:aria-pressed="detail === s.idx"
					@click="pick(s)"
				>
					<span class="tcp__no">{{ s.no }}</span>
					<strong>{{ s.name }}</strong>
					<span class="tcp__actor">{{ s.actor }}</span>
					<span class="tcp__fate">{{ fateMeta[s.fate].label }}</span>
				</button>
			</li>
		</ol>

		<article class="tcp__panel" :style="{ '--c': curFate.color }">
			<header>
				<b class="pc-serif">{{ cur.no }} · {{ cur.name }}</b>
				<span class="tcp__badge">{{ curFate.label }}：{{ curFate.hint }}</span>
			</header>
			<p class="tcp__what">{{ cur.what }}</p>
			<dl class="tcp__kv">
				<dt>典型失败</dt>
				<dd>{{ cur.fail }}</dd>
				<dt>判据</dt>
				<dd>{{ cur.guard }}</dd>
			</dl>
		</article>

		<p class="pc-note">
			关卡划分是对常见实现的归纳，不同框架的层次边界可能合并或再拆。可迁移的结论是这条区分本身：
			<b>参数与执行类的失败应当回灌，授权与配对类的失败不应当</b>。把授权失败也写回给模型，
			会得到一个反复换措辞试探边界的循环；把参数失败当成协议错误抛掉，则放弃了模型本可以自行完成的一次修正。
		</p>
	</div>
</template>

<style scoped>
.tcp { margin: 1rem 0; background: var(--cs-color-bg); border: 1px solid var(--cs-color-border); border-radius: var(--cs-radius-lg); padding: var(--cs-space-5) var(--cs-space-6); }
.tcp__head { display: flex; flex-wrap: wrap; align-items: flex-start; justify-content: space-between; gap: var(--cs-space-3); }
.tcp__lead { flex: 1 1 22rem; margin: 0; font-size: var(--cs-text-base); line-height: var(--cs-leading-relaxed); color: var(--cs-color-text-muted); }
.tcp__lead b { color: var(--cs-color-text); }
.tcp__filter { flex: 0 0 auto; font-size: var(--cs-text-xs); }
.tcp__flow { list-style: none; display: grid; grid-template-columns: repeat(7, 1fr); gap: var(--cs-space-2); margin: var(--cs-space-5) 0 0; padding: 0; }
.tcp__flow li { min-width: 0; }
.tcp__node { width: 100%; height: 100%; display: grid; gap: var(--cs-space-1); text-align: left; cursor: pointer; background: var(--cs-color-bg); border: 1px solid var(--cs-color-border); border-top: 3px solid var(--c); border-radius: var(--cs-radius-md); padding: var(--cs-space-3) var(--cs-space-2); min-height: var(--cs-tap-target); transition: var(--cs-transition-colors); }
.tcp__node:hover { background: var(--cs-color-bg-soft); }
.tcp__node.is-open { background: color-mix(in srgb, var(--c) 12%, transparent); border-color: var(--c); }
.tcp__no { font-family: var(--cs-font-mono); font-size: var(--cs-text-3xs); color: var(--c); }
.tcp__node strong { font-size: var(--cs-text-sm); line-height: var(--cs-leading-tight); }
.tcp__actor { font-size: var(--cs-text-3xs); color: var(--cs-color-text-muted); }
.tcp__fate { font-size: var(--cs-text-3xs); color: var(--c); }
.tcp__panel { margin-top: var(--cs-space-4); background: var(--cs-color-bg); border: 1px solid var(--cs-color-border); border-left: 3px solid var(--c); border-radius: var(--cs-radius-lg); padding: var(--cs-space-5) var(--cs-space-6); }
.tcp__panel header { display: flex; flex-wrap: wrap; align-items: baseline; justify-content: space-between; gap: var(--cs-space-3); }
.tcp__panel header b { font-size: var(--cs-text-lg); }
.tcp__badge { font-family: var(--cs-font-mono); font-size: var(--cs-text-3xs); color: var(--c); border: 1px solid var(--c); border-radius: var(--cs-radius-pill); padding: var(--cs-space-1) var(--cs-space-3); }
.tcp__what { margin: var(--cs-space-3) 0; font-size: var(--cs-text-base); line-height: var(--cs-leading-relaxed); color: var(--cs-color-text); }
.tcp__kv { display: grid; grid-template-columns: 4.5rem minmax(0, 1fr); gap: var(--cs-space-2) var(--cs-space-3); margin: 0; padding-top: var(--cs-space-3); border-top: 1px dashed var(--cs-color-border); }
.tcp__kv dt { font-family: var(--cs-font-mono); font-size: var(--cs-text-3xs); color: var(--cs-color-text-subtle); }
.tcp__kv dd { margin: 0; font-size: var(--cs-text-sm); line-height: var(--cs-leading-normal); color: var(--cs-color-text-muted); }
@media (max-width: 860px) {
	.tcp__flow { grid-template-columns: repeat(2, 1fr); }
}
@media (max-width: 520px) {
	.tcp__flow { grid-template-columns: 1fr; }
	.tcp__kv { grid-template-columns: 1fr; gap: var(--cs-space-1) 0; }
}
</style>
