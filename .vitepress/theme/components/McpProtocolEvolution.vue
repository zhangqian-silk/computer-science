<script setup lang="ts">
import { computed, ref } from "vue"

/**
 * MCP 从「有状态富功能」到「无状态可缓存」的演进，按维度逐项对照。
 *
 * 这张表不是版本更新日志，而是要呈现一条设计主线：
 * 几乎每一项变化都指向同一个目标——让任意一次请求自包含，
 * 从而可以被缓存、被水平扩展、被任意一个后端实例处理。
 * 理解了这条主线，就能预判某个尚未核对的字段大概会朝哪个方向演化。
 *
 * 维度取值依据核验当日的官方规范与其变更说明，逐条对应正文的来源编号。
 */

type Row = {
	key: string
	dim: string
	before: string
	after: string
	why: string
	/** 这一项服务于哪条目标 */
	goal: "stateless" | "cache" | "scale"
}

const goals: Record<Row["goal"], { label: string; color: string }> = {
	stateless: { label: "请求自包含", color: "var(--cs-color-brand)" },
	cache: { label: "结果可缓存", color: "var(--cs-color-info)" },
	scale: { label: "可水平扩展", color: "var(--cs-color-success)" }
}

const rows: Row[] = [
	{
		key: "handshake", dim: "握手", goal: "stateless",
		before: "连接建立后先做一次初始化握手，协商协议版本与能力，之后的请求依赖这次协商的结果。",
		after: "取消握手。每个请求在元数据里自带协议版本与客户端能力；另设一个发现方法供客户端提前查询服务端支持哪些版本。",
		why: "握手结果是连接上的状态。取消它，任何一个后端实例都能独立处理任意一次请求。"
	},
	{
		key: "session", dim: "会话", goal: "scale",
		before: "传输层有会话标识，列表类请求的结果可以随连接而变。",
		after: "取消协议层会话与会话标识。需要跨调用状态的服务端改为签发显式句柄，作为普通工具参数传回。",
		why: "会话标识既是扩展障碍也是攻击面。改成显式句柄后，状态的归属与生存期变成了服务端要明确声明的事。"
	},
	{
		key: "cache", dim: "缓存", goal: "cache",
		before: "靠列表变更通知告知客户端刷新，没有统一的新鲜度表达。",
		after: "列表与读取类结果统一携带存活时长与缓存范围两个字段，声明可缓存多久、能否被共享的中间层缓存。",
		why: "无状态之后列表结果不再随连接变化，这才使得中间层缓存成为可能——两项变化是配套的。"
	},
	{
		key: "order", dim: "工具顺序", goal: "cache",
		before: "未作要求。",
		after: "建议以确定的顺序返回工具列表。",
		why: "顺序不稳定会让客户端侧缓存与模型侧的前缀缓存双双失效。一条很小的约定，收益在缓存命中率上。"
	},
	{
		key: "server2client", dim: "服务端发起的请求", goal: "stateless",
		before: "服务端可以主动向客户端发起请求，例如索取根目录、请求补充输入、或请求一次模型采样。",
		after: "改为多轮往返：服务端返回一个「需要输入」的中间结果，客户端带着补充信息重试原请求。",
		why: "服务端主动发起要求一条活着的双向通道。改成由客户端驱动的重试之后，通道不再是必需品。"
	},
	{
		key: "async", dim: "长任务", goal: "scale",
		before: "试验性的任务机制在核心协议内，取结果的方法是阻塞式的。",
		after: "移出核心，成为可选扩展；改为轮询取状态，新增一个在执行途中回送输入的方法。",
		why: "阻塞取结果本质上还是把一次长执行绑在一条连接上。改成轮询之后，任务真正与连接解耦。"
	},
	{
		key: "resume", dim: "流恢复", goal: "stateless",
		before: "响应流断开后可以凭事件 ID 续传、补发遗漏消息。",
		after: "移除续传与补发。流断了就是丢了，客户端必须以新的请求 ID 重发。",
		why: "续传要求服务端记住这条流发到哪儿了，又是一份连接态。代价是重发的成本被明确摊给了调用方。"
	},
	{
		key: "deprecate", dim: "能力收缩", goal: "stateless",
		before: "服务端可反向索取根目录、发起模型采样、设置日志级别。",
		after: "这三项进入弃用窗口，建议改为用工具参数传路径、直接对接模型接口、以及按请求指定日志级别。",
		why: "被弃用的恰好都是「服务端需要客户端持续在场」的能力。方向与前几条一致。"
	}
]

const filter = ref<"all" | Row["goal"]>("all")
const detail = ref(0)
const shown = computed(() => (filter.value === "all" ? rows : rows.filter((r) => r.goal === filter.value)))
const cur = computed(() => rows[detail.value])

function selectRow(r: Row) {
	detail.value = rows.findIndex((x) => x.key === r.key)
}
</script>

<template>
	<div class="mpe">
		<div class="infra-tabs mpe__tabs" role="group" aria-label="按设计目标筛选">
			<button type="button" :aria-pressed="filter === 'all'" @click="filter = 'all'">全部 {{ rows.length }} 项</button>
			<button
				v-for="(g, k) in goals"
				:key="k"
				type="button"
				:aria-pressed="filter === k"
				@click="filter = (k as Row['goal'])"
			>{{ g.label }}</button>
		</div>

		<table class="mpe__table">
			<thead>
				<tr>
					<th>维度</th>
					<th>此前</th>
					<th>此后</th>
				</tr>
			</thead>
			<tbody>
				<tr
					v-for="r in shown"
					:key="r.key"
					:class="{ 'is-open': cur.key === r.key }"
					@click="selectRow(r)"
				>
					<th scope="row">
						<span class="mpe__dim">{{ r.dim }}</span>
						<span class="mpe__goal" :style="{ color: goals[r.goal].color }">{{ goals[r.goal].label }}</span>
					</th>
					<td>{{ r.before }}</td>
					<td>{{ r.after }}</td>
				</tr>
			</tbody>
		</table>

		<article class="mpe__panel" :style="{ '--c': goals[cur.goal].color }">
			<header>
				<b class="pc-serif">{{ cur.dim }}</b>
				<span class="mpe__badge">服务于「{{ goals[cur.goal].label }}」</span>
			</header>
			<p class="mpe__why">{{ cur.why }}</p>
		</article>

		<p class="pc-note">
			表中各项依据核验当日的官方规范与其变更说明，具体字段名以规范原文为准，且后续修订可能再变。
			真正可迁移的是这条主线：<b>把状态从连接上摘下来，一次请求就能独立成立</b>，
			于是缓存、负载均衡、水平扩展这些常规基础设施才用得上。代价也很清楚——
			以前由协议兜底的续传与服务端推送，现在成了应用层要自己处理的事。
		</p>
	</div>
</template>

<style scoped>
.mpe { margin: 1rem 0; }
.mpe__tabs { margin-bottom: var(--cs-space-4); }
.mpe__table { width: 100%; margin: 0; border-collapse: collapse; display: table; font-size: var(--cs-text-sm); }
.mpe__table th, .mpe__table td { border: 1px solid var(--cs-color-border); padding: var(--cs-space-3); text-align: left; vertical-align: top; line-height: var(--cs-leading-normal); }
.mpe__table thead th { font-size: var(--cs-text-3xs); letter-spacing: .06em; color: var(--cs-color-text-subtle); background: var(--cs-color-bg-soft); font-weight: 500; }
.mpe__table tbody tr { cursor: pointer; transition: var(--cs-transition-colors); }
.mpe__table tbody tr:hover { background: var(--cs-color-bg-soft); }
.mpe__table tbody tr.is-open { background: var(--cs-color-brand-soft); }
.mpe__table tbody th { white-space: nowrap; }
.mpe__table td { color: var(--cs-color-text-muted); }
.mpe__dim { display: block; font-size: var(--cs-text-base); color: var(--cs-color-text); }
.mpe__goal { display: block; margin-top: var(--cs-space-1); font-family: var(--cs-font-mono); font-size: var(--cs-text-3xs); font-weight: 400; }
.mpe__panel { margin-top: var(--cs-space-4); background: var(--cs-color-bg); border: 1px solid var(--cs-color-border); border-left: 3px solid var(--c); border-radius: var(--cs-radius-lg); padding: var(--cs-space-5) var(--cs-space-6); }
.mpe__panel header { display: flex; flex-wrap: wrap; align-items: baseline; justify-content: space-between; gap: var(--cs-space-3); }
.mpe__panel header b { font-size: var(--cs-text-lg); }
.mpe__badge { font-family: var(--cs-font-mono); font-size: var(--cs-text-3xs); color: var(--c); border: 1px solid var(--c); border-radius: var(--cs-radius-pill); padding: var(--cs-space-1) var(--cs-space-3); }
.mpe__why { margin: var(--cs-space-3) 0 0; font-size: var(--cs-text-base); line-height: var(--cs-leading-relaxed); color: var(--cs-color-text); }
@media (max-width: 720px) {
	.mpe__table thead { display: none; }
	.mpe__table, .mpe__table tbody, .mpe__table tr, .mpe__table th, .mpe__table td { display: block; width: 100%; }
	.mpe__table tr { margin-bottom: var(--cs-space-3); border: 1px solid var(--cs-color-border); border-radius: var(--cs-radius-md); overflow: hidden; }
	.mpe__table th, .mpe__table td { border: none; border-bottom: 1px solid var(--cs-color-border); }
	.mpe__table tr td:last-child { border-bottom: none; }
	.mpe__table td::before { display: block; font-family: var(--cs-font-mono); font-size: var(--cs-text-3xs); color: var(--cs-color-text-subtle); margin-bottom: var(--cs-space-1); }
	.mpe__table td:nth-of-type(1)::before { content: "此前"; }
	.mpe__table td:nth-of-type(2)::before { content: "此后"; }
}
</style>
