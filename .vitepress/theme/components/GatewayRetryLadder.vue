<script setup lang="ts">
import { computed, ref } from "vue"
import { seriesColor } from "../series-palette"

/**
 * 模型网关：两层嵌套的重试循环，以及各实现的公开默认值。
 *
 * 组件要说明的是一个容易被忽略的结构：重试不是一个旋钮而是**两层**——
 * 传输层（单个 HTTP 请求）与流式层（一次采样的整条 SSE），
 * 两者的失败语义不同，因此计数、退避与上限都必须分开。
 * 把两层混成一个数字，会得到「重试次数看起来是 5，实际最坏是 5×4」这种意外。
 *
 * 各实现的键名与默认值取自官方文档与官方仓库源码（核验于 2026-09-14）。
 * 这些值随版本变化，落地前应按当日文档与所用版本复核。
 */

type Impl = {
	key: string
	name: string
	/** 每层的配置键与默认值；null 表示该实现在这一层没有公开的独立旋钮 */
	transport: { label: string; value: string } | null
	stream: { label: string; value: string } | null
	idle: { label: string; value: string } | null
	backoff: string
	note: string
}

const impls: Impl[] = [
	{
		key: "codex",
		name: "实现 A（Rust，逐 provider 配置）",
		transport: { label: "request_max_retries", value: "4（上限 100）" },
		stream: { label: "stream_max_retries", value: "5（上限 100）" },
		idle: { label: "stream_idle_timeout_ms", value: "300000（5 分钟）" },
		backoff: "指数退避并带随机抖动；另有一条独立的连接重连路径，延迟从 5 秒起翻倍、封顶 60 秒",
		note: "两层旋钮完全分离，且都挂在 provider 维度——换一个网关地址就换一套重试参数。上限 100 的硬帽是为了防止把值设成一个事实上永不放弃的数字。"
	},
	{
		key: "cc",
		name: "实现 B（环境变量配置）",
		transport: { label: "CLAUDE_CODE_MAX_RETRIES", value: "10" },
		stream: null,
		idle: { label: "CLAUDE_STREAM_IDLE_TIMEOUT_MS", value: "300000，且这同时是下限" },
		backoff: "官方文档未公开退避曲线的具体形式",
		note: "空闲超时的默认值同时是最小值，低于它的配置会被静默抬回——理由是要容纳长思考停顿与代理缓冲。这是一个把「用户配错」当作常态来设计的选择。另有请求总超时 API_TIMEOUT_MS，默认 600000 毫秒。"
	},
	{
		key: "pi",
		name: "实现 C（TypeScript，settings.json）",
		transport: { label: "retry.provider.maxRetries", value: "0（刻意关闭）" },
		stream: { label: "retry.maxRetries", value: "3" },
		idle: { label: "httpIdleTimeoutMs", value: "300000，设 0 可关闭" },
		backoff: "baseDelayMs × 2^(n−1)，默认 2000 毫秒起（2s / 4s / 8s），单次封顶 maxAgentDelayMs 默认 60000",
		note: "刻意把 SDK 层重试设为 0，理由写在文档里：SDK 若先替你重试掉配额类错误，Agent 层就看不到它，可能一直阻塞到配额窗口重置。这是「让错误浮到能正确分类的那一层」的具体例子。"
	},
	{
		key: "lg",
		name: "实现 D（编排框架）",
		transport: null,
		stream: null,
		idle: null,
		backoff: "由所选模型客户端与节点级重试策略决定",
		note: "框架本身不持有模型连接，重试落在所用的模型客户端与节点重试策略上。它真正定义的是另一件事——失败之后从哪里继续，见后文的持久化与恢复。"
	}
]

const active = ref("codex")
const cur = computed(() => impls.find((i) => i.key === active.value)!)

/** 演示两层嵌套的放大效应：外层每重试一次，内层可能各自再耗尽一轮 */
const streamN = ref(3)
const transportN = ref(2)
const worst = computed(() => (streamN.value + 1) * (transportN.value + 1))

const bars = computed(() => [
	{ name: "只看流式层", n: streamN.value + 1, color: seriesColor(0) },
	{ name: "只看传输层", n: transportN.value + 1, color: seriesColor(2) },
	{ name: "两层相乘的最坏情况", n: worst.value, color: seriesColor(4) }
])
const maxN = computed(() => Math.max(...bars.value.map((b) => b.n)))
</script>

<template>
	<div class="grl">
		<div class="grl__tabs">
			<button
				v-for="i in impls"
				:key="i.key"
				type="button"
				class="grl__tab"
				:class="{ 'is-on': active === i.key }"
				:aria-pressed="active === i.key"
				@click="active = i.key"
			>{{ i.name }}</button>
		</div>

		<div class="grl__layers">
			<div class="grl__layer">
				<span>传输层 · 单个 HTTP 请求</span>
				<template v-if="cur.transport">
					<code>{{ cur.transport.label }}</code>
					<b>{{ cur.transport.value }}</b>
				</template>
				<em v-else>未提供独立旋钮</em>
			</div>
			<div class="grl__layer">
				<span>流式层 · 一次采样的整条流</span>
				<template v-if="cur.stream">
					<code>{{ cur.stream.label }}</code>
					<b>{{ cur.stream.value }}</b>
				</template>
				<em v-else>未提供独立旋钮</em>
			</div>
			<div class="grl__layer">
				<span>空闲看门狗</span>
				<template v-if="cur.idle">
					<code>{{ cur.idle.label }}</code>
					<b>{{ cur.idle.value }}</b>
				</template>
				<em v-else>未提供独立旋钮</em>
			</div>
		</div>

		<p class="grl__backoff"><i>退避曲线</i>{{ cur.backoff }}</p>
		<p class="grl__note"><i>这个取值说明了什么</i>{{ cur.note }}</p>

		<div class="grl__calc">
			<div class="infra-controls">
				<div class="infra-control">
					<label for="grl-stream">流式层重试次数 <b class="cs-num">{{ streamN }}</b></label>
					<input id="grl-stream" v-model.number="streamN" type="range" min="0" max="8" step="1" />
				</div>
				<div class="infra-control">
					<label for="grl-transport">传输层重试次数 <b class="cs-num">{{ transportN }}</b></label>
					<input id="grl-transport" v-model.number="transportN" type="range" min="0" max="8" step="1" />
				</div>
			</div>

			<div class="cs-stack cs-stack--tight">
				<div v-for="b in bars" :key="b.name" class="grl__row">
					<span class="grl__dot" :style="{ background: b.color }"></span>
					<span class="grl__name">{{ b.name }}</span>
					<span class="infra-bar grl__bar">
						<i :style="{ width: `${(b.n / maxN) * 100}%`, background: b.color }"></i>
					</span>
					<b class="cs-num grl__val">{{ b.n }} 次</b>
				</div>
			</div>
		</div>

		<p class="pc-note">
			两层旋钮各自都不大，<b>相乘之后才是真实的最坏请求数</b>：当前设置下最坏可发出
			<b class="cs-num">{{ worst }}</b> 次上游请求。这带来两个后果，一个是账单，
			一个是时间——用户看到的是一次「卡住」，而系统正在两层退避里逐级等待。
			因此这两个数字要和<b>整轮的时间预算</b>一起设，而不是各自调到「看起来安全」。
			还有一条独立于次数的判断：<b>配额耗尽与瞬时限流可能是同一个 HTTP 状态码</b>，
			但前者不可重试。把它们混为一谈，退避只是在配额恢复前空耗预算。
		</p>
	</div>
</template>

<style scoped>
.grl { margin: 1rem 0; }
.grl__tabs { display: flex; flex-wrap: wrap; gap: var(--cs-space-1); }
.grl__tab { min-height: var(--cs-tap-target); padding: var(--cs-space-1) var(--cs-space-4); cursor: pointer; font-size: var(--cs-text-xs); color: var(--cs-color-text-muted); background: var(--cs-color-bg); border: 1px solid var(--cs-color-border); border-radius: var(--cs-radius-pill); transition: var(--cs-transition-colors); }
.grl__tab:hover { background: var(--cs-color-bg-soft); }
.grl__tab.is-on { background: var(--cs-color-brand-soft); border-color: var(--cs-color-brand); color: var(--cs-color-brand); font-weight: 600; }
.grl__layers { display: grid; gap: var(--cs-space-3); margin-top: var(--cs-space-5); }
.grl__layer { display: grid; gap: var(--cs-space-1); padding: var(--cs-space-4); border: 1px solid var(--cs-color-border); border-left: 3px solid var(--cs-color-brand); border-radius: var(--cs-radius-md); background: var(--cs-color-bg); }
.grl__layer > span { font-size: var(--cs-text-2xs); color: var(--cs-color-text-subtle); }
.grl__layer code { font-family: var(--cs-font-mono); font-size: var(--cs-text-xs); color: var(--cs-color-text-muted); background: none; padding: 0; overflow-wrap: anywhere; }
.grl__layer b { font-size: var(--cs-text-md); color: var(--cs-color-brand); }
.grl__layer em { font-size: var(--cs-text-sm); font-style: normal; color: var(--cs-color-text-subtle); }
.grl__backoff, .grl__note { margin: var(--cs-space-3) 0 0; padding: var(--cs-space-4); border-radius: var(--cs-radius-sm); font-size: var(--cs-text-base); line-height: var(--cs-leading-relaxed); }
.grl__backoff { background: var(--cs-color-neutral-soft); }
.grl__note { background: var(--cs-color-warning-soft); }
.grl__backoff i, .grl__note i { display: block; font-family: var(--cs-font-mono); font-size: var(--cs-text-3xs); letter-spacing: .08em; font-style: normal; margin-bottom: var(--cs-space-1); color: var(--cs-color-text-subtle); }
.grl__note i { color: var(--cs-color-warning); }
.grl__calc { margin-top: var(--cs-space-6); display: grid; gap: var(--cs-space-5); }
.grl__calc .infra-control label b { color: var(--cs-color-brand); }
.grl__row { display: flex; align-items: center; gap: var(--cs-space-3); font-size: var(--cs-text-sm); }
.grl__dot { width: 10px; height: 10px; border-radius: var(--cs-radius-circle); flex: none; }
.grl__name { flex: 0 0 auto; width: 9.5rem; font-size: var(--cs-text-xs); color: var(--cs-color-text-muted); }
.grl__bar { flex: 1; margin-top: 0; min-width: 0; }
.grl__val { flex: 0 0 auto; color: var(--cs-color-brand); }
@media (max-width: 640px) {
	.grl__row { flex-wrap: wrap; }
	.grl__name { width: auto; flex: 1; }
	.grl__bar { flex: 1 0 100%; }
}
@media (min-width: 861px) {
	.grl__layers { grid-template-columns: repeat(3, 1fr); }
}
</style>
