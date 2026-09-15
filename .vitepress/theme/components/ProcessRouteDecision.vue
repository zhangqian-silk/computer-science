<script setup lang="ts">
import { computed, ref } from "vue"

/**
 * 命令的两类走向：会结束的与不会结束的。
 *
 * 这件事值得单独画出来，是因为它的错误形态非常固定——
 * 把一个**不会自己结束**的服务当成会结束的命令来跑，
 * 于是「执行 → 等待 → 超时 → 处置」这条链从第一步就选错了路。
 *
 * 右侧展示超时之后的两种处置：**自动转后台**与**直接终止**。
 * 这两种处置都真实存在于公开实现中，差别不是优劣，
 * 而是「超时」被解释成了什么：资源保护，还是「这条命令本该更快」。
 */

type Kind = {
	key: string
	idx: number
	name: string
	examples: string
	ends: boolean
	route: string
	observe: string
	risk: string
}

const kinds: Kind[] = [
	{
		key: "finite",
		idx: 0,
		name: "会结束的命令",
		examples: "构建、测试、安装依赖、git 操作、格式化",
		ends: true,
		route: "前台同步执行，带超时。返回退出码与输出。",
		observe: "结果本身就是观察：退出码区分成败，失败时把错误输出一并回填，模型据此改而不是盲目重试。",
		risk: "主要风险是输出体量——构建日志动辄数万行，不加约束就能一次吃满窗口。"
	},
	{
		key: "infinite",
		idx: 1,
		name: "不会结束的服务",
		examples: "开发服务器、watch 构建、日志跟随、数据库",
		ends: false,
		route: "显式后台启动，立即拿到句柄；既不等它，也不因为超时杀它。",
		observe: "靠轮询输出里的就绪标志、探测端口或调健康检查接口来判断可用，并带上重试上限与退避。",
		risk: "主要风险是残留：没有句柄就既读不到输出也停不掉，会话结束后继续占着端口。"
	}
]

type Policy = {
	key: string
	name: string
	one: string
	detail: string
	tone: "pass" | "warn"
}

const policies: Policy[] = [
	{
		key: "background",
		name: "超时后转入后台",
		one: "超时被理解为「不该继续占着前台」",
		tone: "pass",
		detail:
			"命令没被杀掉，而是移到后台继续跑，返回里带上任务标识与输出文件路径，模型可以接着做别的事、稍后回来看。这条路径对「其实还差几秒就好了」的构建最友好。代价是需要为后台任务定义清楚的生命周期——谁启动的、什么时候回收、会话结束时怎么办。注意有些命令不适合自动转后台（例如整条命令无法被完整解析时），这时只能按超时处理。",
	},
	{
		key: "kill",
		name: "超时即终止",
		one: "超时被理解为「这条命令本该更快」",
		tone: "warn",
		detail:
			"到点直接终止并报错。它的好处是资源边界清晰、不留后台状态；坏处是把「慢」和「卡死」当成了同一件事——一个只是慢的构建被杀掉之后，重跑一次往往还是会超时，于是陷入反复。采用这种处置时，分档超时与「该用后台的显式用后台」这两件事就变得必要。"
	}
]

const kindPick = ref(1)
const policyPick = ref(0)
const curKind = computed(() => kinds[kindPick.value])
const curPolicy = computed(() => policies[policyPick.value])

const mismatch = computed(() => !curKind.value.ends)
</script>

<template>
	<div class="prd">
		<div class="prd__kinds">
			<button
				v-for="(k, i) in kinds"
				:key="k.key"
				type="button"
				class="prd__kind"
				:class="{ 'is-open': kindPick === i }"
				:style="{ '--c': `var(--cs-series-${k.idx + 1})` }"
				:aria-pressed="kindPick === i"
				@click="kindPick = i"
			>
				<span class="prd__kind-tag cs-mono">{{ k.ends ? "有终点" : "无终点" }}</span>
				<strong>{{ k.name }}</strong>
				<span class="prd__kind-eg">{{ k.examples }}</span>
			</button>
		</div>

		<article class="prd__panel" :style="{ '--c': `var(--cs-series-${curKind.idx + 1})` }">
			<dl class="prd__dl">
				<div><dt>正确走向</dt><dd>{{ curKind.route }}</dd></div>
				<div><dt>怎么观察</dt><dd>{{ curKind.observe }}</dd></div>
				<div><dt>主要风险</dt><dd>{{ curKind.risk }}</dd></div>
			</dl>
		</article>

		<div
			class="cs-state"
			:class="mismatch ? 'cs-state--fail' : 'cs-state--pass'"
		>
			<b>{{ mismatch ? "用前台同步跑它会怎样" : "用前台同步跑它没有问题" }}</b>
			<span v-if="mismatch">
				整条链从第一步就错了：它永远不会返回，于是必然走到超时，而超时之后无论是被杀还是被转后台，
				这次调用都没有拿到它本该拿到的东西——服务是否已经就绪。手工在命令尾部加一个 <code>&amp;</code> 同样不够：
				进程确实活着，但没有句柄，之后既读不到输出也无法定点停掉。
			</span>
			<span v-else>
				它有确定的终点，退出码与输出就是完整的观察结果。只需要把输出体量约束住，并给足够长但有上限的超时。
			</span>
		</div>

		<div class="prd__policy">
			<span class="prd__policy-label">超时之后怎么处置</span>
			<div class="infra-tabs">
				<button
					v-for="(p, i) in policies"
					:key="p.key"
					type="button"
					:class="{ 'is-active': policyPick === i }"
					:aria-pressed="policyPick === i"
					@click="policyPick = i"
				>
					{{ p.name }}
				</button>
			</div>
			<div class="cs-state" :class="`cs-state--${curPolicy.tone}`">
				<b>{{ curPolicy.one }}</b>
				<span>{{ curPolicy.detail }}</span>
			</div>
		</div>

		<p class="pc-note">
			两种超时处置都真实存在于公开实现中，并且同一个实现可能对不同命令采取不同处置。
			它们的分歧点在于<b>「超时」这个信号被解释成了什么</b>，而不在于哪一种更正确。
			真正稳定的结论是前一层：<b>先判断这条命令有没有终点，再决定它走哪条路</b>——这一步判断错了，后面的超时策略无论怎么调都救不回来。
		</p>
	</div>
</template>

<style scoped>
.prd { margin: 1rem 0; background: var(--cs-color-bg); border: 1px solid var(--cs-color-border); border-radius: var(--cs-radius-lg); padding: var(--cs-space-5) var(--cs-space-6); }
.prd__kinds { display: grid; grid-template-columns: repeat(2, 1fr); gap: var(--cs-space-2); }
.prd__kind { display: grid; gap: var(--cs-space-1); text-align: left; cursor: pointer; background: var(--cs-color-bg); border: 1px solid var(--cs-color-border); border-top: 3px solid var(--c); border-radius: var(--cs-radius-lg); padding: var(--cs-space-3); min-height: var(--cs-tap-target); transition: var(--cs-transition-colors); }
.prd__kind:hover { background: var(--cs-color-bg-soft); }
.prd__kind.is-open { background: color-mix(in srgb, var(--c) 10%, transparent); border-color: var(--c); }
.prd__kind-tag { font-size: var(--cs-text-3xs); letter-spacing: .08em; color: var(--c); }
.prd__kind strong { font-size: var(--cs-text-md); line-height: var(--cs-leading-tight); }
.prd__kind-eg { font-size: var(--cs-text-2xs); line-height: var(--cs-leading-normal); color: var(--cs-color-text-muted); }
.prd__panel { margin: var(--cs-space-4) 0; background: var(--cs-color-bg); border: 1px solid var(--cs-color-border); border-left: 3px solid var(--c); border-radius: var(--cs-radius-lg); padding: var(--cs-space-4) var(--cs-space-5); }
.prd__dl { margin: 0; display: grid; gap: var(--cs-space-2); }
.prd__dl > div { display: grid; grid-template-columns: 5rem minmax(0, 1fr); gap: var(--cs-space-3); }
.prd__dl dt { font-size: var(--cs-text-2xs); color: var(--cs-color-text-subtle); }
.prd__dl dd { margin: 0; font-size: var(--cs-text-sm); line-height: var(--cs-leading-normal); color: var(--cs-color-text); }
.prd .cs-state { display: grid; gap: var(--cs-space-1); font-size: var(--cs-text-sm); line-height: var(--cs-leading-relaxed); }
.prd .cs-state code { font-family: var(--cs-font-mono); font-size: var(--cs-text-2xs); }
.prd__policy { margin-top: var(--cs-space-5); display: grid; gap: var(--cs-space-2); }
.prd__policy-label { font-size: var(--cs-text-xs); color: var(--cs-color-text-muted); }
@media (max-width: 720px) {
	.prd__kinds { grid-template-columns: 1fr; }
	.prd__dl > div { grid-template-columns: 1fr; gap: var(--cs-space-1); }
}
</style>
