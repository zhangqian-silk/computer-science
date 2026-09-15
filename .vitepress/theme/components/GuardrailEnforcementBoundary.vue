<script setup lang="ts">
import { computed, ref } from "vue"

/**
 * 护栏与强制边界：同一条「不许动生产库」的意图，落在不同层上强度完全不同。
 *
 * 组件让读者对同一个规避手法逐层检验：提示词、模型侧自律、hook 拦截、
 * 声明式规则、沙箱。越往下越难绕过，代价也越高。核心结论是
 * 「拦截式扩展是护栏，不是强制边界」——它只覆盖宿主已接线的那几条路径。
 */

type Layer = {
	key: string
	name: string
	kind: string
	color: string
	/** 该层对每种规避手法是否挡得住 */
	holds: Record<string, boolean | null>
	how: string
	cost: string
}

type Evasion = {
	key: string
	short: string
	full: string
}

const evasions: Evasion[] = [
	{ key: "direct", short: "直接执行", full: "模型直接发出那条命令，没有任何伪装。" },
	{ key: "variant", short: "换种写法", full: "同样的效果换个命令、换个参数顺序、走一层别名或环境变量。" },
	{ key: "another", short: "换条工具路径", full: "不走被接线的那个工具，改用另一个能达成等价效果的工具——脚本、编辑器、包管理器的钩子。" },
	{ key: "injected", short: "外部内容诱导", full: "取回的网页或工具结果里带着指令，诱导模型主动去做那件事。" }
]

const layers: Layer[] = [
	{
		key: "prompt",
		name: "写在提示词里",
		kind: "概率性约束",
		color: "var(--cs-color-danger)",
		holds: { direct: true, variant: null, another: null, injected: false },
		how: "在系统提示或规程文件里写明「不许对生产库执行写操作」。模型多数时候会遵守。",
		cost: "近乎为零，改一行字就生效。也正因如此，它承担不了任何需要保证的约束。"
	},
	{
		key: "modelside",
		name: "交给模型自己判断",
		kind: "概率性约束",
		color: "var(--cs-color-danger)",
		holds: { direct: true, variant: null, another: null, injected: false },
		how: "让一次额外的模型调用来评判这个动作是否越界，比如用一个小模型做前置判定。",
		cost: "每次判定都要付一次采样。它比死板的模式匹配更懂语义，但仍然是概率性的——而且它自己也会被诱导。"
	},
	{
		key: "hook",
		name: "拦截式扩展",
		kind: "确定性，但覆盖不完全",
		color: "var(--cs-color-warning)",
		holds: { direct: true, variant: true, another: false, injected: true },
		how: "在工具调用前插入一段代码，检查入参并返回拒绝。同一份输入必定得到同一个判定，不受采样波动影响。",
		cost: "每次工具调用都要起一个进程，需要管超时。真正的短板在覆盖面：它只能看见宿主接线过的那些工具路径，有实现在文档里直言这是护栏而非完整的强制边界。"
	},
	{
		key: "rule",
		name: "声明式规则",
		kind: "确定性，但覆盖不完全",
		color: "var(--cs-color-warning)",
		holds: { direct: true, variant: null, another: false, injected: true },
		how: "以模式声明放行、询问用户、禁止，由宿主统一执行。多条命中时取最严的一条。",
		cost: "比写脚本省事，也更好审计。但它的判断力止于模式本身——穷举危险写法是一场注定追不上的竞赛。"
	},
	{
		key: "sandbox",
		name: "沙箱与凭据边界",
		kind: "强制边界",
		color: "var(--cs-color-success)",
		holds: { direct: true, variant: true, another: true, injected: true },
		how: "根本不给出口：进程由操作系统机制约束可写范围与出网能力，生产库的凭据压根不在这个环境里。",
		cost: "最高。要搭隔离环境、管凭据、处理一批「本来能跑现在跑不了」的合法场景。但它是唯一不依赖模型配合的一层。"
	}
]

const evIdx = ref(2)
const ev = computed(() => evasions[evIdx.value])
const mark = (v: boolean | null) => (v === true ? "挡得住" : v === false ? "挡不住" : "看写法")
const cls = (v: boolean | null) => (v === true ? "is-hold" : v === false ? "is-leak" : "is-maybe")
</script>

<template>
	<div class="geb">
		<div class="geb__evasions">
			<span class="geb__lab">选一种规避手法，看各层分别挡不挡得住</span>
			<div class="geb__ev-row">
				<button
					v-for="(e, i) in evasions"
					:key="e.key"
					type="button"
					class="geb__ev"
					:class="{ 'is-on': evIdx === i }"
					:aria-pressed="evIdx === i"
					@click="evIdx = i"
				>{{ e.short }}</button>
			</div>
			<p class="geb__ev-full">{{ ev.full }}</p>
		</div>

		<div class="geb__layers">
			<article
				v-for="l in layers"
				:key="l.key"
				class="geb__layer"
				:style="{ '--c': l.color }"
			>
				<header>
					<div>
						<strong>{{ l.name }}</strong>
						<span class="geb__kind">{{ l.kind }}</span>
					</div>
					<span class="geb__mark" :class="cls(l.holds[ev.key])">{{ mark(l.holds[ev.key]) }}</span>
				</header>
				<p class="geb__how">{{ l.how }}</p>
				<p class="geb__cost"><i>代价</i>{{ l.cost }}</p>
			</article>
		</div>

		<p class="pc-note">
			这张表最重要的一行不是最上面那条，而是<b>中间两层</b>。提示词靠不住是常识，
			沙箱可靠也是常识；真正容易误判的是拦截式扩展——它确定性地执行，于是很容易被当成强制边界，
			而它实际拦的只是<b>宿主已经接线的那几条路径</b>。一个等价效果换条路走就绕过去了，
			并且不会有任何报错提示你它被绕过了。<b>可以用拦截式扩展减少失误，但不能用它防住对手</b>；
			需要保证的地方，把能力本身拿走比在半路检查更可靠。
		</p>
	</div>
</template>

<style scoped>
.geb { margin: 1rem 0; }
.geb__evasions { padding: var(--cs-space-3) var(--cs-space-4); background: var(--cs-color-bg-soft); border-radius: var(--cs-radius-lg); margin-bottom: var(--cs-space-4); }
.geb__lab { font-family: var(--cs-font-mono); font-size: var(--cs-text-3xs); letter-spacing: .08em; color: var(--cs-color-text-subtle); }
.geb__ev-row { display: flex; flex-wrap: wrap; gap: var(--cs-space-2); margin-top: var(--cs-space-2); }
.geb__ev { min-height: var(--cs-tap-target); cursor: pointer; padding: var(--cs-space-1) var(--cs-space-4); font-size: var(--cs-text-sm); color: var(--cs-color-text-muted); background: var(--cs-color-bg); border: 1px solid var(--cs-color-border); border-radius: var(--cs-radius-pill); transition: var(--cs-transition-colors); }
.geb__ev:hover { background: var(--cs-color-bg-soft); }
.geb__ev.is-on { background: var(--cs-color-brand); border-color: var(--cs-color-brand); color: var(--cs-color-on-brand); font-weight: 600; }
.geb__ev-full { margin: var(--cs-space-3) 0 0; font-size: var(--cs-text-sm); line-height: var(--cs-leading-normal); color: var(--cs-color-text-muted); }
.geb__layers { display: grid; gap: var(--cs-space-2); }
.geb__layer { background: var(--cs-color-bg); border: 1px solid var(--cs-color-border); border-left: 3px solid var(--c); border-radius: var(--cs-radius-lg); padding: var(--cs-space-4); }
.geb__layer header { display: flex; flex-wrap: wrap; align-items: flex-start; justify-content: space-between; gap: var(--cs-space-3); }
.geb__layer strong { display: block; font-size: var(--cs-text-md); line-height: var(--cs-leading-tight); }
.geb__kind { font-family: var(--cs-font-mono); font-size: var(--cs-text-3xs); color: var(--c); }
.geb__mark { flex: none; font-size: var(--cs-text-3xs); border-radius: var(--cs-radius-pill); padding: var(--cs-space-1) var(--cs-space-3); border: 1px solid currentColor; }
.geb__mark.is-hold { color: var(--cs-color-success); }
.geb__mark.is-leak { color: var(--cs-color-danger); }
.geb__mark.is-maybe { color: var(--cs-color-warning); }
.geb__how { margin: var(--cs-space-3) 0 var(--cs-space-2); font-size: var(--cs-text-sm); line-height: var(--cs-leading-relaxed); color: var(--cs-color-text); }
.geb__cost { margin: 0; font-size: var(--cs-text-sm); line-height: var(--cs-leading-normal); color: var(--cs-color-text-muted); }
.geb__cost i { display: block; font-family: var(--cs-font-mono); font-size: var(--cs-text-3xs); letter-spacing: .08em; font-style: normal; color: var(--cs-color-text-subtle); margin-bottom: var(--cs-space-1); }
</style>
