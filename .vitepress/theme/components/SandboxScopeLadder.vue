<script setup lang="ts">
import { computed, ref } from "vue"

/**
 * 隔离粒度与逃逸面。
 *
 * 这张图只回答一个问题：**被关进去的是什么**。
 * 四档粒度从「只关一条命令通道」到「连内核都不共享」，
 * 关键在于每一档「留在外面的东西」——那就是它的逃逸面。
 *
 * 最常见的错误是只把命令执行放进沙箱，而文件写入工具、外部工具服务、生命周期钩子
 * 都在外面裸跑：此时不必攻破沙箱，绕过它就行。
 */

type Scope = {
	key: string
	name: string
	inside: string[]
	outside: string[]
	needsContainer: string
	cost: string
	escape: string
	tone: "fail" | "warn" | "pass"
	verdict: string
}

const scopes: Scope[] = [
	{
		key: "cmd",
		name: "只关命令通道",
		inside: ["模型生成的 shell 命令", "这些命令派生的子进程"],
		outside: ["运行时自带的文件读写工具", "外部工具服务进程", "生命周期钩子脚本", "运行时进程自身"],
		needsContainer: "不需要",
		cost: "最低。多数平台上装好原语即可启用",
		escape: "不必攻破，绕过即可：用文件写入工具改一个开机加载的脚本、改持续集成配置、改版本控制目录里的钩子，下一次这些文件被执行时就在沙箱之外了。",
		tone: "fail",
		verdict: "最容易被误认为「已经有沙箱了」的一档。它挡住的是一条通道，不是一类能力。"
	},
	{
		key: "proc",
		name: "关整个运行时进程",
		inside: ["运行时进程自身", "全部内置工具", "外部工具服务", "钩子脚本", "所有派生子进程"],
		outside: ["宿主机上的其他程序", "内核本身"],
		needsContainer: "不需要",
		cost: "低。同样依赖操作系统原语，但作用域从一条通道扩到整个进程树",
		escape: "面小得多：能力面被整体收在同一套规则里，不存在「换个工具就出去了」的路径。剩下的风险主要来自内核层面的缺陷与配置疏漏。",
		tone: "warn",
		verdict: "本地开发的合理默认。它把「隔离的是命令还是能力」这个问题回答成了后者。"
	},
	{
		key: "container",
		name: "容器化开发环境",
		inside: ["整个开发环境：依赖、工具链、运行时"],
		outside: ["共享的宿主内核"],
		needsContainer: "需要",
		cost: "中。要维护镜像与挂载，还要处理凭据如何进入容器",
		escape: "文件系统与进程视图都换了一套，宿主目录默认不可见。共享内核仍是理论上的逃逸面；此外容器内若把边界当成唯一防线而关掉内层沙箱，容器内的任何东西都能拿到容器内的全部凭据。",
		tone: "warn",
		verdict: "当容器本身就是预期的安全边界时，不必在里面再叠一层，但要接受「容器内即全部可达」这个后果。"
	},
	{
		key: "vm",
		name: "独立内核环境",
		inside: ["完整环境加独立内核"],
		outside: ["几乎没有共享面"],
		needsContainer: "需要，且需要虚拟化支持",
		cost: "最高。启动更慢、资源更重、镜像与生命周期都要治理",
		escape: "最小。代价换来的是不共享内核，因此内核层面的缺陷不再直接构成跨界风险。",
		tone: "pass",
		verdict: "面向明确不可信的输入：任意仓库、外部提交、来源不明的依赖。"
	}
]

const cur = ref("cmd")
const curScope = computed(() => scopes.find((s) => s.key === cur.value)!)
const toneLabel: Record<string, string> = { fail: "逃逸面最大", warn: "逃逸面收窄", pass: "逃逸面最小" }
</script>

<template>
	<div class="isb">
		<p class="isb__hint">
			四档粒度的差别不在「用了什么技术」，而在<b>被关进去的是什么</b>。
			真正决定安全性的是每一档留在外面的那一列。
		</p>

		<div class="isb__tabs">
			<button
				v-for="s in scopes"
				:key="s.key"
				type="button"
				class="isb__tab"
				:class="[`is-${s.tone}`, { 'is-on': cur === s.key }]"
				:aria-pressed="cur === s.key"
				@click="cur = s.key"
			>{{ s.name }}</button>
		</div>

		<div class="isb__grid">
			<section class="isb__col isb__col--in">
				<h4><span class="isb__mark" aria-hidden="true">▣</span>关在里面</h4>
				<ul>
					<li v-for="i in curScope.inside" :key="i">{{ i }}</li>
				</ul>
			</section>
			<section class="isb__col isb__col--out">
				<h4><span class="isb__mark" aria-hidden="true">▢</span>留在外面</h4>
				<ul>
					<li v-for="o in curScope.outside" :key="o">{{ o }}</li>
				</ul>
			</section>
		</div>

		<div class="isb__meta">
			<span><i>需要容器</i>{{ curScope.needsContainer }}</span>
			<span><i>配置代价</i>{{ curScope.cost }}</span>
		</div>

		<div class="isb__escape" :class="`is-${curScope.tone}`">
			<i>{{ toneLabel[curScope.tone] }} · 逃逸面长什么样</i>
			<p>{{ curScope.escape }}</p>
		</div>

		<p class="isb__verdict">{{ curScope.verdict }}</p>

		<p class="pc-note">
			这四档回答的都是同一个问题——<b>边界画在哪里</b>。但边界之外还有一个它管不到的维度：
			隔离决定「能做什么」，不决定「被谁指挥」。一份被读进来的外部内容里写着的指令，
			在任何一档隔离下都同样能影响模型的下一步动作，因为它走的不是被隔离的那条路，
			而是上下文本身。所以这张图应当配一句相反方向的话：
			<b>提高隔离等级不会降低被外部内容误导的概率，它只降低误导成功之后的破坏半径。</b>
		</p>
	</div>
</template>

<style scoped>
.isb { margin: 1rem 0; }
.isb__hint { margin: 0 0 var(--cs-space-4); font-size: var(--cs-text-sm); line-height: var(--cs-leading-normal); color: var(--cs-color-text-muted); }
.isb__tabs { display: flex; flex-wrap: wrap; gap: var(--cs-space-1); }
.isb__tab { min-height: var(--cs-tap-target); padding: var(--cs-space-1) var(--cs-space-4); cursor: pointer; font-size: var(--cs-text-xs); color: var(--cs-color-text-muted); background: var(--cs-color-bg); border: 1px solid var(--cs-color-border); border-radius: var(--cs-radius-pill); transition: var(--cs-transition-colors); }
.isb__tab:hover { background: var(--cs-color-bg-soft); }
.isb__tab.is-on.is-fail { background: var(--cs-color-danger-soft); border-color: var(--cs-color-danger); color: var(--cs-color-danger); font-weight: 600; }
.isb__tab.is-on.is-warn { background: var(--cs-color-warning-soft); border-color: var(--cs-color-warning); color: var(--cs-color-warning); font-weight: 600; }
.isb__tab.is-on.is-pass { background: var(--cs-color-success-soft); border-color: var(--cs-color-success); color: var(--cs-color-success); font-weight: 600; }
.isb__grid { display: grid; gap: var(--cs-space-3); margin-top: var(--cs-space-4); }
.isb__col { padding: var(--cs-space-4) var(--cs-space-5); border: 1px solid var(--cs-color-border); border-radius: var(--cs-radius-lg); background: var(--cs-color-bg); }
.isb__col--in { border-left: 3px solid var(--cs-color-success); }
.isb__col--out { border-left: 3px solid var(--cs-color-danger); }
.isb__col h4 { display: flex; align-items: center; gap: var(--cs-space-2); margin: 0 0 var(--cs-space-3); font-size: var(--cs-text-sm); color: var(--cs-color-text); }
.isb__col--in .isb__mark { color: var(--cs-color-success); }
.isb__col--out .isb__mark { color: var(--cs-color-danger); }
.isb__col ul { margin: 0; padding-left: var(--cs-space-5); }
.isb__col li { font-size: var(--cs-text-sm); line-height: var(--cs-leading-normal); color: var(--cs-color-text-muted); }
.isb__meta { display: flex; flex-wrap: wrap; gap: var(--cs-space-3) var(--cs-space-6); margin-top: var(--cs-space-4); }
.isb__meta span { font-size: var(--cs-text-sm); color: var(--cs-color-text-muted); }
.isb__meta i { display: block; font-family: var(--cs-font-mono); font-size: var(--cs-text-3xs); letter-spacing: .08em; font-style: normal; color: var(--cs-color-text-subtle); }
.isb__escape { margin-top: var(--cs-space-4); padding: var(--cs-space-4) var(--cs-space-5); border-radius: var(--cs-radius-md); }
.isb__escape.is-fail { background: var(--cs-color-danger-soft); }
.isb__escape.is-warn { background: var(--cs-color-warning-soft); }
.isb__escape.is-pass { background: var(--cs-color-success-soft); }
.isb__escape i { display: block; font-family: var(--cs-font-mono); font-size: var(--cs-text-3xs); letter-spacing: .08em; font-style: normal; margin-bottom: var(--cs-space-2); }
.isb__escape.is-fail i { color: var(--cs-color-danger); }
.isb__escape.is-warn i { color: var(--cs-color-warning); }
.isb__escape.is-pass i { color: var(--cs-color-success); }
.isb__escape p { margin: 0; font-size: var(--cs-text-base); line-height: var(--cs-leading-relaxed); color: var(--cs-color-text); }
.isb__verdict { margin: var(--cs-space-4) 0 0; padding-top: var(--cs-space-3); border-top: 1px dashed var(--cs-color-border); font-size: var(--cs-text-sm); line-height: var(--cs-leading-normal); color: var(--cs-color-text-muted); }
@media (min-width: 861px) {
	.isb__grid { grid-template-columns: 1fr 1fr; }
}
</style>
