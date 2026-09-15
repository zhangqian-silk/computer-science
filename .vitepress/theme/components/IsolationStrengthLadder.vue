<script setup lang="ts">
import { computed, ref } from "vue"

/**
 * 隔离强度阶梯：从「什么都不隔离」到「独立内核」，以及每一档的代价。
 *
 * 这张图要纠正一个常见的简化——把「有没有沙箱」当成一个布尔量。
 * 实际上隔离是一条连续谱，每上一档换来的是更强的边界，
 * 付出的是启动延迟、兼容性损失与运维复杂度。
 * 选型的问题从来不是「要不要沙箱」，而是「这段代码的不可信程度配得上哪一档」。
 *
 * 更关键的是**平台差异**：同一个产品在 macOS 与 Linux 上走的是完全不同的内核机制，
 * 因而边界形状、失败模式与依赖前提都不一样。把某一个平台上的实测行为
 * 当成产品的通用行为，是这个主题上最常见的错误。
 *
 * 机制名称与平台对应关系取自各实现的官方文档与官方仓库源码（核验于 2026-09-14）。
 * 强度与开销的刻度是为呈现相对关系所设的教学化估计，不代表任何实测基准。
 */

type Rung = {
	key: string
	name: string
	boundary: string
	/** 相对强度与相对开销，1–5 的教学化刻度 */
	strength: number
	cost: number
	breaks: string
	fits: string
	platform?: { os: string; how: string }[]
}

const rungs: Rung[] = [
	{
		key: "none",
		name: "0 · 不隔离",
		boundary: "与启动它的用户同权限：能读能写的范围就是你的范围",
		strength: 1,
		cost: 1,
		breaks: "一条越界命令就能改掉仓库外的文件、读到凭据文件、或把内容发到外网",
		fits: "只有在外层已经提供边界时才成立——例如整个进程本来就跑在一次性容器里",
		platform: [{ os: "全平台", how: "无机制；边界完全由外层环境提供" }]
	},
	{
		key: "policy",
		name: "1 · 应用层规则",
		boundary: "宿主在执行前按命令字符串与路径匹配规则，决定放行、询问或拒绝",
		strength: 2,
		cost: 1,
		breaks: "判定发生在执行「之前」，依据是命令看起来要做什么。一条命令实际做的事超出它名字的暗示时，规则就落空了",
		fits: "作为策略层与审计入口很有价值，作为唯一的安全边界不够",
		platform: [{ os: "全平台", how: "纯用户态匹配，不依赖内核能力，因而也不受内核版本限制" }]
	},
	{
		key: "os",
		name: "2 · 操作系统原语",
		boundary: "内核在系统调用层拦截：越界的文件访问与网络连接根本不会成功",
		strength: 4,
		cost: 2,
		breaks: "平台能力参差；某些工具在沙箱内会以难以归因的方式失败（例如剪贴板写不进去、版本控制操作因无法替换文件而报只读）",
		fits: "本机开发场景的主力档位——它约束的是「已经运行起来的进程」，因此模型选了什么命令都不影响边界是否成立",
		platform: [
			{ os: "macOS", how: "Seatbelt：通过 sandbox-exec 加载一份策略描述；由系统自带，无需额外安装" },
			{ os: "Linux", how: "两条不同路线并存——一类实现用 Landlock 管文件访问、seccomp 过滤网络相关系统调用；另一类用 bubblewrap 建命名空间，并需要 socat 之类的中继把流量导向代理。后者在部分发行版上还需为 bwrap 单独放开用户命名空间权限" },
			{ os: "WSL2", how: "沿用 Linux 路线；WSL1 因缺少所需内核能力而不被支持" },
			{ os: "Windows", how: "各实现取向不同，有的走受限令牌，有的建议改用 WSL2；这一档在 Windows 上的成熟度明显低于前两者" }
		]
	},
	{
		key: "userkernel",
		name: "3 · 用户态内核",
		boundary: "系统调用先被一层用户态内核接管，宿主内核暴露面大幅收窄",
		strength: 4,
		cost: 3,
		breaks: "部分系统调用不被支持或性能下降；对 I/O 密集型任务的开销尤为明显",
		fits: "需要在隔离强度与启动速度之间取平衡的通用执行环境",
		platform: [{ os: "Linux", how: "以独立用户态内核代理系统调用；不依赖硬件虚拟化" }]
	},
	{
		key: "vm",
		name: "4 · 独立内核",
		boundary: "轻量虚拟机：每个实例有自己的内核，逃逸要先突破虚拟化边界",
		strength: 5,
		cost: 4,
		breaks: "启动慢、资源占用高、镜像与网络配置都要另做一套",
		fits: "执行完全不可信的代码，尤其是面向外部用户、且实例按会话创建与销毁的场景",
		platform: [{ os: "服务端", how: "轻量虚拟机或等价方案；通常由云上的沙箱产品提供，而不是本机自建" }]
	}
]

const active = ref("os")
const cur = computed(() => rungs.find((r) => r.key === active.value)!)
</script>

<template>
	<div class="isl">
		<div class="isl__ladder">
			<button
				v-for="r in rungs"
				:key="r.key"
				type="button"
				class="isl__rung"
				:class="{ 'is-on': active === r.key }"
				:aria-pressed="active === r.key"
				@click="active = r.key"
			>
				<b>{{ r.name }}</b>
				<span class="isl__gauges">
					<span class="isl__gauge">
						<i>强度</i>
						<span class="isl__pips">
							<em
								v-for="n in 5"
								:key="`s${n}`"
								:class="{ 'is-lit': n <= r.strength }"
							></em>
						</span>
					</span>
					<span class="isl__gauge">
						<i>开销</i>
						<span class="isl__pips isl__pips--cost">
							<em
								v-for="n in 5"
								:key="`c${n}`"
								:class="{ 'is-lit': n <= r.cost }"
							></em>
						</span>
					</span>
				</span>
			</button>
		</div>

		<div class="isl__detail">
			<p class="isl__boundary"><i>边界在哪里</i>{{ cur.boundary }}</p>
			<p class="isl__breaks"><i>代价与破绽</i>{{ cur.breaks }}</p>
			<p class="isl__fits"><i>什么时候够用</i>{{ cur.fits }}</p>
		</div>

		<div v-if="cur.platform" class="isl__platforms">
			<p class="isl__ptitle">同一档位在不同平台上的实现路径</p>
			<div v-for="p in cur.platform" :key="p.os" class="isl__platform">
				<b>{{ p.os }}</b>
				<span>{{ p.how }}</span>
			</div>
		</div>

		<p class="pc-note">
			两条容易被略过的判断。其一，<b>档位 1 与档位 2 不是同一种东西的强弱版本</b>：
			前者在执行前按「命令看起来要做什么」判定，后者在执行中由内核按「进程实际访问了什么」拦截。
			前者能被一条名不副实的命令绕过，后者不能——所以两者是互补关系，不是替代关系。
			其二，<b>同一档位在不同平台上是不同的机制</b>，连依赖前提都不一样：
			有的平台自带、有的要另装两个包、有的还要先放开内核权限。
			因此「这个产品有沙箱」不是一句可移植的结论，必须落到平台与版本上说。
			还有一条工程上的必答题：<b>依赖缺失或平台不支持时，默认是降级运行还是硬失败？</b>
			默认降级对个人使用方便，但对「把沙箱当作安全门禁」的部署来说是个静默的缺口。
		</p>
	</div>
</template>

<style scoped>
.isl { margin: 1rem 0; }
.isl__ladder { display: grid; gap: var(--cs-space-2); }
.isl__rung { display: flex; flex-wrap: wrap; align-items: center; justify-content: space-between; gap: var(--cs-space-3); min-height: var(--cs-tap-target); padding: var(--cs-space-3) var(--cs-space-4); cursor: pointer; text-align: left; background: var(--cs-color-bg); border: 1px solid var(--cs-color-border); border-left: 3px solid var(--cs-color-border-strong); border-radius: var(--cs-radius-md); transition: var(--cs-transition-colors); }
.isl__rung:hover { background: var(--cs-color-bg-soft); }
.isl__rung.is-on { border-left-color: var(--cs-color-brand); background: var(--cs-color-brand-soft); }
.isl__rung b { font-size: var(--cs-text-base); color: var(--cs-color-text); }
.isl__rung.is-on b { color: var(--cs-color-brand); }
.isl__gauges { display: flex; gap: var(--cs-space-5); }
.isl__gauge { display: flex; align-items: center; gap: var(--cs-space-2); }
.isl__gauge i { font-family: var(--cs-font-mono); font-size: var(--cs-text-3xs); font-style: normal; color: var(--cs-color-text-subtle); }
.isl__pips { display: flex; gap: 3px; }
.isl__pips em { width: 8px; height: 8px; border-radius: var(--cs-radius-circle); background: var(--cs-color-neutral-soft); border: 1px solid var(--cs-color-border); }
.isl__pips em.is-lit { background: var(--cs-color-success); border-color: var(--cs-color-success); }
.isl__pips--cost em.is-lit { background: var(--cs-color-warning); border-color: var(--cs-color-warning); }
.isl__detail { display: grid; gap: var(--cs-space-3); margin-top: var(--cs-space-5); }
.isl__boundary, .isl__breaks, .isl__fits { margin: 0; padding: var(--cs-space-4); border-radius: var(--cs-radius-sm); font-size: var(--cs-text-base); line-height: var(--cs-leading-relaxed); }
.isl__boundary { background: var(--cs-color-info-soft); }
.isl__breaks { background: var(--cs-color-warning-soft); }
.isl__fits { background: var(--cs-color-success-soft); }
.isl__detail i { display: block; font-family: var(--cs-font-mono); font-size: var(--cs-text-3xs); letter-spacing: .08em; font-style: normal; margin-bottom: var(--cs-space-1); }
.isl__boundary i { color: var(--cs-color-info); }
.isl__breaks i { color: var(--cs-color-warning); }
.isl__fits i { color: var(--cs-color-success); }
.isl__platforms { margin-top: var(--cs-space-5); display: grid; gap: var(--cs-space-2); }
.isl__ptitle { margin: 0 0 var(--cs-space-1); font-family: var(--cs-font-mono); font-size: var(--cs-text-3xs); letter-spacing: .08em; color: var(--cs-color-text-subtle); }
.isl__platform { display: grid; gap: var(--cs-space-1); padding: var(--cs-space-3) var(--cs-space-4); border: 1px solid var(--cs-color-border); border-radius: var(--cs-radius-sm); background: var(--cs-color-bg-soft); }
.isl__platform b { font-size: var(--cs-text-sm); color: var(--cs-color-brand); }
.isl__platform span { font-size: var(--cs-text-xs); line-height: var(--cs-leading-normal); color: var(--cs-color-text-muted); }
@media (min-width: 861px) {
	.isl__detail { grid-template-columns: repeat(3, 1fr); }
	.isl__platform { grid-template-columns: 7rem minmax(0, 1fr); align-items: baseline; gap: var(--cs-space-4); }
}
</style>
