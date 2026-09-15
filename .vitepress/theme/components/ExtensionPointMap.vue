<script setup lang="ts">
import { computed, ref } from "vue"

/**
 * 扩展点地图：按「在 Agent 循环的哪个位置开洞」把扩展机制归位。
 *
 * 这张图要回答的不是「有哪些扩展机制」，而是「一个需求应该落在循环的哪一段」。
 * 因此横轴是循环本身的推进顺序（会话 → 轮 → 循环内每次工具调用 → 收尾），
 * 纵轴是四类开洞方式；每个格子给出该位置真实存在的扩展点及其能否阻断。
 */

type Power = "observe" | "mutate" | "block"

type Point = {
	key: string
	/** 循环中的位置 */
	stage: string
	name: string
	power: Power
	what: string
	/** 该位置的典型误用 */
	trap: string
}

type Axis = {
	key: string
	label: string
	sub: string
	color: string
	/** 这一类扩展的共同契约 */
	contract: string
	points: Point[]
}

const powerMeta: Record<Power, { label: string; cls: string }> = {
	observe: { label: "只可观察", cls: "is-observe" },
	mutate: { label: "可改写", cls: "is-mutate" },
	block: { label: "可阻断", cls: "is-block" }
}

const axes: Axis[] = [
	{
		key: "time",
		label: "时间点开洞",
		sub: "hooks / 中间件 / 生命周期回调",
		color: "var(--cs-color-brand)",
		contract: "宿主在到达某个位置时把一份结构化上下文交给外部代码，再按它的返回值决定继续、改写还是中止。契约的核心不是「能收到通知」，而是「返回值被如何解释」。",
		points: [
			{
				key: "t1", stage: "会话开始", name: "SessionStart 一类",
				power: "mutate",
				what: "注入本次会话的背景：当前分支、未关issue、环境变量。这是少数几个标准输出会真正进入上下文的位置。",
				trap: "每次会话都跑，写成慢脚本会让每次启动都卡一下。"
			},
			{
				key: "t2", stage: "每轮开始", name: "UserPromptSubmit 一类",
				power: "block",
				what: "在模型看到用户输入之前检查它：拦截误粘贴的密钥，或按当前目录追加一段规则。",
				trap: "把它当成唯一的输入过滤层——同一轮里工具还会带回外部文本，那些不经过这里。"
			},
			{
				key: "t3", stage: "工具调用前", name: "PreToolUse 一类",
				power: "block",
				what: "循环内触发最频繁的位置。可以放行、询问用户、拒绝，通常还能改写入参后再执行。",
				trap: "当成强制边界。它覆盖的是宿主已接线的那几类工具，等价能力常有别的路径。"
			},
			{
				key: "t4", stage: "工具调用后", name: "PostToolUse 一类",
				power: "mutate",
				what: "结果已经产生，这里只能追加反馈或替换掉回给模型的那份结果。",
				trap: "以为这里的「拒绝」能撤销刚才的写入。副作用已经落地了。"
			},
			{
				key: "t5", stage: "本轮收尾", name: "Stop 一类",
				power: "block",
				what: "模型认为自己答完了。这里可以判定验收未过、把它顶回去继续干。",
				trap: "不看「是否已因此续跑过」的标志位，于是一轮接一轮永不终止。"
			}
		]
	},
	{
		key: "capability",
		label: "能力集开洞",
		sub: "tools / MCP / skills",
		color: "var(--cs-color-success)",
		contract: "改变的是「模型这一轮能看见哪些可调用项、以及它知道该怎么用」。能力集是每轮随载荷发出的，因此它同时是一笔常驻预算。",
		points: [
			{
				key: "c1", stage: "进程内", name: "本地工具注册",
				power: "mutate",
				what: "最直接的一层：把一个函数连同它的名称、描述与参数模式登记进注册表。",
				trap: "名称重叠、描述含糊，模型选错工具的概率随之上升。"
			},
			{
				key: "c2", stage: "跨进程", name: "MCP 服务端",
				power: "mutate",
				what: "把能力放到另一个进程甚至另一台机器上，用一套协议协商、发现、调用。解决的是连通性。",
				trap: "接一堆服务端，把几百个工具定义常驻进每一次请求。"
			},
			{
				key: "c3", stage: "按需加载", name: "规程与知识包",
				power: "observe",
				what: "补的不是通道而是判断：这件事在这里该按什么步骤做。只在相关时才载入正文。",
				trap: "用它承载必须每次生效的硬约束——带匹配环节的机制总有漏召的那一次。"
			},
			{
				key: "c4", stage: "运行时变更", name: "动态增删",
				power: "mutate",
				what: "按当前阶段、当前权限筛选这一轮暴露哪些工具，或响应服务端的能力变更通告。",
				trap: "工具集在一次会话中反复变化，会一并打破前缀缓存。"
			}
		]
	},
	{
		key: "package",
		label: "打包分发开洞",
		sub: "plugin / marketplace",
		color: "var(--cs-color-info)",
		contract: "不新增任何运行时能力，只解决「一组扩展如何被整体安装、版本化、在团队间复用」。它是分发原语，不是创作原语。",
		points: [
			{
				key: "p1", stage: "清单", name: "manifest",
				power: "observe",
				what: "声明标识、组件路径与安装期要向使用者索取的配置项。常见设计是清单可省略、按默认目录自动发现。",
				trap: "组件目录放进了清单所在的元数据目录里——多数实现要求它们在包根。"
			},
			{
				key: "p2", stage: "路径", name: "根与数据目录",
				power: "observe",
				what: "安装位置不可预测，所以要靠变量寻址。注意区分「随每次更新变化的安装目录」与「跨更新存活的数据目录」。",
				trap: "把依赖装进安装目录，插件一升级全没了。"
			},
			{
				key: "p3", stage: "作用域", name: "个人 / 项目 / 受管",
				power: "block",
				what: "同一个包可以按个人、项目、企业受管三种作用域安装，决定谁能用、能否随仓库共享、能否被本地关掉。",
				trap: "以为项目作用域等于团队强制——受管层才是不可被本地覆盖的那一层。"
			},
			{
				key: "p4", stage: "依赖", name: "跨包依赖与迁移",
				power: "observe",
				what: "包之间可以声明依赖并按语义化版本约束；改名时靠一张迁移映射维持既有安装不失效。",
				trap: "不固定版本，于是每次提交都被当成一个新版本。"
			}
		]
	},
	{
		key: "control",
		label: "管控面开洞",
		sub: "沙箱 / 审批 / 规则 / 信任",
		color: "var(--cs-color-warning)",
		contract: "与前三类正交：前三类决定「能做什么、什么时候做」，这一类决定「允不允许、由谁批准、出事后能追到哪」。它是唯一不依赖模型配合的一层。",
		points: [
			{
				key: "g1", stage: "执行边界", name: "沙箱",
				power: "block",
				what: "由操作系统机制划定技术边界：能写哪里、能不能联网。各平台落到不同的内核设施上。",
				trap: "把沙箱与审批混为一谈。前者是能力上限，后者是何时要问。"
			},
			{
				key: "g2", stage: "人机之间", name: "审批策略",
				power: "block",
				what: "规定哪些动作必须先问。可以按类别细分，也可以把一部分请求交给一个独立的审查者代为判断。",
				trap: "把高频低危动作也设成必问，使用者很快学会无脑点同意。"
			},
			{
				key: "g3", stage: "声明式", name: "规则",
				power: "block",
				what: "以模式匹配声明放行、询问用户、禁止。多条规则同时命中时，成熟实现普遍取最严的那条。",
				trap: "指望模式匹配穷举危险命令——同一个效果换种写法就绕过去了。"
			},
			{
				key: "g4", stage: "供应链", name: "信任模型",
				power: "block",
				what: "扩展是会执行的代码。一种做法是按内容哈希记录信任：未审阅过的跳过，内容一改就要重新确认。",
				trap: "把「安装过」当成「信任过」。两者是分开的决定。"
			}
		]
	}
]

const axisIdx = ref(0)
const pointIdx = ref(2)
const axis = computed(() => axes[axisIdx.value])
const point = computed(() => axis.value.points[pointIdx.value])

function pickAxis(i: number) {
	axisIdx.value = i
	pointIdx.value = 0
}
</script>

<template>
	<div class="epm">
		<div class="epm__axes">
			<button
				v-for="(a, i) in axes"
				:key="a.key"
				type="button"
				class="epm__axis"
				:class="{ 'is-on': axisIdx === i }"
				:style="{ '--c': a.color }"
				:aria-pressed="axisIdx === i"
				@click="pickAxis(i)"
			>
				<strong>{{ a.label }}</strong>
				<span>{{ a.sub }}</span>
			</button>
		</div>

		<p class="epm__contract"><i>这一类的共同契约</i>{{ axis.contract }}</p>

		<div class="epm__track">
			<button
				v-for="(p, i) in axis.points"
				:key="p.key"
				type="button"
				class="epm__point"
				:class="[{ 'is-on': pointIdx === i }, powerMeta[p.power].cls]"
				:style="{ '--c': axis.color }"
				:aria-pressed="pointIdx === i"
				@click="pointIdx = i"
			>
				<span class="epm__stage">{{ p.stage }}</span>
				<strong>{{ p.name }}</strong>
				<span class="epm__power">{{ powerMeta[p.power].label }}</span>
			</button>
		</div>

		<article class="epm__detail" :style="{ '--c': axis.color }">
			<header>
				<b class="pc-serif">{{ point.stage }} · {{ point.name }}</b>
				<span class="epm__badge" :class="powerMeta[point.power].cls">{{ powerMeta[point.power].label }}</span>
			</header>
			<p class="epm__what">{{ point.what }}</p>
			<p class="epm__trap"><i>典型误用</i>{{ point.trap }}</p>
		</article>

		<p class="pc-note">
			四类开洞方式不是并列的备选项，而是<b>四个不同的问题</b>：缺的是时机、是能力、是分发，还是授权。
			判断一个需求该落在哪一类，最快的办法是问它<b>失败时会怎样</b>——时间点上的扩展失败会漏掉一次检查，
			能力上的扩展失败会让模型根本够不着目标系统，分发上的扩展失败只影响安装，
			而管控面的扩展失败会让一次不该发生的写操作真的发生。风险量级不在一个数量级上，设计时的谨慎程度也不该相同。
		</p>
	</div>
</template>

<style scoped>
.epm { margin: 1rem 0; }
.epm__axes { display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: var(--cs-space-2); }
.epm__axis { display: grid; gap: var(--cs-space-1); text-align: left; cursor: pointer; min-height: var(--cs-tap-target); padding: var(--cs-space-3); background: var(--cs-color-bg); border: 1px solid var(--cs-color-border); border-top: 3px solid var(--c); border-radius: var(--cs-radius-lg); transition: var(--cs-transition-colors); }
.epm__axis:hover { background: var(--cs-color-bg-soft); }
.epm__axis.is-on { background: color-mix(in srgb, var(--c) 10%, transparent); border-color: var(--c); }
.epm__axis strong { font-size: var(--cs-text-md); line-height: var(--cs-leading-tight); }
.epm__axis span { font-size: var(--cs-text-3xs); color: var(--cs-color-text-muted); font-family: var(--cs-font-mono); }
.epm__contract { margin: var(--cs-space-4) 0; padding: var(--cs-space-3) var(--cs-space-4); background: var(--cs-color-bg-soft); border-radius: var(--cs-radius-lg); font-size: var(--cs-text-sm); line-height: var(--cs-leading-relaxed); color: var(--cs-color-text-muted); }
.epm__contract i { display: block; font-family: var(--cs-font-mono); font-size: var(--cs-text-3xs); letter-spacing: .08em; font-style: normal; color: var(--cs-color-text-subtle); margin-bottom: var(--cs-space-1); }
.epm__track { display: grid; grid-template-columns: repeat(auto-fit, minmax(0, 1fr)); gap: var(--cs-space-2); }
.epm__point { display: grid; gap: var(--cs-space-1); text-align: left; cursor: pointer; min-height: var(--cs-tap-target); padding: var(--cs-space-3); background: var(--cs-color-bg); border: 1px solid var(--cs-color-border); border-left: 3px solid var(--cs-color-border-strong); border-radius: var(--cs-radius-sm); transition: var(--cs-transition-colors); }
.epm__point:hover { background: var(--cs-color-bg-soft); }
.epm__point.is-on { border-color: var(--c); border-left-color: var(--c); background: color-mix(in srgb, var(--c) 8%, transparent); }
.epm__point.is-observe { border-left-color: var(--cs-color-info); }
.epm__point.is-mutate { border-left-color: var(--cs-color-warning); }
.epm__point.is-block { border-left-color: var(--cs-color-danger); }
.epm__stage { font-family: var(--cs-font-mono); font-size: var(--cs-text-3xs); color: var(--cs-color-text-subtle); }
.epm__point strong { font-size: var(--cs-text-sm); line-height: var(--cs-leading-tight); overflow-wrap: anywhere; }
.epm__power { font-size: var(--cs-text-3xs); color: var(--cs-color-text-muted); }
.epm__detail { margin-top: var(--cs-space-4); background: var(--cs-color-bg); border: 1px solid var(--cs-color-border); border-left: 3px solid var(--c); border-radius: var(--cs-radius-lg); padding: var(--cs-space-5) var(--cs-space-6); }
.epm__detail header { display: flex; flex-wrap: wrap; align-items: baseline; justify-content: space-between; gap: var(--cs-space-3); }
.epm__detail header b { font-size: var(--cs-text-lg); }
.epm__badge { font-family: var(--cs-font-mono); font-size: var(--cs-text-3xs); border-radius: var(--cs-radius-pill); padding: var(--cs-space-1) var(--cs-space-3); border: 1px solid currentColor; }
.epm__badge.is-observe { color: var(--cs-color-info); }
.epm__badge.is-mutate { color: var(--cs-color-warning); }
.epm__badge.is-block { color: var(--cs-color-danger); }
.epm__what { margin: var(--cs-space-3) 0; font-size: var(--cs-text-base); line-height: var(--cs-leading-relaxed); color: var(--cs-color-text); }
.epm__trap { margin: 0; padding-top: var(--cs-space-3); border-top: 1px dashed var(--cs-color-border); font-size: var(--cs-text-sm); line-height: var(--cs-leading-normal); color: var(--cs-color-text-muted); }
.epm__trap i { display: block; font-family: var(--cs-font-mono); font-size: var(--cs-text-3xs); letter-spacing: .08em; font-style: normal; color: var(--cs-color-text-subtle); margin-bottom: var(--cs-space-1); }
@media (max-width: 860px) {
	.epm__axes { grid-template-columns: repeat(2, minmax(0, 1fr)); }
	.epm__track { grid-template-columns: repeat(2, minmax(0, 1fr)); }
}
@media (max-width: 520px) {
	.epm__axes { grid-template-columns: 1fr; }
	.epm__track { grid-template-columns: 1fr; }
	.epm__detail { padding: var(--cs-space-4); }
}
</style>
