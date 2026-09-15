<script setup lang="ts">
import { computed, ref } from "vue"

/**
 * 四个实现在七个工程维度上的取向对照。
 *
 * 这张矩阵的用法不是「挑出最好的那个」，而是读出**分歧点**：
 * 凡是四家取法一致的维度，说明该结论已经稳定，可以当作通则；
 * 凡是取法相左的维度，说明它是一个还没有标准答案的设计选择，
 * 迁移时必须逐个核对，不能凭在某一家上的经验直接推断。
 *
 * 取向描述取自各实现的官方文档与官方仓库源码（核验于 2026-09-14），
 * 并刻意写成「这个选择说明了什么」而不是逐字复述配置项。
 * 四个实现迭代都很快，具体默认值应以所用版本的官方文档为准。
 */

type Dim = {
	key: string
	name: string
	question: string
	cells: Record<string, string>
	consensus: string
}

const implNames = [
	{ key: "a", label: "A · 系统级 CLI" },
	{ key: "b", label: "B · 应用级 CLI" },
	{ key: "c", label: "C · 极简 harness" },
	{ key: "d", label: "D · 编排框架" }
]

const dims: Dim[] = [
	{
		key: "gateway",
		name: "模型网关",
		question: "换一个模型供应方需要改什么？",
		cells: {
			a: "provider 是一等配置对象：地址、鉴权、线路协议、重试与超时都挂在 provider 上，切换即换一整套参数",
			b: "以环境变量为主：地址、鉴权与几个超时各自独立，粒度更粗但改起来快",
			c: "一份模型清单文件，支持把密钥写成「执行某条命令取回」；文档明确说明不替你做缓存与失败重用，因为不同命令需要不同策略",
			d: "框架不持有连接，换供应方等于换模型客户端"
		},
		consensus: "四家都把「地址、鉴权、模型标识」做成了可替换的配置，因此「把网关指向自建代理是被普遍支持的用法」；分歧在于配置的粒度落在 provider、进程还是调用点。"
	},
	{
		key: "retry",
		name: "重试与超时",
		question: "重试是一层还是两层？谁来决定放弃？",
		cells: {
			a: "传输层与流式层两个独立计数，都带百位数的硬上限；连接失败另有一条从秒级起步、封顶一分钟的重连路径",
			b: "一个总的重试次数，加一个空闲看门狗——后者的默认值同时是下限，配低了会被静默抬回",
			c: "刻意把 SDK 层重试关掉，只留 Agent 层三次指数退避；理由是不让 SDK 先吞掉配额类错误",
			d: "交给模型客户端与节点级策略；框架本身管的是失败之后从哪继续"
		},
		consensus: "共识是「瞬时故障要退避重试」，分歧是「在哪一层重试」。C 的取向最值得注意：它主张让错误浮到能正确分类的那一层，而不是在最底层就被吞掉——因为底层分不清限流与配额耗尽。"
	},
	{
		key: "sandbox",
		name: "沙箱与隔离",
		question: "边界由内核施加还是由规则匹配？",
		cells: {
			a: "内核级，且按平台分道：一类平台用系统自带的策略机制，另一类用内核访问控制加系统调用过滤。另提供分级的写入范围与受保护路径",
			b: "内核级，同样分平台；文件与网络是两个可独立开关的层。默认在依赖缺失时降级运行并告警，可配置为硬失败",
			c: "核心不内置沙箱——默认以启动它的用户权限运行，隔离交给扩展或外层环境。它提供的是项目信任机制，那管的是「加载谁的配置」，不是「能碰什么文件」",
			d: "不涉及——框架不规定工具怎么执行，隔离由工具实现方自负"
		},
		consensus: "这是分歧最大的一维，也是最不能凭经验迁移的一维。两家把内核级隔离做进核心，一家明确不做、留给扩展，一家不在讨论范围内。「这个工具有沙箱」这句话必须落到具体产品、具体平台、具体版本才有意义。"
	},
	{
		key: "approval",
		name: "审批与人机交互",
		question: "人被接回来的时机由什么决定？",
		cells: {
			a: "沙箱与审批是两根正交的杠杆：前者定能力边界，后者定越界怎么办。审批还可以按类别分别放行或自动拒绝",
			b: "权限规则与沙箱互补：规则在执行前按命令判定、覆盖所有工具，沙箱在执行中由内核拦截、只覆盖命令类工具。能被沙箱容纳的命令可自动放行，容不下的退回常规审批流",
			c: "核心保持最小，审批能力由扩展提供；社区扩展里既有规则匹配式的，也有包装操作系统沙箱的",
			d: "把「暂停等人」做成一等的执行原语：运行挂起，外部给出答复后继续"
		},
		consensus: "共识是「有副作用的动作时机应当由人决定」。分歧在实现位置：三家把它做成运行时的拦截点，一家把它做成执行图里的一个可持久化的暂停状态——后者的好处是「等人」这段时间可以跨进程存活。"
	},
	{
		key: "persist",
		name: "持久化与恢复",
		question: "崩溃之后能回到哪一点？",
		cells: {
			a: "追加式记录，逐条落盘；写者锁裁决同一会话的归属，防止两个进程交错写入",
			b: "会话记录加文件快照，按用户提示分段；快照有数量上限与保留期，且只覆盖文件编辑工具的改动",
			c: "树形会话：每条记录带父指针，可跳回任一历史位置继续，也可把放弃的分支先总结成一条摘要",
			d: "落盘粒度可配，从每步同步落盘到只在结束时落盘；恢复时所在的节点从头重跑"
		},
		consensus: "共识是「会话状态必须落在进程之外」。真正的分歧在恢复语义：三家的恢复是「重建视图后接着走」，一家是「把那个执行单元再跑一遍」——后者意味着中断前的副作用会重复发生，这必须在写代码时就纳入设计。"
	},
	{
		key: "context",
		name: "上下文治理",
		question: "窗口不够时，先丢什么？",
		cells: {
			a: "多种压缩路径并存，另有独立的令牌预算机制；压缩被建模成一次有生命周期的操作，前后都可挂钩子",
			b: "自动压缩之外，还可以对指定区段做定向摘要；摘要不改磁盘，原始消息仍留在记录里",
			c: "阈值是「窗口减去预留」，触发后从最新往回数够一定量的近期消息保留，其余摘要；摘要前把工具结果按字符数截断，因为它们通常是最大的贡献者",
			d: "不内置，由使用者在状态设计里自行决定"
		},
		consensus: "共识有两条：「压缩的切点不能落在工具结果与其调用之间」，以及「工具返回值是上下文膨胀的主要来源」。分歧在触发时机与是否可被外部接管。"
	},
	{
		key: "observe",
		name: "可观测性",
		question: "出了问题，能不能复现当时的判断依据？",
		cells: {
			a: "重试、压缩等关键动作有结构化事件；记录文件本身就是一条可回放的完整轨迹",
			b: "支持标准的遥测导出；调试日志可按会话开启，用于定位沙箱跳过了哪些路径这类具体问题",
			c: "压缩的开始与失败都有事件，扩展可以订阅并配对成完整的遥测记录",
			d: "与外部追踪平台集成较深，图的每一步执行天然构成一段可检视的轨迹"
		},
		consensus: "共识是「先有轨迹再谈优化」：四家都把「每一步的决策、动作、观察」持久化成可回放的东西。这一维的分歧最小，也说明它已经是稳定结论。"
	}
]

const active = ref("sandbox")
const cur = computed(() => dims.find((d) => d.key === active.value)!)
</script>

<template>
	<div class="icm">
		<div class="icm__tabs">
			<button
				v-for="d in dims"
				:key="d.key"
				type="button"
				class="icm__tab"
				:class="{ 'is-on': active === d.key }"
				:aria-pressed="active === d.key"
				@click="active = d.key"
			>{{ d.name }}</button>
		</div>

		<p class="icm__question">{{ cur.question }}</p>

		<div class="icm__grid">
			<div v-for="im in implNames" :key="im.key" class="icm__cell">
				<b>{{ im.label }}</b>
				<span>{{ cur.cells[im.key] }}</span>
			</div>
		</div>

		<p class="icm__consensus"><i>共识与分歧</i>{{ cur.consensus }}</p>

		<p class="pc-note">
			读这张矩阵的方法是找<b>分歧</b>而不是找赢家。四家取法一致的地方——
			会话状态要落到进程之外、有副作用的动作时机归人、先有轨迹再谈优化——
			可以当作已经稳定的结论。取法相左的地方，尤其是沙箱与恢复语义这两维，
			说明它们仍是开放的设计选择：<b>凭在某一家上的经验去推断另一家，会得到相反的结论。</b>
			这也解释了为什么跨实现迁移时，真正花时间的往往不是接口适配，而是重新核对这些行为契约。
		</p>
	</div>
</template>

<style scoped>
.icm { margin: 1rem 0; }
.icm__tabs { display: flex; flex-wrap: wrap; gap: var(--cs-space-1); }
.icm__tab { min-height: var(--cs-tap-target); padding: var(--cs-space-1) var(--cs-space-4); cursor: pointer; font-size: var(--cs-text-xs); color: var(--cs-color-text-muted); background: var(--cs-color-bg); border: 1px solid var(--cs-color-border); border-radius: var(--cs-radius-pill); transition: var(--cs-transition-colors); }
.icm__tab:hover { background: var(--cs-color-bg-soft); }
.icm__tab.is-on { background: var(--cs-color-brand-soft); border-color: var(--cs-color-brand); color: var(--cs-color-brand); font-weight: 600; }
.icm__question { margin: var(--cs-space-5) 0 var(--cs-space-4); padding: var(--cs-space-3) var(--cs-space-4); border-left: 3px solid var(--cs-color-brand); border-radius: var(--cs-radius-xs); background: var(--cs-color-bg-soft); font-size: var(--cs-text-md); }
.icm__grid { display: grid; gap: var(--cs-space-2); }
.icm__cell { display: grid; gap: var(--cs-space-1); padding: var(--cs-space-4); border: 1px solid var(--cs-color-border); border-radius: var(--cs-radius-md); background: var(--cs-color-bg); }
.icm__cell b { font-size: var(--cs-text-sm); color: var(--cs-color-brand); }
.icm__cell span { font-size: var(--cs-text-xs); line-height: var(--cs-leading-relaxed); color: var(--cs-color-text-muted); }
.icm__consensus { margin: var(--cs-space-4) 0 0; padding: var(--cs-space-4); border-radius: var(--cs-radius-sm); background: var(--cs-color-info-soft); font-size: var(--cs-text-base); line-height: var(--cs-leading-relaxed); }
.icm__consensus i { display: block; font-family: var(--cs-font-mono); font-size: var(--cs-text-3xs); letter-spacing: .08em; font-style: normal; margin-bottom: var(--cs-space-1); color: var(--cs-color-info); }
@media (min-width: 861px) {
	.icm__grid { grid-template-columns: repeat(2, 1fr); }
}
</style>
