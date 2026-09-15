<script setup lang="ts">
import { computed, ref } from "vue"

/**
 * 发现路径与优先级链：同名技能在多个作用域同时存在时，最终生效的是哪一个。
 *
 * 组件承载两件在文字里很难说清的事：
 *   1. 作用域是一条**有序链**，不是一组平级目录；
 *   2. 冲突消解规则在不同宿主之间**不一致**——有的按「越靠近机器/组织越优先」，
 *      有的按「越靠近项目越优先」，还有的干脆不合并、两条都保留。
 * 所以「同名技能谁生效」不是一个可以凭直觉迁移的结论，必须按宿主核对。
 *
 * 数值与层级取自各宿主公开文档（核验于 2026-09-14），仅呈现结构关系。
 */

type Host = {
	key: string
	name: string
	/** 从低优先级到高优先级排列 */
	chain: { scope: string; path: string; note: string }[]
	rule: string
	collide: string
	/** 冲突消解的方向：inward = 越靠近项目越优先，outward = 越靠近组织越优先，keep = 不合并 */
	direction: "inward" | "outward" | "keep"
}

const hosts: Host[] = [
	{
		key: "a",
		name: "宿主 A（组织优先型）",
		chain: [
			{ scope: "内置", path: "随宿主分发", note: "所有用户都有；可被同名条目覆盖，但其别名不受影响" },
			{ scope: "项目", path: "<项目>/.claude/skills/<名>/SKILL.md", note: "仅本项目；工作目录下的嵌套目录也会被发现" },
			{ scope: "个人", path: "~/.claude/skills/<名>/SKILL.md", note: "跨全部项目" },
			{ scope: "企业", path: "由受管设置指定", note: "组织内全部用户" }
		],
		rule: "企业覆盖个人，个人覆盖项目——方向朝外：越靠近组织的越优先。",
		collide: "插件带来的技能走 `插件名:技能名` 命名空间，因此不与上述任何层级冲突，可以并存。",
		direction: "outward"
	},
	{
		key: "b",
		name: "宿主 B（就近优先型）",
		chain: [
			{ scope: "系统", path: "随宿主分发的内置技能", note: "优先级最低" },
			{ scope: "管理员", path: "/etc/codex/skills", note: "机器或容器级的共享默认" },
			{ scope: "用户", path: "$HOME/.agents/skills", note: "该用户在任何仓库中都可用" },
			{ scope: "仓库根", path: "$REPO_ROOT/.agents/skills", note: "全仓库共享，子目录均可见" },
			{ scope: "上层目录", path: "$CWD/../.agents/skills", note: "嵌套结构里的共享区域" },
			{ scope: "当前目录", path: "$CWD/.agents/skills", note: "只与某个模块或微服务相关" }
		],
		rule: "从当前工作目录一路向上扫到仓库根，再叠加用户、管理员与系统层——方向朝内：越靠近工作目录的越具体。",
		collide: "同名技能不做合并，两者都可能出现在选择器里——这与「后者覆盖前者」是完全不同的语义。",
		direction: "keep"
	},
	{
		key: "c",
		name: "宿主 C（别名并存型）",
		chain: [
			{ scope: "内置", path: "随 CLI 分发", note: "优先级最低" },
			{ scope: "扩展", path: "已安装扩展内附带", note: "随扩展启停" },
			{ scope: "用户", path: "~/.gemini/skills/ 或 ~/.agents/skills/", note: "同层内通用别名目录优先" },
			{ scope: "工作区", path: ".gemini/skills/ 或 .agents/skills/", note: "随版本控制分享给团队" }
		],
		rule: "同名时取优先级更高的位置；同一层级内，跨工具通用的别名目录优先于产品专属目录。",
		collide: "激活需要用户确认：确认后正文与目录结构进入历史，同时该目录被加入可读路径白名单。",
		direction: "inward"
	}
]

const active = ref("b")
const cur = computed(() => hosts.find((h) => h.key === active.value)!)
/** 展示时统一按「优先级由低到高」自下而上排列，突出这是一条链 */
const rows = computed(() => [...cur.value.chain].reverse())

const dirLabel: Record<Host["direction"], string> = {
	inward: "就近者优先",
	outward: "组织者优先",
	keep: "不合并，并存"
}
</script>

<template>
	<div class="sdc">
		<div class="sdc__tabs">
			<button
				v-for="h in hosts"
				:key="h.key"
				type="button"
				class="sdc__tab"
				:class="{ 'is-on': active === h.key }"
				:aria-pressed="active === h.key"
				@click="active = h.key"
			>{{ h.name }}</button>
		</div>

		<div class="sdc__chain">
			<div
				v-for="(r, i) in rows"
				:key="r.scope"
				class="sdc__link"
				:class="{ 'is-top': i === 0 }"
			>
				<span class="sdc__rank cs-mono">{{ rows.length - i }}</span>
				<div class="sdc__body">
					<b>{{ r.scope }}</b>
					<code>{{ r.path }}</code>
					<span class="sdc__note">{{ r.note }}</span>
				</div>
			</div>
		</div>

		<p class="sdc__axis">
			<span>↑ 序号越大优先级越高</span>
			<span class="sdc__dir">冲突消解：<b>{{ dirLabel[cur.direction] }}</b></span>
		</p>

		<div class="sdc__rules">
			<p class="sdc__rule"><i>优先级规则</i>{{ cur.rule }}</p>
			<p class="sdc__collide"><i>同名冲突</i>{{ cur.collide }}</p>
		</div>

		<p class="pc-note">
			三种宿主的层级数量、目录名与消解方向都不同：A 是<b>组织者优先</b>，B <b>不合并同名项</b>，
			C 在同层内让<b>跨工具通用目录</b>胜出。这意味着「同名技能谁生效」是一条
			<b>必须逐宿主核对的产品行为，而不是可以凭直觉迁移的通则</b>。
			三者的共同点只有两条：作用域是一条有序链而非一组平级目录；越靠近具体工作范围的层级，越应当只放该范围专属的内容。
		</p>
	</div>
</template>

<style scoped>
.sdc { margin: 1rem 0; }
.sdc__tabs { display: flex; flex-wrap: wrap; gap: var(--cs-space-1); }
.sdc__tab { min-height: var(--cs-tap-target); padding: var(--cs-space-1) var(--cs-space-4); cursor: pointer; font-size: var(--cs-text-xs); color: var(--cs-color-text-muted); background: var(--cs-color-bg); border: 1px solid var(--cs-color-border); border-radius: var(--cs-radius-pill); transition: var(--cs-transition-colors); }
.sdc__tab:hover { background: var(--cs-color-bg-soft); }
.sdc__tab.is-on { background: var(--cs-color-brand-soft); border-color: var(--cs-color-brand); color: var(--cs-color-brand); font-weight: 600; }
.sdc__chain { margin-top: var(--cs-space-5); display: grid; gap: var(--cs-space-2); }
.sdc__link { display: grid; grid-template-columns: 1.8rem minmax(0, 1fr); gap: var(--cs-space-3); align-items: start; padding: var(--cs-space-3) var(--cs-space-4); background: var(--cs-color-bg); border: 1px solid var(--cs-color-border); border-left: 3px solid var(--cs-color-border-strong); border-radius: var(--cs-radius-md); }
.sdc__link.is-top { border-left-color: var(--cs-color-brand); background: var(--cs-color-brand-soft); }
.sdc__rank { font-size: var(--cs-text-sm); color: var(--cs-color-text-subtle); padding-top: 1px; }
.sdc__link.is-top .sdc__rank { color: var(--cs-color-brand); font-weight: 700; }
.sdc__body { display: grid; gap: var(--cs-space-1); }
.sdc__body b { font-size: var(--cs-text-md); }
.sdc__body code { font-family: var(--cs-font-mono); font-size: var(--cs-text-2xs); color: var(--cs-color-text-muted); background: none; padding: 0; }
.sdc__note { font-size: var(--cs-text-xs); line-height: var(--cs-leading-normal); color: var(--cs-color-text-subtle); }
.sdc__axis { display: flex; flex-wrap: wrap; justify-content: space-between; gap: var(--cs-space-3); margin: var(--cs-space-3) 0 var(--cs-space-5); font-family: var(--cs-font-mono); font-size: var(--cs-text-3xs); color: var(--cs-color-text-subtle); }
.sdc__dir b { color: var(--cs-color-brand); }
.sdc__rules { display: grid; gap: var(--cs-space-3); }
.sdc__rule, .sdc__collide { margin: 0; padding: var(--cs-space-3) var(--cs-space-4); border-radius: var(--cs-radius-sm); font-size: var(--cs-text-base); line-height: var(--cs-leading-relaxed); }
.sdc__rule { background: var(--cs-color-info-soft); }
.sdc__collide { background: var(--cs-color-warning-soft); }
.sdc__rule i, .sdc__collide i { display: block; font-family: var(--cs-font-mono); font-size: var(--cs-text-3xs); letter-spacing: .08em; font-style: normal; margin-bottom: var(--cs-space-1); }
.sdc__rule i { color: var(--cs-color-info); }
.sdc__collide i { color: var(--cs-color-warning); }
@media (min-width: 861px) {
	.sdc__rules { grid-template-columns: 1fr 1fr; }
}
</style>
