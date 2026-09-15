<script setup lang="ts">
import { computed, ref } from "vue"

/**
 * 几种实现的横向对照：存储形态、可审计性、检索能力、谁来维护。
 *
 * 这张图想说明的是一条容易被忽略的分野：编码类 Agent 的长期记忆大量收敛到
 * 「Markdown 文件 + 版本控制」，而通用助手与陪伴类走向抽取式的向量 / 图存储。
 * 分野的原因不是技术成熟度，而是**错误代价的形态不同**——切换下面的视角可以看到这一点。
 */

type Impl = {
	key: string
	name: string
	family: "coding" | "assistant" | "middleware"
	store: string
	form: string
	retrieval: string
	who: string
	audit: number
	semantic: number
	autonomy: number
	note: string
}

const familyMeta = {
	coding: { label: "编码类", color: "var(--cs-color-success)" },
	assistant: { label: "通用助手", color: "var(--cs-color-info)" },
	middleware: { label: "记忆中间件", color: "var(--cs-color-warning)" }
} as const

const impls: Impl[] = [
	{
		key: "claude-md", name: "分层指令文件", family: "coding",
		store: "仓库与用户目录下的 Markdown 文件", form: "程序性规程为主",
		retrieval: "不检索：按目录层级在会话启动时全量注入，子目录文件按需加载",
		who: "人为主，模型可提议",
		audit: 3, semantic: 0, autonomy: 1,
		note: "可直接 diff、可 review、可随代码一起演进；代价是没有语义检索，且体积会挤占每一次调用的固定预算。"
	},
	{
		key: "agents-md", name: "跨工具约定文件", family: "coding",
		store: "仓库根与子目录的 AGENTS.md", form: "程序性规程",
		retrieval: "不检索：最近的那份文件优先生效",
		who: "人工维护",
		audit: 3, semantic: 0, autonomy: 0,
		note: "它的价值主要在互操作性：同一份文件被多家工具读取，避免每换一个工具就重写一遍项目约定。"
	},
	{
		key: "auto-memory", name: "自动记忆目录", family: "coding",
		store: "本机的 Markdown 目录，一份索引加若干主题文件", form: "语义事实 + 情景 + 偏好",
		retrieval: "索引常驻注入，主题文件由模型按需读取",
		who: "模型自主写，人可编辑",
		audit: 3, semantic: 1, autonomy: 2,
		note: "介于纯人工文件与自动抽取之间：模型自己决定记什么，但落地形态仍是可读可改的纯文本。"
	},
	{
		key: "memory-tool", name: "文件式记忆工具", family: "middleware",
		store: "由应用方决定：磁盘、数据库、对象存储", form: "任意，取决于用法",
		retrieval: "模型自己发起读写，属于按需取用",
		who: "模型发起，应用执行",
		audit: 2, semantic: 1, autonomy: 3,
		note: "接口只规定操作语义，存储与校验全在应用侧。灵活，但路径校验、体积上限、过期清理都要自己实现。"
	},
	{
		key: "kv-store", name: "命名空间键值存储", family: "middleware",
		store: "按命名空间组织的 JSON 文档，可选向量索引", form: "语义事实、偏好",
		retrieval: "命名空间前缀枚举 + 可选语义检索",
		who: "应用代码显式读写",
		audit: 2, semantic: 2, autonomy: 1,
		note: "把「会话状态持久化」与「跨会话记忆存储」明确拆成两套设施，是这一路线最重要的设计贡献。"
	},
	{
		key: "extract-vector", name: "抽取式向量记忆", family: "middleware",
		store: "向量库，条目是模型抽出的短句事实", form: "语义事实为主",
		retrieval: "向量相似度 + 实体共现加权",
		who: "模型自动抽取与增删改",
		audit: 1, semantic: 3, autonomy: 3,
		note: "写入路径上就有一次模型调用来判断 ADD / UPDATE / DELETE / NOOP，所以它不只是存储，还是一套决策逻辑。"
	},
	{
		key: "temporal-graph", name: "时序知识图", family: "middleware",
		store: "图：实体为节点，带双时间轴的关系为边", form: "实体关系 + 情景",
		retrieval: "向量 + 全文 + 图遍历三路合一",
		who: "自动抽取，矛盾时标注失效而非删除",
		audit: 3, semantic: 3, autonomy: 3,
		note: "双时间轴（事实何时为真、系统何时知道）让「当时以为」和「现在成立」可以分开回答，代价是抽取与对齐的复杂度。"
	},
	{
		key: "memory-blocks", name: "常驻记忆块", family: "middleware",
		store: "带标签、描述与字符上限的结构化块，直接拼进提示", form: "人设、偏好、任务状态",
		retrieval: "不检索：始终可见",
		who: "模型用工具自行改写，可设只读",
		audit: 2, semantic: 1, autonomy: 3,
		note: "「始终可见」是它与检索式记忆的根本差别：不会漏召，但每一块都长期占用窗口，所以必须有字符上限。"
	},
	{
		key: "dual-channel", name: "双通道产品记忆", family: "assistant",
		store: "显式保存的条目 + 对历史对话的引用两套并存", form: "偏好、语义事实",
		retrieval: "两个通道分别可开关，历史引用侧按相关性动态取用",
		who: "用户显式要求，或系统自动判断",
		audit: 2, semantic: 2, autonomy: 2,
		note: "把「用户明确要求记住的」与「系统自己觉得有用的」分成两个通道，是个值得借鉴的产品判断——前者应当稳定且可管理，后者可以浮动。"
	},
	{
		key: "rules-modes", name: "多模式规则", family: "coding",
		store: "带 frontmatter 的规则文件", form: "程序性规程",
		retrieval: "四种加载模式：常驻、按文件匹配、由模型按描述决定、手动引用",
		who: "人工编写，可由模型生成",
		audit: 3, semantic: 1, autonomy: 1,
		note: "「按描述让模型自己决定要不要加载」这一模式是程序性记忆的一种检索：把规则的元数据当索引，正文按需展开。"
	}
]

type Axis = "audit" | "semantic" | "autonomy"
const axes: { key: Axis; label: string; hint: string }[] = [
	{ key: "audit", label: "可审计性", hint: "人能否直接看懂、直接改、直接 diff" },
	{ key: "semantic", label: "语义检索能力", hint: "能否按意思而非按位置找到内容" },
	{ key: "autonomy", label: "模型自主程度", hint: "写什么、何时写由模型还是由人决定" }
]

const axis = ref<Axis>("audit")
const familyFilter = ref<string | null>(null)

const sorted = computed(() => {
	const list = familyFilter.value ? impls.filter((i) => i.family === familyFilter.value) : [...impls]
	return list.sort((a, b) => b[axis.value] - a[axis.value])
})

const detail = ref("claude-md")
const cur = computed(() => impls.find((i) => i.key === detail.value)!)
const curAxis = computed(() => axes.find((a) => a.key === axis.value)!)
const levelLabel = (v: number) => ["无", "弱", "中", "强"][v]
</script>

<template>
	<div class="mic">
		<div class="mic__head">
			<div class="mic__seg" role="tablist" aria-label="选择排序维度">
				<button
					v-for="a in axes"
					:key="a.key"
					type="button"
					role="tab"
					:aria-selected="axis === a.key"
					:class="{ 'is-on': axis === a.key }"
					@click="axis = a.key"
				>{{ a.label }}</button>
			</div>
			<div class="mic__filters">
				<button
					type="button"
					class="mic__filter"
					:class="{ 'is-on': familyFilter === null }"
					@click="familyFilter = null"
				>全部</button>
				<button
					v-for="(m, k) in familyMeta"
					:key="k"
					type="button"
					class="mic__filter"
					:class="{ 'is-on': familyFilter === k }"
					:style="{ '--c': m.color }"
					@click="familyFilter = k"
				>{{ m.label }}</button>
			</div>
		</div>

		<p class="mic__axis-hint">按<b>{{ curAxis.label }}</b>排序 —— {{ curAxis.hint }}</p>

		<ul class="mic__rows">
			<li
				v-for="im in sorted"
				:key="im.key"
				class="cs-legend-row mic__row"
				:class="{ 'is-open': detail === im.key }"
				:style="{ '--c': familyMeta[im.family].color }"
			>
				<button type="button" class="mic__row-btn" :aria-pressed="detail === im.key" @click="detail = im.key">
					<span class="mic__dot" aria-hidden="true"></span>
					<span class="mic__row-name">{{ im.name }}</span>
					<span class="mic__row-fam">{{ familyMeta[im.family].label }}</span>
					<span class="infra-bar mic__row-bar">
						<i :style="{ width: `${(im[axis] / 3) * 100}%` }"></i>
					</span>
					<span class="mic__row-val cs-mono">{{ levelLabel(im[axis]) }}</span>
				</button>
			</li>
		</ul>

		<article class="mic__detail" :style="{ '--c': familyMeta[cur.family].color }">
			<header>
				<b class="pc-serif">{{ cur.name }}</b>
				<span class="mic__badge">{{ familyMeta[cur.family].label }}</span>
			</header>
			<dl>
				<dt>存储在哪</dt><dd>{{ cur.store }}</dd>
				<dt>承载哪类记忆</dt><dd>{{ cur.form }}</dd>
				<dt>怎么进入上下文</dt><dd>{{ cur.retrieval }}</dd>
				<dt>谁来维护</dt><dd>{{ cur.who }}</dd>
			</dl>
			<p class="mic__note">{{ cur.note }}</p>
		</article>

		<p class="pc-note">
			三档强弱是按公开文档所描述的机制作的相对定位，不是评分，也不代表优劣——它们服务的错误代价形态不同。
			把过滤器切到「编码类」再看可审计性一栏，分野就很清楚：<b>这一族几乎全部落在纯文本文件上</b>。
			原因不难理解：代码类任务里一条错误的长期规程会被反复执行到生产环境，所以「人能直接读到、直接改掉、在 code review 里被看见」
			比「能按语义模糊召回」重要得多。反过来，陪伴与通用助手类的记忆错误代价是体验层面的，可以接受抽取带来的不确定性，
			换取覆盖面与召回的自然度。<b>技术路线的分歧，来自错误代价的分歧。</b>
		</p>
	</div>
</template>

<style scoped>
.mic { margin: 1rem 0; }
.mic__head { display: flex; flex-wrap: wrap; align-items: center; justify-content: space-between; gap: var(--cs-space-3); }
.mic__seg { display: inline-flex; border: 1px solid var(--cs-color-border); border-radius: var(--cs-radius-pill); overflow: hidden; }
.mic__seg button { min-height: var(--cs-tap-target); padding: 0 var(--cs-space-4); font-size: var(--cs-text-sm); background: var(--cs-color-bg); color: var(--cs-color-text-muted); border: 0; cursor: pointer; transition: var(--cs-transition-colors); }
.mic__seg button.is-on { background: var(--cs-color-brand-soft); color: var(--cs-color-brand); font-weight: 600; }
.mic__filters { display: inline-flex; flex-wrap: wrap; gap: var(--cs-space-1); }
.mic__filter { --c: var(--cs-color-border-strong); min-height: var(--cs-tap-target); padding: 0 var(--cs-space-3); cursor: pointer; font-size: var(--cs-text-3xs); color: var(--cs-color-text-muted); background: var(--cs-color-bg); border: 1px solid var(--cs-color-border); border-radius: var(--cs-radius-pill); transition: var(--cs-transition-colors); }
.mic__filter.is-on { border-color: var(--c); background: color-mix(in srgb, var(--c) 14%, transparent); color: var(--cs-color-text); }
.mic__axis-hint { margin: var(--cs-space-3) 0; font-size: var(--cs-text-sm); color: var(--cs-color-text-muted); }
.mic__rows { list-style: none; margin: 0; padding: 0; }
.mic__row { padding: 0; border-bottom: 1px dashed var(--cs-color-border); }
.mic__row:last-child { border-bottom: 0; }
.mic__row-btn { display: grid; grid-template-columns: auto minmax(6rem, 1fr) auto minmax(4rem, 1.6fr) 2.2rem; align-items: center; gap: var(--cs-space-3); width: 100%; min-height: var(--cs-tap-target); padding: var(--cs-space-2) var(--cs-space-1); text-align: left; cursor: pointer; background: none; border: 0; transition: var(--cs-transition-colors); }
.mic__row-btn:hover { background: var(--cs-color-bg-soft); }
.mic__row.is-open .mic__row-btn { background: color-mix(in srgb, var(--c) 10%, transparent); }
.mic__dot { width: 9px; height: 9px; border-radius: var(--cs-radius-circle); background: var(--c); flex: none; }
.mic__row-name { font-size: var(--cs-text-sm); color: var(--cs-color-text); }
.mic__row-fam { font-size: var(--cs-text-3xs); color: var(--cs-color-text-subtle); white-space: nowrap; }
.mic__row-bar { min-width: 3rem; }
.mic__row-val { font-size: var(--cs-text-3xs); color: var(--cs-color-text-muted); text-align: right; }
.mic__detail { margin-top: var(--cs-space-4); background: var(--cs-color-bg); border: 1px solid var(--cs-color-border); border-left: 3px solid var(--c); border-radius: var(--cs-radius-lg); padding: var(--cs-space-5) var(--cs-space-6); }
.mic__detail header { display: flex; flex-wrap: wrap; align-items: baseline; justify-content: space-between; gap: var(--cs-space-3); margin-bottom: var(--cs-space-3); }
.mic__detail header b { font-size: var(--cs-text-lg); }
.mic__badge { font-family: var(--cs-font-mono); font-size: var(--cs-text-3xs); color: var(--c); border: 1px solid var(--c); border-radius: var(--cs-radius-pill); padding: var(--cs-space-1) var(--cs-space-3); }
.mic__detail dl { display: grid; grid-template-columns: 7.5rem minmax(0, 1fr); gap: var(--cs-space-2) var(--cs-space-4); margin: 0; }
.mic__detail dt { font-family: var(--cs-font-mono); font-size: var(--cs-text-3xs); color: var(--cs-color-text-subtle); }
.mic__detail dd { margin: 0; font-size: var(--cs-text-sm); line-height: var(--cs-leading-normal); color: var(--cs-color-text); }
.mic__note { margin: var(--cs-space-3) 0 0; padding-top: var(--cs-space-3); border-top: 1px dashed var(--cs-color-border); font-size: var(--cs-text-sm); line-height: var(--cs-leading-relaxed); color: var(--cs-color-text-muted); }
@media (max-width: 720px) {
	.mic__row-btn { grid-template-columns: auto minmax(0, 1fr) 2.2rem; }
	.mic__row-fam, .mic__row-bar { display: none; }
	.mic__detail dl { grid-template-columns: 1fr; gap: 0; }
	.mic__detail dt { margin-top: var(--cs-space-3); }
}
</style>
