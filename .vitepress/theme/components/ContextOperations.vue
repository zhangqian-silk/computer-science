<script setup lang="ts">
import { ref } from "vue"

type Op = {
	key: string
	zh: string
	verb: string
	color: string
	loss: string
	def: string
	means: string[]
	treats: string
	cost: string
}

// 四大操作（LangChain 归纳）。推荐施加次序即数组顺序：无损优先，有损靠后。
const ops: Op[] = [
	{
		key: "select", zh: "Select", verb: "挑进来", color: "var(--cs-color-success)", loss: "基本无损",
		def: "每一步只把当下需要的信息拉进窗口，而不是把能拿到的都塞进去。",
		means: ["记忆的选择性召回", "工具描述检索，只送最相关的", "知识 RAG：AST 切分 + 关键词 + 图谱 + 重排", "从运行时 state 里按需暴露字段"],
		treats: "混淆", cost: "基本无损；但召回本身会出错"
	},
	{
		key: "isolate", zh: "Isolate", verb: "隔开来", color: "var(--cs-color-info)", loss: "无损",
		def: "把上下文拆开，不让所有东西挤在同一个窗口里互相干扰。",
		means: ["子 Agent：独立窗口，只回传结论", "沙箱环境：大对象留作变量", "state schema 分字段隔离"],
		treats: "冲突、混淆", cost: "token 上升、协调复杂度上升"
	},
	{
		key: "compress", zh: "Compress", verb: "压下去", color: "var(--cs-color-warning)", loss: "有损",
		def: "只保留完成任务所必需的 token，把历史压成更短的等价表示。",
		means: ["摘要：整轨迹 / 局部 / 交接边界", "裁剪：丢更早消息、剪枝器", "工具结果清理（最轻的一档）"],
		treats: "干扰、成本", cost: "有损：可能丢掉后来才显重要的细节"
	},
	{
		key: "write", zh: "Write", verb: "写出去", color: "var(--cs-color-brand)", loss: "有额外读写",
		def: "把信息存到窗口之外，需要时再读回来，让窗口只留活跃工作集。",
		means: ["草稿本（会话内）", "长期记忆（跨会话）", "文件系统作为外部记忆"],
		treats: "干扰、容量", cost: "额外读写往返；需设计何时写、何时读"
	}
]

const active = ref(0)
</script>

<template>
	<div class="pc-ops">
		<div class="pc-ops__track" role="tablist" aria-label="上下文四大操作，按推荐施加次序排列">
			<template v-for="(op, i) in ops" :key="op.key">
				<button
					type="button"
					role="tab"
					:aria-selected="active === i"
					class="pc-ops__tab"
					:class="{ 'is-active': active === i }"
					:style="{ '--op': op.color }"
					@click="active = i"
				>
					<span class="pc-ops__verb">{{ op.verb }}</span>
					<strong>{{ op.zh }}</strong>
					<span class="pc-ops__loss">{{ op.loss }}</span>
				</button>
				<span v-if="i < ops.length - 1" class="pc-ops__arrow" aria-hidden="true">→</span>
			</template>
		</div>
		<article class="pc-ops__panel" :style="{ '--op': ops[active].color }">
			<header class="pc-ops__panel-head">
				<b class="pc-serif">{{ ops[active].zh }} · {{ ops[active].verb }}</b>
				<span class="pc-ops__badge">{{ ops[active].loss }}</span>
			</header>
			<p class="pc-ops__def">{{ ops[active].def }}</p>
			<ul class="pc-ops__means">
				<li v-for="m in ops[active].means" :key="m">{{ m }}</li>
			</ul>
			<div class="pc-ops__meta">
				<span><i>主要治</i>{{ ops[active].treats }}</span>
				<span><i>代价</i>{{ ops[active].cost }}</span>
			</div>
		</article>
		<p class="pc-note">四者不是并列备选，而有推荐次序：Select → Isolate → Compress → Write。理由是前两步基本无损，后两步有损或有额外成本。很多团队一上来就做摘要，反而丢了「当时看着不重要、后来才致命」的信息。</p>
	</div>
</template>

<style scoped>
.pc-ops { margin: 1rem 0; }
.pc-ops__track { display: flex; align-items: stretch; flex-wrap: wrap; gap: .35rem; }
.pc-ops__tab { flex: 1 1 130px; min-width: 120px; text-align: left; background: var(--cs-color-bg); border: 1px solid var(--cs-color-border); border-top: 3px solid var(--op); border-radius: 10px; padding: .6rem .7rem; cursor: pointer; transition: var(--cs-transition-colors); }
.pc-ops__tab:hover { background: var(--cs-color-bg-soft); }
.pc-ops__tab.is-active { background: color-mix(in srgb, var(--op) 10%, transparent); border-color: var(--op); }
.pc-ops__verb { display: block; font-size: .74rem; color: var(--cs-color-text-subtle); }
.pc-ops__tab strong { display: block; font-family: var(--cs-font-mono); font-size: .98rem; color: var(--op); margin: .1rem 0; }
.pc-ops__loss { display: block; font-size: .68rem; color: var(--cs-color-text-muted); }
.pc-ops__arrow { align-self: center; font-family: var(--cs-font-mono); color: var(--cs-color-text-subtle); padding: 0 .1rem; }
.pc-ops__panel { margin-top: .8rem; background: var(--cs-color-bg); border: 1px solid var(--cs-color-border); border-left: 3px solid var(--op); border-radius: 10px; padding: 1rem 1.1rem; }
.pc-ops__panel-head { display: flex; align-items: center; justify-content: space-between; gap: .6rem; }
.pc-ops__panel-head b { font-size: 1.05rem; }
.pc-ops__badge { font-family: var(--cs-font-mono); font-size: .68rem; letter-spacing: .04em; color: var(--op); border: 1px solid var(--op); border-radius: var(--cs-radius-pill); padding: .1rem .5rem; }
.pc-ops__def { margin: .5rem 0 .6rem; font-size: .88rem; line-height: 1.7; color: var(--cs-color-text); }
.pc-ops__means { margin: 0 0 .7rem; padding-left: 1.1rem; }
.pc-ops__means li { font-size: .83rem; line-height: 1.65; color: var(--cs-color-text-muted); margin-bottom: .2rem; }
.pc-ops__meta { display: flex; flex-wrap: wrap; gap: .5rem 1.4rem; border-top: 1px dashed var(--cs-color-border); padding-top: .6rem; }
.pc-ops__meta span { font-size: .82rem; color: var(--cs-color-text-muted); }
.pc-ops__meta i { display: block; font-family: var(--cs-font-mono); font-size: .64rem; letter-spacing: .08em; text-transform: uppercase; font-style: normal; color: var(--cs-color-text-subtle); margin-bottom: .1rem; }
@media (max-width: 640px) {
	.pc-ops__arrow { display: none; }
	.pc-ops__tab { flex-basis: 44%; }
}
</style>
