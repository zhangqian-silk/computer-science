<script setup lang="ts">
import { computed, ref } from "vue"

/**
 * 检索增强的完整链路：离线四步把语料变成可查的副本，在线五步把查询变成一段载荷。
 *
 * 组件要传达的是「每一步各自能坏在哪里」——因为这条链路的排障，
 * 全部依赖于把故障归到某一步，而各步的现象在最终输出上看起来是一样的：答不上来。
 * 因此每一步同时给出：它做什么、它失败时的表面现象、以及验证它是否是元凶的动作。
 */

type Stage = {
	key: string
	phase: "offline" | "online"
	idx: string
	name: string
	does: string
	/** 这一步坏掉时，用户侧看到什么 */
	symptom: string
	/** 怎么确认问题就在这一步 */
	probe: string
	/** 这一步最常见的具体失误 */
	pitfall: string
}

const stages: Stage[] = [
	{
		key: "parse", phase: "offline", idx: "1", name: "解析",
		does: "把各种格式的原件读成带结构的文本：标题层级、表格行列、代码块、图注都要留住。",
		symptom: "表格类问题一律答错，或答案里出现串行错位的数字。",
		probe: "直接打开解析产物看一眼，不要看最终答案——结构是否还在，一眼可辨。",
		pitfall: "把版面复杂的文档直接抽成纯字符串，表格与层级在第一步就没了，后面每一步都无从补救。"
	},
	{
		key: "chunk", phase: "offline", idx: "2", name: "切分",
		does: "按语义边界把长文切成检索单元，通常沿标题或语法结构切，并保留片段到原文的定位信息。",
		symptom: "答案总是缺一半，或者引用的位置跳到了相邻小节。",
		probe: "把命中片段原样打印出来，看它是否是一个自足的意思单元。",
		pitfall: "在语义中间硬切。一个结论被切成两段之后，两段各自都不足以支撑答案。"
	},
	{
		key: "embed", phase: "offline", idx: "3", name: "向量化",
		does: "把每个片段编码成向量。片段长度须与所用模型的有效窗口相称，超出的部分会被静默截断。",
		symptom: "长片段的后半部分永远召不回来，且没有任何报错。",
		probe: "取一个长片段，用只出现在末尾的独有词去查，看能否命中。",
		pitfall: "换模型当成常规升级。相似度空间变了，旧向量全部作废，必须整体重建。"
	},
	{
		key: "index", phase: "offline", idx: "4", name: "建索引",
		does: "写入近邻索引与倒排索引，并带上过滤所需的元数据：来源、时间、权限、版本。",
		symptom: "删掉的文档还能被搜到，或加了过滤条件后结果直接为空。",
		probe: "拿被删文档的标识去查索引本身，而不是去查最终答案。",
		pitfall: "只删了原件没删派生的片段与向量；或先取前若干条再过滤，把结果过滤成了空。"
	},
	{
		key: "rewrite", phase: "online", idx: "5", name: "查询改写",
		does: "把口语化、省略主语的问题补全成可检索的表达，必要时拆成几个子问题并行去查。",
		symptom: "追问句（「那它呢」）永远答不对，独立提问却正常。",
		probe: "把改写后的查询打印出来，看省略的主体有没有被补回去。",
		pitfall: "对本来已经清晰的短查询也强行改写，反而把精确的词面改没了。"
	},
	{
		key: "recall", phase: "online", idx: "6", name: "多路召回",
		does: "稠密向量负责语义、稀疏检索负责词面精确命中，各自取一批候选，再按名次融合成一个序列。",
		symptom: "错误码、版本号、人名这类精确串搜不到，换成描述性问法却能搜到。",
		probe: "把两路结果分开看：如果稀疏那一路根本没启用，问题就在这里。",
		pitfall: "只用稠密一路。语义相近对精确标识符无效，而这类查询在真实流量里占比不低。"
	},
	{
		key: "rerank", phase: "online", idx: "7", name: "重排",
		does: "让查询与候选片段一起进模型做一次细粒度打分，把「相关但答非所问」的压下去。",
		symptom: "正确片段确实被召回了，但排在很后面，最终没进载荷。",
		probe: "看正确片段在召回阶段与重排阶段各自的名次——这一步的价值就体现在名次差上。",
		pitfall: "候选集给得太小。重排只能在给定候选里挑，召回没捞到的，它一个也变不出来。"
	},
	{
		key: "assemble", phase: "online", idx: "8", name: "组装载荷",
		does: "把最终少数片段连同来源标识拼进本次请求，同时留出生成所需的余量。",
		symptom: "日志里检索一切正常，模型却说不知道。",
		probe: "打印本次实际发出的内容。这一个动作能把「没召回」和「召回了却没进载荷」一次分开。",
		pitfall: "预算算错，片段被静默截断；或丢掉了来源标识，使得后续无法溯源。"
	},
	{
		key: "generate", phase: "online", idx: "9", name: "生成与溯源",
		does: "要求只依据给定内容作答、每个论断附来源、无依据时明确说没有，并保留拒答分支。",
						symptom: "答案流畅且看起来合理，但与所引片段并不一致。",
		probe: "逐句比对论断与它所引的片段，看是否真的被支撑。",
		pitfall: "没有拒答分支。检索不到时模型仍然作答，而这种错误恰好最难被发现。"
	}
]

const offline = computed(() => stages.filter((s) => s.phase === "offline"))
const online = computed(() => stages.filter((s) => s.phase === "online"))

const openKey = ref("recall")
const cur = computed(() => stages.find((s) => s.key === openKey.value) ?? stages[0])

/** 排障视角：只看现象，还是连同验证动作一起看 */
const showProbe = ref(true)
</script>

<template>
	<div class="rpl">
		<div class="rpl__phases">
			<section class="rpl__phase">
				<header>
					<b>离线</b>
					<span>把语料变成可查的副本，与任何一次提问无关</span>
				</header>
				<div class="rpl__row">
					<button
						v-for="s in offline"
						:key="s.key"
						type="button"
						class="rpl__stage"
						:class="{ 'is-open': openKey === s.key }"
						:aria-pressed="openKey === s.key"
						@click="openKey = s.key"
					>
						<span class="rpl__idx">{{ s.idx }}</span>
						<span class="rpl__name">{{ s.name }}</span>
					</button>
				</div>
			</section>

			<section class="rpl__phase rpl__phase--online">
				<header>
					<b>在线</b>
					<span>把一次查询变成一段载荷，每次请求都重跑</span>
				</header>
				<div class="rpl__row">
					<button
						v-for="s in online"
						:key="s.key"
						type="button"
						class="rpl__stage"
						:class="{ 'is-open': openKey === s.key }"
						:aria-pressed="openKey === s.key"
						@click="openKey = s.key"
					>
						<span class="rpl__idx">{{ s.idx }}</span>
						<span class="rpl__name">{{ s.name }}</span>
					</button>
				</div>
			</section>
		</div>

		<label class="cs-toggle rpl__sw">
			<input v-model="showProbe" type="checkbox">
			<span>同时显示「怎么确认是这一步」</span>
		</label>

		<section class="rpl__panel" :class="cur.phase === 'offline' ? 'is-offline' : 'is-online'">
			<header>
				<b>{{ cur.idx }} · {{ cur.name }}</b>
				<span class="rpl__badge">{{ cur.phase === "offline" ? "离线，一次构建长期复用" : "在线，每次请求重跑" }}</span>
			</header>

			<p class="rpl__does">{{ cur.does }}</p>

			<div class="rpl__facts">
				<p class="cs-state cs-state--warn"><i>失败时的表面现象</i>{{ cur.symptom }}</p>
				<p v-if="showProbe" class="cs-state cs-state--pass"><i>怎么确认是这一步</i>{{ cur.probe }}</p>
				<p class="cs-state cs-state--fail"><i>最常见的具体失误</i>{{ cur.pitfall }}</p>
			</div>
		</section>

		<p class="pc-note">
			九步的失败在最终输出上几乎是同一种现象——答不上来或答错，
			所以这条链路的排障不能从「换个更好的嵌入模型」开始，而要先把故障归到某一步。
			最省力的第一个动作永远是<b>打印本次实际发出的内容</b>：它一次就能把第 8 步之前与之后分开。
		</p>
	</div>
</template>

<style scoped>
.rpl { margin: 1rem 0; }
.rpl__phases { display: grid; gap: var(--cs-space-4); }
.rpl__phase { background: var(--cs-color-bg-soft); border: 1px solid var(--cs-color-border); border-left: 3px solid var(--cs-color-info); border-radius: var(--cs-radius-lg); padding: var(--cs-space-3) var(--cs-space-4); }
.rpl__phase--online { border-left-color: var(--cs-color-brand); }
.rpl__phase > header { display: flex; flex-wrap: wrap; align-items: baseline; gap: var(--cs-space-3); margin-bottom: var(--cs-space-3); }
.rpl__phase > header b { font-size: var(--cs-text-md); color: var(--cs-color-text); }
.rpl__phase > header span { font-size: var(--cs-text-3xs); color: var(--cs-color-text-subtle); }
.rpl__row { display: flex; flex-wrap: wrap; gap: var(--cs-space-2); }
.rpl__stage { display: inline-flex; align-items: center; gap: var(--cs-space-2); min-height: var(--cs-tap-target); padding: 0 var(--cs-space-4); cursor: pointer; background: var(--cs-color-bg); border: 1px solid var(--cs-color-border); border-radius: var(--cs-radius-pill); transition: var(--cs-transition-colors); }
.rpl__stage:hover { background: var(--cs-color-bg-soft); }
.rpl__stage.is-open { background: var(--cs-color-brand-soft); border-color: var(--cs-color-brand); }
.rpl__idx { font-family: var(--cs-font-mono); font-size: var(--cs-text-3xs); color: var(--cs-color-text-subtle); }
.rpl__stage.is-open .rpl__idx { color: var(--cs-color-brand); }
.rpl__name { font-size: var(--cs-text-sm); color: var(--cs-color-text); }
.rpl__stage.is-open .rpl__name { color: var(--cs-color-brand); font-weight: 600; }
.rpl__sw { display: inline-flex; align-items: center; gap: var(--cs-space-1); min-height: var(--cs-tap-target); margin: var(--cs-space-4) 0 var(--cs-space-3); font-size: var(--cs-text-3xs); color: var(--cs-color-text-subtle); cursor: pointer; }
.rpl__panel { background: var(--cs-color-bg); border: 1px solid var(--cs-color-border); border-top: 3px solid var(--cs-color-brand); border-radius: var(--cs-radius-lg); padding: var(--cs-space-5) var(--cs-space-6); }
.rpl__panel.is-offline { border-top-color: var(--cs-color-info); }
.rpl__panel > header { display: flex; flex-wrap: wrap; align-items: baseline; justify-content: space-between; gap: var(--cs-space-3); }
.rpl__panel > header b { font-size: var(--cs-text-lg); color: var(--cs-color-text); }
.rpl__badge { font-size: var(--cs-text-3xs); color: var(--cs-color-text-muted); border: 1px solid var(--cs-color-border); border-radius: var(--cs-radius-pill); padding: var(--cs-space-1) var(--cs-space-3); }
.rpl__does { margin: var(--cs-space-3) 0 var(--cs-space-4); font-size: var(--cs-text-base); line-height: var(--cs-leading-relaxed); color: var(--cs-color-text); }
.rpl__facts { display: grid; gap: var(--cs-space-2); }
.rpl__facts p { margin: 0; font-size: var(--cs-text-sm); line-height: var(--cs-leading-relaxed); }
.rpl__facts i { display: block; font-family: var(--cs-font-mono); font-size: var(--cs-text-3xs); letter-spacing: .08em; font-style: normal; color: var(--cs-color-text-subtle); margin-bottom: var(--cs-space-1); }
@media (max-width: 520px) {
	.rpl__panel { padding: var(--cs-space-4); }
	.rpl__stage { padding: 0 var(--cs-space-3); }
}
</style>
