<script setup lang="ts">
const groups = [
	{
		nature: "静态",
		en: "Static · 写好即固定",
		color: "var(--cs-color-info)",
		desc: "每次调用都原封不动地存在，是成本的固定项，也是行为地基。靠一次性写好。",
		items: [
			{ name: "系统提示", token: "固定，位于最前部", gov: "校准「高度」、分节、删冗余" },
			{ name: "工具定义 / MCP", token: "固定但常被低估：几十个工具就是数千 token，且常驻最前部", gov: "裁最小可用集、按前缀分组遮蔽、对描述做检索" },
			{ name: "示例（few-shot）", token: "固定", gov: "用典型且多样的少量范例，而非边界情况清单" }
		]
	},
	{
		nature: "动态",
		en: "Dynamic · 按召回策略变化",
		color: "var(--cs-color-success)",
		desc: "每一步该把哪些外部信息拉进来，取决于召回策略。靠预取 / 即时 / 混合三种范式。",
		items: [
			{ name: "检索到的知识", token: "每轮可变，取决于召回策略", gov: "预取 / 即时 / 混合；控制召回条数与重排" },
			{ name: "记忆", token: "按需注入，规模可控", gov: "分层（会话内草稿 / 跨会话长期）、选择性召回" }
		]
	},
	{
		nature: "累积",
		en: "Accumulative · 随步数单调增长",
		color: "var(--cs-color-warning)",
		desc: "随循环单调膨胀，是成本大头，也是失效根源。绝大多数线上问题都出在这一类。",
		items: [
			{ name: "消息历史", token: "单调膨胀，随步数线性增长", gov: "压缩、裁剪、复述目标" },
			{ name: "工具返回值", token: "最易失控：单个网页或 PDF 的观测就可能撑爆窗口", gov: "结果清理、可还原压缩、外置到文件 / 沙箱" }
		]
	}
]
const shift = [
	{ step: "第 1 步", stat: 62, dyn: 20, acc: 18 },
	{ step: "第 30 步", stat: 12, dyn: 20, acc: 68 }
]
</script>

<template>
	<div class="pc-anat">
		<div class="pc-anat__grid">
			<section v-for="g in groups" :key="g.nature" class="pc-anat__col" :style="{ '--nat': g.color }">
				<header class="pc-anat__head">
					<span class="pc-anat__tag">{{ g.nature }}</span>
					<span class="pc-anat__en">{{ g.en }}</span>
				</header>
				<p class="pc-anat__desc">{{ g.desc }}</p>
				<article v-for="it in g.items" :key="it.name" class="pc-anat__item">
					<b class="pc-serif">{{ it.name }}</b>
					<p class="pc-anat__tok">{{ it.token }}</p>
					<p class="pc-anat__gov"><span>治理</span>{{ it.gov }}</p>
				</article>
			</section>
		</div>

		<figure class="pc-anat__shift">
			<figcaption class="pc-anat__shift-ttl pc-serif">同一任务，第 1 步与第 30 步的 token 构成</figcaption>
			<div v-for="s in shift" :key="s.step" class="pc-anat__row">
				<span class="pc-anat__step">{{ s.step }}</span>
				<div class="pc-anat__track" role="img" :aria-label="`${s.step}：静态 ${s.stat}%、动态 ${s.dyn}%、累积 ${s.acc}%`">
					<span class="seg s" :style="{ width: s.stat + '%' }">静态 {{ s.stat }}%</span>
					<span class="seg d" :style="{ width: s.dyn + '%' }">动态 {{ s.dyn }}%</span>
					<span class="seg a" :style="{ width: s.acc + '%' }">累积 {{ s.acc }}%</span>
				</div>
			</div>
			<figcaption class="pc-note">比例为定性示意，非任何系统的实测分布。它只说明一个结构性事实：静态部分绝对值不变、占比被稀释，累积部分从边缘变成主体——真正的战场是消息历史与工具返回值。</figcaption>
		</figure>
	</div>
</template>

<style scoped>
.pc-anat { margin: 1rem 0; }
.pc-anat__grid { display: grid; grid-template-columns: repeat(3, 1fr); gap: .8rem; }
.pc-anat__col { background: var(--cs-color-bg); border: 1px solid var(--cs-color-border); border-top: 3px solid var(--nat); border-radius: 10px; padding: 1rem 1.05rem; }
.pc-anat__head { display: flex; align-items: baseline; gap: .5rem; flex-wrap: wrap; }
.pc-anat__tag { font-weight: 700; font-size: 1rem; color: var(--nat); }
.pc-anat__en { font-family: var(--cs-font-mono); font-size: .68rem; letter-spacing: .04em; color: var(--cs-color-text-subtle); }
.pc-anat__desc { margin: .5rem 0 .8rem; font-size: .82rem; line-height: 1.65; color: var(--cs-color-text-muted); }
.pc-anat__item { border-top: 1px dashed var(--cs-color-border); padding-top: .6rem; margin-top: .6rem; }
.pc-anat__item:first-of-type { border-top: 0; padding-top: 0; margin-top: 0; }
.pc-anat__item b { display: block; font-size: .92rem; }
.pc-anat__tok { margin: .2rem 0 .35rem; font-size: .8rem; line-height: 1.6; color: var(--cs-color-text-muted); }
.pc-anat__gov { margin: 0; font-size: .8rem; line-height: 1.6; color: var(--cs-color-text-muted); }
.pc-anat__gov span { display: inline-block; font-family: var(--cs-font-mono); font-size: .64rem; letter-spacing: .08em; text-transform: uppercase; color: var(--nat); margin-right: .4rem; }
.pc-anat__shift { margin: 1rem 0 0; background: var(--cs-color-bg); border: 1px solid var(--cs-color-border); border-radius: 10px; padding: 1rem 1.1rem .85rem; }
.pc-anat__shift-ttl { font-size: 1rem; font-weight: 700; margin: 0 0 .8rem; }
.pc-anat__row { display: grid; grid-template-columns: 64px 1fr; gap: .7rem; align-items: center; margin-bottom: .5rem; }
.pc-anat__step { font-family: var(--cs-font-mono); font-size: .74rem; color: var(--cs-color-text-muted); text-align: right; }
.pc-anat__track { display: flex; height: 30px; border-radius: var(--cs-radius-sm); overflow: hidden; border: 1px solid var(--cs-color-border); }
.pc-anat__track .seg { display: flex; align-items: center; justify-content: center; font-size: .68rem; color: var(--cs-color-on-brand); white-space: nowrap; overflow: hidden; }
.pc-anat__track .seg.s { background: var(--cs-color-info); }
.pc-anat__track .seg.d { background: var(--cs-color-success); }
.pc-anat__track .seg.a { background: var(--cs-color-warning); }
@media (max-width: 860px) {
	.pc-anat__grid { grid-template-columns: 1fr; }
}
</style>
