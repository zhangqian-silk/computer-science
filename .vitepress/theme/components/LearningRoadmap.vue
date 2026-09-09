<script setup lang="ts">
import { computed, ref, useId } from "vue"
import { withBase } from "vitepress"

const selected = ref(0)
const id = useId()
const chapters = [
	{ name: "基础", label: "建立共同语言", question: "怎样从数据、目标和梯度恢复一次学习？", index: "/ai/foundations/", paths: [
		["学习问题与分类", "/ai/foundations/learning-problem-map", "区分监督来源、输出接口、架构和系统"],
		["数学与符号", "/ai/foundations/math-notation", "读出对象、形状和归约轴"],
		["训练闭环", "/ai/foundations/neural-network-basics", "从局部导数到一次更新"],
		["Tensor 与数值", "/ai/foundations/tensor-and-memory", "将公式落到布局、字节和精度"],
		["方法演化", "/ai/foundations/nlp-history", "按瓶颈连接技术，而非只记年份"]
	], check: "能用一个小样本算出前向、loss、梯度与更新，并解释验证集不参与这次更新。" },
	{ name: "表示", label: "对象怎样进入计算", question: "ID、词向量、token 状态和检索向量有什么区别？", index: "/ai/representation/", paths: [
		["Tokenization", "/ai/representation/tokenization", "训练词表与应用词表是不同阶段"],
		["Embedding 与 word2vec", "/ai/representation/embedding", "训练信号塑造可复用表示"],
		["矩阵分解视角", "/ai/representation/embedding-matrix-factorization", "理解理想内积、噪声分布与低秩约束"],
		["文本与多模态", "/ai/representation/text-embedding", "池化、对比、局部交互与跨模态接口"],
		["向量检索", "/ai/representation/vector-retrieval", "分开编码误差、索引近似与重排"]
	], check: "能解释一组正负样本怎样改变得分，并定位检索失败发生在哪一层。" },
	{ name: "机制", label: "信息与参数如何组织", question: "一个机制到底改变了什么数学规则或成本？", index: "/ai/mechanism/", paths: [
		["Attention", "/ai/mechanism/attention", "从分数到权重，再到 Value 读取"],
		["可见性与稀疏", "/ai/mechanism/self-attention", "检查因果路径与跨层传播"],
		["位置与扩长", "/ai/mechanism/positional-encoding", "坐标、相对偏置和旋转分开理解"],
		["稳定性与通道变换", "/ai/mechanism/normalization-and-residual", "Norm、残差与 FFN 的职责"],
		["容量与适配", "/ai/mechanism/moe", "MoE 选择计算，LoRA 约束更新"]
	], check: "能追踪 shape、梯度和成本，并指出一个该机制不能保证的结论。" },
	{ name: "模型", label: "组合成完整学习接口", question: "同样的机制为什么会产生不同训练与推理行为？", index: "/ai/model/", paths: [
		["固定窗口", "/ai/model/n-gram", "计数、平滑与神经参数共享"],
		["递归记忆", "/ai/model/rnn", "状态压缩、BPTT 与门控"],
		["条件生成", "/ai/model/seq2seq", "源端编码、目标前缀与搜索"],
		["Transformer 家族", "/ai/model/transformer", "BERT/GPT 的可见性、目标与接口"],
		["状态空间", "/ai/model/state-space-model", "递推、卷积和选择性 scan"]
	], check: "能从训练样本走到推理结果，区分论文原版与后来改造。" },
	{ name: "Infra", label: "从公式到运行系统", question: "数据放在哪里，谁来计算，什么时候释放？", index: "/ai/infra/", paths: [
		["加速器与性能", "/ai/infra/accelerator/", "形状、字节、上界与 kernel"],
		["通信与训练", "/ai/infra/training/", "状态所有权、精度、并行和快照"],
		["推理运行时", "/ai/infra/inference/", "Prefill、KV、调度与解码"],
		["服务平台", "/ai/infra/platform/", "版本、资源、路由与安全边界"],
		["实验与项目", "/ai/infra/labs/", "先验证语义，再补目标硬件证据"]
	], check: "能画出状态生命周期，并用同一 workload 对比质量、延迟、吞吐和失败。" },
	{ name: "评估", label: "把理解变成证据", question: "这个分数究竟支持哪一个能力主张？", index: "/ai/evaluation/", paths: [
		["语言概率", "/ai/evaluation/language-model-evaluation", "NLL、PPL、计分单位与污染"],
		["表示几何", "/ai/evaluation/embedding-geometry", "近邻与探针不是因果证明"],
		["检索", "/ai/evaluation/retrieval-evaluation", "覆盖、排名和候选流转"],
		["生成", "/ai/evaluation/generation-evaluation", "字面、事实、证据与任务完成"],
		["长上下文", "/ai/evaluation/long-context-evaluation", "长度 × 位置 × 任务与成本"]
	], check: "能写出公平对照、失败切片和结论边界，而非只报告平均分。" },
	{ name: "Agent", label: "模型怎样参与动作", question: "模型的建议何时才会变成已授权的真实结果？", index: "/ai/agent/", paths: [
		["系统闭环", "/ai/agent/", "决策、校验、执行、观察与终止"],
		["检索证据", "/ai/representation/vector-retrieval", "外部知识不是参数记忆"],
		["生成验收", "/ai/evaluation/generation-evaluation", "回答正确不等于工具成功"],
		["安全边界", "/ai/infra/platform/security-and-multitenancy", "输入、缓存与执行权限分层"]
	], check: "能用环境状态验证完成，并在失败或无权限时明确停止。" }
] as const
const current = computed(() => chapters[selected.value])
</script>
<template>
	<section class="learning-map" :aria-labelledby="`${id}-title`">
		<header><span>AI · 知识导航</span><h2 :id="`${id}-title`">沿问题建立知识，而非沿名称记忆</h2><p>选择分区查看主干路径。完整主题在各分区入口中展开；这些分区相互依赖，不是互斥分类。</p></header>
		<div class="learning-map__tabs" role="group" aria-label="选择学习分区"><button v-for="(chapter, i) in chapters" :key="chapter.name" type="button" :aria-pressed="selected === i" :aria-controls="`${id}-content`" @click="selected = i"><span>{{ String(i + 1).padStart(2, "0") }}</span>{{ chapter.name }}</button></div>
		<div :id="`${id}-content`" class="learning-map__content">
			<p class="learning-map__label">{{ current.label }}</p><h3>{{ current.question }}</h3>
			<ol><li v-for="([name, href, question], i) in current.paths" :key="href"><span>{{ i + 1 }}</span><div><a :href="withBase(href)">{{ name }}</a><p>{{ question }}</p></div></li></ol>
			<p class="learning-map__check"><strong>读完应能：</strong>{{ current.check }}</p>
			<a :href="withBase(current.index)" class="learning-map__all">进入{{ current.name }}完整目录 →</a>
		</div>
	</section>
</template>
<style scoped>
.learning-map { margin: 1.5rem 0 2rem; border: 1px solid var(--vp-c-divider); border-radius: 8px; background: var(--vp-c-bg); overflow: hidden; }
.learning-map header { padding: 1.4rem; background: var(--vp-c-bg-soft); }
.learning-map header > span, .learning-map__label { font-size: .75rem; letter-spacing: .04em; color: var(--vp-c-brand-1); }
.learning-map header h2 { margin: .45rem 0; padding: 0; border: 0; font-size: 1.4rem; line-height: 1.6; }
.learning-map header p { margin: .4rem 0 0; color: var(--vp-c-text-2); font-size: .9rem; }
.learning-map__tabs { display: flex; flex-wrap: wrap; border-block: 1px solid var(--vp-c-divider); }
.learning-map__tabs button { flex: 1 0 82px; min-height: 54px; padding: .5rem; font-size: .85rem; cursor: pointer; border-bottom: 2px solid transparent; }
.learning-map__tabs button span { margin-right: .4rem; font-size: .7rem; color: var(--vp-c-text-2); }
.learning-map__tabs button[aria-pressed='true'] { border-bottom-color: var(--vp-c-brand-1); background: var(--vp-c-brand-soft); color: var(--vp-c-brand-1); }
.learning-map button:focus-visible, .learning-map a:focus-visible { outline: 2px solid var(--vp-c-brand-1); outline-offset: -2px; }
.learning-map__content { padding: 1.25rem 1.4rem; }
.learning-map__label { margin: 0; }
.learning-map__content h3 { margin: .35rem 0 1.2rem; font-size: 1.12rem; }
.learning-map ol { list-style: none; margin: 0; padding: 0; }
.learning-map li { display: flex; gap: .9rem; margin: 0; padding: .75rem 0; border-top: 1px solid var(--vp-c-divider); }
.learning-map li > span { color: var(--vp-c-text-3); font-family: var(--vp-font-family-mono); }
.learning-map li p { margin: .2rem 0 0; font-size: .82rem; color: var(--vp-c-text-2); }
.learning-map__check { font-size: .88rem; padding-top: .8rem; border-top: 1px solid var(--vp-c-divider); }
.learning-map__all { display: inline-block; padding: .5rem 0; font-size: .9rem; }
@media (max-width: 640px) { .learning-map header, .learning-map__content { padding: 1rem; } .learning-map header h2 { font-size: 1.2rem; } }
</style>
