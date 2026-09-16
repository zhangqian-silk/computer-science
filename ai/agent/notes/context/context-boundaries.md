# context、memory 与 knowledge 的边界

这三个词经常被换着用，但它们指的是三套不同的机制：解决的问题、本体位置、读写路径、寿命都不同。混用会在错误的层里排障：召回没进载荷却去调抽取逻辑，知识库覆盖不全却去加长期记忆。

---

## 一、从无状态到持久化

一次模型调用是纯函数：输入一段内容，输出一段内容，调用结束后什么都不留下。上下文窗口是一次调用能接收的信息上限，context 用完即弃。但只靠 context 走不通：窗口装不下全部历史，塞进去的信息埋在长上下文中段也不一定用得上，会话结束后 context 就不存在了。于是必须存在 context 之外的、生命周期更长的信息来源。

<MemoryGenesisSteps />

这些外部来源按两个属性分裂成两类：

- **绑定主体，还是全体共享。** 「用户花生过敏」只对这一个用户成立；「退货政策是 7 天无理由」对所有用户都一样。
- **会被改写，还是只会有新版本。** 「用户搬到上海了」是对既有信息的改写；「产品手册出了 v3」是新版本替换旧版本。

由此得到两类数据源的定义：

- **memory（记忆）**：context 之外、绑定特定主体、会被交互改写的带时效信息，判据是「是否仍然成立」。
- **knowledge（知识）**：context 之外、不绑定主体的共享语料，线上只读、由离线流程灌入[[1]](#ref-1)，判据是「是否忠实于源文档」。

三套机制回答三个彼此独立的问题——扩大窗口只缓解 context 的容量压力，加记忆存储只解决跨会话存续，接知识库只解决领域事实覆盖：

- context：**这一次**让模型看见什么；
- memory：跨越**多次调用**的经历如何存续；
- knowledge：参数里**没有**的客观事实从哪来。

---

## 二、边界：对照表与判据

<MemoryContextBoundary />

context 是唯一的出口：memory 与 knowledge 都必须进入 context 才能影响输出，三者最终汇合于本次载荷的容量瓶颈。

对照表可以浓缩成两条判据，都不依赖术语定义：

1. **本体在 context 之内还是之外。** memory、knowledge 的本体在外部存储，进入 context 的只是投影——删掉这份投影，本体还在；改掉这份投影，本体不变。context 没有外部本体，每轮现场拼装。因此变更要落在本体上：长期约束不能只写进 context，错误记忆要改存储而不是改召回结果。
2. **移除之后能不能重建。** context 每轮都在重建；knowledge 重新检索就能复现；memory 记录一次性经历，「上次试这个方案压测反而更慢」删掉后没有第二个来源。可重建的是派生物，不可重建的才是本体——memory 可能是唯一副本，context 的丢弃没有代价。

三个常见的错误判据：「能不能被丢弃」（三者都能丢弃）、「存放时间长短」（介质与寿命不对应）、「是否需要检索」（高命中率的规程常驻不检索，但仍是长期记忆）。

---

## 三、三者如何协同

一次典型的 Agent 调用（以客服场景为例，用户问「我上次那个订单的退货政策适用吗」），装配过程是**两条链路并行、结果分区注入**：

<svg viewBox="0 0 920 500" role="img" aria-label="context 作为装配层，memory 与 knowledge 作为两个数据源经装配进入模型" style="width:100%;height:auto">
	<rect x="0" y="0" width="920" height="500" fill="var(--cs-color-bg-soft)"/>
	<text x="14" y="20" font-family="JetBrains Mono, monospace" font-size="10.5" fill="var(--cs-color-text-subtle)" letter-spacing="1.2">DATA SOURCES / 数据源层 · 本体在此 · 生命周期长于单次调用</text>
	<rect x="14" y="30" width="288" height="132" fill="var(--cs-color-bg)" stroke="var(--cs-color-success)" stroke-width="1.6"/>
	<rect x="14" y="30" width="288" height="26" fill="var(--cs-color-success-soft)"/>
	<text x="28" y="48" font-family="Noto Sans SC, sans-serif" font-size="12.5" font-weight="700" fill="var(--cs-color-success)">MEMORY · 记忆</text>
	<g font-family="Noto Sans SC, sans-serif" font-size="11" fill="var(--cs-color-text)">
		<text x="28" y="74">绑定主体：user / agent / thread</text>
		<text x="28" y="93">Agent 自读写，交互中产生</text>
		<text x="28" y="112">带时效：valid_from / valid_to</text>
		<text x="28" y="131">判据：是否仍然成立</text>
		<text x="28" y="152" font-size="10.5" fill="var(--cs-color-success)">偏好 · 项目约定 · 共同经历 · 待办</text>
	</g>
	<rect x="618" y="30" width="288" height="132" fill="var(--cs-color-bg)" stroke="var(--cs-color-warning)" stroke-width="1.6"/>
	<rect x="618" y="30" width="288" height="26" fill="var(--cs-color-warning-soft)"/>
	<text x="632" y="48" font-family="Noto Sans SC, sans-serif" font-size="12.5" font-weight="700" fill="var(--cs-color-warning)">KNOWLEDGE · 知识</text>
	<g font-family="Noto Sans SC, sans-serif" font-size="11" fill="var(--cs-color-text)">
		<text x="632" y="74">不绑定主体：共享语料</text>
		<text x="632" y="93">离线灌入，线上只读</text>
		<text x="632" y="112">版本语义：reindex 重建</text>
		<text x="632" y="131">判据：是否忠实于源文档</text>
		<text x="632" y="152" font-size="10.5" fill="var(--cs-color-warning)">产品手册 · 代码库 · 制度文档 · Wiki</text>
	</g>
	<rect x="326" y="52" width="268" height="88" fill="var(--cs-color-bg-soft)" stroke="var(--cs-color-border)" stroke-dasharray="4 3"/>
	<text x="460" y="78" text-anchor="middle" font-family="Noto Sans SC, sans-serif" font-size="11.5" font-weight="700" fill="var(--cs-color-text)">两条独立召回链路</text>
	<text x="460" y="99" text-anchor="middle" font-family="Noto Sans SC, sans-serif" font-size="10.8" fill="var(--cs-color-text)">各自打分 · 各自预算 · 各自 rerank</text>
	<text x="460" y="120" text-anchor="middle" font-family="Noto Sans SC, sans-serif" font-size="10.8" fill="var(--cs-color-text)">注入时按来源分区标注</text>
	<path d="M158 162 L158 196 L400 196 L400 214" fill="none" stroke="var(--cs-color-success)" stroke-width="1.6"/>
	<path d="M762 162 L762 196 L520 196 L520 214" fill="none" stroke="var(--cs-color-warning)"/>
	<polygon points="400,220 396,212 404,212" fill="var(--cs-color-success)"/>
	<polygon points="520,220 516,212 524,212" fill="var(--cs-color-warning)"/>
	<text x="14" y="246" font-family="JetBrains Mono, monospace" font-size="10.5" fill="var(--cs-color-text-subtle)" letter-spacing="1.2">ASSEMBLY LAYER / 装配层 · 每轮重建 · 生命周期 = 一次调用</text>
	<rect x="14" y="256" width="892" height="130" fill="var(--cs-color-bg)" stroke="var(--cs-color-border-strong)" stroke-width="1.8" stroke-dasharray="6 4"/>
	<text x="30" y="278" font-family="Noto Sans SC, sans-serif" font-size="12.5" font-weight="700" fill="var(--cs-color-text)">CONTEXT · 本次调用的输入装配</text>
	<g font-family="Noto Sans SC, sans-serif" font-size="10.8">
		<rect x="30" y="292" width="160" height="40" fill="var(--cs-color-bg-soft)" stroke="var(--cs-color-text-subtle)"/>
		<text x="110" y="309" text-anchor="middle" font-weight="700" fill="var(--cs-color-text)">系统指令 / 工具定义</text>
		<text x="110" y="325" text-anchor="middle" font-size="10" fill="var(--cs-color-text-subtle)">最稳定，放最前</text>
		<rect x="200" y="292" width="160" height="40" fill="var(--cs-color-success-soft)" stroke="var(--cs-color-success)"/>
		<text x="280" y="309" text-anchor="middle" font-weight="700" fill="var(--cs-color-success)">召回的记忆</text>
		<text x="280" y="325" text-anchor="middle" font-size="10" fill="var(--cs-color-success)">来自左侧数据源</text>
		<rect x="370" y="292" width="160" height="40" fill="var(--cs-color-warning-soft)" stroke="var(--cs-color-warning)"/>
		<text x="450" y="309" text-anchor="middle" font-weight="700" fill="var(--cs-color-warning)">检索到的知识</text>
		<text x="450" y="325" text-anchor="middle" font-size="10" fill="var(--cs-color-warning)">来自右侧数据源</text>
		<rect x="540" y="292" width="176" height="40" fill="var(--cs-color-bg-soft)" stroke="var(--cs-color-text-subtle)"/>
		<text x="628" y="309" text-anchor="middle" font-weight="700" fill="var(--cs-color-text)">会话历史 / 工具结果</text>
		<text x="628" y="325" text-anchor="middle" font-size="10" fill="var(--cs-color-text-subtle)">滑窗 · 压缩 · 清理</text>
		<rect x="726" y="292" width="160" height="40" fill="var(--cs-color-bg-soft)" stroke="var(--cs-color-text-subtle)"/>
		<text x="806" y="309" text-anchor="middle" font-weight="700" fill="var(--cs-color-text)">当轮用户输入</text>
		<text x="806" y="325" text-anchor="middle" font-size="10" fill="var(--cs-color-text-subtle)">每轮唯一必新</text>
	</g>
	<text x="30" y="356" font-family="Noto Sans SC, sans-serif" font-size="11" fill="var(--cs-color-text)">这五块共享同一个 token 预算；context 自己<tspan font-weight="700" fill="var(--cs-color-text)">不存储任何东西</tspan>，它只决定这一轮放什么、放多少、按什么顺序放</text>
	<text x="30" y="375" font-family="Noto Sans SC, sans-serif" font-size="11" fill="var(--cs-color-text)">顺序有工程含义：越稳定的内容放越前，才能被 prompt cache 的前缀匹配复用</text>
	<path d="M460 386 L460 412" fill="none" stroke="var(--cs-color-border-strong)" stroke-width="1.8"/>
	<polygon points="460,418 455,409 465,409" fill="var(--cs-color-border-strong)"/>
	<rect x="326" y="424" width="268" height="34" fill="var(--cs-color-success)"/>
	<text x="460" y="446" text-anchor="middle" font-family="Noto Sans SC, sans-serif" font-size="12.5" font-weight="700" fill="var(--cs-color-bg)">一次无状态的 LLM 调用</text>
	<text x="460" y="482" text-anchor="middle" font-family="Noto Sans SC, sans-serif" font-size="11.5" font-weight="700" fill="var(--cs-color-danger)">调用结束，装配层随即消失；两侧数据源不受影响</text>
</svg>

```text
用户输入
│
├─→ 记忆召回链路（memory）
│     query 改写（补主体：这个 user 的「上次订单」指什么）
│     → 按 user_id 强制过滤（隔离红线，必须在检索层做而非事后过滤）
│     → 向量 + BM25 混合检索 → 融合
│     → 按 relevance × recency × importance × type 加权
│     → 过滤已失效条目（valid_to 已过）
│     → rerank + 去冗 → 截断到记忆预算
│
├─→ 知识召回链路（knowledge）
│     query 改写（补术语：退货政策 / 退换货规则）
│     → 按文档 ACL 过滤（该用户的租户 / 可见范围）
│     → 向量 + BM25 混合检索 → 融合
│     → 按语义相关性排序（无需 recency 加权，取当前有效版本即可）
│     → rerank → 截断到知识预算
│
└─→ 业务系统实时查询（既不是 memory 也不是 knowledge）
      订单状态、物流、余额 —— 这类强一致数据必须实时查，绝不进记忆
```

装配进 context（顺序按缓存友好度排列）：

```text
[1] 系统指令 + 工具定义        ← 最稳定，放最前
[2] 常驻规则 / 项目约定         ← 会话内不变
[3] 检索到的知识（标注：权威文档，附来源）
[4] 召回的记忆（标注：历史推断，可能过时）
[5] 会话历史（滑窗 / 压缩后）
[6] 当轮用户输入 + 实时业务数据  ← 每轮必新，放最后
```

三条容易被忽略的装配细节：

- **两条链路要分别标注来源与权威等级。** 模型看到的只是文本，不标注就分不清「制度文档说 7 天」和「记忆说用户以为是 15 天」哪个该采信。标注之后配一条明确的优先级：**当轮指令 > 权威知识 > 记忆**。
- **两条链路各设独立预算上限。** 共用一个 top-k 会让一方挤占另一方——知识召回出 10 段长文档，记忆就一条都进不去。独立预算还能在线上退化时分别定位。
- **强一致业务数据不走这两条链路。** 订单状态、库存、余额必须实时查业务系统。把它们写进记忆是常见的严重设计错误——记忆是「关于用户的推断」，不是「业务事实的副本」，副本一定会与真实状态不一致。

---

## 四、几个例子

- **「把这个函数的返回值改成 error」**：只在本次任务内有效，完成后再保留只会干扰——context。
- **「用户上周说过更喜欢先看结论」**：跨会话有效、只对这个用户成立、来源是交互本身——memory，没有任何外部文档能提供它。
- **某框架 v3 的 API 变更说明**：跨会话有效、对所有人一样、有外部权威源——knowledge。抽成 memory 会让每个用户各存一份，文档更新时记忆也不跟着更新。
- **「上次试过在网关层做重试，压测下超时反而上升，回滚了」**：memory 里的情景经历，不可重建——删掉没有第二个来源；但也不能记成「重试有害」的普适规则，那是过度泛化。
- **「我们团队所有服务都用 error 返回值，不用 panic」**：程序性记忆，命中率接近百分之百，所以常驻注入，而不是放进检索赌它每次都被召回。

边界清楚之后，失效定位是直接的推论：「答得头头是道但依据是旧的」是 memory 的有效性问题，需要时间与矛盾信号，不是更好的嵌入模型；「明明存了却不用」是召回侧问题，去查存储里有没有那条是白费功夫；「领域事实答错、用户偏好正常」是 knowledge 侧问题。三套机制通常不会同时失效，正常与异常的分界直接指出坏在哪一层。

---

## 参考文献

<a id="ref-1"></a>[1] [Retrieval-Augmented Generation for Knowledge-Intensive NLP](https://arxiv.org/abs/2005.11401)（论文，arXiv:2005.11401，核验于 2026-09-14）。把参数化记忆与外部语料稠密向量索引（非参数化记忆）结合的研究起点，不代表当前工程形态。
