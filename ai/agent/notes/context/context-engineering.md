# Context Engineering

上下文工程解决「本轮推理让模型看见什么」：从可用信息中选择、组织并维护真正进入请求的 token。它覆盖 Prompt、工具、知识、记忆、历史与工具结果，并在任务推进时持续更新。Prompt 负责表达任务，是上下文中的静态部分；上下文工程还要治理随每轮调用变化和累积的信息。

上下文工程的目标不是填满窗口，而是在不遗漏必要信息的前提下，构造尽可能小的高信号 token 集合。[[22]](../references.md#source-anthropic-context)

## Prompt 是 Context 的一部分

<div class="pc-compare-wrap" role="region" aria-label="Prompt 与 Context 对照表" tabindex="0">
	<table class="pc-compare">
		<thead>
			<tr>
				<th scope="col">对照维度</th>
				<th scope="col" class="pe">Prompt</th>
				<th scope="col" class="ce">Context</th>
			</tr>
		</thead>
		<tbody>
			<tr>
				<th scope="row" class="dim">核心问题</th>
				<td class="pe">这项任务怎样说清楚</td>
				<td class="ce">此刻哪些信息应该进入请求</td>
			</tr>
			<tr>
				<th scope="row" class="dim">操作对象</th>
				<td class="pe">指令、示例、输出契约</td>
				<td class="ce">Prompt、工具、知识、记忆、历史、工具结果</td>
			</tr>
			<tr>
				<th scope="row" class="dim">时间维度</th>
				<td class="pe">通常随任务定义固定</td>
				<td class="ce">每轮推理前重新选择和整理</td>
			</tr>
			<tr>
				<th scope="row" class="dim">主要约束</th>
				<td class="pe">表达清晰，输出格式可解析</td>
				<td class="ce">注意力与容量有限，状态会持续累积</td>
			</tr>
			<tr>
				<th scope="row" class="dim">典型场景</th>
				<td class="pe">单轮分类、抽取、改写与生成</td>
				<td class="ce">多轮 Agent、长任务、知识检索与代码库操作</td>
			</tr>
			<tr>
				<th scope="row" class="dim">典型失败</th>
				<td class="pe">指令歧义、格式偏离</td>
				<td class="ce">目标漂移、错误累积、工具混淆、信息冲突</td>
			</tr>
		</tbody>
	</table>
</div>

表中的差异不是把两者拆成平行模块，而是区分局部问题和整体问题：Prompt 决定静态任务契约怎样表达，Context 决定这份契约与哪些动态、累积信息共同进入本轮请求。排障时仍需区分成因；资料没有进入请求、历史膨胀或证据冲突时，继续扩写 Prompt 只会让固定前缀更重。

---

## 一、上下文由什么构成

对话、知识库和长期记忆只是信息来源；只有真正进入本次模型请求的 token 才是上下文。上下文也不是启动时组装一次便固定：模型调用工具后，新的观察会回到信息池，下一轮推理前必须重新选择、整理和排序。

<figure class="pc-concept-figure">
	<figcaption>上下文在每轮推理前重新策展</figcaption>
	<div class="pc-figure-scroll">
		<svg class="pc-flow-svg" viewBox="0 0 900 280" role="img" aria-labelledby="pc-curation-title pc-curation-desc">
			<title id="pc-curation-title">上下文策展闭环</title>
			<desc id="pc-curation-desc">静态、动态和累积信息经过选择进入上下文窗口，模型推理后可能调用工具，工具结果回到累积信息，下一轮重新策展。</desc>
			<defs>
				<marker id="pc-curation-arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
					<path d="M0 0 L10 5 L0 10 z" class="pc-svg-arrowhead" />
				</marker>
				<marker id="pc-curation-loop-arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
					<path d="M0 0 L10 5 L0 10 z" class="pc-svg-loop-arrowhead" />
				</marker>
			</defs>
			<text x="20" y="22" class="pc-svg-kicker">可用信息池</text>
			<g class="pc-svg-copy">
				<rect x="20" y="38" width="210" height="50" rx="6" class="pc-svg-box" />
				<text x="34" y="59" class="pc-svg-label">静态</text>
				<text x="34" y="78" class="pc-svg-muted">系统提示 · 工具定义 · 示例</text>
				<rect x="20" y="98" width="210" height="50" rx="6" class="pc-svg-box" />
				<text x="34" y="119" class="pc-svg-label">动态</text>
				<text x="34" y="138" class="pc-svg-muted">检索知识 · 召回的记忆</text>
				<rect x="20" y="158" width="210" height="50" rx="6" class="pc-svg-box" />
				<text x="34" y="179" class="pc-svg-label">累积</text>
				<text x="34" y="198" class="pc-svg-muted">消息历史 · 工具返回值</text>
			</g>
			<path d="M242 123 H314" class="pc-svg-arrow" marker-end="url(#pc-curation-arrow)" />
			<text x="278" y="112" text-anchor="middle" class="pc-svg-accent">Select</text>
			<rect x="326" y="30" width="210" height="188" rx="8" class="pc-svg-window" />
			<text x="431" y="20" text-anchor="middle" class="pc-svg-kicker">CONTEXT WINDOW</text>
			<rect x="344" y="48" width="174" height="28" rx="4" class="pc-svg-static" />
			<rect x="344" y="84" width="174" height="28" rx="4" class="pc-svg-dynamic" />
			<rect x="344" y="120" width="174" height="28" rx="4" class="pc-svg-cumulative" />
			<rect x="344" y="156" width="174" height="44" rx="4" class="pc-svg-summary" />
			<g class="pc-svg-on-fill">
				<text x="356" y="67">指令与可用工具</text>
				<text x="356" y="103">当前证据与记忆</text>
				<text x="356" y="139">近期轨迹</text>
				<text x="356" y="175">旧历史摘要</text>
				<text x="356" y="192">+ 可恢复的外部句柄</text>
			</g>
			<path d="M548 123 H610" class="pc-svg-arrow" marker-end="url(#pc-curation-arrow)" />
			<circle cx="660" cy="123" r="38" class="pc-svg-model" />
			<text x="660" y="119" text-anchor="middle" class="pc-svg-label">模型</text>
			<text x="660" y="138" text-anchor="middle" class="pc-svg-muted">推理</text>
			<path d="M700 108 H764" class="pc-svg-arrow" marker-end="url(#pc-curation-arrow)" />
			<path d="M700 140 H764" class="pc-svg-arrow" marker-end="url(#pc-curation-arrow)" />
			<rect x="776" y="87" width="104" height="32" rx="5" class="pc-svg-box" />
			<rect x="776" y="130" width="104" height="32" rx="5" class="pc-svg-box-accent" />
			<text x="828" y="108" text-anchor="middle" class="pc-svg-label">回复</text>
			<text x="828" y="151" text-anchor="middle" class="pc-svg-accent">调用工具</text>
			<path d="M828 168 V244 H126 V216" class="pc-svg-loop" marker-end="url(#pc-curation-loop-arrow)" />
			<text x="480" y="237" text-anchor="middle" class="pc-svg-loop-label">工具结果回流，信息池膨胀，下一轮重新策展</text>
		</svg>
	</div>
	<p>模型只会使用进入窗口的信息；保存、索引或记住某份资料，不等于本轮已经把它交给模型。</p>
</figure>

按变化方式把七个来源归为三类，可以直接推出各自的治理方式：

| 性质 | 来源 | token 行为 | 主要治理手段 |
| --- | --- | --- | --- |
| 静态 | 系统提示 | 固定，通常位于前缀 | 校准高度、分节、删除冗余 |
| 静态 | 工具定义 | 固定且容易被低估 | 裁最小可用集，明确职责与参数边界 |
| 静态 | 少样本示例 | 固定 | 保留少量典型且多样的范例 |
| 动态 | 检索知识 | 每轮随问题与召回策略变化 | 控制候选范围、重排与证据时效 |
| 动态 | 记忆 | 按需注入 | 区分会话草稿与长期记忆，选择性召回 |
| 累积 | 消息历史 | 随任务步数增长 | 裁剪重复内容，复述目标，必要时压缩 |
| 累积 | 工具返回值 | 增长最快，单次结果也可能很大 | 清理原文、外置大对象、保留恢复句柄 |

静态部分包括系统提示、工具定义和少样本示例，它们共同构成相对稳定的任务与动作契约。本页只说明这部分在整体上下文中的位置，重点讨论每轮都会变化或持续累积的信息。

累积部分通常最先失控：网页、日志或 PDF 的一次返回就可能超过真正有用的信息量，早期错误也会随历史反复进入后续推理。工具因此同时影响上下文的两端：

- **工具定义决定动作空间**：名称、参数和职责有重叠时，模型容易选错；如果人类也无法说清两个工具的边界，模型同样难以稳定区分。
- **工具返回决定增长速度**：返回值应优先给结论、结构化字段与可继续读取的句柄，大段原文应留在文件或沙箱中按需读取。

组织信息时还要区分三种状态：用户「希望改成 60 秒」是**意图**，配置文件显示「当前为 30 秒」是**观察**，「问题可能由超时导致」是**推断**。三者不能互相替代；当前、直接且与问题匹配的证据应优先于转述和猜测。

---

## 二、为什么更大的窗口仍不够

窗口容量只回答「能放多少」，不回答「该放什么」。上下文仍需治理，原因有四层；前两层都表现为长输入下的信息利用变差，但机制与排障方向不同。

<div class="pc-constraint-grid">
	<section>
		<span>01</span>
		<h3>注意力预算</h3>
		<p>输入越长，需要处理的 token 关系越多，低信号材料会与关键证据争夺有限的计算与注意力。工程上应问「它是否改变当前决策」，而不只是「它是否相关」。</p>
	</section>
	<section>
		<span>02</span>
		<h3>召回衰减</h3>
		<p>长上下文评测通常呈现渐进的性能下降，而不是到窗口上限才突然失效；位置、信息密度与模型都会影响衰减程度。[[22]](../references.md#source-anthropic-context)</p>
	</section>
	<section>
		<span>03</span>
		<h3>成本与延迟</h3>
		<p>Agent 每一步都会再次携带既有前缀，累积部分不仅占用窗口，还会放大传输、预填和输入计费。应先治理 token 占比最大的来源。</p>
	</section>
	<section>
		<span>04</span>
		<h3>状态腐坏</h3>
		<p>错误、过期、重复和互相矛盾的信息会随循环沉积。扩大窗口不会让这些内容自动一致，这是容量无法解决的问题。[[78]](../references.md#source-breunig-longctx)</p>
	</section>
</div>

因此，「最小」不是越短越好，而是所有保留内容都对当前决策有作用，且必要信息没有缺失。

### 四种失效模式

| 失效模式 | 可观察症状 | 主要处理 |
| --- | --- | --- |
| 中毒 | 已被证伪的结论仍被当作事实反复引用 | 校验工具结果；区分事实与推断；移除错误结论 |
| 干扰 | 历史过长，模型重复旧轨迹或遗忘当前目标 | 裁剪重复内容；把目标、约束和待办复述到近期位置 |
| 混淆 | 从冗余材料中取错依据，或在相似工具间误选 | 收紧召回；裁剪到职责不重叠的最小工具集 |
| 冲突 | 新旧口径同时存在，行为前后不一致 | 显式标明替代关系；保留当前权威来源；污染严重时重建上下文 |

失败证据与错误结论要区别处理：真实发生的失败动作和报错应保留，它们能阻止重复尝试；由失败产生、随后被证伪的解释应删除，否则会形成中毒。

---

## 三、四种治理操作

上下文治理可以归纳为 Select、Isolate、Write 和 Compress。[[80]](../references.md#source-langchain-context)它们不是必须依次执行的流水线，而是针对不同问题的操作：

<figure class="pc-concept-figure">
	<figcaption>四种操作改变窗口的不同边界</figcaption>
	<div class="pc-figure-scroll">
		<svg class="pc-flow-svg pc-ops-svg" viewBox="0 0 900 300" role="img" aria-labelledby="pc-ops-title pc-ops-desc">
			<title id="pc-ops-title">上下文治理的四种操作</title>
			<desc id="pc-ops-desc">Select 控制哪些信息进入窗口，Isolate 把信息分到独立窗口，Write 把信息存到窗口之外，Compress 缩短仍需留在窗口内的信息。</desc>
			<defs>
				<marker id="pc-ops-arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
					<path d="M0 0 L10 5 L0 10 z" class="pc-svg-arrowhead" />
				</marker>
			</defs>
			<rect x="342" y="48" width="216" height="204" rx="9" class="pc-svg-window" />
			<text x="450" y="36" text-anchor="middle" class="pc-svg-kicker">CONTEXT WINDOW</text>
			<rect x="362" y="68" width="176" height="30" rx="4" class="pc-svg-static" />
			<rect x="362" y="108" width="176" height="30" rx="4" class="pc-svg-dynamic" />
			<rect x="362" y="148" width="176" height="30" rx="4" class="pc-svg-cumulative" />
			<rect x="362" y="188" width="176" height="44" rx="4" class="pc-svg-summary" />
			<g class="pc-svg-on-fill">
				<text x="374" y="88">稳定前缀</text>
				<text x="374" y="128">当前证据</text>
				<text x="374" y="168">近期轨迹</text>
				<text x="374" y="207">压缩表示</text>
				<text x="374" y="224">+ 外部句柄</text>
			</g>
			<rect x="28" y="60" width="190" height="64" rx="7" class="pc-svg-operation" />
			<text x="123" y="86" text-anchor="middle" class="pc-svg-operation-name">SELECT</text>
			<text x="123" y="108" text-anchor="middle" class="pc-svg-muted">控制哪些信息进入</text>
			<path d="M224 92 H332" class="pc-svg-arrow" marker-end="url(#pc-ops-arrow)" />
			<rect x="28" y="178" width="190" height="64" rx="7" class="pc-svg-operation" />
			<text x="123" y="204" text-anchor="middle" class="pc-svg-operation-name">WRITE</text>
			<text x="123" y="226" text-anchor="middle" class="pc-svg-muted">存到窗口之外，按需读回</text>
			<path d="M332 210 H224" class="pc-svg-arrow" marker-end="url(#pc-ops-arrow)" />
			<rect x="682" y="60" width="190" height="64" rx="7" class="pc-svg-operation" />
			<text x="777" y="86" text-anchor="middle" class="pc-svg-operation-name">COMPRESS</text>
			<text x="777" y="108" text-anchor="middle" class="pc-svg-muted">缩短仍需保留的信息</text>
			<path d="M672 92 H568" class="pc-svg-arrow-reverse" marker-end="url(#pc-ops-arrow)" />
			<rect x="682" y="178" width="190" height="64" rx="7" class="pc-svg-operation" />
			<text x="777" y="204" text-anchor="middle" class="pc-svg-operation-name">ISOLATE</text>
			<text x="777" y="226" text-anchor="middle" class="pc-svg-muted">拆到独立窗口或状态域</text>
			<path d="M568 210 H672" class="pc-svg-arrow" marker-end="url(#pc-ops-arrow)" />
			<text x="450" y="282" text-anchor="middle" class="pc-svg-muted">先判断问题来自进入、混放、常驻还是体积，再选择操作；优先可逆、可恢复的变化。</text>
		</svg>
	</div>
</figure>

| 操作 | 要解决的问题 | 典型做法 | 主要代价 |
| --- | --- | --- | --- |
| Select（选择） | 不需要的信息进入了当前请求 | 检索与重排、选择性记忆召回、按需暴露工具和状态字段 | 召回可能漏掉关键信息 |
| Isolate（隔离） | 不同任务或证据在同一窗口互相干扰 | 子任务独立窗口、沙箱保存大对象、按 schema 分隔状态 | 增加协调与合并成本 |
| Write（外置） | 信息稍后仍需使用，但不必常驻窗口 | 计划文件、结构化笔记、长期记忆、文件与 URL 句柄 | 需要设计写入、更新与读回时机 |
| Compress（压缩） | 必要历史仍超过预算 | 清理工具原文、裁剪重复消息、生成交接摘要 | 有损，可能丢掉后来才重要的细节 |

选择操作时遵循两个原则：

1. **先移除不需要的信息，再缩写需要的信息。** Select 通常比摘要安全；压缩不是缺少选择策略的补丁。
2. **优先可还原表示。** 网页正文可以移出窗口，但保留 URL；文档内容可以省略，但保留文件路径。只有无法按需恢复时，压缩才意味着永久丢失。

### 动态取用与记忆

| 机制 | 适合的内容 | 优点 | 风险与边界 |
| --- | --- | --- | --- |
| 预取 | 变动慢、几乎每次都需要的项目约束 | 延迟稳定，推理开始时材料已就绪 | 索引和切分可能过期；召回过多会制造混淆 |
| 即时取用 | 规模大、变化快、相关性稀疏的代码与资料 | 只保留路径、查询或链接，支持渐进式披露 | 增加工具往返，依赖搜索原语与启发式质量 |
| 会话草稿 | 当前任务的计划、进度、中间结论 | 更新直接，适合恢复长任务 | 状态不及时更新会产生新的陈旧信息 |
| 长期记忆 | 跨会话稳定的偏好、事实、案例与规则 | 避免反复收集相同信息 | 写入容易、准确召回困难，不应默认全部注入 |

实践中通常混合使用：固定项目说明预先载入，其余材料按需搜索；能由少量固定文件承载的规则，不必急于引入向量检索。目录、文件名、类型和时间等元数据本身也能提供选择信号。[[22]](../references.md#source-anthropic-context)

---

## 四、长任务怎样维持连续性

当任务总信息量必然超过单个窗口时，重点不是保存完整对话，而是保存恢复工作所需的状态：

- **目标与进度**：维护结构化计划、已完成项和下一步，并周期性放回上下文末尾。
- **决策与未决问题**：记录已经采用的方案、放弃原因、仍未解决的错误和验证条件。
- **证据句柄**：大对象写入文件或沙箱，只在上下文中保留路径、查询、版本和必要摘要。
- **交接摘要**：触发压缩时优先保留目标、约束、决策、未决问题、最近状态和恢复入口；从清理工具原始返回开始，再逐步采用更有损的摘要。
- **隔离探索**：只有子任务可独立调查、工具或信息边界确实不同，且结果能用明确协议合并时，才使用子 Agent；它是 Isolate 的实现，不是默认架构。

压缩的触发条件、摘要保留项与恢复协议见[上下文压缩](./compaction.md)；跨会话状态的写入与召回见[Agent 记忆](./memory.md)。长期记忆只保存跨会话仍稳定且值得复用的信息，记忆的难点通常不在写入，而在以后能否准确召回。

---

## 五、缓存、度量与排障

如果模型服务支持前缀缓存，稳定前缀可以减少重复预填带来的延迟与成本。具体命中条件、缓存期限和计费方式取决于提供方，不能把某一产品的单价或实现当作通用规律。上下文侧只需守住三点：稳定内容放在前部；动态内容尽量后置并只追加；序列化保持确定性。中途修改前部的工具定义，不仅会使后续缓存失效，还可能与历史里的旧工具调用产生冲突；动态工具集应结合提供方能力选择描述检索、分组启用或受限解码，而不是假设存在通用的「遮蔽」机制。缓存本身的计算语义见[模型交互](../model/llm-api.md#cache)。[[79]](../references.md#source-manus-context)

优化前必须先有完整轨迹和固定评测集，否则无法判断减少 token 是否损害了任务完成率。核心指标不需要很多：

| 层面 | 指标 | 用途 |
| --- | --- | --- |
| 任务 | 完成率、平均步数、错误恢复率 | 判断系统是否真的更可靠 |
| 上下文 | 每步与峰值 token、各类内容占比 | 找到膨胀来源 |
| 召回与压缩 | 无用召回占比、关键信息保留率、压缩后失败率 | 发现 Select 或 Compress 的副作用 |
| 服务 | 前缀缓存命中率、首字延迟、输入成本 | 验证稳定前缀与缩减输入的收益 |

排障按同一条主线推进：

1. 查看完整轨迹，确认每一步模型实际看到了什么。
2. 按 token 占比找到异常膨胀的来源。
3. 用中毒、干扰、混淆、冲突归类症状。
4. 先做可逆改动：收紧 Select、隔离无关材料、外置大对象；预算仍不足时再压缩。
5. 在固定评测集上做单变量实验，并把修复案例加入回归集。

上下文优化的最终判据不是 token 越少越好，而是在完成率不下降的前提下，以更少的步骤、输入和延迟完成任务。

---

## 参考文献

本文依赖的来源见[参考资料](../references.md)，主要为[上下文工程](../references.md#source-anthropic-context)、[四种治理操作](../references.md#source-langchain-context)、[生产实践](../references.md#source-manus-context)与[长上下文失效分类](../references.md#source-breunig-longctx)。
