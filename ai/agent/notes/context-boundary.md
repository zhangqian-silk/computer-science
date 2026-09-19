---
aside: false
---

# Context、Memory 与 Knowledge 的边界

LLM 调用不保存会话状态。模型在第 N 次调用中能够使用的信息，都必须由调用方写入本次输入；历史、个性化信息和共享资料需要在模型外部维护，并在调用时取回。Context、memory、knowledge 分别表示当轮输入、主体状态和共享资料。

<div class="cmk-intro">
	<article class="cmk-intro__thesis">
		<h2>基本关系</h2>
		<p>Context 是一次调用实际发送的输入；memory 保存绑定主体、可随交互变化的信息；knowledge 保存不绑定主体、按权限共享的资料。本文只讨论三者边界；两类外部信息的内部实现分别在后续专题展开。</p>
	</article>
</div>

---

## 从无状态调用推出三者

一次 LLM 调用可近似视为从输入序列到输出序列的无状态函数。模型权重在请求之间保持只读，运行时不为某次会话保留可被下一次调用直接访问的隐式状态；第 N 次调用面对的模型与第 1 次相同。历史、偏好、文档和工具结果必须由调用方在本次输入中显式提供。

所谓「模型记得刚才说过」，本质上是调用方将历史消息重新放入当前请求。模型连续性的来源不是隐式状态延续，而是外部程序对输入序列的重新构造。该输入序列即 context。

<div class="cmk-derivation">
	<span class="cmk-kind cmk-kind--context">CONTEXT · 当轮输入</span>
	<p>真正发送给模型的 token 序列，包括系统指令、工具定义、召回材料、会话历史、工具观察和当轮输入。它在每轮请求中重新构造，主要决定放入哪些内容、采用何种顺序以及占用多少 token。</p>
</div>

如果上下文窗口不受容量和成本限制，且模型能够充分利用任意长度的输入，每次请求直接携带全部历史即可。实际系统受到三方面约束：

<div class="cmk-constraints">
	<article>
		<span>01</span>
		<h3>容量与成本</h3>
		<p>全量重放历史会使 token 数和延迟随轮次增加，而且多数历史内容与当前请求无关。</p>
		<b>需要外部筛选</b>
	</article>
	<article>
		<span>02</span>
		<h3>有效利用率</h3>
		<p>输入变长后，模型不一定能稳定使用其中的信息；相关内容可能被淹没在长上下文中。</p>
		<b>需要信息提炼</b>
	</article>
	<article>
		<span>03</span>
		<h3>跨会话存续</h3>
		<p>请求结束后 context 不复存在，但用户偏好、项目约定和共享文档需要独立于单次会话持续存在。</p>
		<b>需要外部持久化</b>
	</article>
</div>

上述约束只能推出 context 之外需要持久化信息，不能推出所有外部信息应放入同一个库。按照主体归属和更新方式继续划分，外部信息分为 memory 与 knowledge 两类。

<div class="cmk-defs cmk-defs--split">
	<article class="cmk-card cmk-card--memory">
		<div class="cmk-card__head">
			<span class="cmk-kind cmk-kind--memory">MEMORY</span>
			<h3>主体归属</h3>
		</div>
		<div class="cmk-card__body">
			<p>用户偏好、团队约定和历史经历只对特定主体成立，需要按用户、团队、Agent 或线程隔离，并在后续交互中更新或失效。</p>
			<p class="cmk-card__meta">主体隔离 · 更新 / 失效 · 时间范围</p>
		</div>
	</article>
	<article class="cmk-card cmk-card--knowledge">
		<div class="cmk-card__head">
			<span class="cmk-kind cmk-kind--knowledge">KNOWLEDGE</span>
			<h3>更新语义</h3>
		</div>
		<div class="cmk-card__body">
			<p>产品手册、代码库和制度文档面向有权限的用户共享，一般不在线改写，主要通过文档更新、版本替换和索引重建维护。</p>
			<p class="cmk-card__meta">文档 ACL · 版本替换 · 重建索引 / 来源追溯</p>
		</div>
	</article>
</div>

由此可以先得到两条边界：一类信息绑定主体，并可能随交互失效；另一类信息按权限共享，并随来源版本更新。二者的正式定义和维护方式见下一节；在参与模型计算前，它们都必须经过召回和装配进入 context。

---

## 概念定义与边界对照

区分三者时，存储介质不是主要依据，关键是信息的作用范围、生命周期和维护方式。

<div class="cmk-decision">
	<section class="cmk-step">
		<b>1. Context：当轮输入</b>
		<p>系统指令、工具定义、召回片段、会话历史、工具观察和当轮输入共同组成本次请求的 token 序列。它的存续范围限于单次调用。</p>
	</section>
	<section class="cmk-step">
		<b>2. Memory：主体状态</b>
		<p>用户、团队、Agent 或线程拥有的偏好、事实、经历和约定独立于单次调用存在，须按主体隔离，并通过更新、失效、冲突消解和删除治理。</p>
	</section>
	<section class="cmk-step">
		<b>3. Knowledge：共享资料</b>
		<p>产品文档、代码库、制度和 Wiki 等资料按权限共享，通常经离线同步、版本替换和索引重建维护；回答须忠实于来源并可追溯。</p>
	</section>
</div>

同一对象中可以同时存在主体状态和共享资料，应按信息单元拆分，不应按文件名整体归类。团队文档中的通用流程属于 knowledge；文档形成时的决策依据和历史约束，则可能属于团队 memory。

<div class="cmk-callout">
	<p><strong>载体不决定类别。</strong>同一份项目文件常驻注入时，可以作为约束 Agent 的程序性 memory；按需检索时，则属于 knowledge。代码库索引、提示词和向量库都只是承载或装配手段。</p>
</div>

<div class="cmk-table-wrap" tabindex="0">
	<table class="cmk-table">
		<thead>
			<tr>
				<th>维度</th>
				<th>Context</th>
				<th>Memory</th>
				<th>Knowledge</th>
			</tr>
		</thead>
		<tbody>
			<tr>
				<th>本体</th>
				<td>本次调用的 token 序列</td>
				<td>绑定主体的持久条目</td>
				<td>源文档、源代码及其索引投影</td>
			</tr>
			<tr>
				<th>生命周期</th>
				<td>一次调用；每轮重建</td>
				<td>长于单次会话，可跨会话存在</td>
				<td>与来源和版本同寿</td>
			</tr>
			<tr>
				<th>作用域</th>
				<td>本次请求</td>
				<td><code>user_id</code>、<code>agent_id</code>、<code>thread_id</code></td>
				<td>共享语料，按文档 ACL 或租户权限控制</td>
			</tr>
			<tr>
				<th>维护责任</th>
				<td>装配代码</td>
				<td>由用户或 Agent 在交互中产生，系统控制写入条件</td>
				<td>离线导入、索引流水线或内容运营</td>
			</tr>
			<tr>
				<th>正确性判据</th>
				<td>内容完整、顺序合理且不超过预算</td>
				<td>当前是否仍然成立</td>
				<td>是否忠实于来源的当前版本</td>
			</tr>
			<tr>
				<th>更新方式</th>
				<td>无需更新，下一轮重新装配</td>
				<td>ADD / UPDATE / DELETE / NOOP，配合冲突消解</td>
				<td>版本替换、增量索引或整体 reindex</td>
			</tr>
			<tr>
				<th>主要治理动作</th>
				<td>分区、滑窗、摘要压缩、工具结果清理、缓存排序</td>
				<td>写入控制、时效、置信度、可编辑、按主体级联删除</td>
				<td>切分、引用溯源、ACL、索引更新、忠实度校验</td>
			</tr>
			<tr>
				<th>成本形态</th>
				<td>随轮次重复支付 token 与延迟成本</td>
				<td>写入抽取、判重和存储成本较高，召回量通常较小</td>
				<td>离线索引一次性投入大，边际检索成本较低</td>
			</tr>
			<tr>
				<th>错误代价</th>
				<td>主要影响本轮，下轮可修正装配</td>
				<td>可能跨会话反复生效</td>
				<td>回答偏离来源，可按引用追溯</td>
			</tr>
		</tbody>
	</table>
</div>

### 维护方式

Memory 维护主体状态。系统从交互中抽取候选信息，经过价值判断、判重和冲突处理后写入；之后通过更新、失效、衰减和删除保持其有效性。召回时必须先按主体过滤，再排除失效条目，并向用户提供查看、编辑和删除入口。

Knowledge 维护来源资料。系统从文档、代码库或 Wiki 同步内容，经过解析、切分、元数据抽取和索引构建形成可检索片段；来源更新后，通过版本替换、增量索引或重建索引更新。检索时按文档权限过滤，回答中保留来源和引用位置。

二者的删除语义不同。知识库通常按文档或索引版本删除；记忆删除需要覆盖主记录、向量索引、摘要和派生副本。删除原始对话，并不等于删除由对话抽取出的记忆。

---

## 两个数据源和一个装配层

Memory 和 knowledge 分开维护、分别召回；context 决定本轮注入哪些片段、分配多少 token 以及采用何种顺序。外部数据源中的内容不会直接影响模型，只有进入本次输入后才参与计算。调用结束后 context 消失，memory 和 knowledge 仍然存在。

<div class="cmk-svg-scroll">
<svg class="cmk-svg" viewBox="0 0 920 500" role="img" aria-labelledby="cmk-fig1-title cmk-fig1-desc">
<title id="cmk-fig1-title">Context、Memory 与 Knowledge 的层次关系</title>
<desc id="cmk-fig1-desc">Memory 与 Knowledge 是两个持久数据源，分别通过召回链路进入每轮重建的 Context，再发送给一次无状态 LLM 调用。</desc>
<rect class="fill-bg" x="0" y="0" width="920" height="500"/>
<text class="mono fill-subtle" x="14" y="24" font-size="11" letter-spacing="1.2">DATA SOURCES / 数据源层 · 生命周期长于单次调用</text>
<rect class="fill-bg memory-stroke" x="14" y="38" width="288" height="132" stroke-width="1.6"/>
<rect class="memory-fill" x="14" y="38" width="288" height="28"/>
<text class="memory-text" x="28" y="57" font-size="13" font-weight="700">MEMORY · 记忆</text>
<text class="fill-muted" x="28" y="83" font-size="11.5">绑定主体：user / agent / thread</text>
<text class="fill-muted" x="28" y="103" font-size="11.5">Agent 自读写，交互中产生</text>
<text class="fill-muted" x="28" y="123" font-size="11.5">带时效：valid_from / valid_to</text>
<text class="fill-muted" x="28" y="143" font-size="11.5">判据：是否仍然成立</text>
<text class="memory-text" x="28" y="163" font-size="10.8">偏好 · 项目约定 · 共同经历 · 待办</text>
<rect class="fill-bg knowledge-stroke" x="618" y="38" width="288" height="132" stroke-width="1.6"/>
<rect class="knowledge-fill" x="618" y="38" width="288" height="28"/>
<text class="knowledge-text" x="632" y="57" font-size="13" font-weight="700">KNOWLEDGE · 知识</text>
<text class="fill-muted" x="632" y="83" font-size="11.5">不绑定主体：共享语料</text>
<text class="fill-muted" x="632" y="103" font-size="11.5">离线灌入，线上只读</text>
<text class="fill-muted" x="632" y="123" font-size="11.5">版本语义：reindex 重建</text>
<text class="fill-muted" x="632" y="143" font-size="11.5">判据：是否忠实于源文档</text>
<text class="knowledge-text" x="632" y="163" font-size="10.8">产品手册 · 代码库 · 制度文档 · Wiki</text>
<rect class="fill-soft stroke-border" x="326" y="58" width="268" height="92" stroke-dasharray="4 3"/>
<text class="fill-text" x="460" y="84" text-anchor="middle" font-size="12" font-weight="700">两条独立召回链路</text>
<text class="fill-muted" x="460" y="106" text-anchor="middle" font-size="11">各自打分 · 各自预算 · 各自 rerank</text>
<text class="fill-muted" x="460" y="128" text-anchor="middle" font-size="11">注入时按来源分区标注</text>
<path class="memory-line" d="M158 170 L158 202 L400 202 L400 218" stroke-width="1.6"/>
<polygon class="memory-fill memory-text" points="400,225 395,215 405,215"/>
<path class="knowledge-line" d="M762 170 L762 202 L520 202 L520 218" stroke-width="1.6"/>
<polygon class="knowledge-fill knowledge-text" points="520,225 515,215 525,215"/>
<text class="mono fill-subtle" x="14" y="250" font-size="11" letter-spacing="1.2">ASSEMBLY LAYER / 装配层 · 每轮重建 · 生命周期 = 一次调用</text>
<rect class="fill-bg stroke-strong" x="14" y="260" width="892" height="132" stroke-width="1.8" stroke-dasharray="6 4"/>
<text class="fill-text" x="30" y="284" font-size="13" font-weight="700">CONTEXT · 本次调用的输入装配</text>
<rect class="fill-soft stroke-border" x="30" y="298" width="160" height="42"/>
<text class="fill-text" x="110" y="316" text-anchor="middle" font-size="11" font-weight="700">系统指令 / 工具定义</text>
<text class="fill-subtle" x="110" y="332" text-anchor="middle" font-size="10">最稳定，放最前</text>
<rect class="memory-fill memory-stroke" x="200" y="298" width="160" height="42"/>
<text class="memory-text" x="280" y="316" text-anchor="middle" font-size="11" font-weight="700">召回的记忆</text>
<text class="memory-text" x="280" y="332" text-anchor="middle" font-size="10">来自记忆数据源</text>
<rect class="knowledge-fill knowledge-stroke" x="370" y="298" width="160" height="42"/>
<text class="knowledge-text" x="450" y="316" text-anchor="middle" font-size="11" font-weight="700">检索到的知识</text>
<text class="knowledge-text" x="450" y="332" text-anchor="middle" font-size="10">来自知识数据源</text>
<rect class="fill-soft stroke-border" x="540" y="298" width="176" height="42"/>
<text class="fill-text" x="628" y="316" text-anchor="middle" font-size="11" font-weight="700">会话历史 / 工具结果</text>
<text class="fill-subtle" x="628" y="332" text-anchor="middle" font-size="10">滑窗 · 压缩 · 清理</text>
<rect class="fill-soft stroke-border" x="726" y="298" width="160" height="42"/>
<text class="fill-text" x="806" y="316" text-anchor="middle" font-size="11" font-weight="700">当轮用户输入</text>
<text class="fill-subtle" x="806" y="332" text-anchor="middle" font-size="10">每轮唯一必新</text>
<text class="fill-muted" x="30" y="364" font-size="11">五块内容共享同一个 token 预算；context 自己不存储内容，只决定放什么、放多少、按什么顺序放</text>
<text class="fill-muted" x="30" y="383" font-size="11">稳定内容放前，可变内容放后，以提高 prompt cache 的前缀复用概率 [3]</text>
<path class="stroke-strong" d="M460 392 L460 416" stroke-width="1.8"/>
<polygon class="fill-text" points="460,423 454,413 466,413"/>
<rect class="fill-text" x="326" y="428" width="268" height="36"/>
<text class="fill-bg" x="460" y="451" text-anchor="middle" font-size="13" font-weight="700">一次无状态的 LLM 调用</text>
<text class="danger-text" x="460" y="486" text-anchor="middle" font-size="12" font-weight="700">调用结束，装配层消失；两侧数据源不受影响</text>
</svg>
</div>

会话历史常被称为「短期记忆」或「工作记忆」，但在结构上仍属于 context 中的会话状态。只有从历史中抽出独立于会话存在的偏好、事实或约定并写入记忆库后，它才成为 memory。

Prompt cache 和 KV cache 也不是信息来源。Prompt cache 复用请求前缀的计算结果，KV cache 复用模型推理时的中间状态；缓存命中不会增加输入内容，只影响延迟和计算量。

---

## 一次调用中的装配过程

一次 Agent 调用通常并行准备三类信息：记忆召回、知识检索和业务系统实时查询。前两者来自持久化信息源，工具查询返回外部系统在当前时刻的状态。订单状态、物流、库存、余额和权限结果等强一致数据不写入 memory，也不作为 knowledge 保存，而是作为本轮工具结果 observation 注入。

<pre class="cmk-flow-code"><code>用户输入
│
├─→ 记忆召回（memory recall）
│      按主体取回仍有效的个性化信息
│
├─→ 知识检索（knowledge retrieval）
│      按文档权限取回当前版本的共享资料
│
└─→ 实时工具观察（tool observation）
       读取订单、库存、余额等强一致状态

装配进 context：
  [1] 系统指令 + 工具定义        ← 最稳定，放最前
  [2] 常驻规则 / 项目约定         ← 会话内不变
  [3] 检索到的知识               ← 标注来源与版本
  [4] 召回的记忆                 ← 标注主体与时效
  [5] 会话历史                   ← 滑窗 / 压缩后
  [6] 当轮输入 + 实时 observation ← 每轮必新，放最后</code></pre>

记忆召回和知识检索应分别设置预算与监控指标。共用一个 top-k 时，较长的知识片段可能占用记忆配额，也不利于定位异常来源。某一路超时时，可以在缺少该路信息的情况下继续回答，但不应阻断整个请求。

<div class="cmk-callout cmk-callout--warning">
	<p><strong>实时状态属于工具观察。</strong>「用户上个月住在哪里」可以进入 memory；「订单当前是否已支付」必须查询业务系统。前者描述主体历史，后者反映外部系统当前状态。</p>
</div>

---

## 检索设施相同，治理逻辑不同

Memory 和 knowledge 可以共用 embedding、向量库、BM25、混合排序和 rerank。相同点主要在检索技术层面；条目结构、权限控制、失效方式、删除方式和可信度治理不能混用。

RAG 早期论文将参数外的可检索索引称为 non-parametric memory，这是检索文献中的术语。在 Agent 工程中，仍应按信息内容区分共享 knowledge 与绑定主体的 memory，不能只按术语归类。

早期系统可以共用向量库，但命名空间、主体过滤、条目结构和删除链路必须分离。物理设施可以复用，治理逻辑不能混用；否则按主体删除和按文档重建会相互干扰。

<div class="cmk-svg-scroll">
<svg class="cmk-svg" viewBox="0 0 920 486" role="img" aria-labelledby="cmk-fig2-title cmk-fig2-desc">
<title id="cmk-fig2-title">共享设施层与上层治理分叉</title>
<desc id="cmk-fig2-desc">Memory、Context 和 Knowledge 在数据模型层各自独立；Memory 与 Knowledge 共用检索设施；三者最终都进入同一个 context window。</desc>
<rect class="fill-bg" x="0" y="0" width="920" height="486"/>
<text class="mono danger-text" x="14" y="24" font-size="11" letter-spacing="1.2">LAYER 3 · 各自独有：数据模型与治理</text>
<rect class="fill-bg memory-stroke" x="14" y="36" width="288" height="140" stroke-width="1.5"/>
<rect class="memory-fill" x="14" y="36" width="288" height="26"/>
<text class="memory-text" x="28" y="54" font-size="12" font-weight="700">MEMORY 独有</text>
<text class="fill-muted" x="28" y="78" font-size="10.8">· type / scope / confidence</text>
<text class="fill-muted" x="28" y="97" font-size="10.8">· valid_from / valid_to</text>
<text class="fill-muted" x="28" y="116" font-size="10.8">· provenance 溯源到源消息</text>
<text class="fill-muted" x="28" y="135" font-size="10.8">· ADD / UPDATE / DELETE / NOOP</text>
<text class="fill-muted" x="28" y="154" font-size="10.8">· 冲突消解 · 衰减 · 软删除审计</text>
<text class="fill-muted" x="28" y="173" font-size="10.8">· 按主体级联删除</text>
<rect class="fill-bg stroke-strong" x="316" y="36" width="288" height="140" stroke-width="1.5"/>
<rect class="fill-soft" x="316" y="36" width="288" height="26"/>
<text class="fill-text" x="330" y="54" font-size="12" font-weight="700">CONTEXT 独有</text>
<text class="fill-muted" x="330" y="78" font-size="10.8">· 本轮 token 预算分配</text>
<text class="fill-muted" x="330" y="97" font-size="10.8">· 分区顺序与缓存前缀</text>
<text class="fill-muted" x="330" y="116" font-size="10.8">· 滑窗 / 摘要压缩 / 工具清理</text>
<text class="fill-muted" x="330" y="135" font-size="10.8">· 来源标注与优先级</text>
<text class="fill-muted" x="330" y="154" font-size="10.8">· 预算内截断</text>
<text class="fill-muted" x="330" y="173" font-size="10.8">· 无持久化，无失效语义</text>
<rect class="fill-bg knowledge-stroke" x="618" y="36" width="288" height="140" stroke-width="1.5"/>
<rect class="knowledge-fill" x="618" y="36" width="288" height="26"/>
<text class="knowledge-text" x="632" y="54" font-size="12" font-weight="700">KNOWLEDGE 独有</text>
<text class="fill-muted" x="632" y="78" font-size="10.8">· 文档 → 切分 → 向量</text>
<text class="fill-muted" x="632" y="97" font-size="10.8">· 文档元数据与 ACL</text>
<text class="fill-muted" x="632" y="116" font-size="10.8">· 版本管理与增量索引</text>
<text class="fill-muted" x="632" y="135" font-size="10.8">· 引用与原文位置</text>
<text class="fill-muted" x="632" y="154" font-size="10.8">· faithfulness 校验</text>
<text class="fill-muted" x="632" y="173" font-size="10.8">· 源删则索引删</text>
<line class="danger-line" x1="14" y1="190" x2="906" y2="190" stroke-dasharray="7 4"/>
<text class="danger-text" x="460" y="207" text-anchor="middle" font-size="12" font-weight="700">分界线以上互不通用，以下共享检索或出口</text>
<text class="mono fill-subtle" x="14" y="232" font-size="11" letter-spacing="1.2">LAYER 2 · memory 与 knowledge 共用：检索设施</text>
<rect class="fill-soft stroke-border" x="14" y="242" width="892" height="86"/>
<rect class="fill-bg stroke-border" x="34" y="260" width="192" height="34"/>
<text class="fill-text" x="130" y="282" text-anchor="middle" font-size="11.5" font-weight="700">embedding 模型</text>
<rect class="fill-bg stroke-border" x="242" y="260" width="192" height="34"/>
<text class="fill-text" x="338" y="282" text-anchor="middle" font-size="11.5" font-weight="700">向量库 · 近邻检索</text>
<rect class="fill-bg stroke-border" x="450" y="260" width="192" height="34"/>
<text class="fill-text" x="546" y="282" text-anchor="middle" font-size="11.5" font-weight="700">BM25 + 融合排序</text>
<rect class="fill-bg stroke-border" x="658" y="260" width="228" height="34"/>
<text class="fill-text" x="772" y="282" text-anchor="middle" font-size="11.5" font-weight="700">rerank · 去冗 · 截断</text>
<text class="fill-muted" x="460" y="316" text-anchor="middle" font-size="11">技术选型可复用，数据模型与治理不可复用</text>
<text class="mono fill-subtle" x="14" y="356" font-size="11" letter-spacing="1.2">LAYER 1 · 三者共用：唯一出口</text>
<rect class="fill-elevated stroke-border" x="14" y="366" width="892" height="66"/>
<text class="fill-text" x="460" y="392" text-anchor="middle" font-size="12.5" font-weight="700">context window —— 最终都变成同一段 token 序列</text>
<text class="fill-muted" x="460" y="414" text-anchor="middle" font-size="11">模型不能仅凭文本形式区分权威文档与用户偏好，因此必须显式标注来源</text>
<rect class="fill-text" x="14" y="446" width="892" height="30"/>
<text class="fill-bg" x="460" y="466" text-anchor="middle" font-size="12" font-weight="700">相似性来自共享下层，差异性只存在于上层模型与治理</text>
</svg>
</div>

材料进入 context 后都会转换为文本，模型无法仅凭格式判断来源和可信度。因此注入时需要标明来源、时间和权威性，并规定冲突处理顺序。通常由当轮指令确定任务要求，由当前权威知识约束事实答案，由 memory 提供个性化背景；用户明确纠正旧信息时，应触发复核或更新。

---

## 边界示例

<div class="cmk-cases">
	<article class="cmk-case">
		<h4>常驻项目指令</h4>
		<p><span class="cmk-badge cmk-badge--memory">按使用判定</span>每轮强制注入的项目约定可以作为约束 Agent 的程序性 memory；同一份文件仅在按需检索时阅读，则按 knowledge 使用。分类依据不是文件名，而是装配路径和适用主体。</p>
	</article>
	<article class="cmk-case">
		<h4>代码库语义索引</h4>
		<p><span class="cmk-badge cmk-badge--knowledge">KNOWLEDGE</span>其来源是代码提交，索引由流水线重建，不绑定个人，也不需要逐条失效。它是代码知识的检索投影，不应称为 Agent 的长期记忆。</p>
	</article>
	<article class="cmk-case">
		<h4>自动生成的 Wiki</h4>
		<p><span class="cmk-badge cmk-badge--knowledge">拆分判定</span>能从源码重新生成的接口说明是 knowledge；「团队当时为什么放弃某个方案」这类不能由代码反推的决策史，具有集体记忆属性。</p>
	</article>
	<article class="cmk-case">
		<h4>组织级规则与团队决策</h4>
		<p><span class="cmk-badge cmk-badge--memory">集体记忆</span>组织规则或团队决策有主体、负责人和失效时间，虽然名称常叫 Knowledge，治理方式更接近集体程序性记忆。能由源码重新生成的说明归 knowledge，无法从现状反推的决策史归 memory。</p>
	</article>
	<article class="cmk-case">
		<h4>压缩摘要与历史消息</h4>
		<p><span class="cmk-badge cmk-badge--context">CONTEXT</span>摘要服务于当前会话的后续装配，本体仍是会话历史。它是有损压缩结果，不是独立记忆；需要跨会话生效的内容应显式写入 memory。</p>
	</article>
	<article class="cmk-case">
		<h4>Prompt cache 与 KV cache</h4>
		<p><span class="cmk-badge cmk-badge--context">计算优化</span>两类缓存都不引入新信息。Prompt cache 复用请求前缀，KV cache 复用模型计算；它们只改变成本与延迟，不改变信息归属。</p>
	</article>
</div>

---

## 参考来源

1. Anthropic. [Effective context engineering for AI agents](https://www.anthropic.com/engineering/effective-context-engineering-for-ai-agents)。
2. Lewis, P. et al. (2020). [Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks](https://arxiv.org/abs/2005-11401)。
3. Anthropic. [How Claude Code uses prompt caching](https://code.claude.com/docs/en/prompt-caching)。
