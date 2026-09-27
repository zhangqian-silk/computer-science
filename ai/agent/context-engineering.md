---
aside: false
---

# Agent 上下文工程

上下文工程是 Agent 根据当前任务选择模型可见信息、构造本轮输入，并用调用结果准备下一轮输入的过程。它解决的是：任务状态保存在模型之外时，本轮需要让模型看到什么，行动之后又该更新什么。输入既要覆盖目标、约束、可用能力和必要证据，也要符合权限、时效与窗口预算。

---

## 无状态调用与工作集循环

从应用管理任务状态的角度看，模型调用是无状态的：本次没有提供的历史、文件和工具结果，模型不能自动读取。即使服务托管历史，调用方仍须明确模型本轮实际可见的内容。一次调用中实际提供的消息、工具定义及其他可见输入，称为 Context；保存任务记录与构造 Context 是两件事。

无状态不意味着所有请求都要多轮。可以直接回答的任务只需一次调用；需要读取外部信息、执行动作并核对结果时，才形成从候选信息构造本轮输入、执行后记录观察、再构造下一轮输入的循环。

<figure class="ce-figure">
	<div class="ce-figure__scroll" tabindex="0" aria-label="逐轮上下文装配图，可横向滚动">
		<svg class="ce-svg" viewBox="0 0 940 300" role="img" aria-labelledby="ce-cycle-title ce-cycle-desc">
			<title id="ce-cycle-title">候选信息进入本轮上下文，工具观察进入下一轮</title>
			<desc id="ce-cycle-desc">左侧是规则、目标、资料、记忆与行动轨迹。通过权限和时效过滤及选择后组成当轮上下文，交给模型产生回复或工具调用。只有工具执行的观察经校验后回流，下轮重新装配。</desc>
			<defs>
				<marker id="ce-cycle-arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto">
					<path d="M0 0 L10 5 L0 10 Z" fill="var(--cs-color-brand)"/>
				</marker>
			</defs>
			<text x="20" y="29" class="ce-svg__head">候选信息 · 可长期保存</text>
			<g class="ce-svg__box">
				<rect x="20" y="49" width="116" height="39"/><rect x="146" y="49" width="116" height="39"/>
				<rect x="20" y="97" width="116" height="39"/><rect x="146" y="97" width="116" height="39"/>
				<rect x="20" y="145" width="242" height="39"/>
			</g>
			<text x="34" y="74">规则与工具</text><text x="160" y="74">当前目标</text>
			<text x="34" y="122">资料与记忆</text><text x="160" y="122">近期轨迹</text>
			<text x="34" y="170">工具观察 · 验证记录</text>
			<path class="ce-svg__arrow" d="M269 137 H321" marker-end="url(#ce-cycle-arrow)"/>
			<text x="295" y="124" text-anchor="middle" class="ce-svg__small ce-svg__brand">筛选</text>
			<rect x="332" y="49" width="219" height="185" fill="var(--cs-color-bg)" stroke="var(--cs-color-text)" stroke-width="1.5"/>
			<text x="348" y="73" class="ce-svg__head">本轮 Context</text>
			<rect x="348" y="84" width="187" height="28" class="ce-svg__band--rule"/>
			<rect x="348" y="121" width="187" height="28" class="ce-svg__band--capability"/>
			<rect x="348" y="158" width="187" height="28" class="ce-svg__band--evidence"/>
			<rect x="348" y="195" width="187" height="28" class="ce-svg__band--history"/>
			<text x="361" y="103" class="ce-svg__on-brand">适用规则与能力</text>
			<text x="361" y="140">目标与任务状态</text>
			<text x="361" y="177">证据、来源与版本</text>
			<text x="361" y="214">必要的行动轨迹</text>
			<path class="ce-svg__arrow" d="M561 142 H596" marker-end="url(#ce-cycle-arrow)"/>
			<circle cx="646" cy="142" r="42" fill="var(--cs-color-bg-elevated)" stroke="var(--cs-color-border-strong)" stroke-width="1.5"/>
			<text x="646" y="148" text-anchor="middle" class="ce-svg__head">模型</text>
			<path class="ce-svg__arrow" d="M690 128 L747 105" marker-end="url(#ce-cycle-arrow)"/>
			<path class="ce-svg__arrow" d="M690 154 L747 173" marker-end="url(#ce-cycle-arrow)"/>
			<g class="ce-svg__box">
				<rect x="759" y="85" width="158" height="40"/><rect x="759" y="153" width="158" height="40"/>
			</g>
			<text x="776" y="111">回复／交付</text><text x="776" y="179">执行工具</text>
			<path class="ce-svg__feedback" d="M838 198 V269 H139 V192" marker-end="url(#ce-cycle-arrow)"/>
			<text x="480" y="259" text-anchor="middle" class="ce-svg__small ce-svg__brand">校验观察与更新状态 → 下一轮重新装配</text>
		</svg>
	</div>
	<figcaption>定性示意：候选信息留在窗口外，只有当轮选中的内容进入模型输入。</figcaption>
</figure>

例如修复订单接口的失败测试，第一轮要读取代码和测试结果，修改后还要依据真实的复测结果判断是否完成。原始目标贯穿各轮，但失败日志、待验证原因与验收状态会随观察变化；不能只把新消息不断追加到旧输入后面。

---

## 当轮输入与外部状态

图中的 Context 属于一次调用的短暂工作集；完整资料、任务记录和可复用状态留在窗口外。窗口外的信息按归属和更新责任区分：

<dl class="ce-boundary">
	<div class="ce-boundary__item">
		<dt>任务轨迹与状态</dt>
		<dd>轨迹记录动作、工具结果和用户修正；状态提炼当前进度。二者留在运行时，下一轮按需投影到输入中。</dd>
	</div>
	<div class="ce-boundary__item">
		<dt>Memory</dt>
		<dd>与用户、团队或任务主体关联且可能随交互更新的信息。装入前核对主体、授权范围和是否仍然有效。</dd>
	</div>
	<div class="ce-boundary__item">
		<dt>Knowledge</dt>
		<dd>代码、文档等随来源版本维护的共享资料。装入前核对权限、来源和版本。</dd>
	</div>
</dl>

Memory 和 Knowledge 即使采用相同存储方式，也不能只按载体区分：前者随主体变化，后者随源材料更新。以修复接口为例，代码与接口文档属于 Knowledge，用户持续有效的工作偏好属于 Memory，复测是否通过则是工具刚返回的观察。它们只有经过授权和有效性检查、进入本轮请求后，才成为 Context。

由此，窗口大小只限制一次能提供多少信息，并不能产生缺失的事实。若任务所需资料不可访问，或验证尚未执行，就应获取信息、执行核对或明确未知，而不是用模型推断填补空白。

---

## 工作集的输入来源

图中的候选信息还需按本轮任务取舍：约束决定能做什么，工具与技能说明可采取的动作，资料与观察提供依据，状态与轨迹维持进度。模块并非每轮都要全量出现，装载条件也不同。

<table class="ce-table">
	<thead>
		<tr><th scope="col">来源</th><th scope="col">装入本轮的条件</th></tr>
	</thead>
	<tbody>
		<tr><th scope="row">指令与示例（Prompt）</th><td>保留本轮适用的规则、输出约束和必要示例，避免旧示例挤占证据空间。</td></tr>
		<tr><th scope="row">工具（Tools）</th><td>提供本轮可能调用的能力及参数契约；近义工具过多会占据窗口并增加误选，执行权限仍由运行时控制。</td></tr>
		<tr><th scope="row">技能（Skills）</th><td>先提供可发现的短索引，任务命中后再装入所需流程与验收步骤；流程不会自动授予工具权限。</td></tr>
		<tr><th scope="row">知识与记忆</th><td>分别按来源版本与主体范围筛选；只装入可核对且相关的片段，避免旧信息冒充当前事实。</td></tr>
		<tr><th scope="row">轨迹与状态</th><td>保留当前阶段、尚未解决的问题和必要的调用—结果关系；完整旧日志留在窗口外并保留恢复入口。</td></tr>
		<tr><th scope="row">当前事件</th><td>包含用户新要求或工具新观察，并明确它替代、补充还是推翻已有状态；观察内容不取得指令权限。</td></tr>
	</tbody>
</table>

资料、技能正文和工具目录可以预先装入，也可以先提供入口、命中时再读取。预载减少中途往返，但消耗输入预算并引入干扰；按需读取降低当轮负担，却增加延迟和漏查风险。稳定且常用的内容适合较早提供，体量大、偶尔才相关的内容适合延迟装入；选择取决于任务分布与读取成本。

---

## 选择、排列与状态更新

来源确定后，装配仍要以当前决策为起点：定位失败需要报错及相关代码，选择修改需要适用的规则和工具，决定结束则需要复测回执。同一任务的各阶段共用目标与任务记录，不共用一份固定的模型输入。

先按主体权限、来源版本和时效排除不合格的候选，再选取与当前决策有关的细节。装配后的输入须保留规则的适用范围、证据的来源、推断的待验证身份，以及工具调用与结果的对应关系。网页和工具结果中的命令式文字仍属于资料，不能据此获得执行权限。

模型提出行动后，运行时校验并执行，把观察写入轨迹；用户修正目标或新观察推翻旧结论时，更新有效状态。下一轮从更新后的记录重新选择内容，而不是在上轮输入末尾无限追加。已完成阶段的细节可以退出窗口，必要的失败证据和恢复入口仍保存在外部。

### 轨迹与状态栏

轨迹记录「发生过什么」：用户修正、工具调用、执行结果与证据。状态栏提炼「现在处于什么状态」：运行时从轨迹与当前约束生成短投影，供下一轮选择行动。长轨迹不必全文进入窗口，但状态栏不能替代可回查的原始证据。

例如修复订单接口测试时，运行时可生成下面的状态投影；记录编号可回读原始日志，计数和时间由运行时计算：

<section class="ce-status" aria-label="修复订单接口测试的任务状态示例">
	<header class="ce-status__head">
		<span>任务状态 · 修改后复测</span>
		<span>记录 #18</span>
	</header>
	<dl class="ce-status__grid">
		<div class="ce-status__item ce-status__item--wide">
			<dt>当前目标</dt><dd>修复订单接口失败测试</dd>
		</div>
		<div class="ce-status__item ce-status__item--verified">
			<dt>已验证</dt><dd>测试 A 通过（#17）；测试 B 失败（#18）</dd>
		</div>
		<div class="ce-status__item ce-status__item--pending">
			<dt>待验证</dt><dd>测试 B 的失败原因</dd>
		</div>
		<div class="ce-status__item">
			<dt>工具调用</dt><dd>4 / 8</dd>
		</div>
		<div class="ce-status__item">
			<dt>剩余时间</dt><dd>90 秒</dd>
		</div>
	</dl>
</section>

工具执行或用户纠正后，运行时替换旧状态栏。已验证结果应能追溯到执行回执；原因猜测在验证前不应晋升为事实。状态栏里的「4 / 8」可帮助模型安排步骤，但调用次数、权限和时限仍由运行时强制检查。

---

## 输入容量与计算复用

逐轮重建工作集面对两种成本：输入容量有限，相同内容反复处理也有计算开销。容量预算要为指令、能力、证据、状态和近期轨迹分配空间，并预留本次输出预算。先保留关键证据，再移除重复示例、无关日志与失效状态；未达到窗口上限时，额外内容仍可能干扰判断。

KV Cache 是推理计算产生的中间状态，不是任务记忆。服务若支持跨请求前缀缓存，相同且仍有效的输入前缀可能复用对应计算；在前部修改工具定义，则后续部分可能需要重新处理。下图仅示意这一**有条件的计算复用**，不表示所有服务采用相同的匹配规则。

<figure class="ce-figure">
	<div class="ce-figure__scroll" tabindex="0" aria-label="相同前缀和中途修改对缓存复用的影响，可横向滚动">
		<svg class="ce-svg" viewBox="0 0 940 284" role="img" aria-labelledby="ce-cache-title ce-cache-desc">
			<title id="ce-cache-title">前部改动与尾部追加影响跨请求缓存的范围</title>
			<desc id="ce-cache-desc">三行输入依次为上一轮、前部工具定义已改的下一轮，以及仅在尾部新增观察的下一轮。工具定义改动时仅先前相同的系统规则有机会命中；仅尾部变化时稳定前缀有机会复用，变化的尾部需要重新处理。</desc>
			<text x="20" y="24" class="ce-svg__head">上轮输入 · 缓存候选前缀</text>
			<g class="ce-svg__box">
				<rect x="20" y="36" width="166" height="34"/><rect x="196" y="36" width="155" height="34"/>
				<rect x="361" y="36" width="148" height="34"/><rect x="519" y="36" width="194" height="34"/>
				<rect x="723" y="36" width="195" height="34"/>
			</g>
			<text x="34" y="59">系统规则</text><text x="210" y="59">工具定义</text>
			<text x="375" y="59">Skill 索引</text><text x="533" y="59">资料与记忆</text>
			<text x="737" y="59">轨迹与观察</text>
			<text x="20" y="106" class="ce-svg__head">下一轮 A · 工具定义改变</text>
			<rect x="20" y="118" width="166" height="34" fill="var(--cs-color-success-soft)" stroke="var(--cs-color-success)"/>
			<g fill="var(--cs-color-danger-soft)" stroke="var(--cs-color-danger)">
				<rect x="196" y="118" width="155" height="34"/><rect x="361" y="118" width="148" height="34"/>
				<rect x="519" y="118" width="194" height="34"/><rect x="723" y="118" width="195" height="34"/>
			</g>
			<text x="34" y="141">相同 · 可命中</text><text x="210" y="141">工具已改</text>
			<text x="375" y="141">重新处理</text><text x="533" y="141">重新处理</text>
			<text x="737" y="141">重新处理</text>
			<text x="20" y="188" class="ce-svg__head">下一轮 B · 仅在尾部新增观察</text>
			<g fill="var(--cs-color-success-soft)" stroke="var(--cs-color-success)">
				<rect x="20" y="200" width="166" height="34"/><rect x="196" y="200" width="155" height="34"/>
				<rect x="361" y="200" width="148" height="34"/><rect x="519" y="200" width="194" height="34"/>
			</g>
			<rect x="723" y="200" width="195" height="34" fill="var(--cs-color-info-soft)" stroke="var(--cs-color-info)"/>
			<text x="34" y="223">系统规则</text><text x="210" y="223">工具定义</text>
			<text x="375" y="223">Skill 索引</text><text x="533" y="223">资料与记忆</text>
			<text x="737" y="223">新观察 · 新计算</text>
			<text x="20" y="267" class="ce-svg__small">颜色对应：绿色＝可复用的相同前缀；红色＝中途改变后需重新处理；蓝色＝新增尾部。</text>
		</svg>
	</div>
	<figcaption>定性示意：缓存规则、有效期和隔离范围取决于推理服务。</figcaption>
</figure>

因此，可将跨轮仍适用的规则、常用工具说明和短技能索引放在相对稳定的前部，把本轮检索片段、状态和新观察放在变化较快的后部。这只是装填策略，仍须服从接口的消息角色与授权边界；规则、工具契约或用户要求一旦变化，就应更新输入，不能为了命中缓存保留旧内容。缓存只节约计算，不扩大模型可处理的工作集。

---

## 长任务工作集治理

长任务的证据与轨迹持续增长。要让当前决策所需的信息保持可见，就需要选择、外置、压缩或隔离不同内容；四种操作可以组合，并无固定顺序。

<figure class="ce-figure">
	<div class="ce-figure__scroll" tabindex="0" aria-label="上下文治理操作图，可横向滚动">
		<svg class="ce-svg" viewBox="0 0 940 310" role="img" aria-labelledby="ce-ops-title ce-ops-desc">
			<title id="ce-ops-title">选择、外置、压缩、隔离作用于工作集的不同边界</title>
			<desc id="ce-ops-desc">中间是本轮工作集。左上选择控制哪些信息进入；左下外置把大对象留在窗口外。右上压缩缩短旧轨迹；右下隔离把独立分支留在另一工作集，只回传必要结果。</desc>
			<defs>
				<marker id="ce-ops-arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto">
					<path d="M0 0 L10 5 L0 10 Z" fill="var(--cs-color-brand)"/>
				</marker>
			</defs>
			<rect x="361" y="48" width="218" height="213" fill="var(--cs-color-bg)" stroke="var(--cs-color-text)" stroke-width="1.5"/>
			<text x="470" y="74" text-anchor="middle" class="ce-svg__head">本轮工作集</text>
			<rect x="378" y="88" width="184" height="31" class="ce-svg__band--rule"/>
			<rect x="378" y="130" width="184" height="31" class="ce-svg__band--capability"/>
			<rect x="378" y="172" width="184" height="31" class="ce-svg__band--evidence"/>
			<rect x="378" y="214" width="184" height="31" class="ce-svg__band--history"/>
			<text x="394" y="109" class="ce-svg__on-brand">规则与当前目标</text>
			<text x="394" y="151">相关证据</text>
			<text x="394" y="193">近期观察</text>
			<text x="394" y="235">旧阶段的摘要</text>
			<g class="ce-svg__box">
				<rect x="28" y="76" width="214" height="70"/><rect x="28" y="183" width="214" height="70"/>
				<rect x="698" y="76" width="214" height="70"/><rect x="698" y="183" width="214" height="70"/>
			</g>
			<text x="43" y="103" class="ce-svg__head ce-svg__brand">SELECT · 选择</text>
			<text x="43" y="128" class="ce-svg__small">只引入当前需要的候选</text>
			<text x="43" y="210" class="ce-svg__head ce-svg__brand">WRITE · 外置</text>
			<text x="43" y="235" class="ce-svg__small">大对象留下可读入口</text>
			<text x="713" y="103" class="ce-svg__head ce-svg__brand">COMPRESS · 压缩</text>
			<text x="713" y="128" class="ce-svg__small">旧轨迹变短，摘要有损</text>
			<text x="713" y="210" class="ce-svg__head ce-svg__brand">ISOLATE · 隔离</text>
			<text x="713" y="235" class="ce-svg__small">分支独立，只回传结果</text>
			<path class="ce-svg__arrow" d="M251 111 H351" marker-end="url(#ce-ops-arrow)"/>
			<path class="ce-svg__arrow" d="M352 218 H253" marker-end="url(#ce-ops-arrow)"/>
			<path class="ce-svg__arrow" d="M589 111 H688" marker-end="url(#ce-ops-arrow)"/>
			<path class="ce-svg__feedback" d="M589 218 H688" marker-end="url(#ce-ops-arrow)"/>
			<text x="470" y="293" text-anchor="middle" class="ce-svg__small">操作依据：信息能否恢复、恢复成本，以及当前决策是否需要细节</text>
		</svg>
	</div>
	<figcaption>定性示意：操作按不同边界改变当轮工作集，不表示固定执行顺序。</figcaption>
</figure>

选择过少会漏掉决定性证据；隔离独立分支可减少相互干扰，但汇总时仍要带回结果、来源与未解决事项，且付出额外调用和核对成本。外置大日志或文档时，恢复入口必须仍可读取；仅有路径或 URL 不够，内容可能已变更或失去访问权限，需要精确复现的材料应保留版本或快照。

压缩对已结束阶段的轨迹做有损提炼，应保留目标、已验证事实、待处理问题、关键失败和恢复入口。真实失败的动作与报错是证据，不能与被证伪的原因推断一起抹去；不可重做的执行回执也不能当成可重新获取的普通材料。工具调用与结果仍须成对保留，以免下轮丢失因果和接口要求的对应关系。

---

## 失效与验证

过滤可能漏掉证据，压缩可能保留错误推断，缓存稳定性也可能诱使旧状态滞留。判断装配是否有效，首先要回放实际输入，而不是只看最终回答。下图给出症状与排查起点；同一症状可能涉及多种失效。

<figure class="ce-figure">
	<div class="ce-figure__scroll" tabindex="0" aria-label="上下文失效排查图，可横向滚动">
		<svg class="ce-svg" viewBox="0 0 940 318" role="img" aria-labelledby="ce-diag-title ce-diag-desc">
			<title id="ce-diag-title">从任务症状定位上下文失效与检查点</title>
			<desc id="ce-diag-desc">四条示例链路：反复询问已给约束对应遗漏，检查入选内容和摘要；错误推断持续引用对应污染，检查来源与验证；相近工具选错对应干扰，检查候选工具；新要求后沿用旧口径对应冲突，检查版本替代关系。实际故障可能同时涉及多类。</desc>
			<defs>
				<marker id="ce-diag-arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto">
					<path d="M0 0 L10 5 L0 10 Z" fill="var(--cs-color-border-strong)"/>
				</marker>
			</defs>
			<text x="22" y="25" class="ce-svg__head">可见症状</text>
			<text x="359" y="25" class="ce-svg__head">可能失效</text>
			<text x="635" y="25" class="ce-svg__head">先核对什么</text>
			<g class="ce-svg__box">
				<rect x="20" y="41" width="284" height="48"/><rect x="20" y="104" width="284" height="48"/>
				<rect x="20" y="167" width="284" height="48"/><rect x="20" y="230" width="284" height="48"/>
				<rect x="632" y="41" width="287" height="48"/><rect x="632" y="104" width="287" height="48"/>
				<rect x="632" y="167" width="287" height="48"/><rect x="632" y="230" width="287" height="48"/>
			</g>
			<g fill="var(--cs-color-danger-soft)" stroke="var(--cs-color-danger)">
				<rect x="357" y="41" width="184" height="48" rx="3"/><rect x="357" y="104" width="184" height="48" rx="3"/>
				<rect x="357" y="167" width="184" height="48" rx="3"/><rect x="357" y="230" width="184" height="48" rx="3"/>
			</g>
			<text x="35" y="71">反复询问已给出的约束</text>
			<text x="35" y="134">错误推断被继续引用</text>
			<text x="35" y="197">相近工具选错、重复旧动作</text>
			<text x="35" y="260">新要求后仍使用旧口径</text>
			<text x="373" y="71" class="ce-svg__danger">遗漏 · 关键证据</text>
			<text x="373" y="134" class="ce-svg__danger">污染 · 推断当事实</text>
			<text x="373" y="197" class="ce-svg__danger">干扰 · 候选过多</text>
			<text x="373" y="260" class="ce-svg__danger">冲突 · 新旧并列</text>
			<text x="648" y="71">实际输入与压缩摘要</text>
			<text x="648" y="134">来源、验证与作废记录</text>
			<text x="648" y="197">工具描述与相关轨迹</text>
			<text x="648" y="260">生效范围与替代关系</text>
			<g class="ce-svg__line" marker-end="url(#ce-diag-arrow)">
				<path d="M310 65 H346"/><path d="M310 128 H346"/><path d="M310 191 H346"/><path d="M310 254 H346"/>
				<path d="M548 65 H620"/><path d="M548 128 H620"/><path d="M548 191 H620"/><path d="M548 254 H620"/>
			</g>
			<text x="470" y="305" text-anchor="middle" class="ce-svg__small">示例映射：同一症状可能有多个原因，需回放真实请求与执行结果</text>
		</svg>
	</div>
	<figcaption>定性示意：这些映射是排查起点，不是唯一归因。</figcaption>
</figure>

回放时应记录每轮实际载荷、候选与入选内容、过滤或压缩理由、来源版本、工具结果及任务验收。网页、附件和工具结果若包含改变任务的命令，还须检查资料是否越过指令边界，以及执行层是否正确拒绝未授权动作。观测记录本身也应遵守相同的权限约束。

评估装配策略时，固定任务、模型、输出预算与验收条件，分别测试遗漏关键证据、混入过期版本、增加近义工具或压缩后重读等扰动。先比较任务完成与执行、引用的正确性，再比较 token、缓存命中、延迟和费用。更短的输入若增加遗漏，或更高的缓存命中若保留旧状态，都不是改进；证据位置的效果也须在具体模型和任务下测量。

---

## 参考资料

- Bojie Li，[《AI Agent Book》第二章：上下文决定 Agent 能力上限的关键](https://bojieli.github.io/ai-agent-book/astro/book/chapter2/#%E4%B8%8A%E4%B8%8B%E6%96%87%E5%86%B3%E5%AE%9A-agent-%E8%83%BD%E5%8A%9B%E4%B8%8A%E9%99%90%E7%9A%84%E5%85%B3%E9%94%AE)。任务信息可获得性与逐轮上下文构造。
- Bojie Li，[《AI Agent Book》第二章：Agent 状态栏](https://bojieli.github.io/ai-agent-book/astro/book/chapter2/#agent-%E7%8A%B6%E6%80%81%E6%A0%8F%E9%80%9A%E8%BF%87%E5%85%83%E4%BF%A1%E6%81%AF%E5%A2%9E%E5%BC%BA-agent-%E8%BD%A8%E8%BF%B9%E7%AE%A1%E7%90%86)。用进度、剩余约束与阶段信息辅助长任务轨迹管理。
- Anthropic，[Effective context engineering for AI agents](https://www.anthropic.com/engineering/effective-context-engineering-for-ai-agents)。输入选择、按需读取与长任务信息治理的实践讨论。
- Anthropic，[Prompt caching](https://platform.claude.com/docs/en/build-with-claude/prompt-caching)。跨请求前缀匹配、缓存断点与接口中的输入层级；规则随服务和版本变化。
- vLLM，[Automatic Prefix Caching](https://docs.vllm.ai/en/latest/design/prefix_caching/)。基于前缀的 KV 状态复用机制，不能与应用层记忆混为一谈。
- LangChain，[Context Engineering for Agents](https://blog.langchain.com/context-engineering-for-agents/)。选择、写出、压缩与隔离的操作分类。
- Liu et al.，[Lost in the Middle: How Language Models Use Long Contexts](https://arxiv.org/abs/2307.03172)。特定任务和模型下的证据位置效应。

<style>
.VPDoc .vp-doc:has(.ce-figure) h1 {
	border-bottom: 0;
	font-size: clamp(2rem, 3vw, 2.75rem);
	line-height: var(--cs-leading-tight);
}

.VPDoc .vp-doc:has(.ce-figure) h1 + p {
	padding: var(--cs-space-5) var(--cs-space-6);
	border-left: 3px solid var(--cs-color-brand);
	background: var(--cs-color-bg-soft);
	font-size: var(--cs-text-lg);
}

.VPDoc .vp-doc:has(.ce-figure) > div > p {
	max-width: 52rem;
}

.VPDoc .vp-doc:has(.ce-figure) h2 {
	margin-top: var(--cs-space-8);
	padding: 0 0 0 var(--cs-space-4);
	border-top: 0;
	border-left: 3px solid var(--cs-color-brand);
}

.ce-boundary {
	display: grid;
	grid-template-columns: repeat(3, minmax(0, 1fr));
	margin: var(--cs-space-8) 0;
	border: 1px solid var(--cs-color-border-strong);
}

.ce-boundary__item {
	min-width: 0;
	padding: var(--cs-space-5);
	border-right: 1px solid var(--cs-color-border);
	border-top: 3px solid var(--cs-color-border-strong);
}

.ce-boundary__item:nth-child(2) {
	border-top-color: var(--cs-color-brand);
}

.ce-boundary__item:nth-child(3) {
	border-top-color: var(--cs-color-success);
	border-right: 0;
}

.ce-boundary dt {
	margin: 0 0 var(--cs-space-3);
	font-size: var(--cs-text-lg);
	font-weight: 700;
}

.ce-boundary dd {
	margin: 0;
	color: var(--cs-color-text-muted);
	line-height: var(--cs-leading-relaxed);
}

.ce-status {
	margin: var(--cs-space-8) 0;
	border: 1px solid var(--cs-color-border-strong);
	background: var(--cs-color-bg);
}

.ce-status__head {
	display: flex;
	justify-content: space-between;
	gap: var(--cs-space-4);
	padding: var(--cs-space-4) var(--cs-space-5);
	background: var(--cs-color-brand);
	color: var(--cs-color-on-brand);
	font: 700 var(--cs-text-sm)/var(--cs-leading-normal) var(--cs-font-mono);
}

.ce-status__grid {
	display: grid;
	grid-template-columns: repeat(2, minmax(0, 1fr));
	margin: 0;
}

.ce-status__item {
	min-width: 0;
	padding: var(--cs-space-5);
	border-top: 1px solid var(--cs-color-border);
}

.ce-status__item--wide {
	grid-column: 1 / -1;
	background: var(--cs-color-bg-soft);
}

.ce-status__item--verified {
	border-left: 3px solid var(--cs-color-success);
}

.ce-status__item--pending {
	border-left: 3px solid var(--cs-color-warning);
}

.ce-status dt {
	margin: 0 0 var(--cs-space-2);
	color: var(--cs-color-text-muted);
	font: 700 var(--cs-text-sm)/var(--cs-leading-normal) var(--cs-font-mono);
}

.ce-status dd {
	margin: 0;
	font-weight: 600;
}

.VPDoc .vp-doc .ce-table {
	width: 100%;
	border-collapse: collapse;
}

.VPDoc .vp-doc .ce-table th,
.VPDoc .vp-doc .ce-table td {
	padding: var(--cs-space-3) var(--cs-space-4);
	border: 0;
	border-bottom: 1px solid var(--cs-color-border);
	background: transparent;
	text-align: left;
	vertical-align: top;
}

.VPDoc .vp-doc .ce-table thead th {
	border-bottom-color: var(--cs-color-border-strong);
	background: var(--cs-color-bg-soft);
}

.VPDoc .vp-doc .ce-table tbody th {
	width: 12rem;
	font-weight: 600;
}

.ce-figure {
	margin: var(--cs-space-8) 0;
	border: 1px solid var(--cs-color-border-strong);
	background: var(--cs-color-bg-elevated);
}

.ce-figure__scroll {
	max-width: 100%;
	overflow-x: auto;
	overflow-y: hidden;
	padding: var(--cs-space-6);
}

.ce-figure__scroll:focus-visible {
	outline: var(--cs-focus-ring-width) solid var(--cs-color-brand);
	outline-offset: var(--cs-focus-ring-offset);
}

.ce-svg {
	display: block;
	width: 100%;
	min-width: 0;
	height: auto;
}

.ce-svg text {
	font-family: inherit;
	font-size: 16px;
	fill: var(--cs-color-text);
}

.ce-svg .ce-svg__head {
	font-size: 18px;
	font-weight: 700;
}

.ce-svg .ce-svg__small {
	font-size: 14px;
	fill: var(--cs-color-text-muted);
}

.ce-svg text.ce-svg__on-brand {
	fill: var(--cs-color-on-brand);
}

.ce-svg__band--rule {
	fill: var(--cs-color-brand);
}

.ce-svg__band--capability {
	fill: var(--cs-color-warning-soft);
}

.ce-svg__band--evidence {
	fill: var(--cs-color-info-soft);
}

.ce-svg__band--history {
	fill: var(--cs-color-neutral-soft);
}

.ce-svg .ce-svg__brand {
	fill: var(--cs-color-brand);
}

.ce-svg .ce-svg__danger {
	fill: var(--cs-color-danger);
}

.ce-svg__box rect {
	fill: var(--cs-color-bg);
	stroke: var(--cs-color-border-strong);
	stroke-width: 1.5px;
}

.ce-svg__arrow,
.ce-svg__feedback {
	fill: none;
	stroke: var(--cs-color-brand);
	stroke-width: 2px;
}

.ce-svg__feedback {
	stroke-dasharray: 5 5;
}

.ce-svg__line {
	fill: none;
	stroke: var(--cs-color-border-strong);
	stroke-width: 1.4px;
}

.ce-figure figcaption {
	margin: 0;
	padding: var(--cs-space-3) var(--cs-space-5) var(--cs-space-4);
	border-top: 1px dashed var(--cs-color-border-strong);
	color: var(--cs-color-text-muted);
	font-size: var(--cs-text-sm);
	line-height: var(--cs-leading-normal);
}

@media (max-width: 1100px) {
	.ce-svg {
		min-width: 920px;
	}

	.ce-figure figcaption::after {
		content: " · 左右滑动查看完整图";
		color: var(--cs-color-brand);
	}
}

@media (max-width: 640px) {
	.ce-boundary,
	.ce-status__grid {
		grid-template-columns: 1fr;
	}

	.ce-boundary__item,
	.ce-boundary__item:nth-child(3) {
		border-right: 0;
	}

	.ce-boundary__item + .ce-boundary__item {
		border-top-width: 1px;
		border-left: 3px solid var(--cs-color-brand);
	}

	.ce-boundary__item:nth-child(3) {
		border-left-color: var(--cs-color-success);
	}

	.ce-status__item--wide {
		grid-column: auto;
	}

	.ce-status__item--verified {
		border-left: 3px solid var(--cs-color-success);
	}

	.ce-status__item--pending {
		border-left: 3px solid var(--cs-color-warning);
	}

	.VPDoc .vp-doc .ce-table tbody th {
		width: 7.5rem;
		white-space: normal;
	}
}
</style>
