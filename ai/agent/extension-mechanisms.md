# Agent 扩展机制：Hook 与 Middleware

Agent 运行时在「决策—执行—观察」循环的确定位置上挂载外部代码：运行到该位置时把结构化上下文交给扩展，再按其返回值决定继续、改写还是中止。这样的挂载位置称为 **Hook**（钩子、生命周期回调）；把多个 Hook 组织在同一位置的规则称为 **Middleware**（中间件）或责任链。

扩展机制本身不提供具体功能。参数校验、授权审批、沙箱确认是[工具调用机制](./tool-mechanics)模块借助「工具执行前」这个 Hook 实现的；规则注入与上下文改写是上下文装配模块借助「每轮开始」Hook 实现的。本页只定义三件事：**挂载位置、返回值契约、多个扩展的组合方式**。

---

## 扩展点地图

Hook 的位置沿循环分布。位置决定扩展能拿到什么、能否改变控制流；同一位置在不同产品上可能只允许观察、允许改写或允许阻断。

<div class="agx-scroll">
<svg class="agx-svg" viewBox="0 0 1080 420" role="img" aria-labelledby="agx-fig1-title agx-fig1-desc">
<title id="agx-fig1-title">Agent 循环上的 Hook 位置</title>
<desc id="agx-fig1-desc">用户输入、上下文组装、模型调用、工具决策、工具执行、结果回填与收尾构成主循环；各节点之间的挂载点可观察、改写或阻断，工具执行是唯一产生副作用的节点。</desc>
<defs>
<marker id="agx-ar" markerWidth="8" markerHeight="8" refX="6.5" refY="3" orient="auto"><path d="M0,0 L7,3 L0,6 z" class="agx-edge"/></marker>
<marker id="agx-arg" markerWidth="8" markerHeight="8" refX="6.5" refY="3" orient="auto"><path d="M0,0 L7,3 L0,6 z" class="agx-edge-brand"/></marker>
</defs>
<g>
<rect x="18" y="70" width="118" height="46" rx="2" class="agx-node"/><text class="agx-lbl" x="77" y="91" text-anchor="middle">用户输入</text><text class="agx-sm" x="77" y="106" text-anchor="middle">input</text>
<rect x="176" y="70" width="118" height="46" rx="2" class="agx-node"/><text class="agx-lbl" x="235" y="91" text-anchor="middle">上下文组装</text><text class="agx-sm" x="235" y="106" text-anchor="middle">system prompt</text>
<rect x="334" y="70" width="118" height="46" rx="2" class="agx-node agx-node--brand"/><text class="agx-lbl agx-fill-text" x="393" y="91" text-anchor="middle">模型调用</text><text class="agx-sm" x="393" y="106" text-anchor="middle">provider request</text>
<rect x="492" y="70" width="118" height="46" rx="2" class="agx-node"/><text class="agx-lbl" x="551" y="91" text-anchor="middle">工具决策</text><text class="agx-sm" x="551" y="106" text-anchor="middle">tool calls</text>
<rect x="650" y="70" width="118" height="46" rx="2" class="agx-node agx-node--danger"/><text class="agx-lbl agx-fill-text" x="709" y="91" text-anchor="middle">工具执行</text><text class="agx-sm agx-c-danger" x="709" y="106" text-anchor="middle">副作用发生点</text>
<rect x="808" y="70" width="118" height="46" rx="2" class="agx-node"/><text class="agx-lbl" x="867" y="91" text-anchor="middle">结果回填</text><text class="agx-sm" x="867" y="106" text-anchor="middle">observation</text>
<rect x="966" y="70" width="98" height="46" rx="2" class="agx-node"/><text class="agx-lbl" x="1015" y="91" text-anchor="middle">收尾</text><text class="agx-sm" x="1015" y="106" text-anchor="middle">stop / settled</text>
</g>
<g class="agx-edge" stroke-width="1.2" marker-end="url(#agx-ar)" fill="none">
<path d="M136,93 L172,93"/><path d="M294,93 L330,93"/><path d="M452,93 L488,93"/>
<path d="M610,93 L646,93"/><path d="M768,93 L804,93"/><path d="M926,93 L962,93"/>
</g>
<path d="M867,116 L867,140 L393,140 L393,120" class="agx-loop" marker-end="url(#agx-arg)"/>
<text class="agx-sm agx-c-brand" x="630" y="135">存在未完成工具调用时，观察回填后继续下一轮</text>
<g class="agx-fill-brand">
<circle cx="156" cy="93" r="5"/><circle cx="314" cy="93" r="5"/><circle cx="472" cy="93" r="5"/>
<circle cx="630" cy="93" r="5"/><circle cx="788" cy="93" r="5"/><circle cx="946" cy="93" r="5"/>
</g>
<g class="agx-c-brand" stroke-width="1" stroke-dasharray="3 3" opacity=".55">
<path d="M156,100 L156,164"/><path d="M314,100 L314,164"/><path d="M472,100 L472,164"/>
<path d="M630,100 L630,164"/><path d="M788,100 L788,164"/><path d="M946,100 L946,164"/>
</g>
<rect x="14" y="164" width="1052" height="64" rx="3" class="agx-band agx-band--brand"/>
<text class="agx-ttl agx-c-brand" x="30" y="186">HOOK · 每个挂载瞬间可观察、改写或阻断，能力上限由该事件的返回值协议决定</text>
<text class="agx-sm" x="30" y="206">典型位置：每轮输入 · 模型请求前后 · 工具执行前 · 工具执行后 · 本轮收尾；另有压缩、会话切换等少数位置只通知、不可阻断</text>
<rect x="14" y="248" width="518" height="92" rx="3" class="agx-band agx-band--info"/>
<text class="agx-ttl agx-c-info" x="30" y="270">工具执行前：唯一能阻止副作用的位置</text>
<text class="agx-sm" x="30" y="290">参数校验、授权审批、人机确认、静态策略都挂在这里</text>
<text class="agx-sm" x="30" y="308">这些功能属于工具调用模块，Hook 只提供拦截时机与返回值通道</text>
<text class="agx-sm" x="30" y="326">拒绝在此生效：动作不执行，原因回填给模型</text>
<rect x="548" y="248" width="518" height="92" rx="3" class="agx-band agx-band--warn"/>
<text class="agx-ttl agx-c-warn" x="564" y="270">工具执行后：只能改模型看到的结果</text>
<text class="agx-sm" x="564" y="290">结果脱敏、裁剪、整形、追加上下文</text>
<text class="agx-sm" x="564" y="308">此时副作用已经落地，拒绝不能撤销已写入的内容</text>
<text class="agx-sm agx-c-danger" x="564" y="326">要防副作用，挂载点必须在执行前</text>
<text class="agx-sm agx-c-danger" x="14" y="372">工具执行是循环中唯一产生外部副作用的节点；它右侧的 Hook 都只能改 observation，不能回滚磁盘与网络动作。</text>
<text class="agx-sm agx-c-subtle" x="14" y="396">压缩、记忆与上下文可见性的机制见 <tspan class="agx-c-info">Context、Memory 与 Knowledge 的边界</tspan>；Skill 是能力与知识供给，不是 Hook。</text>
</svg>
</div>

图 01 · 循环上的 Hook 位置。绿点为挂载瞬间，工具前后两个位置的能力差异是扩展机制最重要的边界。

| 位置 | 能拿到什么 | 典型能力 |
| --- | --- | --- |
| 每轮输入 | 原始用户输入 | 追加规则、拦截输入；命令类入口可能在此之前先行截获 |
| 模型请求前后 | 发送给 Provider 的载荷与响应 | 改写请求、读取或替换响应 |
| 工具执行前 | 工具名与实际参数 | 校验、改写参数、拒绝、转人工审批 |
| 工具执行后 | 工具返回值 | 脱敏、整形、替换回填内容 |
| 本轮收尾 | 模型的结束决定 | 验收未过时驳回，让循环继续 |

部分位置只发出通知、返回值不被消费（例如会话开始结束、压缩前的部分实现）。只通知位置不能承载策略，只能做埋点与审计。

---

## 返回值契约

Hook 的能力不由「能不能收到事件」决定，而由返回值如何被解释决定。同一个事件位置，协议通常区分三种结果。

| 返回结果 | 运行时行为 | 用途 |
| --- | --- | --- |
| 观察（空返回） | 主流程不受影响，继续执行 | 日志、指标、审计埋点 |
| 改写（modify） | 用返回的新参数或新内容替换后继续 | 参数修正、注入上下文、结果整形 |
| 阻断（deny） | 动作不执行，控制流改变 | 权限拒绝、验收驳回、转人工 |

三条契约约束决定了 Hook 是否可靠：

- **改写后必须重新校验**。改写工具参数后直接执行，会让后续环节面对未验证输入。新参数应由运行时重新做 schema 与策略校验，而不是信任前一个 Hook。
- **业务拒绝与处理器故障要可区分**。显式返回 `deny` 表示策略不允许，拒绝原因作为 observation 回填，模型可以更换工具或缩小范围；抛异常表示处理器自身出错，通常直接终结整轮，模型得不到可行动的反馈。只用异常表达一切，两者在协议层无法区分。
- **只读必须由类型保证**。观察回调返回空或返回值不被消费，是技术上的只读；文档里写「请不要修改」只是约定，调用方仍可返回决策对象。只读保证要么由返回类型表达，要么不存在。

少数实现还提供第四种结果 `synthesize`：不执行真实工具，由 Hook 直接返回一个合成结果。它适合缓存与降级，但模型无法分辨结果真伪，必须在 observation 与审计中显式标记。

### 观察与管控是否同源

返回值契约的设计分两派，这是各产品 API 形状差异的根源。

| 设计 | 形态 | 代价 |
| --- | --- | --- |
| 观察与管控分开 | 观察回调返回值不被消费；改写、阻断放在 Middleware、护栏或审批 API | 埋点无法改坏业务，但在观察点改写必须换另一套 API |
| 观察与管控合一 | 同一回调返回空即观察、返回非空即短路；责任链中以是否调用 `next` 放行 | 心智简单、支持重试与缓存，但漏调 `next` 会静默跳过真实执行 |

---

## 多扩展的组合：Middleware 与责任链

一个事件位置挂多个扩展时，差异才显现。组合方式有三种，Middleware 是其中可改写控制流的两种。

| 组合形态 | 执行方式 | 返回路径 | 失败的默认含义 |
| --- | --- | --- | --- |
| 观察者 | 各收一份通知，互不影响，可并发 | 不回传 | 记日志后继续（fail-open） |
| 责任链 | 按优先级顺序执行，任一拒绝即短路 | 决策在链上累积 | 漏处理等于静默拒绝 |
| 中间件（洋葱模型） | 进入时正序，调用 `next` 进入下游 | 返回时逆序，每层可包住下游结果 | 不调 `next` 即阻断 |

中间件的每一层可以调用下游零次、一次或多次，因此重试、缓存、限流、降级天然写在某一层里；责任链更适合按优先级给出互斥裁决。

### 顺序与聚合

多个 Hook 都能决策时，聚合方向必须写进契约，否则「多加一个收紧 Hook」可能产生相反效果：

- **取严**：任一拒绝即短路，符合权限策略的直觉。多个 Hook 并发执行时，取最先到达的拒绝即可。
- **取宽**：可见工具或允许动作取并集。多个以「收紧」为目的的过滤器取并集会反向放宽，只有显式的全拒结果能压过其余 Hook。
- **顺序未定义**：同事件多 Hook 并发且不保证次序时，它们之间不能有先后依赖，改写也不能建立在彼此输出之上。

链式改写还要求传递规则明确：后一个 Hook 看到的是原始参数还是前一个改写后的参数、字段是合并还是整体替换，决定了多层改写能否叠加。

### 失败取向

Hook 失败后的默认行为按用途区分，不能统一处理：

- **观测型 Hook fail-open**：埋点失败记日志后继续，否则观测设施会变成业务的单点故障。
- **策略型 Hook fail-closed**：护栏不可用时不放行，但必须配套有限超时，否则一个卡住的 Hook 会永久挂住循环。
- **失败原因可区分**：回填给模型的信息应写明「护栏不可用」，而不是「请求不合法」，避免模型把系统故障误判成策略拒绝后尝试绕行。

隐式 fail-open 最危险：解析失败后默认放行，或失败模式可配置但默认值未文档化，都会让使用者以为配了护栏、实际没有。

---

## 决策的表达载体

Hook 以什么进程形式运行，决定决策如何传回运行时，也决定超时、崩溃和信任的边界。载体与组合形态正交：进程内函数和外部进程都可以组织成责任链。

| 载体 | 决策表达方式 | 超时与崩溃 |
| --- | --- | --- |
| 进程内函数 | 返回决策对象、修改入参、抛异常、不调用 `next` | 与宿主同权限，崩溃直达宿主；进程内难以安全取消，处理器级超时通常缺失 |
| 每次启动的外部进程 | 退出码 + stdout 上的结构化 JSON | 语言自由，可设超时后强杀，崩溃不影响宿主；可按脚本内容哈希建立信任 |

外部进程的跨边界协议有两个必须知道的契约：

- **退出码三分，不是二分**。约定的成功码表示无意见放行，约定的阻断码才真正拦截，其他非零码只算非阻断告警，动作按原参数继续。习惯上的 `exit 1` 在这套协议里不阻断。
- **输出污染导致隐式放行**。结构化结果只从 stdout 解析，处理器被要求除最终 JSON 外不得输出其他内容；一条调试输出使 JSON 解析失败时，拒绝决策丢失，运行时按无意见处理，配置正确但行为相反。

进程内模块还能运行期注册工具、持有长驻状态、操作交互界面；外部进程每次启动都是新进程，做不到这些。这是能力与治理的交换：进程内换得上限，付出的是隔离、超时与细粒度信任的缺失。真实强制边界仍由沙箱、容器或受管执行器提供，Hook 本身不是安全边界。

---

## 扩展点与相邻机制的边界

下列对象都可能出现在 Agent 运行时里，但它们不是 Hook，回答的问题不同：

- **参数校验、授权审批、沙箱确认**：属于[工具调用机制](./tool-mechanics)模块，在「工具执行前」Hook 里实现。它们是 Hook 的使用者，扩展机制只提供拦截时机和 `allow/deny/modify` 通道；挂起等人、跨进程恢复属于审批实现自身的语义。
- **规则文件、上下文压缩、记忆装配**：属于上下文供给，在「每轮开始」「压缩前后」等位置影响模型能看见什么，边界见 [Context、Memory 与 Knowledge 的边界](./context-boundary)。
- **Skill**：按需加载的过程性知识包，由模型按描述选择，触发是概率性的，不能承载护栏；机制见 [Agent Skills](./skill)。
- **Command**：用户显式调用的入口，确定执行但不阻断主循环。
- **Subagent**：在独立上下文中执行子任务的组织方式，不能阻断父循环。
- **MCP**：工具、资源与提示模板跨进程接入 Host 的能力协议，属于能力供给，不含生命周期拦截。

判据只有一个：**该机制是否在循环的确定事件上被运行时调用，并以返回值改变控制流**。是，则为 Hook；否，则是借 Hook 实现功能的其他模块，或与 Hook 正交的能力层。

---

## 附录：产品口径

Hook 的位置命名、返回字段、退出码约定与默认超时随版本快速变化。下表只记录 2026-09 核验时的形态归属，具体字段以所用版本的官方文档为准。

<details>
<summary>形态归属与组合口径</summary>

| 产品形态 | 代表 | 扩展机制上的标志性取舍 |
| --- | --- | --- |
| 外部进程 CLI | Claude Code · Codex · Gemini CLI | 每次 fork，退出码 + JSON 决策；Hook 全部加载并发运行，拒绝优先 |
| 长驻独立进程 | Amp | 决策为显式 action 联合类型；同事件顺序未定义，清理预算与重复投递写入契约 |
| 宿主内模块 | Pi Coding Agent · opencode · Cline | 生命周期事件密、可运行期注册；阻断有的只能抛异常；隔离与超时外移 |
| 应用内嵌函数 | OpenAI Agents SDK · LangChain/LangGraph · Semantic Kernel · Google ADK · Vercel AI SDK | 扩展作者即应用作者，同信任域；观察回调与管控 Middleware 分合各有选择 |

</details>

## 参考文献

- Pi Coding Agent. [Extensions](https://pi.dev/docs/latest/extensions)
- Anthropic. [Claude Code Hooks](https://docs.anthropic.com/en/docs/claude-code/hooks)
- Google. [Gemini CLI hooks reference](https://github.com/google-gemini/gemini-cli/tree/main/docs/hooks)
- opencode. [Plugins](https://opencode.ai/docs/plugins)
- Amp. [Plugin API](https://ampcode.com/manual/plugin-api)
- Cline. [SDK Plugins](https://docs.cline.bot/sdk/plugins)
- OpenAI. [OpenAI Agents SDK](https://openai.github.io/openai-agents-python/)
- LangChain. [LangChain documentation](https://docs.langchain.com/oss/python/langchain/overview)
- Microsoft. [Semantic Kernel filters](https://learn.microsoft.com/en-us/semantic-kernel/concepts/enterprise-readiness/filters)
- Google. [ADK callbacks](https://google.github.io/adk-docs/callbacks/)
- Vercel. [AI SDK middleware](https://ai-sdk.dev/docs/ai-sdk-core/middleware)

<style scoped>
.agx-scroll {
	width: min(100%, 960px);
	margin: var(--cs-space-8) 0 var(--cs-space-4);
	overflow-x: auto;
	padding-bottom: var(--cs-space-2);
}

.agx-svg {
	display: block;
	width: 100%;
	min-width: 720px;
	height: auto;
}

.agx-svg text {
	font-family: var(--vp-font-family-base);
}

.agx-fill-text { fill: var(--cs-color-text); }
.agx-fill-brand { fill: var(--cs-color-brand); }

.agx-c-brand { fill: var(--cs-color-brand); }
.agx-c-info { fill: var(--cs-color-info); }
.agx-c-warn { fill: var(--cs-color-warning); }
.agx-c-danger { fill: var(--cs-color-danger); }
.agx-c-subtle { fill: var(--cs-color-text-subtle); }

.agx-edge { stroke: var(--cs-color-border-strong); fill: none; }
.agx-edge-brand { stroke: var(--cs-color-brand); fill: none; }
.agx-loop {
	stroke: var(--cs-color-brand);
	stroke-width: 1.2;
	fill: none;
	stroke-dasharray: 5 4;
}

.agx-node {
	fill: var(--cs-color-bg-soft);
	stroke: var(--cs-color-border-strong);
}
.agx-node--brand {
	fill: var(--cs-color-brand-soft);
	stroke: var(--cs-color-brand);
}
.agx-node--danger {
	fill: var(--cs-color-danger-soft);
	stroke: var(--cs-color-danger);
}

.agx-band { stroke-width: 1; }
.agx-band--brand { fill: var(--cs-color-brand-soft); stroke: var(--cs-color-brand); }
.agx-band--info { fill: var(--cs-color-info-soft); stroke: var(--cs-color-info); }
.agx-band--warn { fill: var(--cs-color-warning-soft); stroke: var(--cs-color-warning); }

.agx-lbl {
	font-size: 11px;
	fill: var(--cs-color-text);
}
.agx-sm {
	font-size: 9.5px;
	fill: var(--cs-color-text-subtle);
}
.agx-ttl {
	font-size: 11.5px;
	font-weight: 600;
	fill: var(--cs-color-text);
}

@media (min-width: 1280px) {
	.agx-scroll {
		width: min(960px, calc(100vw - 28rem));
	}
}
</style>
