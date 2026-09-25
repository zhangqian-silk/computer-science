---
title: Agent 工程接入
aside: false
prev: false
next: false
---

<style>
/* 保留参考手册的编号、栈图、矩阵、时间线和树形布局；样式只作用于本页。 */
.vp-doc .ca { color: var(--cs-color-text); line-height: var(--cs-leading-relaxed); min-width: 0; }
.vp-doc .ca-header { border-bottom: 2px solid var(--cs-color-text); padding-bottom: var(--cs-space-8); }
.vp-doc .ca-header h1 { margin: 0; line-height: var(--cs-leading-tight); }
.vp-doc .ca-header em { color: var(--cs-color-brand); font-style: normal; }
.vp-doc .ca-sec { padding: var(--cs-space-9) 0; border-bottom: var(--cs-border); }
.vp-doc .ca-sec-head { align-items: baseline; margin-bottom: var(--cs-space-6); }
.vp-doc .ca-sec-head h2 { margin: 0; padding: 0; border: 0; line-height: var(--cs-leading-tight); }
.vp-doc .ca-sec-num { flex: none; border: 1px solid var(--cs-color-brand); padding: var(--cs-space-1) var(--cs-space-3); color: var(--cs-color-brand); font: bold var(--cs-text-xs)/var(--cs-leading-normal) var(--cs-font-mono); }
.vp-doc .ca p { margin: var(--cs-space-6) 0; }
.vp-doc .ca h3 { margin: var(--cs-space-8) 0 var(--cs-space-5); }
.vp-doc .ca h4 { margin: 0 0 var(--cs-space-4); }
.vp-doc .ca-grid { margin: var(--cs-space-6) 0; grid-template-columns: repeat(auto-fit, minmax(min(100%, 13rem), 1fr)); }
.vp-doc .ca-card { min-width: 0; padding: var(--cs-space-6); border: var(--cs-border); background: var(--cs-color-bg); }
.vp-doc .ca-card p { font-size: var(--cs-text-base); margin: var(--cs-space-3) 0; }
.vp-doc .ca-stack { margin: var(--cs-space-6) 0; border: var(--cs-border-strong); }
.vp-doc .ca-layer { display: grid; grid-template-columns: 3rem 9rem minmax(0, 1fr); border-bottom: var(--cs-border); }
.vp-doc .ca-layer:last-child { border-bottom: 0; }
.vp-doc .ca-lv { display: grid; place-items: center; background: var(--cs-color-brand); color: var(--cs-color-on-brand); font: bold var(--cs-text-xs)/var(--cs-leading-normal) var(--cs-font-mono); }
.vp-doc .ca-ln, .vp-doc .ca-ld { padding: var(--cs-space-5); }
.vp-doc .ca-ln { font-weight: 700; border-right: var(--cs-border); }
.vp-doc .ca-ld { color: var(--cs-color-text-muted); font-size: var(--cs-text-base); }
.vp-doc .ca-ld small { display: block; margin-top: var(--cs-space-2); font: var(--cs-text-xs)/var(--cs-leading-normal) var(--cs-font-mono); }
.vp-doc .ca-callout { border-left: 3px solid var(--cs-color-brand); padding: var(--cs-space-5) var(--cs-space-6); background: var(--cs-color-brand-soft); margin: var(--cs-space-6) 0; }
.vp-doc .ca-callout--warn { border-left-color: var(--cs-color-warning); background: var(--cs-color-warning-soft); }
.vp-doc .ca-callout b { display: block; margin-bottom: var(--cs-space-3); }
.vp-doc .ca-callout p { margin: 0; }
.vp-doc .ca-tw { overflow-x: auto; margin: var(--cs-space-6) 0; border: var(--cs-border); }
.vp-doc .ca-tw table { display: table; border-collapse: collapse; width: 100%; min-width: 38rem; margin: 0; font-size: var(--cs-text-base); }
.vp-doc .ca-tw caption { text-align: left; padding: var(--cs-space-5); font-weight: 700; }
.vp-doc .ca-tw th { background: var(--cs-color-text); color: var(--cs-color-bg); text-align: left; font-size: var(--cs-text-xs); }
.vp-doc .ca-tw td, .vp-doc .ca-tw th { padding: var(--cs-space-5); border: var(--cs-border); line-height: var(--cs-leading-normal); }
.vp-doc .ca-tw tbody tr { background: var(--cs-color-bg); }
.vp-doc .ca-tw tbody tr:nth-child(even) { background: var(--cs-color-bg-soft); }
.vp-doc .ca-tree { white-space: pre; overflow-x: auto; padding: var(--cs-space-6); border-left: 3px solid var(--cs-color-brand); background: var(--cs-color-bg-soft); color: var(--cs-color-text); font: var(--cs-text-base)/var(--cs-leading-relaxed) var(--cs-font-mono); margin: var(--cs-space-6) 0; }
.vp-doc .ca-tl { list-style: none; padding: 0; margin: var(--cs-space-6) 0; }
.vp-doc .ca-tl li { position: relative; display: grid; grid-template-columns: 5rem minmax(0, 1fr); gap: var(--cs-space-6); padding: 0 0 var(--cs-space-7); margin: 0; }
.vp-doc .ca-tl li::before { content: ""; position: absolute; left: 5rem; top: var(--cs-space-2); bottom: 0; border-left: var(--cs-border-strong); }
.vp-doc .ca-tl li:last-child::before { bottom: auto; height: var(--cs-space-3); }
.vp-doc .ca-tl-label { color: var(--cs-color-brand); font: bold var(--cs-text-xs)/var(--cs-leading-normal) var(--cs-font-mono); }
.vp-doc .ca-tl-body { padding-left: var(--cs-space-3); }
.vp-doc .ca-tl-body b { display: block; }
.vp-doc .ca-tl-body p { margin: var(--cs-space-2) 0 0; font-size: var(--cs-text-base); color: var(--cs-color-text-muted); }
.vp-doc .ca [tabindex="0"]:focus-visible { outline: var(--cs-focus-ring-width) solid var(--cs-color-brand); outline-offset: var(--cs-focus-ring-offset); }
@media (max-width: 640px) {
	.vp-doc .ca-layer { grid-template-columns: 2.5rem minmax(0, 1fr); }
	.vp-doc .ca-lv { grid-row: span 2; }
	.vp-doc .ca-ln { border-right: 0; padding-bottom: 0; }
	.vp-doc .ca-ld { padding-top: var(--cs-space-2); }
	.vp-doc .ca-sec-head { flex-wrap: wrap; }
	.vp-doc .ca-sec-head h2 { flex: 1; min-width: 0; }
}
</style>

<div class="ca">

<header class="ca-header">
<h1>Agent <em>工程接入</em></h1>
<p>基于大语言模型的 Agent 围绕任务运行「决策—执行—观察」循环：模型提出动作，运行时校验、授权并执行，再将结果返回模型。工程接入是在业务应用、交互客户端或调度系统与 Agent 运行时之间建立调用契约，管理任务输入、会话状态、操作权限、执行环境与任务结果。</p>
<p>宿主负责用户交互或业务编排，运行时负责执行循环，工具连接搜索、文档、业务 API 或代码环境。Coding Agent 是其中以代码修改为目标的一类；文件编辑和终端控制属于场景能力，不是所有 Agent 接入的前提。</p>
<p>接入方式决定宿主能够观察和控制哪些状态；沙箱限定动作能够触及的资源；持久化与验收机制决定故障后能否继续，以及任务是否真正完成。</p>
</header>

<section class="ca-sec">
<div class="ca-sec-head cs-row"><span class="ca-sec-num">01</span><h2 id="boundaries">职责划分与调用方向</h2></div>
<p>模型接口、Agent 运行时、应用协议、传输和隔离分别承担不同职责。下图是职责分类，不是固定的网络协议栈；隔离边界横跨运行时及其工具，SDK 则可能封装其中多个部分。</p>
<div class="ca-stack" role="group" aria-label="接入系统的五类职责">
<div class="ca-layer"><div class="ca-lv">模型</div><div class="ca-ln">生成接口</div><div class="ca-ld">接收上下文，生成回答或工具调用提议。调用提议本身不执行操作。<small>消息 · 工具说明 · 输出约束</small></div></div>
<div class="ca-layer"><div class="ca-lv">执行</div><div class="ca-ln">Runtime / Harness</div><div class="ca-ld">维护执行循环，调度工具，组装上下文，实施权限、预算与终止条件。<small>工具执行 · 任务状态 · 验收</small></div></div>
<div class="ca-layer"><div class="ca-lv">语义</div><div class="ca-ln">应用协议</div><div class="ca-ld">约定会话、事件、审批与取消。JSON-RPC 提供消息封装，ACP 等协议进一步定义方法和生命周期。<small>ACP · 产品原生协议 · 产品特定 JSON 事件流</small></div></div>
<div class="ca-layer"><div class="ca-lv">传输</div><div class="ca-ln">通信通道</div><div class="ca-ld">传递字节或事件。分帧由通道上使用的协议约定；PTY 还提供终端语义。<small>stdio 管道 · PTY · HTTP/SSE · WebSocket</small></div></div>
<div class="ca-layer"><div class="ca-lv">边界</div><div class="ca-ln">资源隔离</div><div class="ca-ld">限制文件、网络、凭据和系统调用的可达范围。接口相同不意味着权限相同。<small>进程沙箱 · 容器 · 虚拟机</small></div></div>
</div>
<p>更换通信通道不应改变操作含义，但会改变断线、认证和进程生命周期的处理方式。更换运行时则可能改变工具、审批与恢复能力，需要重新检查协议映射。</p>
<div class="ca-grid infra-cards">
<div class="ca-card"><h3>宿主 ↔ Agent</h3><p>宿主提交任务并接收进展；Agent 按契约请求审批或补充输入。接口可以是进程内调用或远程服务；ACP 是面向代码编辑器与 Coding Agent 的协议实例，并非通用接入的必选项。[2]</p></div>
<div class="ca-card"><h3>Agent ↔ 工具服务</h3><p>MCP 提供工具与资源等能力的发现和交互。它不是编辑器会话协议，但一个完整 Agent 也可以被包装为 MCP 工具。[3]</p></div>
<div class="ca-card"><h3>Agent ↔ Agent</h3><p>委派需要明确输入、结果、权限和共享状态。独立上下文不等于资源隔离，多个执行者仍可能修改同一文档、业务记录或工作区。</p></div>
</div>
</section>

<section class="ca-sec">
<div class="ca-sec-head cs-row"><span class="ca-sec-num">02</span><h2 id="integration-surfaces">接入方式与控制能力</h2></div>
<p>接入首先区分自建运行时与复用现有运行时。复用时再检查其程序接口、事件、审批和恢复契约。SDK 是封装形式，ACP 是应用协议，两者可以叠加使用；MCP 的工具扩展方向已在前节区分，不与宿主接入方式并列评分。</p>
<div class="ca-tw" tabindex="0" role="region" aria-label="接入方式对照，可横向滚动">
<table>
<caption>同一控制问题下的实现取舍</caption>
<thead><tr><th scope="col">方式</th><th scope="col">状态与控制</th><th scope="col">接入代价</th><th scope="col">适用条件</th></tr></thead>
<tbody>
<tr><td>模型 API + 自建 harness</td><td>宿主自行定义工具、审批、状态与恢复</td><td>控制范围大，也要自行实现安全、工具执行与故障处理</td><td>现有运行时不能满足业务约束</td></tr>
<tr><td>SDK</td><td>通过类型化方法、事件或回调接入；能力取决于底层运行时</td><td>减少协议处理代码，受语言、版本和封装范围约束</td><td>语言匹配，SDK 覆盖所需能力</td></tr>
<tr><td>远程任务 API</td><td>提交任务、查询状态、获取结果；按契约提供事件订阅、补充输入与取消</td><td>执行环境由服务端管理；需处理身份、网络故障、重复提交与结果保留</td><td>业务系统、跨语言服务、后台长任务</td></tr>
<tr><td>Headless CLI</td><td>启动子进程，读取结构化输出；续跑和审批取决于具体接口</td><td>部署直接，需管理进程、输出分帧与版本兼容</td><td>CI、批处理、非交互任务</td></tr>
<tr><td>产品原生 RPC / app-server</td><td>使用产品提供的会话、事件和审批方法</td><td>覆盖产品特有能力，但绑定其方法集</td><td>单一运行时的深度集成</td></tr>
<tr><td>ACP</td><td>使用编辑器与 Coding Agent 的会话接口；可选能力在初始化时协商</td><td>减少重复适配；产品独有能力仍需扩展或降级</td><td>代码编辑器与 Coding Agent 的互操作</td></tr>
<tr><td>PTY</td><td>读取终端输出、发送输入；任务状态可能需要从画面推断</td><td>不依赖专用 API，但依赖终端行为、界面布局和完成判定</td><td>缺少结构化接口，或需要展示真实终端</td></tr>
</tbody>
</table>
</div>
<p>是否支持中途审批、明确的任务结束事件和会话恢复，比接口名称更能决定选型。单向进度流也可以通过独立 HTTP 请求回传审批；关键是应用契约是否提供完整回路，而不是必须使用某种传输。</p>
<h3>原生协议与适配器</h3>
<p>原生实现直接访问运行时状态；适配器在外部协议与已有入口之间转换。适配器便于独立升级，但只能映射底层已暴露的能力。接入前逐项检查任务与会话映射、结果格式、取消传播、审批以及工具配置；文件与终端委派只在对应场景检查。不能把某一个适配器的能力缺口归纳为所有适配器的限制。</p>
<h3>PTY 与非交互 CLI</h3>
<p>复杂 TUI 会移动光标和局部重绘，单纯剥离 ANSI 或匹配输出字符串未必能还原画面，必要时使用终端模拟器维护屏幕网格。PTY 子进程仍有退出状态，但持续运行的 TUI 内部一次任务完成，未必会退出进程；提示符、静默窗口与超时只能作为启发式证据。</p>
<p>供程序调用的 CLI 应提供明确的非交互模式、结构化结果、退出码和输出上限。缺少必需参数时返回错误，避免停在输入提示上；终端尺寸变化、子进程退出与任务验收应分别处理。</p>
</section>

<section class="ca-sec">
<div class="ca-sec-head cs-row"><span class="ca-sec-num">03</span><h2 id="protocol-and-session">协议生命周期与会话状态</h2></div>
<p>接入契约应分别标识调用、任务与会话：一次调用可以创建长时间运行的任务，一个会话也可以包含多次任务。网络请求返回不等于任务完成；不需要连续对话的任务，也不必建立持久会话。</p>
<p>使用 JSON-RPC 时，request 带有 <code>id</code>，接收方以相同标识返回结果或错误；notification 不带 <code>id</code>，不应得到响应。取消是应用层操作，可以被定义为请求或通知，并不是第三种 JSON-RPC 消息封装。JSON-RPC 只是可选封装，不是 Agent 接入的前提。[1]</p>
<h3>会话与审批回路</h3>
<p>下列流程是带会话与审批的接入设计示意，不是统一协议要求。无状态任务可以省略会话步骤；固定能力可以由版本契约约定。ACP 等具体协议提供各自的方法和生命周期，不能直接套用到所有运行时。[2]</p>
<ol class="ca-tl" aria-label="宿主与 Agent 的会话交互顺序">
<li><span class="ca-tl-label">01 · 初始化</span><div class="ca-tl-body"><b>宿主 ↔ Agent：版本与能力协商</b><p>确认支持的功能；认证按实现要求完成。未声明的可选能力不能直接假定可用。</p></div></li>
<li><span class="ca-tl-label">02 · 会话</span><div class="ca-tl-body"><b>宿主 → Agent：按需新建或恢复会话</b><p>建立会话标识、资源范围及相关配置；恢复能力与新建能力分别判断。</p></div></li>
<li><span class="ca-tl-label">03 · 执行</span><div class="ca-tl-body"><b>宿主提交任务，Agent 返回进展与结果</b><p>输入包含目标、业务参数和约束；消息增量、工具状态等更新可能交错到达，客户端按任务和条目标识归并。</p></div></li>
<li><span class="ca-tl-label">04 · 审批</span><div class="ca-tl-body"><b>Agent → 宿主：权限请求；宿主 → Agent：决定</b><p>允许则回到执行流程，拒绝则停止该动作或选择其他路径。审批结果必须绑定具体动作，不能授权后再静默更换参数。</p></div></li>
<li><span class="ca-tl-label">05 · 终止</span><div class="ca-tl-body"><b>正常返回或取消后确认终止</b><p>发出取消不等于工具已经停止；客户端还需接收明确终态。进程、会话和一轮任务的生命周期分别管理。</p></div></li>
</ol>
<p>任务输入应明确目标资源、允许动作与完成条件；结果应区分回答、产物和已执行的外部动作。例如文档更新需要返回文档标识与版本，不能仅凭「已完成」文本验收。文件委派还需约定路径基准、行号基数及读取磁盘还是编辑器未保存内容，防止观察与修改目标不一致。</p>
<h3>流式分帧与背压</h3>
<div class="ca-grid infra-cards">
<div class="ca-card"><h4>分帧与校验</h4><p>JSONL/NDJSON 需要跨读取缓冲，并处理一次读取中的多条记录。限制单条大小；EOF 残留只有在协议允许且 JSON 完整时才接受，否则报告截断。</p></div>
<div class="ca-card"><h4>数据与日志</h4><p>子进程接入可约定 stdout 承载协议、stderr 承载诊断；远程服务则分离响应载荷与服务日志。日志混入协议流会破坏解析，不同通道也不能被当成天然有序的单一事件序列。</p></div>
<div class="ca-card"><h4>背压与重建</h4><p>使用有界队列。展示刷新可以合并，但原始增量只有在可重放或存在完整快照时才能丢弃。完成事件可能只有状态，不一定带完整消息。</p></div>
</div>
<h3>恢复对象</h3>
<div class="ca-tw" tabindex="0" role="region" aria-label="会话恢复对象，可横向滚动">
<table>
<thead><tr><th scope="col">对象</th><th scope="col">保存内容</th><th scope="col">恢复边界</th></tr></thead>
<tbody>
<tr><td>对话与任务状态</td><td>消息、工具记录、待审批动作、预算</td><td>历史可恢复，不证明记录之外的动作没有发生</td></tr>
<tr><td>运行环境</td><td>工具与模型配置、资源范围、连接或进程状态</td><td>配置可重建，失效连接与已退出进程的内存不能仅靠日志恢复</td></tr>
<tr><td>业务资源与外部系统</td><td>资源版本、产物、外部操作标识</td><td>会话恢复不自动回滚文档、业务记录或已发送消息</td></tr>
</tbody>
</table>
</div>
<p>实现若提供 resume 与 fork，需分别确认其续跑和分支范围。对话历史分叉不意味着外部资源也被复制；多个分支可能仍然操作同一业务记录。Coding Agent 的独立 worktree 只能隔离对应代码文件，不能隔离共享的数据库或远端服务。</p>
</section>

<section class="ca-sec">
<div class="ca-sec-head cs-row"><span class="ca-sec-num">04</span><h2 id="security">授权与隔离边界</h2></div>
<p>授权决定当前主体是否可以执行具体动作，隔离限制执行环境实际能够访问哪些资源。两者互补：已获批准的命令也可能包含错误，沙箱允许访问的资源也不应自动获得业务层授权。</p>
<div class="ca-grid infra-cards">
<div class="ca-card"><h4>提示与规则</h4><p>提示词表达操作约定，不能作为强制边界。执行前的程序校验可以确定地拒绝动作，但效果取决于解析是否正确、所有等价路径是否被覆盖。</p></div>
<div class="ca-card"><h4>人工审批</h4><p>审批绑定主体、动作参数、目标和有效期。无人在场的任务应预先定义允许范围；不弹审批不等于允许全部操作。</p></div>
<div class="ca-card"><h4>资源隔离</h4><p>通过文件系统权限、进程隔离和网络出口限制约束实际访问。远端工具在自己的执行环境中授权，不自动继承本地沙箱。</p></div>
</div>
<h3>隔离范围</h3>
<div class="ca-tw" tabindex="0" role="region" aria-label="隔离范围对照，可横向滚动">
<table>
<thead><tr><th scope="col">边界</th><th scope="col">覆盖对象</th><th scope="col">未覆盖部分与代价</th></tr></thead>
<tbody>
<tr><td>单个命令沙箱</td><td>指定命令及受限的派生进程</td><td>其他文件工具、hooks 和远端服务必须另行治理</td></tr>
<tr><td>运行时进程沙箱</td><td>被纳入策略的运行时与本地子进程</td><td>不包含已在外部运行的工具服务；需要检查继承、代理和逃逸路径</td></tr>
<tr><td>容器</td><td>指定挂载、进程与网络视图</td><td>通常共享宿主内核；特权、宿主挂载或控制套接字会削弱边界</td></tr>
<tr><td>虚拟机 / microVM</td><td>独立来宾内核与执行环境</td><td>仍需治理宿主共享目录、凭据与网络；增加镜像、启动和资源成本</td></tr>
</tbody>
</table>
</div>
<p>隔离强度应由任务信任程度、资源价值和部署条件决定。上述边界针对执行环境；只通过业务 API 操作的 Agent，还需由目标服务实施租户、对象和操作级权限，容器不能代替这些检查。</p>
<ul>
<li><strong>文件：</strong>限制可写目录，对运行时配置、凭据和执行钩子实施额外保护。路径规范化只是预检查，还需防检查后符号链接被替换等竞争条件。</li>
<li><strong>网络：</strong>模型请求出口与工具执行出口分别治理。域名白名单之外还要检查重定向、地址解析及目标服务的访问权限。</li>
<li><strong>凭据：</strong>按任务提供最小范围、短时有效的凭据，避免全量继承环境变量。代理隐藏密钥并不消除其代为执行操作的权限。</li>
</ul>
<div class="ca-callout ca-callout--warn">
<b>不可信内容与外部动作</b>
<p>网页、文档、业务消息、仓库文本和工具结果是数据，不能自行提升指令优先级或授予权限。沙箱内仍可发生误修改；若环境同时具备敏感信息和外发能力，也仍有泄漏风险。摘要和内容标记可以降低暴露量，但不是消除 Prompt Injection 的安全保证。</p>
</div>
</section>

<section class="ca-sec">
<div class="ca-sec-head cs-row"><span class="ca-sec-num">05</span><h2 id="editing-and-recovery">工具执行契约与故障恢复</h2></div>
<p>工具执行契约应明确目标资源、参数、前置条件、权限、返回结果与副作用。结构化参数便于校验，但不能替代目标服务授权；同一动作若还能通过 shell 或其他工具完成，也需要覆盖这些执行路径。</p>
<div class="ca-tw" tabindex="0" role="region" aria-label="工具副作用与恢复边界，可横向滚动">
<table>
<thead><tr><th scope="col">动作</th><th scope="col">执行前提</th><th scope="col">验收与恢复边界</th></tr></thead>
<tbody>
<tr><td>查询与检索</td><td>数据访问范围、过滤条件和结果上限</td><td>保留来源与查询条件；重复查询可能因数据更新而得到不同结果</td></tr>
<tr><td>文档或业务记录更新</td><td>对象标识、预期版本与字段约束</td><td>返回更新后的版本；冲突时重新读取，不静默覆盖并发修改</td></tr>
<tr><td>消息发送或外部任务创建</td><td>目标、授权与重复提交策略</td><td>保存服务端操作标识；超时后先查询状态，已送达消息通常不能视为可回滚</td></tr>
<tr><td>文件编辑与命令执行</td><td>路径、原始版本、执行环境和权限</td><td>检查实际差异与测试结果；部分写入和派生进程需单独清理</td></tr>
</tbody>
</table>
</div>
<p>在代码编辑场景，整文件替换、SEARCH/REPLACE、diff 或结构感知编辑只是不同的修改表达。应用前仍需检查版本与定位条件，应用后进行测试。Git 检查点可帮助恢复已纳入记录的文件，但不覆盖忽略文件、未记录修改和远端副作用。</p>
<h3>执行与持久化顺序</h3>
<ol class="ca-tl" aria-label="授权、执行、记录与恢复的先后关系">
<li><span class="ca-tl-label">01 · 提议</span><div class="ca-tl-body"><b>生成调用，校验参数与前置状态</b><p>尚未执行动作；失败可重新生成。重新生成的动作可能不同，仍须重新校验。</p></div></li>
<li><span class="ca-tl-label">02 · 授权</span><div class="ca-tl-body"><b>绑定具体参数并记录审批</b><p>需要跨进程等待时持久化待审批状态。恢复后重新检查权限与目标资源版本是否仍有效。</p></div></li>
<li><span class="ca-tl-label">03 · 执行</span><div class="ca-tl-body"><b>在受限环境中产生副作用</b><p>结果未返回时不能假定没有执行。外部请求可能已成功，本地进程也可能部分修改文件。</p></div></li>
<li><span class="ca-tl-label">04 · 记录</span><div class="ca-tl-body"><b>持久化结果，再供后续恢复使用</b><p>动作与本地记录之间可能存在故障窗口。追加日志仍需完整性检查和落盘策略，不天然等于原子持久化。</p></div></li>
</ol>
<p>恢复时优先查询真实状态，再决定继续、补偿或重试。外部系统支持幂等键时，键必须绑定同一操作；任务标识本身不能防止工具重复执行。暂态错误可以退避重试，但未知副作用必须先确认，持久化检查点也不代表任意执行点都能精确续跑。</p>
<h3>Hook、取消与审计</h3>
<p>Hook 可以在工具前校验或改写参数，在工具后追加检查结果。其能力由产品契约决定；脚本报错、退出码、结构化拒绝及超时的默认处置需要分别验证。改写后的动作须重新授权，审计同时记录原始提议与实际执行参数。</p>
<p>工具、turn 和整任务分别设定时间与费用上限。取消应传播到整个受控执行单元，例如进程组、作业对象或容器，并确认终止与清理结果。后台服务需要独立的生命周期，不能一律按短命令超时回收。</p>
<p>审计记录主体、任务标识、动作、审批、耗时、结果摘要、资源版本和用量，并对敏感入参脱敏。请求响应、任务终态与业务验收分别表达「调用返回」「运行结束」和「目标达成」，三者不能互相代替；子进程退出码也只描述进程结果。</p>
</section>

<section class="ca-sec">
<div class="ca-sec-head cs-row"><span class="ca-sec-num">06</span><h2 id="context-and-cost">上下文预算与缓存复用</h2></div>
<p>上下文包含本轮实际提交的指令、历史、工具说明与观察。全量记录用于审计和恢复，不必全部进入每次模型请求。缩减输入需保留当前任务约束、未完成动作及证据定位，避免把最近一次工具结果当成所有历史事实的替代品。</p>
<pre class="ca-tree" tabindex="0" role="region" aria-label="上下文预算处理流程">工具结果产生
  │ 分页、字段选择、截断标记；完整内容可按需取回
  ▼
组装本轮上下文
  ├─ 预算充足 → 继续执行
  └─ 超预算   → 移除确认无关或已被替代的内容
                   ├─ 已满足预算 → 继续执行
                   └─ 仍超预算   → 摘要并保留证据索引
                                      ├─ 校验通过 → 继续执行
                                      └─ 反复失败 → 停止重试，保留状态</pre>
<p>规则裁剪不增加模型调用，但仍可能丢失信息；模型摘要增加调用成本，也不保证事实保真。子 Agent 可以隔离探索上下文，却会带来额外调用、摘要失真和共享资源冲突，应在子任务能够独立验收时使用。</p>
<h3>前缀稳定性</h3>
<p>前缀缓存复用重复输入对应的计算。稳定规则和工具定义通常放在前部，动态信息放在后部；前缀中发生变化，会缩短可复用范围，并非必然让所有缓存归零。缓存标记、最短长度、有效期和路由条件由供应商决定；存在自动缓存与显式控制等不同方式，不能把某一方式写成统一要求。[4]</p>
<pre class="ca-tree" tabindex="0" role="region" aria-label="稳定前缀和动态尾部关系">稳定规则 / 工具定义 → 固定示例 → 已有历史 → 本轮变化
└──────────── 可复用的公共前缀 ───────────┘
                                   变化从这里开始 ──▶</pre>
<p>只有在相同输入、相同计费口径下才能比较成本。若供应商把输入互斥地分为普通输入、缓存写入和缓存读取，则按三类 token 数分别乘以对应单价后求和；其他计费方式应按供应商定义拆分。输出、工具、子 Agent 和重试费用另计，不能仅凭输入 token 总量推断收益。</p>
<div class="ca-callout">
<b>预算与计量</b>
<p>任务侧记录输入、输出和费用上限；调用侧记录缓存命中量、有效期、工具结果体积和失败重试。工具按需加载减少常驻上下文，但动态改变工具列表也可能缩短稳定前缀，需要按实际命中率与任务正确率评估。</p>
</div>
</section>

<section class="ca-sec">
<div class="ca-sec-head cs-row"><span class="ca-sec-num">07</span><h2 id="platform">多客户端平台与部署边界</h2></div>
<p>平台化在单个运行时之外增加认证、路由、配额和执行环境管理。以下结构是一种可选拆分：调用量较少、单用户且无跨进程恢复要求时，宿主可以直接启动运行时，无需先建立独立网关与 worker 池。</p>
<pre class="ca-tree" tabindex="0" role="region" aria-label="平台请求与恢复路径">业务应用 / 交互客户端 / 任务调度器
      │ 请求与身份
      ▼
网关：认证、配额、会话路由
      │ 控制协议；事件与审批沿连接返回
      ▼
适配层：协议转换、有界队列、多客户端审批仲裁
      │
      ▼
执行单元：Agent 运行时 + 受限资源访问 + 工具权限
      │ 保存 / 恢复
      ▼
状态存储：任务与会话记录、资源版本、产物、外部操作标识</pre>
<p>会话不是进程或租户的同义词。有状态连接可能需要路由到原 worker；迁移则要求恢复运行时配置、资源访问条件和待执行状态。是否每个会话独占进程，由隔离目标与运行时实现决定，不能从「协议有状态」直接推出。</p>
<ul>
<li><strong>并发写入：</strong>对同一资源使用版本检查、锁或串行执行；仅按 session ID 分流不能防文档、业务记录或文件冲突。</li>
<li><strong>审批仲裁：</strong>指定具有授权资格的客户端，记录决定并使重复回答失效；断线或无人审批时明确超时策略。</li>
<li><strong>跨租户隔离：</strong>业务对象、执行环境、凭据、日志与缓存权限均需检查，不把共享会话存储当作共享访问权限。</li>
<li><strong>交付边界：</strong>产出草稿、建议或变更计划与实际生效分别授权；发送消息、更新业务记录和发布产物不能由「已生成结果」自动推出。</li>
</ul>
<h3>接入验收与版本范围</h3>
<p>接入前固定运行时版本及协议 schema，验证正常完成、拒绝审批、取消、部分写入后断线、重连、重复请求、超大输出和队列饱和。参数、默认权限、恢复范围及 SDK 暴露能力随实现变化；此处的职责与失败边界不代替具体版本的契约测试。</p>
<p>持久审批、外部幂等和业务状态恢复需要宿主、运行时与目标服务共同实现，不能仅由界面或传输协议保证。</p>
</section>

<section class="ca-sec">
<h2 id="references">参考依据</h2>
<p>规范与示例核验日期：2026-09-20。来源范围限于消息语义、会话流程、工具扩展和缓存示例，不用于证明其他产品的默认配置。</p>
<ol>
<li>JSON-RPC 2.0 Specification：Request、Notification 与 Response 对象定义。</li>
<li>Agent Client Protocol，官方仓库说明与 Protocol Overview（指向协议 v1）：代码编辑器与 Coding Agent 的接入范围，以及初始化、会话、prompt turn、权限请求与可选能力。协议定位复核日期：2026-09-25。</li>
<li>Model Context Protocol：Architecture Overview、Tools、Resources；用于区分工具扩展与宿主会话接入。</li>
<li>Anthropic 官方 Cookbook：<code>misc/prompt_caching.ipynb</code>，commit <code>6b671ef60ada2a8d3b0c07cadb424172da5135f5</code>。示例区分自动管理断点与显式断点，以及缓存写入和读取用量，不据此推定其他供应商的价格与行为。</li>
</ol>
</section>

</div>
