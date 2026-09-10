# Harness 架构与接口

Harness 是把模型接口、上下文、工具和控制策略组织起来的运行程序。它可以非常小，也可以包含会话、扩展、隔离与后台执行。本页从已解释的机制推导代码职责，不把某种分层当成 Agent 的必要定义。

阅读前可先看 [总览](../index.md) 与 [Loop](../control/react-loop.md)。这里不再解释 Tool Calling 是什么，而是说明这些已经认识的部分怎样形成可维护的软件。

## 单进程 Harness 的基本结构

一次 Agent 运行可以由一个 model adapter、一个 context builder、一个 tool executor 和一个 loop 函数组成。状态先保存在内存，结果交由任务专属 verifier 检查；只有需要重启继续时再引入持久化。

```python
# 参考接口关系。具体对象由应用启动代码注入。
class Harness:
    def __init__(self, model, context_builder, tools, verifier):
        self.model = model
        self.context_builder = context_builder
        self.tools = tools
        self.verifier = verifier

    def one_model_step(self, state):
        request = self.context_builder.build(state, self.tools.catalog())
        reply = self.model.complete(request)
        return request, reply
```

这个类没有必要立刻成为万能框架。核心是依赖可以在测试中替换：模型返回固定 fixture，工具只访问临时资源，verifier 读取独立事实。运行控制仍由 Loop 负责，类名不能代替控制语义。

## 职责划分与变化边界

| 职责 | 主要读取 | 主要产出 | 不应承担 |
| --- | --- | --- | --- |
| Model adapter | 请求、provider配置 | 原生＋规范化响应 | 业务授权、任务分解 |
| Context builder | 当前目标、历史、证据 | 消息、工具描述、manifest | 执行外部动作 |
| Tool gateway | 调用、身份、策略 | 事实回执与结果投影 | 决定整个任务主路径 |
| Loop/Workflow | 响应、观察、控制输入 | 下一步/暂停/结束 | 供应商序列化细节 |
| State store | 已确认事件和产物 | 版本化状态与读取 | 靠模型文字猜事实 |
| Verifier | 候选、证据、任务约束 | 验收结论与缺口 | 默许所有合法JSON正确 |

这样的划分允许更换供应商时主要改 adapter，改变检索策略时主要改 builder，增加权限规则时主要改 gateway。它们可以是同文件内的函数，不要求独立部署。

一段逻辑跨多个职责时，应找出真正的决策点。例如 timeout 既可能是 HTTP 设置，也可能是业务截止时间；前者由 adapter 执行，后者由运行控制派发剩余 deadline。不能每层独立设置默认十分钟，导致总任务无法预测。

## 配置与依赖注入

配置描述组合选择，代码实现机制。建议把模型、工具目录、Prompt、Skills、策略和环境版本放进运行清单，在启动或请求边界生成不可变快照。凭据用引用，不直接写入模型输入或可导出日志。

```python
run_manifest = {
    "model_profile": "provider-A/profile-3",
    "adapter_revision": "a-7",
    "prompt_revision": "p-12",
    "tool_catalog_revision": "t-4",
    "skill_versions": {"config-review": "2"},
    "policy_revision": "policy-18",
    "workspace_revision": "git:abc123",
    "limits": {"steps": 12, "deadline_seconds": 180},
}
```

这个清单描述采用了什么版本，不冻结管理员的即时安全撤销。注册表快照保证调用与描述一致，执行时权限检查保证当前仍允许。两个时间语义应分别实现。

不要让模型修改这份清单以给自己更多权限。模型可以建议更换模型或增加工具，由宿主按预算和策略判断；否则配置系统变成新的越权工具。

## 产品实现的组合边界

Pi 将轻量 Agent 核心、模型适配与更上层编码会话分开，适合观察一个较小循环怎样被嵌入产品；DSH 将更多服务和循环本身开放为插件，适合研究组合与生命周期。它们的不同不是简单的功能优劣，而是可替换深度与维护复杂度的取舍。[[9]](../references.md#source-pi-loop) [[30]](../references.md#source-dsh-tools)

Codex 的 App Server 暴露 Thread/Turn/Item 等客户端可用的执行接口，说明终端界面和核心运行不必是同一层。自建宿主接入成熟 Agent 时，可以驱动原生会话，而不是再次实现其工具循环。具体协议与能力不能仅凭“兼容Agent”名称推断。[[57]](../references.md#source-codex-app) [[58]](../references.md#source-openai-app-article)

Claude Code 的公开运行机制可用于理解产品层行为，但不应据界面表现臆造其全部内部对象和线程模型。选择“自己写Loop”还是“驱动已有Harness”，首先看需要控制哪些边界，以及原生接口是否暴露它们。

## 适配接口与能力协商

统一所有 provider 和 Agent 的单个接口很容易丢掉重要语义。比如某个 Agent 可以 resume，另一个只支持新会话；某个支持主动中断，另一个只能杀进程；某个有结构化 final result，另一个仅有终端文本。

采用能力协商和明确的 `unsupported` 比静默忽略好。统一层可提供启动、事件、结果、取消等基本契约，同时保留原生 session 与 continuation 数据。高级功能逐项声明，不能以“函数签名相同”作为功能等价证明。

## Middleware 的边界

Middleware 适合日志、时间测量或局部转换，但不是所有逻辑都适合串成一个可任意重入的链。预算扣减、授权和操作账本往往需要事务及固定顺序；若可以被任意插件跳过，它们就不能提供硬保证。

为每个扩展阶段定义输入所有权与可变字段。纯观察器不许改回执，参数变换之后必须校验，执行包装器不能脱离调用方取消。详细组合规则见 [Hooks 与 Plugins](../tools/hooks-plugins.md)。

## 单进程与服务化的适用条件

需要多个用户、长任务跨进程恢复、独立资源隔离或异构机器调度时，再考虑服务化。服务化带来的不是把函数改HTTP那么简单，还增加身份转移、重试、消息顺序、租约、幂等和部分失败。

可先把存储和执行接口抽象稳定，再外移某些边界。不要先引入独立“证据服务”，却尚未定义一条工具成功回执是什么；也不要把每个工具都变成微服务，增加网络复杂度而没有隔离收益。

## 测试结构与开发反馈

建立三层测试：adapter 解析fixture；机制单测验证Loop、Gate、上下文和生命周期；集成测试在临时工作区运行真实工具。真实模型测试单独放在可选集合，记录模型与时间，不让普通单元测试依赖网络费用。

OpenAI 的 Harness 工程实践和 Anthropic 长任务文章强调让工作环境中的知识、约束与反馈可用。对自建开发最可操作的借鉴是：文档路径可发现、验证命令可执行、产物可审查、失败有明确反馈；不是把一个长系统提示称为工程系统。[[21]](../references.md#source-openai-harness) [[29]](../references.md#source-anthropic-long-harness)

## 设计审查问题

审查时沿一次运行问：输入由谁构造；工具绑定何时固定；权限在哪里执行；谁拥有当前进程；用户取消影响哪些工作；事实存在哪里；谁确认完成；版本变化如何解释。回答应能落到接口、数据和测试，不能只落到架构图中的盒子。

当出现巨大类同时处理UI、检索、供应商字段、工具授权和重试，就按这些责任拆分；反过来，若只需要一个只读工具循环，就保持简单。Harness 的价值在于明确且可验证的边界，不在于模块数量。

---

## 请求处理与职责协作

```text
输入接受 → 读取目标与历史 → builder生成请求与manifest
       → adapter编码并调用模型 → adapter解释响应
       → Loop决定进入工具分支
       → gateway绑定工具、校验、审批、执行
       → store保存观察 → 下一步builder重新构造
       → 模型提出答案 → verifier检查 → 交付
```

这张顺序图不是部署图。Model、Store和工具可能是远程服务，也可以是本地对象；顺序表达谁对哪个决定负责。若为了异步性能改变顺序，仍需保持授权在执行前、调用结果配对和终态依据等不变量。

## 端口、适配器与实现替换

对业务代码暴露小接口，例如模型“得到一次完整响应”、工具“执行已校验动作”、存储“读写版本化状态”。接口应包含实际需要的失败语义，不用一个`Exception`吞掉拒绝、未知效果和暂时故障。

对于不支持的能力，返回显式unsupported，或者让构造阶段拒绝组合。这样把Pi式Loop换成已有Agent SDK时，会清楚发现哪些控制点无法保留，而不是继续使用同一类名却失去语义。

## 状态所有权与一致性

如果UI与后端都能直接修改任务状态，或者Loop和调度器都独立重试同一工具，就出现两个拥有者。选择一个权威写入点，其他层通过命令或事件请求变更。

模型也不应直接写“事实状态”字段。它输出候选计划、动作或答案，程序依据实际执行更新观察和验证。保持这一分工，后续才能可靠重放和审计。
