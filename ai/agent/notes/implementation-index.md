# 产品实现索引

本页按产品或资料来源定位对应的机制笔记，不重新介绍一遍产品。主阅读入口仍是 [整体结构](index.md)。某个实现只适用于指定提交或公开文档的范围；开源实现、产品文档、作者实践和社区观察的证据强度不同。

## 编码 Agent 与 Harness

| 研究对象 | 对应专题 | 重点观察的实现决策 |
| --- | --- | --- |
| Claude Code | [Skills](tools/skills.md)、[扩展](tools/hooks-plugins.md)、[人工控制](runtime/human-control.md)、[动态工作流](control/dynamic-workflows.md) | 模型可见内容与权限分开；Hooks 阶段；编排脚本持有中间值；同会话恢复的边界 |
| Codex | [Loop](control/react-loop.md)、[工具网关](tools/tool-calling.md)、[Harness](runtime/harness-architecture.md)、[异构接入](runtime/scheduling-protocols.md) | StepContext 固定工具语义；共享／排他执行；Thread/Turn/Item 与外部任务的映射 |
| Pi | [流式](model/streaming.md)、[Loop](control/react-loop.md)、[压缩](context/compaction.md)、[扩展](tools/hooks-plugins.md) | transformContext 与 convertToLlm；Steering/Follow-up；有序历史；摘要边界与扩展错误 |
| DeepSeek Harness | [Loop](control/react-loop.md)、[扩展](tools/hooks-plugins.md)、[子代理](control/subagents-multi-agent.md)、[安全](runtime/security.md) | 工具调度滚动池、排他屏障、冻结结果；可延续子代理与活跃实例；预览安全边界 |

Pi 的固定读取提交为 `6160683a4a8012f0d1cd30c145df18b4ca6f5176`；Codex 工具调度为 `634ebc1865c6ac840ed3ba118f040d527bf4b55d`；DSH 为 `5dda764ed3aa172535a7967b06ff95d9cbfe536a`。这些是可复查的分析对象，不标成最新版本。[[9]](references.md#source-pi-loop) [[14]](references.md#source-codex-tools) [[49]](references.md#source-dsh-scheduler)

建议阅读源码时先定位输入、状态修改和输出，再沿调用链追查副作用。一个函数名字叫 `complete`，不代表它验证了用户任务；一个回调叫 `cancel`，也不能据此认定它已经停止所有子进程。源码节摘取的是关键控制点，不提供完整产品安全认证。

## 调度、工作助理与终端管理

| 研究对象 | 阅读入口 | 需要保持的边界 |
| --- | --- | --- |
| Multica | [调度与协议](runtime/scheduling-protocols.md)、[观测](runtime/observability.md)、[人工控制](runtime/human-control.md) | Backend 的事件流与权威 Result；原生会话恢复；不同阶段的超时与能力差异 |
| Symphony | [Workflow](control/workflows.md)、[调度](runtime/scheduling-protocols.md)、[发布](runtime/versioning-release.md) | 仓库策略、工单资格、工作区与 Agent Runner；不是通用分布式任务系统规范 |
| OpenWorker | [工具网关](tools/tool-calling.md)、[安全](runtime/security.md)、[审批](runtime/human-control.md) | Reviewer 与硬拒绝；文件工具约束不能自然覆盖任意 Shell；自动审批模式的语义 |
| Herdr | [执行环境](tools/execution-environments.md)、[调度与协议](runtime/scheduling-protocols.md) | 终端占用、布局、服务进程与编码 Agent 原生会话是不同生命周期 |
| LangGraph | [Workflow](control/workflows.md)、[恢复](runtime/persistence-recovery.md)、[记忆](context/conversation-memory.md) | 节点与任务级检查点、重放和中断；持久化记录不等于自动避免外部重复效果 |

Multica 和 OpenWorker 的分析分别绑定 `b5a7ee1e0e75347bed5fb4590e2fb9a92b046353` 与 `5bc10d928e0b64aae74313349a3b17bd19643ae2`。统一接口只说明存在共同接入面，不能推断每个后端支持相同能力。[[61]](references.md#source-multica-session) [[31]](references.md#source-openworker-tests)

## 模型厂商和工程作者资料

模型接口页使用 OpenAI、Anthropic、DeepSeek 的官方 API 文档，关注原始响应、推理续接与缓存差异，不以兼容的请求外形代替协议验证。RAG 和 ReAct 的归属与基本方法引用原始论文；底层缓存机制参考模型库和推理框架文档。[[2]](references.md#source-openai-functions) [[5]](references.md#source-claude-tools) [[6]](references.md#source-deepseek-thinking) [[23]](references.md#source-rag-paper) [[48]](references.md#source-react-paper)

Anthropic 的工具、上下文、长期 Agent 和评测文章用于参考设计方法；OpenAI 的 Harness 与 App Server 资料用于分析宿主和接入职责；Factory 的压缩评估提供继续执行的测量思路；Cursor、Cognition、Cloudflare 的实践用来对照并行、任务分工与代码化工具组合。作者报告来自各自环境，不构成统一硬件、模型、预算下的产品基准。[[20]](references.md#source-anthropic-tools) [[29]](references.md#source-anthropic-long-harness) [[66]](references.md#source-anthropic-evals) [[21]](references.md#source-openai-harness) [[28]](references.md#source-factory-compact) [[53]](references.md#source-cursor-scaling) [[54]](references.md#source-cognition-context) [[55]](references.md#source-cloudflare-code)

## 社区资料的证据分级

论坛适合发现具体故障和不熟悉的工作负载，但不能以热度替代复现。先记录作者实际看到什么，再区分作者猜测和维护者确认，最后把观察改写成可注入的实验。

LangChain Forum 中有人报告子代理中间消息在刷新后消失。这个观察提示要分别检查实时事件、持久状态和前端重建；它不能直接证明所有子代理框架都丢数据，也不能证明存更多模型思考文本就是修复方案。可以在本地增加“输出期间刷新”和“最后检查点之后取消”两个测试，核对哪类数据承诺了持久化。[[15]](references.md#source-forum-subagent)

Hacker News 上一个项目作者介绍跨 Claude Code/Codex 的本地工作流上下文存储，并强调绑定具体原生会话而非寻找“最近的文件”。可借鉴的问题是：任务状态迁移、原生会话续接和模型缓存复用是否被错误混成一件事。作者项目的具体正确性没有在本资料中运行验证；评论里的计费推断应回到供应商原始文档检查。[[67]](references.md#source-hn-context-bridge)

另一条 DeepSeek 与编码 Harness 的讨论涉及缓存命中。单个工作负载的观察不能推出“所有其他 Harness 每轮都破坏缓存”。应记录模型、接口、原始请求、工具 schema、命中 token 及冷暖条件，再比较质量、延迟和总成本。[[68]](references.md#source-hn-harness-cache) [[18]](references.md#source-openai-cache)

## X 与不可读取的原始线索

保留了 Pi 会话分享及另一个 Agent/Harness 讨论的 X 原始链接；两次打开均未获得可读正文。这里不据此确认作者论点、功能发布或性能数字，也不把搜索结果标题当成完整文章。[[69]](references.md#source-x-pi-sharing) [[70]](references.md#source-x-harness-lead)

社区观点用于技术论证前，应核验作者、发布时间、完整上下文，以及对应代码或官方文档。正文不可读取的记录只作为线索，不能用于判断安全默认值、协议版本或产品能力。

## 资料维护方式

来源分为固定源码、公开接口契约、作者实验、社区观察、未读取线索。每次更新先判断原结论是否仍成立，再记录新版本和差异，不仅修改“核验日期”。若两个版本的取消或恢复规则不同，应保留明确的适用范围，而不是把所有差异并成一个看似通用的说明。

当源码与文档冲突，先报告冲突和实际测试条件；没有复现时不替上游宣布根因。实验代码只用于证明自己的设计不变量，不能给研究对象增加一项“已通过测试”的能力。
