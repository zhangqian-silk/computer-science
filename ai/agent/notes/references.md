# 参考资料

引用在正文中靠近所支持的结论。固定提交提供可复查的实现范围，官方在线页面表示核验时的公开契约；两者都不等于经过实际产品联调。教学建议与计算示例是笔记中的参考分析，不冒充上游实现。

源码范围、摘要阅读与不可读取的线索分别标明。在线文档可能变化，部署前应核对实际 SDK、二进制和模型能力。资料中涉及的历史产品版本仍按其对应范围解释；核验日期不代表该日期的全部最新能力均已核验。

社区报告用于提出测试问题，不替代维护者确认、源码审查或独立复现。X 正文不可读取的记录只作为线索，不支持技术结论。

---

<a id="source-openai-text"></a>

## [1] OpenAI：Text generation

来源类型：官方文档。核验日期：2026-09-10。在线文档未固定源码提交。

[OpenAI：Text generation](https://developers.openai.com/api/docs/guides/text)

阅读范围：一次 Responses 调用、消息输入和 output_text 与原始 output 的区别。

访问状态：已读取正文或标注的源码范围。

---

<a id="source-openai-functions"></a>

## [2] OpenAI：Function calling

来源类型：官方文档/作者工程实践。核验日期：2026-09-10。在线文档未固定源码提交。

[OpenAI：Function calling](https://developers.openai.com/api/docs/guides/function-calling)

阅读范围：结构化调用、工具结果关联、参数 schema 与流式边界。

访问状态：已读取正文或标注的源码范围。

---

<a id="source-anthropic-agents"></a>

## [3] Anthropic：Building effective agents

来源类型：作者工程实践。核验日期：2026-09-10。在线文档未固定源码提交。

[Anthropic：Building effective agents](https://www.anthropic.com/engineering/building-effective-agents)

阅读范围：Workflow/Agent 区分与组合模式；不作为跨任务性能排名。

访问状态：已读取正文或标注的源码范围。

---

<a id="source-hf-cache"></a>

## [4] Hugging Face：Caching

来源类型：官方机制说明。核验日期：2026-09-10。在线文档未固定源码提交。

[Hugging Face：Caching](https://huggingface.co/docs/transformers/cache_explanation)

阅读范围：仅用于传统自回归注意力的 KV 缓存机制；未核验具体库接口。

访问状态：已读取正文或标注的源码范围。

---

<a id="source-claude-tools"></a>

## [5] Claude：Tool use overview

来源类型：官方文档。核验日期：2026-09-10。在线文档未固定源码提交。

[Claude：Tool use overview](https://platform.claude.com/docs/en/agents-and-tools/tool-use/overview)

阅读范围：client/server tools、tool_use 与 tool_result 的原生表示。

访问状态：已读取正文或标注的源码范围。

---

<a id="source-deepseek-thinking"></a>

## [6] DeepSeek：Thinking mode

来源类型：官方文档/作者工程实践。核验日期：2026-09-10。在线文档未固定源码提交。

[DeepSeek：Thinking mode](https://api-docs.deepseek.com/guides/thinking_mode/)

阅读范围：工具推理上下文中的 reasoning_content 回传要求。

访问状态：已读取正文或标注的源码范围。

---

<a id="source-deepseek-compat"></a>

## [7] DeepSeek：GitHub Copilot CLI 接入注意事项

来源类型：官方文档。核验日期：2026-09-09。版本／提交：`页面核验快照；无固定提交时不称源码固定版`。

[DeepSeek：GitHub Copilot CLI 接入注意事项](https://api-docs.deepseek.com/quick_start/agent_integrations/copilot_cli/)

阅读范围：读取 provider 类型与 reasoning_content 回传不匹配案例。

访问状态：核验范围以所列日期、版本和阅读范围为限，不代表核验日后的行为。

---

<a id="source-openai-state"></a>

## [8] OpenAI：Conversation state

来源类型：官方文档。核验日期：2026-09-10。在线文档未固定源码提交。

[OpenAI：Conversation state](https://developers.openai.com/api/docs/guides/conversation-state)

阅读范围：历史续接、previous_response_id 与费用不是状态免费。

访问状态：已读取正文或标注的源码范围。

---

<a id="source-pi-loop"></a>

## [9] Pi：工具批次执行

来源类型：源码阅读。核验日期：2026-09-10。版本／提交：`6160683a4a8012f0d1cd30c145df18b4ca6f5176`。

[Pi：工具批次执行](https://github.com/earendil-works/pi/blob/6160683a4a8012f0d1cd30c145df18b4ca6f5176/packages/agent/src/agent-loop.ts)

阅读范围：读取指定提交的 145–325、425–720 行；循环控制与工具批次执行。

访问状态：已读取正文或标注的源码范围。

---

<a id="source-openai-structured"></a>

## [10] OpenAI：Structured model outputs

来源类型：官方文档。核验日期：2026-09-10。在线文档未固定源码提交。

[OpenAI：Structured model outputs](https://developers.openai.com/api/docs/guides/structured-outputs)

阅读范围：schema、拒绝、截断和业务校验的边界。

访问状态：已读取正文或标注的源码范围。

---

<a id="source-openai-reasoning"></a>

## [11] OpenAI：Reasoning models

来源类型：官方文档。核验日期：2026-09-10。在线文档未固定源码提交。

[OpenAI：Reasoning models](https://developers.openai.com/api/docs/guides/reasoning)

阅读范围：推理条目、不透明续接和上下文保留；按具体模型核验参数。

访问状态：已读取正文或标注的源码范围。

---

<a id="source-claude-thinking-legacy"></a>

## [12] Claude：Extended thinking (legacy)

来源类型：官方文档。核验日期：2026-09-10。在线文档未固定源码提交。

[Claude：Extended thinking (legacy)](https://platform.claude.com/docs/en/build-with-claude/extended-thinking)

阅读范围：manual thinking 的版本边界；不将 budget_tokens 推广到所有模型。

访问状态：已读取正文或标注的源码范围。

---

<a id="source-openai-stream"></a>

## [13] OpenAI：Streaming API responses

来源类型：官方文档。核验日期：2026-09-10。在线文档未固定源码提交。

[OpenAI：Streaming API responses](https://developers.openai.com/api/docs/guides/streaming-responses)

阅读范围：流式事件和类型化增量处理。

访问状态：已读取正文或标注的源码范围。

---

<a id="source-codex-tools"></a>

## [14] Codex：ToolCallRuntime

来源类型：源码阅读。核验日期：2026-09-10。版本／提交：`634ebc1865c6ac840ed3ba118f040d527bf4b55d`。

[Codex：ToolCallRuntime](https://github.com/openai/codex/blob/634ebc1865c6ac840ed3ba118f040d527bf4b55d/codex-rs/core/src/tools/parallel.rs)

阅读范围：读取 37–214 行：StepContext 绑定、读写门、终态与取消竞争。

访问状态：已读取正文或标注的源码范围。

---

<a id="source-forum-subagent"></a>

## [15] LangChain Forum：子 Agent 中间消息与页面重载

来源类型：社区一手报告。核验日期：2026-09-10。在线文档未固定源码提交。

[LangChain Forum：子 Agent 中间消息与页面重载](https://forum.langchain.com/t/how-to-persist-intermediate-subagent-messages-in-deep-agents-survive-page-reload/2991)

阅读范围：页面正文；用户现象与建议，不认定通用缺陷，不声称独立复现。

访问状态：已读取正文或标注的源码范围。

---

<a id="source-vllm-apc"></a>

## [16] vLLM：Automatic Prefix Caching

来源类型：官方文档。核验日期：2026-09-10。在线文档未固定源码提交。

[vLLM：Automatic Prefix Caching](https://docs.vllm.ai/en/latest/features/automatic_prefix_caching/)

阅读范围：Introduction 与 Limits；前缀 KV 复用作用于 prefill，不消除 decode。

访问状态：已读取正文或标注的源码范围。

---

<a id="source-claude-cache"></a>

## [17] Claude API：Prompt caching

来源类型：官方文档/作者工程实践。核验日期：2026-09-10。在线文档未固定源码提交。

[Claude API：Prompt caching](https://platform.claude.com/docs/en/build-with-claude/prompt-caching)

阅读范围：缓存写入/读取、块边界和生命周期；具体支持依模型。

访问状态：已读取正文或标注的源码范围。

---

<a id="source-openai-cache"></a>

## [18] OpenAI：Prompt caching

来源类型：官方文档/作者工程实践。核验日期：2026-09-10。在线文档未固定源码提交。

[OpenAI：Prompt caching](https://developers.openai.com/api/docs/guides/prompt-caching)

阅读范围：核验日文档包含模型族差异与缓存断点；不使用统一命中保证或固定价格。

访问状态：已读取正文或标注的源码范围。

---

<a id="source-openai-prompt"></a>

## [19] OpenAI：Prompt engineering

来源类型：官方文档。核验日期：2026-09-10。在线文档未固定源码提交。

[OpenAI：Prompt engineering](https://developers.openai.com/api/docs/guides/prompt-engineering)

阅读范围：指令、示例、输出要求与提示版本。

访问状态：已读取正文或标注的源码范围。

---

<a id="source-anthropic-tools"></a>

## [20] Anthropic：Writing tools for agents

来源类型：官方文档/作者工程实践。核验日期：2026-09-10。在线文档未固定源码提交。

[Anthropic：Writing tools for agents](https://www.anthropic.com/engineering/writing-tools-for-agents)

阅读范围：工具设计与评测实践；厂商经验不是跨任务最优证明。

访问状态：已读取正文或标注的源码范围。

---

<a id="source-openai-harness"></a>

## [21] OpenAI：Harness engineering

来源类型：厂商工程实践。核验日期：2026-09-10。版本／提交：`页面核验快照；无固定提交时不称源码固定版`。

[OpenAI：Harness engineering](https://openai.com/index/harness-engineering/)

阅读范围：读取仓库可理解性、机械约束与反馈；不推广其人效数字。

访问状态：已读取正文或标注的源码范围。

---

<a id="source-anthropic-context"></a>

## [22] Anthropic：Effective context engineering for AI agents

来源类型：官方文档/作者工程实践。核验日期：2026-09-10。在线文档未固定源码提交。

[Anthropic：Effective context engineering for AI agents](https://www.anthropic.com/engineering/effective-context-engineering-for-ai-agents)

阅读范围：上下文选择、按需信息获取与压缩。

访问状态：已读取正文或标注的源码范围。

---

<a id="source-rag-paper"></a>

## [23] Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks

来源类型：论文摘要。核验日期：2026-09-10。在线文档未固定源码提交。

[Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks](https://arxiv.org/abs/2005.11401)

阅读范围：阅读摘要；仅用于方法来源及原始检索器与生成器结合的定义，不引用实验数值。

访问状态：已读取正文或标注的源码范围。

---

<a id="source-langgraph-memory"></a>

## [24] LangChain / LangGraph：Memory overview

来源类型：官方文档。核验日期：2026-09-09。在线文档未固定源码提交。

[LangChain / LangGraph：Memory overview](https://docs.langchain.com/oss/python/concepts/memory)

阅读范围：线程级检查点与跨线程 namespace store、写入时机。

访问状态：核验范围以所列日期、版本和阅读范围为限，不代表核验日后的行为。

---

<a id="source-langgraph-persistence"></a>

## [25] LangGraph：Persistence

来源类型：官方文档/作者工程实践。核验日期：2026-09-10。在线文档未固定源码提交。

[LangGraph：Persistence](https://docs.langchain.com/oss/python/langgraph/persistence)

阅读范围：检查点与持久化；笔记中的外部副作用账本是独立的参考设计，不是该项目的原生实现。

访问状态：已读取正文或标注的源码范围。

---

<a id="source-cc-memory"></a>

## [26] Claude Code：项目指令与自动记忆

来源类型：官方文档。核验日期：2026-09-10。版本／提交：`在线文档`。

[Claude Code：项目指令与自动记忆](https://code.claude.com/docs/en/memory)

阅读范围：区分 CLAUDE.md 与 auto memory、上下文与强制配置、作用域和加载。未实测产品。

访问状态：已读取正文或标注的源码范围。

---

<a id="source-pi-compact"></a>

## [27] Pi：压缩与分支摘要

来源类型：项目文档。核验日期：2026-09-09。版本／提交：`6160683a4a8012f0d1cd30c145df18b4ca6f5176`。

[Pi：压缩与分支摘要](https://github.com/earendil-works/pi/blob/6160683a4a8012f0d1cd30c145df18b4ca6f5176/packages/coding-agent/docs/compaction.md)

阅读范围：读取 1–240 行；firstKeptEntryId、重复压缩、切分点、文件跟踪。

访问状态：核验范围以所列日期、版本和阅读范围为限，不代表核验日后的行为。

---

<a id="source-factory-compact"></a>

## [28] Factory：Evaluating compression

来源类型：官方文档/作者工程实践。核验日期：2026-09-10。在线文档未固定源码提交。

[Factory：Evaluating compression](https://factory.ai/news/evaluating-compression)

阅读范围：用后续任务检验压缩质量；不引用其数据作为独立产品排名。

访问状态：已读取正文或标注的源码范围。

---

<a id="source-anthropic-long-harness"></a>

## [29] Anthropic：Effective harnesses for long-running agents

来源类型：厂商工程实践。核验日期：2026-09-10。版本／提交：`页面核验快照；无固定提交时不称源码固定版`。

[Anthropic：Effective harnesses for long-running agents](https://www.anthropic.com/engineering/effective-harnesses-for-long-running-agents)

阅读范围：读取初始化、进度文件、增量执行和验收实践；不是普适效果基准。

访问状态：已读取正文或标注的源码范围。

---

<a id="source-dsh-tools"></a>

## [30] DeepSeek Harness：工具管线定义

来源类型：源码阅读。核验日期：2026-09-10。版本／提交：`5dda764ed3aa172535a7967b06ff95d9cbfe536a`。

[DeepSeek Harness：工具管线定义](https://github.com/deepseek-ai/deepseek-harness/blob/5dda764ed3aa172535a7967b06ff95d9cbfe536a/packages/core/tools/src/index.ts)

阅读范围：读取指定提交的 1–287 行；canonical output、finalizeContent、并发与协作取消契约。

访问状态：已读取正文或标注的源码范围。

---

<a id="source-openworker-tests"></a>

## [31] OpenWorker：审批与 Shell 边界回归测试

来源类型：源码测试阅读。核验日期：2026-09-09。版本／提交：`5bc10d928e0b64aae74313349a3b17bd19643ae2`。

[OpenWorker：审批与 Shell 边界回归测试](https://github.com/andrewyng/openworker/blob/5bc10d928e0b64aae74313349a3b17bd19643ae2/tests/test_auto_approve.py)

阅读范围：读取 1–200 行；hard floor、配置别名、非法裁决与 OPE-113 的提示保证回归；只阅读，未执行上游测试。

访问状态：核验范围以所列日期、版本和阅读范围为限，不代表核验日后的行为。

---

<a id="source-python-subprocess"></a>

## [32] Python：subprocess

来源类型：语言官方文档。核验日期：2026-09-09。版本／提交：`页面核验快照；无固定提交时不称源码固定版`。

[Python：subprocess](https://docs.python.org/3/library/subprocess.html)

阅读范围：读取参数列表、进程退出、通信与超时；进程树和安全限制由正文单列。

访问状态：核验范围以所列日期、版本和阅读范围为限，不代表核验日后的行为。

---

<a id="source-cc-sandbox"></a>

## [33] Claude Code：Sandboxing

来源类型：官方文档/作者工程实践。核验日期：2026-09-10。在线文档未固定源码提交。

[Claude Code：Sandboxing](https://code.claude.com/docs/en/sandboxing)

阅读范围：文件系统、网络隔离与审批的区别。

访问状态：已读取正文或标注的源码范围。

---

<a id="source-cc-checkpoints"></a>

## [34] Claude Code：Checkpointing

来源类型：官方文档。核验日期：2026-09-10。版本／提交：`页面核验快照；无固定提交时不称源码固定版`。

[Claude Code：Checkpointing](https://code.claude.com/docs/en/checkpointing)

阅读范围：读取文件编辑检查点、恢复范围与限制。

访问状态：已读取正文或标注的源码范围。

---

<a id="source-mcp-release"></a>

## [35] MCP：2026-07-28 specification release

来源类型：维护者发布说明。核验日期：2026-09-10。版本／提交：`2026-07-28`。

[MCP：2026-07-28 specification release](https://blog.modelcontextprotocol.io/posts/2026-07-28/)

阅读范围：无初始化/协议会话、显式状态、缓存列表；规范支持不等于所有宿主兼容。

访问状态：已读取正文或标注的源码范围。

---

<a id="source-mcp-ts-versions"></a>

## [36] MCP TypeScript SDK：Protocol versions

来源类型：官方 SDK 文档。核验日期：2026-09-10。版本／提交：`SDK v2`。

[MCP TypeScript SDK：Protocol versions](https://ts.sdk.modelcontextprotocol.io/v2/protocol-versions)

阅读范围：legacy 与 modern 两族协议、自动发现与回退。

访问状态：已读取正文或标注的源码范围。

---

<a id="source-mcp-security-202511"></a>

## [37] MCP：Security Best Practices（2025-11-25）

来源类型：版本化规范。核验日期：2026-09-09。版本／提交：`页面核验快照；无固定提交时不称源码固定版`。

[MCP：Security Best Practices（2025-11-25）](https://modelcontextprotocol.io/docs/2025-11-25/tutorials/security/security_best_practices)

阅读范围：读取 confused deputy、token passthrough 等威胁；不冒充 2026-07-28 传输规范。

访问状态：核验范围以所列日期、版本和阅读范围为限，不代表核验日后的行为。

---

<a id="source-acp"></a>

## [38] Agent Client Protocol：Introduction

来源类型：官方文档/作者工程实践。核验日期：2026-09-10。在线文档未固定源码提交。

[Agent Client Protocol：Introduction](https://agentclientprotocol.com/get-started/introduction)

阅读范围：宿主与 Agent 协作的协议定位；不声称各宿主完整实现。

访问状态：已读取正文或标注的源码范围。

---

<a id="source-a2a"></a>

## [39] A2A：Specification

来源类型：官方文档/作者工程实践。核验日期：2026-09-10。在线文档未固定源码提交。

[A2A：Specification](https://a2a-protocol.org/latest/specification/)

阅读范围：Agent 间任务与产物通信定位；latest 为可变文档。

访问状态：已读取正文或标注的源码范围。

---

<a id="source-skills-spec"></a>

## [40] Agent Skills：格式规范

来源类型：官方文档。核验日期：2026-09-10。在线文档未固定源码提交。

[Agent Skills：格式规范](https://agentskills.io/specification)

阅读范围：目录、元数据、正文与可选资源；宿主特有字段不视为通用授权协议。

访问状态：已读取正文或标注的源码范围。

---

<a id="source-cc-skills"></a>

## [41] Claude Code：Skills

来源类型：官方文档/作者工程实践。核验日期：2026-09-10。在线文档未固定源码提交。

[Claude Code：Skills](https://code.claude.com/docs/en/skills)

阅读范围：按需加载、调用控制、正文生命周期、宿主扩展字段；以读取日文档为准。

访问状态：已读取正文或标注的源码范围。

---

<a id="source-pi-ext-doc"></a>

## [42] Pi：扩展接口与生命周期

来源类型：项目文档。核验日期：2026-09-09。版本／提交：`6160683a4a8012f0d1cd30c145df18b4ca6f5176`。

[Pi：扩展接口与生命周期](https://github.com/earendil-works/pi/blob/6160683a4a8012f0d1cd30c145df18b4ca6f5176/packages/coding-agent/docs/extensions.md)

阅读范围：读取 1–230 行；同进程权限、工厂初始化、资源启动与收尾。

访问状态：核验范围以所列日期、版本和阅读范围为限，不代表核验日后的行为。

---

<a id="source-pi-ext-runner"></a>

## [43] Pi：ExtensionRunner

来源类型：源码阅读。核验日期：2026-09-09。版本／提交：`6160683a4a8012f0d1cd30c145df18b4ca6f5176`。

[Pi：ExtensionRunner](https://github.com/earendil-works/pi/blob/6160683a4a8012f0d1cd30c145df18b4ca6f5176/packages/coding-agent/src/core/extensions/runner.ts)

阅读范围：读取 1–180、720–1020 行；惰性上下文、失效检查、事件传播、结果改写。

访问状态：核验范围以所列日期、版本和阅读范围为限，不代表核验日后的行为。

---

<a id="source-cc-hooks"></a>

## [44] Claude Code：Hooks reference

来源类型：官方文档/作者工程实践。核验日期：2026-09-10。在线文档未固定源码提交。

[Claude Code：Hooks reference](https://code.claude.com/docs/en/hooks)

阅读范围：事件、参数改写、执行前决策与后处理边界。

访问状态：已读取正文或标注的源码范围。

---

<a id="source-cc-hooks-guide"></a>

## [45] Claude Code：Hooks guide

来源类型：官方文档/作者工程实践。核验日期：2026-09-09。在线文档未固定源码提交。

[Claude Code：Hooks guide](https://code.claude.com/docs/en/hooks-guide)

阅读范围：匹配 Hook 的并行运行、配置与调试。

访问状态：核验范围以所列日期、版本和阅读范围为限，不代表核验日后的行为。

---

<a id="source-langgraph-functional"></a>

## [46] LangGraph：Functional API 与重放

来源类型：官方文档。核验日期：2026-09-10。版本／提交：`页面核验快照；无固定提交时不称源码固定版`。

[LangGraph：Functional API 与重放](https://docs.langchain.com/oss/python/langgraph/functional-api)

阅读范围：读取确定性、任务结果重放、幂等性与常见错误部分。

访问状态：已读取正文或标注的源码范围。

---

<a id="source-langgraph-interrupts"></a>

## [47] LangGraph：Interrupts

来源类型：官方文档。核验日期：2026-09-10。版本／提交：`页面核验快照；无固定提交时不称源码固定版`。

[LangGraph：Interrupts](https://docs.langchain.com/oss/python/langgraph/interrupts)

阅读范围：读取恢复从节点起点重执行、人工响应与副作用约束。

访问状态：已读取正文或标注的源码范围。

---

<a id="source-react-paper"></a>

## [48] ReAct: Synergizing Reasoning and Acting in Language Models

来源类型：原始论文。核验日期：2026-09-10。版本／提交：`2022/2023`。

[ReAct: Synergizing Reasoning and Acting in Language Models](https://arxiv.org/abs/2210.03629)

阅读范围：用于 ReAct 的方法归属，不用于说明当前产品功能。

访问状态：已读取正文或标注的源码范围。

---

<a id="source-dsh-scheduler"></a>

## [49] DeepSeek Harness：批次调度

来源类型：源码阅读。核验日期：2026-09-10。版本／提交：`5dda764ed3aa172535a7967b06ff95d9cbfe536a`。

[DeepSeek Harness：批次调度](https://github.com/deepseek-ai/deepseek-harness/blob/5dda764ed3aa172535a7967b06ff95d9cbfe536a/packages/core/agent-loop/src/tool-calls.ts)

阅读范围：读取 1–245 行：滚动池、顺序提交、排他屏障和取消后停止派发工作。

访问状态：已读取正文或标注的源码范围。

---

<a id="source-cc-runtime"></a>

## [50] Claude Code：How Claude Code works

来源类型：官方文档。核验日期：2026-09-10。在线文档未固定源码提交。

[Claude Code：How Claude Code works](https://code.claude.com/docs/en/how-claude-code-works)

阅读范围：公开 Loop、工具、上下文、会话与环境行为，不等于内部源码。

访问状态：已读取正文或标注的源码范围。

---

<a id="source-dsh-subagent"></a>

## [51] DeepSeek Harness：可延续子代理

来源类型：项目文档。核验日期：2026-09-09。版本／提交：`5dda764ed3aa172535a7967b06ff95d9cbfe536a`。

[DeepSeek Harness：可延续子代理](https://github.com/deepseek-ai/deepseek-harness/blob/5dda764ed3aa172535a7967b06ff95d9cbfe536a/docs/subsystems/subagent.md)

阅读范围：读取开头及 1–220 行可返回内容；能力拒绝、Session/Activation、接收所有权、drain 与最终 flush 边界。

访问状态：核验范围以所列日期、版本和阅读范围为限，不代表核验日后的行为。

---

<a id="source-cc-teams"></a>

## [52] Claude Code：Agent Teams

来源类型：官方文档/作者工程实践。核验日期：2026-09-10。在线文档未固定源码提交。

[Claude Code：Agent Teams](https://code.claude.com/docs/en/agent-teams)

阅读范围：读取角色、通信和限制；页面含多个产品版本边界，TeamCreate/TeamDelete 的适用范围须按对应版本核对，不能推广为通用接口。

访问状态：已读取正文或标注的源码范围。

---

<a id="source-cursor-scaling"></a>

## [53] Cursor：Scaling long-running autonomous coding

来源类型：厂商实验记录。核验日期：2026-09-10。版本／提交：`页面核验快照；无固定提交时不称源码固定版`。

[Cursor：Scaling long-running autonomous coding](https://cursor.com/blog/scaling-agents)

阅读范围：读取共享文件协调的失败、职责拆分与实验边界；不是产品 SLA。

访问状态：已读取正文或标注的源码范围。

---

<a id="source-cognition-context"></a>

## [54] Cognition：Don’t Build Multi-Agents

来源类型：作者观点与实践。核验日期：2026-09-10。版本／提交：`页面核验快照；无固定提交时不称源码固定版`。

[Cognition：Don’t Build Multi-Agents](https://cognition.com/blog/dont-build-multi-agents)

阅读范围：读取上下文丢失、隐含决策冲突；2025 年观点，不用于描述当前 Claude Code 能力。

访问状态：已读取正文或标注的源码范围。

---

<a id="source-cloudflare-code"></a>

## [55] Cloudflare：Code Mode

来源类型：厂商工程实践。核验日期：2026-09-10。版本／提交：`页面核验快照；无固定提交时不称源码固定版`。

[Cloudflare：Code Mode](https://blog.cloudflare.com/code-mode/)

阅读范围：读取代码组合工具与执行隔离思想；不引用宣传性节省百分比。

访问状态：已读取正文或标注的源码范围。

---

<a id="source-cc-workflows"></a>

## [56] Claude Code：Dynamic Workflows

来源类型：官方文档/作者工程实践。核验日期：2026-09-10。在线文档未固定源码提交。

[Claude Code：Dynamic Workflows](https://code.claude.com/docs/en/workflows)

阅读范围：读取全文，重点核验脚本控制边界、null 结果及同会话启动顺序连续前缀重放；这是公开行为契约，不代表已审查闭源内部实现。

访问状态：已读取正文或标注的源码范围。

---

<a id="source-codex-app"></a>

## [57] Codex App Server

来源类型：官方文档。核验日期：2026-09-09。在线文档未固定源码提交。

[Codex App Server](https://learn.chatgpt.com/docs/app-server)

阅读范围：Thread/Turn/Item、请求通知与原生会话；当前开发者文档重定向目标。

访问状态：核验范围以所列日期、版本和阅读范围为限，不代表核验日后的行为。

---

<a id="source-openai-app-article"></a>

## [58] OpenAI：App Server 的线程、轮次、事件和审批

来源类型：厂商工程实践。核验日期：2026-09-09。版本／提交：`页面核验快照；无固定提交时不称源码固定版`。

[OpenAI：App Server 的线程、轮次、事件和审批](https://openai.com/index/unlocking-the-codex-harness/)

阅读范围：读取交互协议、客户端重连与 Harness 分离部分。

访问状态：核验范围以所列日期、版本和阅读范围为限，不代表核验日后的行为。

---

<a id="source-python-asyncio"></a>

## [59] Python：Coroutines and Tasks

来源类型：语言官方文档。核验日期：2026-09-09。版本／提交：`页面核验快照；无固定提交时不称源码固定版`。

[Python：Coroutines and Tasks](https://docs.python.org/3/library/asyncio-task.html)

阅读范围：读取 TaskGroup、gather、取消和超时语义；实验使用 Python 3.11+ 可用子集。

访问状态：核验范围以所列日期、版本和阅读范围为限，不代表核验日后的行为。

---

<a id="source-herdr-state"></a>

## [60] Herdr：Session state

来源类型：官方文档/作者工程实践。核验日期：2026-09-10。在线文档未固定源码提交。

[Herdr：Session state](https://herdr.dev/docs/session-state/)

阅读范围：客户端断开、服务器重启、原生续接与实验性 handoff 的差别。

访问状态：已读取正文或标注的源码范围。

---

<a id="source-multica-session"></a>

## [61] Multica：Backend 会话与结果契约

来源类型：源码阅读。核验日期：2026-09-10。版本／提交：`b5a7ee1e0e75347bed5fb4590e2fb9a92b046353`。

[Multica：Backend 会话与结果契约](https://github.com/multica-ai/multica/blob/b5a7ee1e0e75347bed5fb4590e2fb9a92b046353/server/pkg/agent/agent.go)

阅读范围：读取 1–215 行；Backend、ExecOptions、多个超时、会话连续性，以及 Messages／Result／TerminalObserved 的边界。

访问状态：已读取正文或标注的源码范围。

---

<a id="source-symphony"></a>

## [62] Symphony 服务规格

来源类型：项目规格。核验日期：2026-09-09。在线文档未固定源码提交。

[Symphony 服务规格](https://github.com/openai/symphony/blob/main/SPEC.md)

阅读范围：读取 1–155 行；调度器非目标、工作区、轮询及重启恢复。文件内容 blob 为 cd24131a1e2358cbfecc4f6efb028fc9fc6edefc；main 是可变链接，不作为固定 commit。

访问状态：核验范围以所列日期、版本和阅读范围为限，不代表核验日后的行为。

---

<a id="source-herdr-api"></a>

## [63] Herdr：Socket API

来源类型：官方文档/作者工程实践。核验日期：2026-09-10。在线文档未固定源码提交。

[Herdr：Socket API](https://herdr.dev/docs/socket-api/)

阅读范围：占用者身份绑定的 wait、prompt/wait 组合与重连快照。

访问状态：已读取正文或标注的源码范围。

---

<a id="source-dsh-safety"></a>

## [64] DeepSeek Harness：安全说明

来源类型：项目安全说明。核验日期：2026-09-10。版本／提交：`5dda764ed3aa172535a7967b06ff95d9cbfe536a`。

[DeepSeek Harness：安全说明](https://github.com/deepseek-ai/deepseek-harness/blob/5dda764ed3aa172535a7967b06ff95d9cbfe536a/SAFETY.md)

阅读范围：明确为未安全审计的开发者预览；沙箱与审批不是隔离保证。

访问状态：已读取正文或标注的源码范围。

---

<a id="source-otel-genai-repo"></a>

## [65] OpenTelemetry：GenAI semantic conventions repository

来源类型：项目文档。核验日期：2026-09-09。版本／提交：`页面核验快照；无固定提交时不称源码固定版`。

[OpenTelemetry：GenAI semantic conventions repository](https://github.com/open-telemetry/semantic-conventions-genai/blob/main/README.md)

阅读范围：读取 README；核验覆盖 Agent／模型／MCP 约定，未审查完整实现。

访问状态：核验范围以所列日期、版本和阅读范围为限，不代表核验日后的行为。

---

<a id="source-anthropic-evals"></a>

## [66] Anthropic：Demystifying evals for AI agents

来源类型：官方文档/作者工程实践。核验日期：2026-09-10。在线文档未固定源码提交。

[Anthropic：Demystifying evals for AI agents](https://www.anthropic.com/engineering/demystifying-evals-for-ai-agents)

阅读范围：任务、轨迹、环境与 grader 的评估边界。

访问状态：已读取正文或标注的源码范围。

---

<a id="source-hn-context-bridge"></a>

## [67] Hacker News：作者介绍本地跨会话上下文工具

来源类型：社区作者经验。核验日期：2026-09-10。在线文档未固定源码提交。

[Hacker News：作者介绍本地跨会话上下文工具](https://news.ycombinator.com/item?id=47836740)

阅读范围：阅读作者对 SQLite 工作流状态、原生会话绑定和恢复包的说明；未运行其项目，不采纳评论中的计费推断。

访问状态：已读取正文或标注的源码范围。

---

<a id="source-hn-harness-cache"></a>

## [68] Hacker News：DeepSeek 与编码 Harness 缓存讨论

来源类型：社区讨论。核验日期：2026-09-10。在线文档未固定源码提交。

[Hacker News：DeepSeek 与编码 Harness 缓存讨论](https://news.ycombinator.com/item?id=48257509)

阅读范围：阅读原帖及缓存争论；只作实验线索，不使用个体命中率或评论推断作为普遍结论。

访问状态：已读取正文或标注的源码范围。

---

<a id="source-x-pi-sharing"></a>

## [69] X：Pi 作者分享会话资料的原始链接

来源类型：社交平台线索。核验日期：2026-09-10。在线文档未固定源码提交。

[X：Pi 作者分享会话资料的原始链接](https://x.com/badlogicgames/status/2037811643774652911)

阅读范围：打开返回无可读正文；仅保存线索，不用于支持技术或性能结论。

访问状态：正文不可读取；不作为事实证据。

---

<a id="source-x-harness-lead"></a>

## [70] X：Agent/Harness 原始讨论链接

来源类型：社交平台线索。核验日期：2026-09-10。在线文档未固定源码提交。

[X：Agent/Harness 原始讨论链接](https://x.com/Vtrivedy10/status/2038346865775874285)

阅读范围：打开返回无可读正文，作者身份和论点未据此确认；仅保存待核验线索。

访问状态：正文不可读取；不作为事实证据。
