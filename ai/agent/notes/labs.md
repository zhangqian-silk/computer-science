# 代码示例与实验

正文中的 Python 片段分为原生 API 示例、教学伪代码和数据结构示例。除明确指向 [`examples/`](https://github.com/zhangqian-silk/computer-science/tree/master/ai/agent/notes/examples) 的完整程序外，不应把单个片段直接当成生产代码。依赖参数旁的注释说明它由模型适配、检索服务、权限系统还是存储实现提供。

## 离线运行

本模块的 [`examples/`](https://github.com/zhangqian-silk/computer-science/tree/master/ai/agent/notes/examples) 目录使用 Python 3.11+ 标准库，无需 API Key：

```bash
cd ai/agent/notes/examples
python demo.py
python -m unittest discover -s . -v
```

[`demo.py`](https://github.com/zhangqian-silk/computer-science/blob/master/ai/agent/notes/examples/demo.py) 运行单次生成、工具直接返回与观察驱动闭环。这里采用确定性模拟模型，便于检查调用次数和消息；它不是模型质量基准。假模型会按照预设回复执行，不说明真实模型一定能选择正确动作。

## 代码阅读地图

| 正文机制 | 参考实现 | 应观察什么 |
| --- | --- | --- |
| [响应完整性](model/outputs.md) | `normalize_reply` | 截断、重复调用 ID、拒绝与执行提议混合时是否拒绝派发 |
| [工具执行](tools/tool-calling.md) | `ToolGateway`、`one_shot_tool` | 执行前校验，权限即时检查，结果可以直接交付 |
| [ReAct Loop](control/react-loop.md) | `run_loop`、`validate_pairs` | 调用原文与结果成对保留；停止提议不是验收成功 |
| [上下文选择](context/context-engineering.md) | `select_groups` | 必要内容不静默删除，工具调用组不被部分截断 |
| [检索融合](context/rag-retrieval.md) | `rrf` | 排名融合，而不是直接相加不同量纲分数 |
| [结果覆盖](control/subagents-multi-agent.md) | `require_coverage` | null、重复、未验证结果不能变成整体成功 |
| [注册表版本](tools/hooks-plugins.md) | `Registry` | 旧快照语义与新注册隔离；旧清理不能删除新代次 |
| [审批](runtime/human-control.md) | `ApprovalStore` | 绑定最终身份与参数；同意不能无限重复消费 |
| [预算](runtime/performance-cost.md) | `Budget` | 并发预留、释放和一次性结算；超额真实记录 |
| [增量处理](model/streaming.md) | `JsonLineStream` | UTF-8 分块、条目结束与响应结束区别 |
| [恢复对账](runtime/persistence-recovery.md) | `OperationLedger`、`FakeRemote`、`recover` | 不确定操作先查状态，不是自动再写一次 |

测试方法覆盖这些不变量及反例，不为代码行数或断言数量提供质量保证。全部实现位于 [`core.py`](https://github.com/zhangqian-silk/computer-science/blob/master/ai/agent/notes/examples/core.py)，对应测试位于 [`test_core.py`](https://github.com/zhangqian-silk/computer-science/blob/master/ai/agent/notes/examples/test_core.py)。

## 崩溃实验的验证范围

恢复测试创建两个独立 SQLite 文件：一个代表本地账本，另一个代表支持幂等键的模拟远端。受控崩溃子进程 [`crash_worker.py`](https://github.com/zhangqian-silk/computer-science/blob/master/ai/agent/notes/examples/crash_worker.py) 先记录派发意图，然后提交模拟远端效果，在保存本地回执前以退出码 17 结束。父进程重新打开文件，读取远端结果并提交回执，检查模拟远端效果数量仍为 1。

这个实验真实执行了一个进程退出窗口，但没有模拟所有网络行为。它成立的原因包括远端模拟器提供唯一键和可查询结果；不是因为本地数据库有一条 operation 记录，就能使任意外部服务实现 exactly-once。远端没有找到结果时，示例返回 `needs_reconciliation`，不自动重试写入。

## 未覆盖的验证范围

工具验证只是精确字段和 JSON 可表示性检查，不是完整 JSON Schema 校验器；真实业务需要字段类型、资源权限、版本前置条件以及授权身份。注册表快照复制的是绑定，不是隔离可变闭包，也没有实现完整卸载排空。

流实验采用自定义 JSONL 教学协议，并不是 SSE 或某个供应商的协议。Python 语法可解析不意味着片段中的外部依赖已经实现。审批没有真实认证，预算没有接入真实账单，模型没有连接在线推理服务。

可选的 [`live_api.py`](https://github.com/zhangqian-silk/computer-science/blob/master/ai/agent/notes/examples/live_api.py) 使用官方 OpenAI SDK 接口，要求单独固定 SDK 版本、设置账号权限和环境变量，可能收费；它未进入离线测试，也没有在此资料中执行。[[1]](references.md#source-openai-text)

## 真实产品联调规格

[`product-integration-cases.jsonl`](https://github.com/zhangqian-silk/computer-science/blob/master/ai/agent/notes/templates/product-integration-cases.jsonl) 提供 16 个待执行场景，包括流式截断、审批后改参、取消与后台工具、未知写入、上下文压缩、过期检索、缺失工作者、冷暖缓存、MCP 版本、旧执行者、租户边界、插件热更新、会话恢复和伪造日志。

这些条目统一标为「未联调」。运行时应记录二进制／SDK／模型版本、输入、时限、实际事件、资源变化与结论，不能只填一个通过或失败。测试范围改变以后，应重新定义可比较的基线。

## 调优实验设计

先固定一组真实失败与成功任务，再选择一个变量：如检索 top-k、工具结果预览长度、压缩阈值、并行度或模型推理预算。记录最终任务质量、p50/p95 延迟、模型与工具费用、失败类别以及原始 usage。

每个变体重复运行，保持相同模型与工具版本；含缓存实验同时标注冷暖条件。单次速度提升可能只是缓存偶然命中，费用下降也可能来自省略了必要步骤。优先比较已经验证的交付成本，而不是一个孤立请求的 token 数。

## 实验结论与实现边界

失败应回到所属机制，而不是统一归因于 Prompt。工具参数关联错误首先检查输出适配；权限错误检查工具网关；长对话遗漏检查上下文与记忆；完成后被超时覆盖检查终态与清理边界。只有证据表明模型获得了合适输入却仍做错语义判断，再评估提示、模型或规划策略。
