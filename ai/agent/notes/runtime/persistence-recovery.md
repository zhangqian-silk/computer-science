# 持久化与故障恢复

持久化保存运行所需状态，恢复根据已知状态继续工作，对账确认不确定的外部效果。三者不是同一个功能。一个原生会话可以恢复聊天，却仍不知道上次超时的发布是否实际成功。

本页将恢复边界落在模型请求、工具操作、任务和通知上；记忆内容语义见 [Memory](../context/conversation-memory.md)，多工作者资格与租约见 [调度](scheduling-protocols.md)。

## 恢复对象与状态范围

至少考虑用户目标及其版本、已接受输入、模型/工具调用记录、关键回执、待审批和未知操作、产物及其版本。UI增量可根据产品目标选择保留或丢弃，但必须说明恢复后是否还能展示用户曾看到的部分回答。

模型内部 KV 一般不属于应用可持久恢复的对象。工具使用的文件、浏览器登录态和进程也不必与会话一同存活。恢复记录应说明“可重建什么”和“必须重新验证什么”，不能只给一个笼统 resume 按钮。

## 任务、尝试、调用与操作标识

任务 ID 标识业务目标，attempt ID 标识一次执行尝试，native session ID 标识原生会话，operation ID 标识一次经授权的业务效果。模型 call ID 仅用于调用与结果配对。

如果 HTTP 响应丢失，下一次模型可能生成新 call ID；用它当幂等键会重复创建资源。反过来，用参数 hash 去重会吞掉用户明确要求的第二次相同操作。operation ID 应在业务意图和授权确定后由宿主生成并持久化，重复尝试沿用，同一内容的新意图重新生成。

## 副作用状态模型

```text
prepared → dispatched → succeeded
                  ├──→ failed_definitive
                  └──→ unknown → reconciliation → confirmed outcome
```

`unknown` 表示操作可能发生，但回执未获得或未保存。必须阻止把它自动转成“没执行”。准备记录也不证明已经发出，需保存适当的发送阶段或 provider request reference。

```python
operation = {
    "id": "op-42",
    "intent": "create_review_draft",
    "action_digest": "sha256:...",
    "state": "unknown",
    "provider_key": "op-42",
    "remote_reference": None,
    "last_error": "response_lost",
}
```

写入和网络发送之间无法凭一个本地事务天然保证原子性。需要远端幂等键、查询接口、业务去重、补偿或人工处理。没有这些能力时，应明确承认不能安全自动重试，而不是声称框架提供 exactly-once。

## 副作用对账算法

```python
# 教学伪代码：remote.query 必须区分权威不存在与暂时查不到。
def reconcile(operation, remote, store):
    observed = remote.query(operation["provider_key"])
    if observed["status"] == "succeeded":
        store.confirm(operation["id"], observed["receipt"])
        return "confirmed"
    if observed["status"] == "definitively_not_applied":
        store.mark_retryable(operation["id"])
        return "safe_to_retry_with_same_intent"
    store.keep_unknown(operation["id"], observed)
    return "needs_reconciliation_or_human"
```

读到404是否代表明确不存在取决于接口：可能有最终一致性、权限变化或查询范围不对。对账策略要使用业务语义和等待窗口，不把所有“未找到”都转成重发许可。

只有远端承诺同一幂等键重复调用返回同一效果时，才可在其范围与保留期内安全重试。幂等有效期过后仍要核验；也不能把不同参数偷偷复用同一key。

## Checkpoint 的内容与重放边界

Checkpoint 可以保存图节点结果、函数任务结果或事件快照。LangGraph 的持久化与函数式API文档区分这些边界，恢复时可能重新执行入口或节点，而不从源代码某一行继续。任务外副作用和非确定分支仍需自行管理。[[25]](../references.md#source-langgraph-persistence) [[46]](../references.md#source-langgraph-functional)

模型生成本身非确定且可能计费。已保存的模型响应可以复用以减少重复成本，但若配置、任务或工具目录变化，应显式决定是否重用。恢复对话时保留合法调用结果，不能把半个响应伪装成完整历史。

事件重放应该重建状态，而不是再次执行动作。`tool_requested` 事件不应在每次UI刷新时触发工具。将“事实投影器”和“动作派发器”分开，有助于避免这种严重错误。

## 状态与通知一致性

先把任务标为完成，再写完成事件，进程可能在两者之间崩溃。任务库完成了，UI却一直等通知。参考方案是将状态转移和 outbox 通知写入同一数据库事务，后台反复投递，消费者按事件ID去重。

```python
# 教学伪代码：事务只覆盖本地状态与待投递事件。
with database.transaction() as tx:
    tx.update_task(task_id, state="execution_finished")
    tx.insert_outbox({
        "event_id": event_id,
        "task_id": task_id,
        "kind": "execution_finished",
    })
```

outbox解决通知一致性，不解决远端工具的 exactly-once。没有必要为最小Agent一开始建设outbox，但多客户端和持久任务平台应选择一种明确的一致性方案。

## 部分文件与存储故障

JSONL 便于追加和诊断，但最后一行可能因崩溃不完整。恢复应验证 framing、序号和内容校验，仅对明确的未完成尾部按协议修复，不把中间损坏悄悄跳过。数据库事务同样有隔离级别、磁盘同步和备份恢复条件，不因使用SQLite/Postgres就自动拥有所需SLO。

产物要先完成写入并校验，再发布可读取引用；否则记录里有artifact ID，实际文件却不存在。删除和保留策略也应覆盖日志、产物和摘要中的敏感信息，不能为恢复无限保留所有数据。

## 产品恢复语义对照

Claude Code checkpointing 的覆盖范围与任意 Shell/数据库回滚不同。Herdr 文档又把脱离客户端后进程仍运行、服务器重启后布局恢复、屏幕历史重放、原生Agent会话恢复分开。这些实例说明“恢复成功”的对象必须写清楚。[[34]](../references.md#source-cc-checkpoints) [[60]](../references.md#source-herdr-state)

DeepSeek Harness 的可延续子会话也区分持久Session与当前Activation；管理器释放进程资源后，持久记录可能仍在。不要把会话可读、执行器活跃和任务完成视为同一个状态。[[51]](../references.md#source-dsh-subagent)

## 故障注入的最小矩阵

在这些边界强制中断：准备操作前；准备已写但未发出；远端成功但本地未保存；本地完成但通知未发；通知发出但未标已投递；产物写到一半；模型流部分输出；恢复时权限已改变。每项都检查实际效果数量、可恢复状态和用户看到的说明。

对外部写入，最重要的断言不是“重启不报错”，而是“未知时不盲重发，成功有可核验回执”。用模拟远端可以先测试状态机，再在隔离的真实服务环境验证幂等和最终一致性；不能用前者替代后者。

## 调优的正确顺序

先定义可接受数据损失、恢复时间和人工介入范围，再优化批量写、快照频率和压缩。减少同步写可能降低延迟，却增加崩溃窗口。无条件每token落盘又会扩大开销和隐私风险。区分显示增量与承诺动作，按不同可靠性要求保存，才能解释取舍。

---

## 崩溃窗口的可运行实验

将本地操作账本和模拟远端各放在独立SQLite文件。工作进程先写prepared，再向模拟远端用operation ID提交；远端事务已提交后，工作进程立即退出，不写本地成功回执。新进程打开本地账本，看到未确认操作，查询模拟远端，确认原效果并更新本地。

断言远端效果计数仍为1，第二进程没有发送新的业务意图。这个实验能验证参考状态机和本地重开，但不能证明真实服务的幂等保留期、最终一致性和网络语义；后者需要单独联调。

## 发送前后崩溃的处理差异

若只有prepared而没有发送证据，恢复仍不能总是假定没发送，因为进程可能在发送之后、写dispatched之前退出。这里需要将操作流程与远端幂等/查询设计一起考虑。单靠把数据库字段更新顺序调一调，无法跨两套系统制造原子事务。

有些业务可使用事务外发箱由专门发送器处理，但发送器仍可能遇到远端成功后本地确认丢失，最终还是需要远端去重或对账。因此outbox与幂等是互补，不是替代。

## 只读动作的恢复与版本

重新读取文件通常无副作用，但恢复后文件可能变化。复用旧读取结果和重新读取当前内容各有意义：前者用于解释过去决策，后者用于新的动作。记录版本并区分历史观察与当前观察，避免恢复任务把旧事实当作现在。
