# 任务调度与 Agent 接入

任务调度决定哪个任务由哪个执行者在什么资源限制下运行；Agent Loop 决定一个任务内部的下一动作。两者可以在同一进程，也可以分成控制服务与工作者。本页从单任务运行管理扩展到队列与异构接口，不把终端管理、会话续接和业务调度混为一谈。

## 任务、尝试、原生会话与进程

任务代表用户目标，尝试代表一次执行，原生会话保存某个Agent的交互状态，进程是当前承载执行的操作系统对象。一个任务可以多次尝试复用同一原生会话，也可以因不兼容新建会话，但必须声明连续性变化。

一个进程退出不证明任务失败，可能已经写完产物；一个会话仍存在不证明执行正在进行；一个终端恢复布局也不证明原进程活着。Herdr 的状态文档正是把这些恢复路径分开。[[60]](../references.md#source-herdr-state)

## 统一接入接口与权威终态

```python
# 数据契约示例：事件和最终结果分开。
execution = {
    "task_id": "task-7",
    "attempt_id": "attempt-2",
    "runtime": "codex-app-server",
    "native_session": "native-31",
    "status": "running",
}
terminal_result = {
    "execution_status": "completed",
    "verification_status": "not_run",
    "cleanup_status": "pending",
    "artifact_refs": ["patch-8"],
}
```

“执行完成”不自动等于验证通过，清理未完成也不必把已知执行结果变成超时。Multica 指定适配接口将可选消息流与唯一最终结果分开，并设置 `TerminalObserved` 来保护已经确定的终态不被晚到Watchdog改写。[[61]](../references.md#source-multica-session)

这种接口设计比从终端里搜索“Done”可靠。对于只有PTY的产品，状态识别必须保留不确定性；优先寻找官方机器协议或明确的结果产物，不能凭安静几秒就宣布完成。

## 原生接口与通用协议

Codex App Server、Pi RPC、ACP、A2A和MCP承载的职责不同。ACP面向宿主和Agent交互，A2A面向Agent服务间协作，MCP主要接入工具和资源。统一Backend可以把它们包装成运行句柄，但必须逐项标明resume、interrupt、结构化结果、审批和usage能力。[[57]](../references.md#source-codex-app) [[38]](../references.md#source-acp) [[39]](../references.md#source-a2a)

```python
capabilities = {
    "resume": "native",
    "interrupt": "cooperative",
    "structured_final": True,
    "approval_events": True,
    "background_tool_cleanup": False,
}
```

`False`、未知和不支持需要区分。某个参数在Backend里被忽略，不应在上层UI继续显示“已启用限制”。Multica源码对若干选项的适用Backend有明确注释，提醒接入者不要假定相同ExecOptions对所有产品都有效。[[61]](../references.md#source-multica-session)

## 队列领取与执行资格

最小调度器可以定期扫描可运行任务，在原子操作中领取，记录owner与租约到期时间，再派发。依赖未完成、额度不足或正在等待审批的任务不应占用普通执行槽位。

```python
# 教学伪代码：claim 的比较与更新在同一个存储事务内完成。
def claim_next(store, worker_id, now, lease_seconds):
    return store.claim_one(
        eligible_states={"queued", "retry_ready"},
        owner=worker_id,
        lease_until=now + lease_seconds,
        increment_generation=True,
    )
```

generation用于拒绝旧执行者晚到的结果。但只在任务库比较generation，不能阻止失去租约的进程继续写文件或调用外部API。需要下游写入网关检查执行资格、隔离并回收旧进程，或将副作用绑定稳定operation ID。租约本身不是硬终止。

还需公平性：单个租户的大量任务不能占满所有槽位。可采用分租户队列、配额、加权调度和最大在途数。复杂算法之前，先测基本队列等待和饥饿情况。

## Deadline 与 Watchdog

总deadline控制任务最长时间，心跳判断执行者是否仍可联系，语义无进展监测关注有用事件，工具timeout限制具体执行。它们保护不同目标，不能全部复用一个“300秒没输出就重启”。

Multica接口将握手、首轮无进展、运行静默、工具活动和中断等待区分，是一个直接可借鉴的实例。统计tokens不断增长可能是有效长推理，也可能是循环；没有token可能是工具仍正常运行。因此Watchdog需要执行状态，而不只是stdout字符。[[61]](../references.md#source-multica-session)

原生Agent返回权威终态后先保存它，再清理进程和流。清理失败应记为cleanup问题，不覆盖已经确定的执行结果。重试政策也要看是否有实际副作用和原生连续性，不能超时后一律fresh session。

## Symphony、Multica、OpenWorker 与 Herdr 的职责

Symphony的公开规范围绕任务来源、每项工作区、执行Runner、并发、重试和状态协调；其工作流策略放在仓库中。它说明外层调度器可以驱动编码Agent，而不重写Agent内部Loop，也不必规定所有业务交付都直接进入Done。[[62]](../references.md#source-symphony)

Multica适合观察异构原生Agent的接口差异和执行管理。OpenWorker的案例重点是本地工作助理与工具治理，不能因也支持自动化就当成相同的分布式调度器。Herdr主要提供终端/会话管理和集成，其服务器重启恢复与原生会话恢复有明确边界。[[31]](../references.md#source-openworker-tests) [[60]](../references.md#source-herdr-state) [[63]](../references.md#source-herdr-api)

比较这些项目，应按任务来源、执行接口、状态来源、权限、恢复和产物验收逐项看，而不是列一张“都支持多个Agent”的功能表。

## 排队、限流与背压

模型供应商有请求和token额度，工具服务也有限额，机器CPU/内存又有上限。任务槽位不应只有一个全局数。可以分别管理模型请求并发、工具并发、外部服务额度和工作区资源。

当下游限流时，将可安全等待的任务移到带退避的队列，避免每个Agent在内部同时疯狂重试。退避要有抖动和上限，并使用剩余deadline；已进入未知副作用状态的调用不能按普通限流重发。

事件消费者慢不应占住执行槽直到资源耗尽。为订阅者设置有界缓冲，允许从权威快照重同步，保留任务最终结果。输出流和任务执行应该是两个生命周期。

## 恢复与降级的接口

原生resume失败需要分类：会话不存在或已不兼容，可能需要新会话；网络、限流和认证问题不必然意味着历史坏了。重新创建时明确告诉上层哪些连续性丢失，重新注入必要状态和资料，不假装“无缝继续”。

任务的状态与产物能否独立于原生对话存在，决定降级后的可靠性。如果全部事实都只在原生聊天记录里，换执行器会变得困难；但把所有供应商信息压成统一摘要，也可能丢掉必要续接数据。保留统一事实与原生历史两条边界更可解释。

## 测试与运行指标

测试同时领取、租约过期而旧进程未停、重复完成、迟到超时、用户取消、原生会话损坏、结果先到而日志仍排队、某个Backend不支持选项。断言不只是任务状态正确，还要检查执行次数和资源清理。

指标包括队列等待p95、每租户在途数、模型/工具槽位利用率、限流重试、僵尸执行者、重复派发、未知操作、终态后清理时间。调优必须与业务正确性一起看：更高吞吐若来自跳过审批或丢弃失败任务，不是有效改进。

---

## 工作者竞争与租约失效

Worker A领取generation5后卡住；租约过期，B领取generation6。A恢复后提交结果，任务库应拒绝generation5。但如果A继续直接写共享目录，任务库拒绝并不能撤销写入。这正是需要下游fencing或隔离回收的原因。

可以让写网关检查当前generation，或给每次尝试独立工作区，仅允许当前拥有者发布产物。远端业务效果仍用operation ID对账，不能把新的attempt ID当成新操作无条件重发。

## 公平性与吞吐实验

建立两个租户：A持续提交长任务，B偶尔提交短任务。只用全局FIFO可能让B等待很久；按租户轮转或配额可以改善隔离，但会改变整体吞吐。记录每租户队列p95、完成率和资源利用，而不是只看总任务数。

同样测试模型额度低于工作者数量时的情况：过多工作者可能都在重试限流，没有提升有效吞吐。将等待任务停留在调度队列比启动大量空转Agent更可解释。

## 原生 Agent 接入验收

先测进程启动/握手，再测一次普通任务、一次工具任务、一次审批、一次中断、一次resume和一次失败恢复。每阶段保存原生事件与最终结果，检查工作目录、模型选择、工具权限和usage是否实际生效。

当某Backend只能通过stdout估计状态时，把这种限制暴露给上层，不在统一界面上伪装成与结构化App Server同等可靠。统一体验应建立在真实能力上。
