# 工具调用与执行网关

Tool Calling 让模型提出结构化动作，工具网关将该提议转换成受校验、受授权的实际操作。它可用于一次调用后直接返回，也可用于 Workflow 或 Agent Loop。本页完整解释执行链，响应字段见 [输出分类](../model/outputs.md)，批次并发与循环推进见 [Agent Loop](../control/react-loop.md)。

## 工具描述与执行契约

模型需要名称、用途、参数结构及必要的语义说明；宿主还需要函数实现、权限要求、超时、是否可并行、重试语义和输出上限。不要把 API key 或内部授权规则完整写进工具描述，也不要误以为 JSON Schema 覆盖了全部执行风险。

```python
# 教学数据：模型只看到 public 字段。
tool = {
    "public": {
        "name": "get_order",
        "description": "读取一个订单的状态，不修改订单。",
        "parameters": {
            "type": "object",
            "properties": {"order_id": {"type": "string"}},
            "required": ["order_id"],
            "additionalProperties": False,
        },
    },
    "runtime": {
        "revision": "2",
        "effect": "read",
        "timeout_seconds": 5,
        "parallel_safe": True,
    },
}
```

“只读”也需要验证：查询可能暴露私有信息，下载可能触发计费，浏览器 GET 也不保证业务无副作用。工具声明是设计契约，不是自动安全证明。

工具粒度应服务任务。一个巨大的 `do_anything` 参数难校验，几百个几乎重复的低层工具难选择。通常先提供业务含义清楚、返回可操作结果的能力；是否合并多个 API，要考虑减少往返与丢失中间审批/验证机会之间的取舍。

## 一次调用的完整生命周期

一个可维护的顺序是：解析提议 → 绑定本轮工具版本 → 参数规范化 → 最终 schema 与业务校验 → 当前权限判定 → 必要审批 → 执行 → 保存事实回执 → 生成模型/用户可见结果。

其中“最终参数”很重要。若 Hook 把路径从 `a.txt` 改成 `.env`，不能沿用修改前的校验或审批。审批应绑定真正执行的工具版本、参数和资源，而不是工具名这一个字符串。

```python
# 教学代码：只读订单工具；get_order_from_db 与 auth 由宿主提供。
def execute_get_order(call, user_id, get_order_from_db, auth):
    if call["name"] != "get_order":
        return {"status": "invalid", "reason": "unknown_tool"}
    args = call["arguments"]
    if set(args) != {"order_id"} or not isinstance(args["order_id"], str):
        return {"status": "invalid", "reason": "bad_arguments"}
    order_id = args["order_id"].strip()
    if not order_id or len(order_id) > 80:
        return {"status": "invalid", "reason": "bad_order_id"}
    if not auth.can_read_order(user_id, order_id):
        return {"status": "denied"}
    order = get_order_from_db(order_id)
    if order is None:
        return {"status": "not_found"}
    return {"status": "observed", "order_id": order_id, "value": order}
```

真实数据库接口最好在同一数据访问边界实施授权，避免检查后资源归属变化形成竞争。示例只是把调用、参数与权限三层显式展开；不能按字符串前缀检查订单归属。

## 调用 ID、操作 ID 与工具版本

调用 ID 关联模型消息和结果；业务操作 ID 标识一次经授权的意图。模型重试可能生成新 call ID，但仍指向同一业务动作；用户第二次明确发起相同参数的操作，则可能需要新 operation ID。单纯用参数 hash 去重，会错误吞掉合法的重复业务请求。

本轮工具版本也要固定。Codex 指定提交的 `ToolCallRuntime` 保留创建时的 `StepContext`，其中包含向模型公布工具时的 router。这避免模型按旧 schema 生成参数，执行时却进入新函数。自建系统可用不可变注册表快照达到相似目的；权限撤销仍需执行前读取当前策略，不应随快照冻结。[[14]](../references.md#source-codex-tools)

## 事实回执与结果表示

工具已经成功写入，随后摘要或渲染失败，不应把这次业务动作标成“尚未执行”并自动重试。应保存独立回执，再处理模型可见表示：

```python
receipt = {
    "operation_id": "op-91",
    "execution_status": "succeeded",
    "resource": "draft-17",
    "resource_revision": "3",
    "evidence": {"remote_request_id": "r-52"},
}
model_view = {
    "status": "succeeded",
    "summary": "评审草稿已创建。",
    "receipt_ref": "receipt:op-91",
}
```

可见摘要是投影，不能任意改写执行状态。DSH 指定提交把规范输出 schema、纯渲染和最终内容处理分开，并提供冻结结果通知。这个接口切片说明区分事实与表示的价值；不是声称所有 Hook 已经具有安全隔离。[[30]](../references.md#source-dsh-tools)

错误结果也应结构化：参数无效、权限拒绝、未找到、暂时不可用、外部效果未知是不同状态。“工具出错，请重试”会让模型无法作出正确后续决策。对于外部效果未知，先查询原操作状态，不能默认再执行一次。

## 工具结果的可用性

工具返回应让下一步容易决定。例如检索结果给来源与可读取范围；测试工具给退出码、失败摘要和完整日志引用；编辑工具给修改前后版本与差异。把一大段 terminal output 不加选择地塞给模型，会增加费用和漏读风险。

输出截断必须标明。保留头部可能丢掉最后测试汇总；只保留尾部又可能丢掉最初根因。可针对工具类型设计提取器，保留关键字段，并将原始内容保存在可授权回查的产物中。摘要不是证明，引用也不是权限。

Anthropic 的工具工程实践把返回内容和工具选择表现纳入评测，这比仅检查函数能调用更接近日常 Agent 开发。具体工具长度与粒度仍需自有任务实测。[[20]](../references.md#source-anthropic-tools)

## 并行与重试的适用条件

两个读工具可能安全并行，两个写同一文件的工具通常需要序列化或版本比较。读写锁只是调度机制，是否可交换仍由工具语义决定。DSH 的并发声明采用显式 opt-in，Codex 的相关实现也区分共享与排他执行；这类选择适合从保守默认开始，再经测试放开。[[30]](../references.md#source-dsh-tools) [[14]](../references.md#source-codex-tools)

重试必须识别在哪个阶段失败。执行前因参数无效可修正再提议；只读短暂错误可有限重试；发送后响应丢失属于未知，除非远端支持幂等或查询，否则不能自动重复。把所有异常在工具 wrapper 中统一 retry 会破坏上层审批与账本语义。

## 不同工具类型的安全边界

OpenWorker 的测试明确区分结构化文件工具的路径限制与任意 Shell 命令：文件写入受范围规则控制，不意味着 Shell 触及的所有路径也被同一规则限制。其 Reviewer 也不能将硬拒绝改为放行。这个具体测试比“支持审批”四字更值得参考。[[31]](../references.md#source-openworker-tests)

因此安全策略应落在实际资源接口和执行环境。允许 `bash` 通常比允许一个受限 `read_file` 大得多；隐藏危险工具不代表其依赖进程无法完成相同行为。环境隔离另见 [执行环境](execution-environments.md)。

## 工具系统的契约测试

测试不仅验证正确参数成功，还要断言禁止动作没有执行。准备未知工具、额外字段、空对象、跨用户资源、参数变换后越权、审批后版本变化、回执成功但渲染异常、响应丢失与重复请求。

记录选择正确率、参数修正次数、每任务工具次数、平均结果长度、执行/展示失败比例、未知效果数量。选择经常错时改描述与粒度；执行经常错时修接口；摘要反复误导时改返回契约。不要把三类问题都用更长 Prompt 掩盖。

资料包的离线实验采用无外部效果的内存工具，验证调用关联和拒绝路径；真实发送、删除、数据库写入必须在受控环境中进行幂等与故障注入测试，不能以模拟通过声称生产安全。

---

## 故障案例：结果处理与重复写入

考虑这样的错误结构：

```python
# 反例：不要用于真实写操作。
def unsafe_send(send_remote, render):
    try:
        receipt = send_remote()
        return render(receipt)
    except Exception:
        return send_remote()   # render失败也会重复发送
```

错误并不在于模型，而在于异常范围把发送与显示混在一起。修复时先把外部操作回执保存，按执行阶段分类错误；显示失败可重新显示原回执，网络未知状态则查询原operation，不能复用同一个通用重试块。

同样，工具返回JSON序列化失败不应触发整个业务动作重跑。约束工具规范输出为可序列化类型，并在执行之前或测试中尽早检查；执行之后才发现错误，则保留实际效果为未知或已成功，而不是回到“未开始”。

## 执行回执的字段语义

`operation_id`连接重试与对账；`tool_revision`记录执行了哪个实现；`arguments_digest`绑定规范化参数；`resource_revision`说明产生或读取的版本；`execution_status`是动作结果；`model_content`仅是给模型的表示。字段可以按项目简化，但作用不要合并。

对只读工具，回执也有意义：它能证明哪一个文件版本被读取，不只是模型声称“看过”。对修改工具，回执最好给实际diff或范围，而非让模型自己总结修改了什么。

## 接口契约与测试矩阵

| 输入/故障 | 预期可见结果 | 必须断言 |
| --- | --- | --- |
| 未知工具名 | invalid/unsupported | 执行次数0 |
| schema合法但无对象权限 | denied | 未读取或写入敏感对象 |
| 执行前Hook改了目标 | 重新授权或拒绝 | 不沿用旧批准 |
| 执行成功，摘要失败 | 原回执仍成功＋展示错误 | 实际效果只发生1次 |
| 超时且效果不确定 | unknown | 不盲目重复执行 |
| 旧工具实例被更新 | 旧快照执行或明确拒绝 | 不混用新旧参数语义 |

这些用例可以先不调用模型就验证。只有执行边界稳定后，评测模型选择工具的准确率才有清楚意义。
