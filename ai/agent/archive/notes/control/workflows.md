# Workflow 与流程控制

Workflow 用程序或已确定的图组织模型调用、工具执行和普通计算。它适合路径可描述、步骤需要独立校验的任务，不是“低级 Agent”。本页讨论控制与数据依赖。

## 节点、边与状态

节点是一次职责清楚的工作，可以是确定性函数、模型调用、工具操作或完整 Agent。边决定哪些结果使后续节点可运行。数据流说明输入从哪里来，控制流说明是否执行；两者相关但不能混淆。

例如「抽取信息 → 查询数据库 → 生成摘要」里，抽取错误不应直接继续查询；数据库未找到记录也不一定是系统失败，可能进入明确的缺失分支。把任何非空字符串当作节点成功，会让上游错误在后续被美化为正常答案。

```python
# 教学代码：步骤依赖显式写在程序中，不需要先引入框架。
def order_summary(request, extract, query_order, explain):
    fields = extract(request)
    if fields["status"] != "valid":
        return {"status": "needs_input", "missing": fields["missing"]}
    order = query_order(fields["order_id"])
    if order is None:
        return {"status": "not_found"}
    answer = explain(request, order)
    return {"status": "answer_proposed", "answer": answer}
```

这里三次函数执行中可以只有两次模型调用。数据库查询和字段必需性检查不需要模型。整个主路径由代码规定，即使 `extract` 使用 tool selection，也不因此变成自主 Agent。

## 常见组合模式与取舍

| 模式 | 控制特点 | 主要代价 |
| --- | --- | --- |
| 顺序 Chain | 后一步依赖前一步 | 延迟累积、误差传播 |
| Router | 从有限候选选择处理器 | 错路由、兜底和分类边界 |
| Parallel | 独立工作同时执行 | 并发预算、结果覆盖、共享状态冲突 |
| Evaluator–Optimizer | 检查后有限修正 | 验证器质量、重复生成和终止 |
| Map–Reduce | 分项处理再集成 | 分块遗漏、汇总丢失细节 |
| Human checkpoint | 人工决定是否继续 | 长等待、过期审批、恢复语义 |

Anthropic 对工作流模式的讨论可作为模式入口，但模式名称不是质量保证。需要先说明比较条件：同样任务是否减少遗漏、是否多付了不必要调用、失败路径是否更明确。[[3]](../references.md#source-anthropic-agents)

## 节点间的数据契约

每个节点至少定义输入版本、输出 schema、成功/业务缺失/失败状态和证据。不要让所有节点读写一个任意字典，然后依赖提示保证“不改别人的字段”。并行场景特别需要各节点独立产物及确定性合并。

```python
step_result = {
    "step_id": "inspect-config",
    "input_revision": "workspace-7",
    "status": "completed",
    "value": {"port": 8080},
    "evidence": ["read-19"],
    "output_revision": "result-3",
}
```

后续节点不能将工作区7的检查结果用于工作区9的补丁验收。缓存节点结果也需要把输入、代码、Prompt 和必要权限范围纳入兼容条件，而不是只按节点名称复用。

对于条件分支，`skipped`、`not_required`、`failed` 和 `missing` 应区分。汇总节点要知道没有结果是因为不该执行，还是任务遗漏。否则丢失一项结果后仍可能写出“全部检查完成”。

## DAG 执行器示例

```python
# 教学代码：串行拓扑执行，不含并发、持久化与副作用重放。
def run_dag(nodes, initial):
    outputs = {}
    pending = dict(nodes)
    while pending:
        ready = [
            key for key, node in pending.items()
            if all(dep in outputs for dep in node["depends_on"])
        ]
        if not ready:
            raise ValueError("依赖缺失或存在环")
        for key in sorted(ready):
            node = pending.pop(key)
            inputs = {dep: outputs[dep] for dep in node["depends_on"]}
            outputs[key] = node["run"](initial, inputs)
    return outputs
```

这段展示“何时可以执行”和“从何处取数据”，不负责判断业务结果。真正节点失败时，应按任务策略停止依赖项、继续独立分支或等待补充，而不是所有异常统一 `continue`。加入循环需显式预算和循环状态，不能让图结构隐藏无界重试。

图只是表示方式。普通函数流程可能更易读；需要可视化、复杂依赖、持久执行时再选框架，不必为三个步骤先建一套调度平台。

## LangGraph：函数式和图式接口的实现区别

LangGraph 的 Functional API 用普通分支与函数保留控制结构，通过 entrypoint/task 引入持久化与中断。Graph API 则显式描述状态与图；其 checkpoint 边界和函数式任务结果保存方式不同。选择应依据状态共享、可视化和恢复需要，而不是哪种看起来更“Agent”。[[46]](../references.md#source-langgraph-functional)

特别要读取恢复语义：重新进入函数或节点不等于从某一行继续。已持久任务可以复用结果，任务之外的副作用仍可能再次发生。这个机制促使开发者把非确定性步骤和外部动作边界显式化，不能只加一个装饰器就宣称 exactly-once。

本笔记的 Python DAG 示例没有实现上述持久能力，它只是便于理解控制。采用真实框架时，应以其当前 SDK 的具体类型、返回值和测试为准。

## 验证与修正循环

有限修正可以提升结构化结果或代码质量，但验证器应检查实际对象。代码生成后的检查用编译、测试、静态分析和人工评审组合；文本检查用明确 rubric 和证据。让同一个模型读取自己的解释然后给“通过”标签，不是独立验证。

```python
# 教学伪代码：每次修正都有独立检查，达到上限返回未验证。
def revise(initial, improve, check, limit=3):
    candidate = initial
    for _ in range(limit):
        verdict = check(candidate)
        if verdict["passed"]:
            return {"status": "verified", "value": candidate}
        candidate = improve(candidate, verdict["feedback"])
    return {"status": "unverified", "value": candidate}
```

如果反馈连续不变，应停止或升级，不应把同一修正重复消耗全部额度。错误属于环境缺依赖时，重写业务代码可能使结果更差。保留失败分类，让修正函数知道该改内容、补资料还是等待环境。

## 人工检查点与发布边界

人工批准应绑定具体候选产物和版本。用户批准了补丁A，程序不能在后台修改成B后继续沿用批准。长等待时重新检查资源版本、任务目标和策略。

LangGraph 中断文档指出恢复可能重新执行节点，故中断前动作要幂等或隔离。可把“准备产物”“请求审批”“提交动作”拆成明确步骤，避免恢复时再次发送消息或创建资源。[[47]](../references.md#source-langgraph-interrupts)

## 调优与选型

测量每条路径的调用数、关键路径耗时、失败位置和验证通过率。并行只优化独立部分，不能把有依赖的读写强行并发。更少节点可能减少开销，但也可能消除关键验证；更多节点可能提升可诊断性，但会增加接口和上下文传递成本。

对固定流程，优先把确定性工作留给程序；对无法枚举的探索，允许一个有预算的 Agent 节点，而不是让整个业务的审批与发布都交给自由规划。Workflow 与 Agent 的混合常比二选一更符合工程实际。

---

## 分支结果与汇总节点

假设三项检查中A成功、B因条件不适用跳过、C执行失败。汇总器不能只把A的结果传给模型并要求“总结全部检查”，而应显式传入三项状态。B不是缺失，C不是没有发现问题。

```python
outcomes = {
    "A": {"state": "verified", "findings": []},
    "B": {"state": "not_applicable", "reason": "没有该配置"},
    "C": {"state": "failed", "reason": "环境依赖缺失"},
}
ready = all(
    item["state"] in {"verified", "not_applicable"}
    for item in outcomes.values()
)
assert ready is False
```

模型可以生成一个说明部分完成的报告，但发布节点仍按程序条件拒绝“全部通过”。这让内容生成与业务状态分开，不会让流畅总结掩盖失败节点。

## 重试输入与版本绑定

一个检查节点失败后重试，输入文件可能已改变；此时它不是原步骤的完全重复。将输入版本、节点实现、Prompt和必要配置记入任务记录。需要复用结果时核对这些条件，不能以相同step ID直接读取历史缓存。

副作用节点重试需要业务operation ID；纯计算节点可以按输入hash缓存；模型节点缓存要考虑上下文和版本。它们都叫“重试”，但正确复用条件不同。

## 函数组合与图式执行

流程短、依赖清楚、几乎无动态分支时，普通函数最容易读和调试。需要可视化依赖、复杂并行、持久状态和人工中断时，图或工作流框架可能值得。应测框架引入的序列化、检查点和版本迁移成本，不把“能画图”当唯一选型理由。

无论形式如何，节点契约、错误类别和验收都必须存在。将逻辑搬到图框架不应使实际控制规则更难找到。
