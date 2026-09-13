# Dynamic Workflows 与代码编排

动态工作流把部分控制流程交由模型生成的任务图或程序表达，再由运行时执行。它与逐轮 ReAct 的区别是「后续路径的控制状态放在哪里」，与普通多 Agent 的区别是「多执行者由什么机制组织」。一个脚本里仍可调用运行 ReAct 的工作者。

本页用 Claude Code Dynamic Workflows 作为公开行为实例，结合可实现的参考设计讲解；Python 例子是教学控制程序，不是其原生 JavaScript API。

## 程序化工具调用、脚本与 Agent 编排

| 机制 | 模型生成什么 | 程序负责什么 |
| --- | --- | --- |
| 原生 Tool Calling | 一个或多个调用及参数 | 校验并执行，再回填结果 |
| Programmatic Tool Calling | 调用工具的代码片段 | 在受限环境运行代码与工具桥接 |
| Dynamic Workflow | 分支、循环、并行和子任务编排 | 管理工作者、状态、预算与结果 |

后两者都涉及代码，但不一定有同样权限或恢复语义。一个过滤搜索结果的工具程序，与运行五十个编码子 Agent 的编排程序，资源成本和生命周期差异很大。Cloudflare Code Mode 的实践可作为工具组合的相邻参考，不应用来证明所有动态 Agent 编排都具备相同隔离。[[55]](../references.md#source-cloudflare-code)

## 对话控制与程序控制

在逐轮协调中，Lead 模型反复接收结果、决定下一个任务，并把中间信息留在上下文。代码化后，列表遍历、条件分支、计数和局部中间值可由程序持有。它可以减少纯协调推理和主会话信息膨胀，但也把错误转化为程序控制错误：漏遍历、错误过滤、无界循环和异常短路。

```python
# 教学伪代码：程序取得可信任务列表后逐项委派；本段先用串行展示。
async def audit_all(files, worker, verify):
    outcomes = []
    for path in files:
        result = await worker({
            "task_id": path,
            "goal": "审查此文件，返回发现和证据",
            "path": path,
        })
        outcomes.append(result)
    return verify(files, outcomes)
```

这里故意先串行，说明代码保存 `files/outcomes`，不是每处理一个文件都请主模型决定要不要继续。并行、预算与恢复可进一步加入，但不会改变“程序负责遍历，Agent负责局部语义工作”的分工。

## Claude Code 的公开控制边界

其文档描述使用 JavaScript 编排子 Agent，中间值留在脚本变量，脚本在独立受限环境运行；脚本本身不直接获得文件或 Shell 访问，工作由子 Agent 执行。文档还说明子 Agent 中止或不可恢复错误可形成 `null` 结果，以及恢复局限于同一会话并按启动顺序复用连续完成前缀。这里只总结执行契约，不声称审查了闭源实现。[[56]](../references.md#source-cc-workflows)

从中最值得借鉴的不是脚本语法，而是把编排能力与实际执行能力分开。脚本可表达 `for/if/await`，不必因此拥有宿主任意模块、凭据和文件。资源限制也不能只写在模型提示里，应由 runtime 真正执行。

## 空结果与任务覆盖检查

假设目标是审查A、B、C，B失败返回空。为了显示简洁而过滤空值可以接受，但不能据过滤后两个结果宣布“全部检查”。完整性要与原始输入集合对照：

```python
# 教学代码：保留每个输入位置的失败信息。
def pair_outcomes(inputs, values):
    if len(inputs) != len(values):
        raise ValueError("返回数量与输入不一致")
    pairs = []
    for item, value in zip(inputs, values):
        pairs.append({
            "task_id": item,
            "status": "missing" if value is None else "returned",
            "result": value,
        })
    return pairs
```

后续还要验证每个非空返回是否属于正确任务、是否提供有效证据、是否满足局部契约。完全靠数组位置也有风险，最好给工作者传稳定 ID，并检查未知和重复项。

## 启动顺序重放的具体含义

如果A、B、C、D依次启动，停止时A/C/D完成、B未完成，连续前缀规则只能复用A；B及之后的任务可能再运行。这个案例说明“已完成结果存在”与“恢复一定复用它”是两件事。同会话的重放也不同于进程退出后的持久任务恢复。[[56]](../references.md#source-cc-workflows)

```python
# 教学代码：演示连续完成前缀，不是完整工作流引擎。
def reusable_prefix(started, completed):
    prefix = []
    for task_id in started:
        if task_id not in completed:
            break
        prefix.append(task_id)
    return prefix

assert reusable_prefix(["A", "B", "C", "D"], {"A", "C", "D"}) == ["A"]
```

自建系统可以选择按稳定任务标识及输入 hash 复用非连续结果，但代价是要验证脚本改变、分支条件、输入版本和依赖结果是否仍相同。若脚本把随机值或当前时间影响调度，原来的 journal 可能不再对应同一次计算。它不是简单多存几个结果就能正确恢复。

更重要的是，重放工作者可能重复外部副作用。只读审查通常较容易重放；提交邮件、发布资源或创建草稿需要业务操作 ID 与幂等/对账。代码化控制不会自动提供 exactly-once。

## 静态检查与运行时限制

任意执行模型生成代码风险很大。可从受限 DAG/DSL 开始：只允许声明节点、依赖、参数和有限控制；由宿主提供明确操作。若选择 JavaScript/Python 沙箱，需要限制模块、文件、网络、CPU、内存、递归、总任务数、并发和 wall-clock deadline。

静态验证检查任务 ID 唯一、依赖合法、能力在允许集合、循环有上限、工作区与输出路径合法。运行时再次计量实际任务数和成本；静态脚本看起来有限，子 Agent 内部仍可能多轮调用并超预算。

审批也要绑定真正的执行动作。批准了“审查文件”，不应自动授权子 Agent 发布修改；一个大脚本不能成为把多个高风险动作打包绕过审批的通道。

## 有界并行与失败收尾

```python
# 教学代码：限制同时进入 worker 的任务，并收集各项业务结果。
import asyncio

async def bounded_map(items, worker, limit):
    if limit < 1:
        raise ValueError("并发上限必须为正")
    semaphore = asyncio.Semaphore(limit)

    async def run_one(item):
        async with semaphore:
            try:
                return {"id": item["id"], "value": await worker(item)}
            except Exception as error:
                return {"id": item["id"], "error": str(error)}

    return await asyncio.gather(*(run_one(item) for item in items))
```

这个例子不吞掉 `CancelledError`，也没有承诺 hard kill。父级取消和 worker 自己的外部资源需要配套所有权与收尾。很大的输入集合还应采用滚动队列，避免一次创建海量协程，即便 semaphore 限制了实际执行数量。

是否遇一项失败就取消其余，取决于任务：独立审查可以保留其余结果；依赖共同基线的发布则可能需要停止整个阶段。失败策略必须明确，不能完全由 `gather` 的默认异常传播偶然决定。

## 对比其他恢复模型

LangGraph 的函数式执行会在恢复时重新进入入口，并复用已保存 task 结果；它要求对非确定性与副作用边界作清晰处理。与会话内脚本重放相比，关键比较项是保存边界、工作者身份、脚本变化和恢复范围，而不是都写着“resumable”。[[46]](../references.md#source-langgraph-functional)

持久调度器还能管理跨进程、跨机器的任务与租约，但复杂度明显更高。不要为同会话内的小型并行检查直接建设完整分布式工作流平台；也不要把轻量重放用于不能承受重复效果的长期业务。

## 调优与适用范围

比较工作者粒度、并发、规划开销、重复上下文和合并成本。小任务有利于复用和局部失败隔离，但每个 Agent 的固定提示、启动和汇总成本可能占主要部分；大任务减少启动但失败损失和上下文压力增加。

实验至少测正常完成、早停后恢复、某一任务失败、脚本修改、父进程退出和权限变化。报告实际覆盖与已验证结果，不只报告“启动了多少个Agent”。动态编排适合把明确可检查的控制结构交给程序，不适合用不可观察的大脚本隐藏任务责任。

---

## 任务日志与脚本版本兼容

若需要跨进程恢复，可把每个工作者调用写成journal条目：

```python
journal_entry = {
    "workflow_revision": "script-8",
    "node_id": "audit:routes/a.py",
    "input_digest": "sha256:...",
    "dependency_digests": ["discover:hash-4"],
    "status": "verified",
    "result_ref": "artifact:audit-17",
}
```

恢复时检查脚本/节点含义、输入、依赖和验证策略。只根据`node_id`复用会把新输入误当旧任务；只根据脚本文件hash又可能过度失效全部结果。选择兼容策略需要明确版本关系并测试，不是简单打开“checkpoint”。

非确定步骤要保存其结果或外部快照。例如发现文件列表在重启后发生变化，应重新计算目标覆盖，不能把新文件遗漏。模型生成的脚本本身也是产物，需要审查和版本记录。

## 动态代码与调度策略边界

脚本中的“最多执行10次”可以帮助表达意图，但宿主仍应有独立硬上限；生成错误或工具返回巨大列表时，脚本可能创建远超预期任务。限制总启动次数、并发、token子预算和产物大小，在超限时保留已完成结果并明确未完成范围。

对复杂业务，先让模型生成受限计划对象，再由可信解释器运行，往往比直接执行通用语言代码更容易校验。选择通用语言的价值是表达力，代价是更大的静态分析、隔离与恢复责任。

## 主会话上下文与编排成本

脚本保存中间变量确实可减少Lead的反复协调输入，但子Agent仍要接收目标、工具和任务上下文，汇总仍可能很大。计量应包含全部工作者和验证，不只看Lead会话token下降。

对只需要三次独立查询的任务，普通程序并行工具可能已经足够；引入完整子Agent会重复支付模型启动和规划。Dynamic Workflow适合需要局部自主工作的编排，不应替代所有普通批处理代码。
