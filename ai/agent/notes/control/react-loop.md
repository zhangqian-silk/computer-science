# ReAct 与 Agent Loop

Agent Loop 组织模型决策、工具执行、观察更新和后续调用。模型能根据新观察选择下一动作，宿主仍控制权限、资源和停止条件。ReAct 研究将推理与行动交错；现代实现常用类型化 Tool Calling 承载动作，而不依赖解析公开的长篇思考文本。[[48]](../references.md#source-react-paper)

本页的前置接口只有 [响应分类](../model/outputs.md) 和 [工具执行](../tools/tool-calling.md)。上下文具体选什么由 [Context](../context/context-engineering.md) 负责，Loop 不应再兼任检索、存储和所有业务逻辑。

## 最小 Agent Loop

```python
# 教学代码。model.complete、tools.execute、verify 的契约见下文。
def run_agent(query, model, tools, verify, max_steps=8):
    messages = [{"role": "user", "content": query}]
    for step in range(max_steps):
        reply = model.complete(messages, tools.descriptions())
        if not reply["complete"]:
            return {"status": "incomplete", "reason": reply["finish"]}
        messages.append(reply["assistant_message"])

        calls = reply["tool_calls"]
        if calls:
            for call in calls:
                observation = tools.execute(call)
                messages.append({
                    "role": "tool",
                    "tool_call_id": call["id"],
                    "content": observation,
                })
            continue

        verdict = verify(reply["text"], messages)
        return {
            "status": "completed" if verdict["passed"] else "unverified",
            "answer": reply["text"],
            "verification": verdict,
        }
    return {"status": "incomplete", "reason": "step_limit"}
```

`model.complete` 已做供应商协议适配；`tools.execute` 负责最终参数、权限、执行和标准化观察；`verify` 是任务专属检查器，不是万能真值函数。这个例子使用完整响应后顺序执行工具，便于看到控制链；实时取消、预算和持久化需要进一步实现。

`continue` 的含义很具体：工具结果已加入历史，再次请求模型。若程序执行完工具就直接返回，则是一次工具路由；这里新增的是根据观察持续推进。

## 工具调用与消息回填

第一轮输入只有用户问题和工具说明。模型提议 `read_file(config.yaml)`。宿主先保存 assistant 调用，执行后追加带相同 call ID 的结果。第二轮模型看到文件中端口值，才提出答复。

```text
消息0 user：查询配置中的端口
消息1 assistant：call c1 → read_file(config.yaml)
消息2 tool：reply to c1 → port: 8080
消息3 assistant：配置端口是8080
```

若消息2是“文件不存在”，模型可以改查候选路径或请求用户说明；权限拒绝不应促使它不断尝试绕过。观察是下一步的依据，不是任何失败都允许无限修正。

同一模型响应可能包含多个调用，必须保存完整 assistant 消息，逐一关联结果。丢掉原始调用而只保存结果，会破坏协议续接；空文本也不代表失败，只要合法动作存在。

## Task、Turn 与 Step

产品对 turn/step 的命名不同：有的 turn 指一次模型生成加工具执行，有的指一个用户输入触发的整个工作过程。文档和观测应先说明本项目约定，不用名称猜语义。

本资料用“模型步骤”指一次请求及其相关工具批次，“用户轮次”指一次用户输入引发的工作，“任务”指可独立验收的目标。一个用户轮次可以有多步，一个任务又可能跨多个用户轮次。这有助于定义不同预算，但不要求初学者先搭六种 ID。

## 串行与并行工具批次

同时出现两个工具调用不意味着它们可以并行。模型可能没有充分理解共享资源依赖。宿主应按工具执行契约决定：只读独立查询允许并行；共享写或全局上下文修改需要屏障；不确定时保守串行。

并行至少产生两种顺序：实际完成顺序和模型历史中的提交顺序。UI 可以在某个查询先完成时立即显示，但模型输入和事件重放应有稳定的关联和顺序。顺序提交又可能产生队头阻塞：第一个调用慢，后面完成结果暂时不能成为连续历史。

```python
# 教学代码：并行完成，但以输入顺序得到结果；函数内部需自行安全收尾。
import asyncio

async def parallel_reads(calls, execute, concurrency=4):
    semaphore = asyncio.Semaphore(concurrency)

    async def one(call):
        async with semaphore:
            return await execute(call)

    tasks = [asyncio.create_task(one(call)) for call in calls]
    try:
        return await asyncio.gather(*tasks)
    finally:
        for task in tasks:
            if not task.done():
                task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
```

这段只适合已确认可并行且支持协作取消的调用。真实外部效果不能靠 `task.cancel()` 回滚；取消后仍需工具接口报告已知结果或未知状态。排他屏障和按结果更新策略的情况需要更专门的调度器。

## Pi：内循环、外循环和消息投影

Pi 固定提交的 `runLoop` 将工具/Steering 处理置于内循环，任务即将停止时再检查 Follow-up 外循环。`prepareNextTurn` 可在下一次请求前进行压缩或配置准备，随后再检查期间新到的 Steering。响应因长度被截断时，调用参数不再执行。[[9]](../references.md#source-pi-loop)

这解释了三个实现决策：控制输入需要有到达与消费边界；压缩不能让刚到的用户要求永久延迟；响应结束和整体任务静止是不同事件。自建时不要把所有新消息都直接拼入正在发送的请求，因为请求已经在途，模型实际看不到“原地更新”的历史。

Pi 的上下文转换和 provider 编码分开，允许应用消息包含不发送模型的内容。借鉴这一点可以让 Loop 只持有逻辑消息，不需要为每个供应商重写一套控制流。

## Codex：步骤绑定与共享／排他执行

Codex 指定提交的工具运行对象保留公布工具时的 StepContext，并按工具是否支持并行取得共享读锁或排他写锁。计量中还能区分派发等待与进入 handler 的时间；取消处理会先检查已知终态，再决定是否中止。[[14]](../references.md#source-codex-tools)

对自建系统的启示是将“排队等执行”和“真正执行中”分开，否则并发越高，工具看起来越慢，却不知道慢在锁等待。共享锁不是业务事务，仍需要工具自身遵守并发安全声明；步骤快照也不应冻结可撤销权限。

## DSH：滚动池、屏障与有序提交

DSH 指定调度源码使用有界滚动池。预检按调用顺序推进，执行可以重叠；结果暂存槽位，只有连续完成部分按模型顺序提交。遇排他调用形成屏障；尚未启动的调用会重新读取并发分类。取消停止补充新工作并排空已启动调用，未启动项产生取消结果；内部调度失败则不伪造正常结果。[[49]](../references.md#source-dsh-scheduler)

这比一次 `Promise.all` 多了可控的并发上限和顺序语义。代价是实现更复杂，慢首项会阻塞提交，动态策略与槽位需要共同验证。并不是所有应用都需要照搬；只有批次数量、共享状态和取消语义确实要求时才引入。

三种实现应围绕同一测试比较：独立读返回乱序；写操作插在读批次中；预检期间收到取消；执行成功后收到取消；工具目录更新；慢调用导致后面结果等待。没有这些测试，仅列“支持并行”无法判断设计。

## 结束条件与进展

没有新工具调用，只说明模型本轮提出了答案。任务是否完成还要看要求：问题回答是否有证据，代码是否通过规定测试，外部写入是否已确认。验证不了时返回 `unverified` 或需要用户检查，而不是假定成功。

最大步骤、总时间和 token/费用预算是硬边界；重复动作和无进展可以触发更早停止。无进展的比较应关注相同目标、相同资源版本、相同工具及结果，而不是把每轮时间戳变化当成新工作。成功检测同样不应仅靠关键词“完成”。

## 实验与调优顺序

先验证单调用配对、错误观察、截断不执行、步骤上限，再加入可并行读和屏障，最后增加用户输入、持久记录与恢复。每增加一层都保留原串行行为作为基线。

记录每个已验证任务的模型步骤数、重复读取数、工具等待与执行时间、并行峰值、取消收尾时间和最终验证成本。减少 Loop 次数若通过扩大一次工具返回实现，可能增加上下文噪声；增加规划一步若减少后续探索，可能总体更省。以同任务验收为依据，不以循环代码行数判断优劣。

---

## 并发调度时序分析

假设工具序列是读A、读B、写C、读D。读A耗时80ms，读B耗时20ms，写C耗时30ms，读D耗时10ms；写C是排他屏障。以下是理想化的调度算例，忽略模型、预检与网络开销：

```text
时间0：读A、读B并行
时间20：读B完成，可更新UI；按模型顺序提交时等待A
时间80：读A完成，提交A/B；开始写C
时间110：写C完成；开始读D
时间120：读D完成
```

串行需要140ms，受屏障并发需要120ms；若不顾依赖把D提前执行，虽然看似更快，却可能读到写C之前的数据。并发是否正确首先取决于工具语义。这个算例不是性能实测，也不适用于所有读写关系。

DSH式顺序提交能保持历史一致，但前面慢读会让后面结果等待；Codex式共享/排他门也需要工具正确声明。分析源码时要找到实际派发与提交点，不只找到并发函数名。

## 步骤级观测数据

`model_started`、`model_completed`、`tool_admitted`、`tool_started`、`tool_settled`、`step_completed`分别标记不同阶段。工具排队时不能提前发“执行中”后把等待算成handler耗时；模型输出完整时也不能提前发“任务完成”。

某工具业务失败可以作为观察让Loop继续；核心存储损坏或adapter无法解释响应则应显式停止，而不是把内部异常文字交给模型自由“修复”。错误进入哪条控制路径由宿主决定。

## 最小 Loop 的工程扩展

第一次升级加入取消：每个派发前检查，已启动工作由executor收尾。第二次加入预算：模型和工具前预留，结束后结算。第三次加入持久化：先定义调用与回执的写入边界，再讨论重启。第四次加入并发：保留串行测试作为语义基线。

这不是教程必经步骤，而是实现依赖说明。一次性只读应用可以停在简单版本；长任务服务不能靠在最小循环外围套一个通用try/retry就声称具备这些能力。
