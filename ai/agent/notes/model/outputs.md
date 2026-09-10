# 模型输出与结构化结果

模型响应是程序需要解释的一组类型化内容，不是一段必然代表最终答案的文字。本页负责从原始响应识别可显示内容、可验证数据、推理续接信息和动作提议；实际动作执行归 [工具系统](../tools/tool-calling.md)，是否继续下一轮归 [Loop](../control/react-loop.md)。

## 内容类型与完成状态

同一响应可能先解释意图，再提出两个工具调用，还携带推理条目。不能用一个 `type = text | reasoning | tool` 枚举描述整个响应，更不能见到文字就立即返回用户、忽略后面的动作。

| 轴 | 常见取值 | 应用如何使用 |
| --- | --- | --- |
| 内容 | 文本、结构化数据、调用、推理相关条目、引用 | 各自进入显示、校验、执行或续接路径 |
| 响应完整性 | 完整、截断、中断、错误 | 决定哪些内容能作为确定结果使用 |
| 模型行为 | 提出答案、请求工具、拒绝、需要输入 | 影响控制策略 |
| 业务验收 | 未检查、通过、不通过、无法验证 | 来自任务验证，不由接口状态代替 |

例如文本是“正在读取配置”，但同一回复含 `read_file`。这句话是过程说明，不是文件已被读取的证据。工具执行成功也不保证答案正确：模型可能误读返回值。反过来，工具没有查到记录可以是正常业务结果，而非运行故障。

## 不同接口的具体表示

OpenAI Responses 用 `output` 数组承载消息和调用等条目，`output_text` 是提取文本的便利属性。Claude Messages 的 assistant `content` 中可以出现 text 和 `tool_use` 块，客户端把 `tool_result` 放回后续 user 内容块。DeepSeek 的兼容对话接口则在 assistant 对象上使用 `content`、`tool_calls`、`reasoning_content` 等字段。不要直接把下面的统一示例当成三家原生格式。[[1]](../references.md#source-openai-text) [[2]](../references.md#source-openai-functions) [[5]](../references.md#source-claude-tools) [[6]](../references.md#source-deepseek-thinking)

```python
# 数据示例：适配后的内容数组。
items = [
    {"kind": "text", "text": "先读取配置。"},
    {"kind": "tool_call", "id": "c-1", "name": "read_file",
     "arguments": {"path": "config.yaml"}},
    {"kind": "continuation", "native": {"opaque": "..."}},
]
```

一个稳健的分派器应逐项处理，最后再决定整体控制。为了便于教学，下例只接收已经完整且完成原生协议验证的消息：

```python
def classify_items(items):
    text_parts, calls, continuation = [], [], []
    seen_ids = set()
    for item in items:
        if item["kind"] == "text":
            text_parts.append(item["text"])
        elif item["kind"] == "tool_call":
            if item["id"] in seen_ids:
                raise ValueError("同一响应出现重复调用 ID")
            seen_ids.add(item["id"])
            calls.append(item)
        elif item["kind"] == "continuation":
            continuation.append(item["native"])
        else:
            raise ValueError("未知执行语义，停止自动分派")
    return "".join(text_parts), calls, continuation
```

这里故意不执行函数、不产生最终成功状态。输出解析器的单一职责，是正确保留并解释供应商已经给出的信息。

## 文本、JSON 与结构化输出

“请用 JSON 返回”是一条提示；JSON mode 约束可解析形式；严格的结构化输出进一步约束 schema。即便 schema 符合，也只能证明结构条件，不证明订单号属于当前用户，也不证明日期存在或引用来自真实证据。

设计 schema 应从下游决策需要的最小字段开始。对未知值给出明确表达，比强制模型填满所有字段更合理。字段过多会增加生成和校验成本，也会迫使模型在缺乏事实时填充貌似合理的值。

```python
# 数据契约示例。供应商支持的 JSON Schema 子集需要另外验证。
port_result_schema = {
    "type": "object",
    "properties": {
        "port": {"type": ["integer", "null"]},
        "source": {"type": ["string", "null"]},
        "status": {"enum": ["observed", "not_found"]},
    },
    "required": ["port", "source", "status"],
    "additionalProperties": False,
}

def validate_port_result(result, allowed_sources):
    # schema 校验在前；这里补充业务语义。
    if result["status"] == "observed":
        if result["port"] is None or not 1 <= result["port"] <= 65535:
            raise ValueError("不是有效端口")
        if result["source"] not in allowed_sources:
            raise ValueError("没有可追溯的输入来源")
    elif result["port"] is not None:
        raise ValueError("not_found 不得附带猜测值")
```

严格输出还必须有独立的拒绝和中断处理。OpenAI 文档明确指出拒绝结果可能不符合预设 schema。把“schema 解析失败”统一送回模型无限修复，会把拒绝、服务错误和字段小问题混在一起。[[10]](../references.md#source-openai-structured)

合理的修正闭环应有限：只对已知的格式或语义问题附带精确反馈，最多重试约定次数；没有证据、无权限和用户需求不明确时应返回相应状态，不能强迫模型生成一个合法数字。

## 推理信息与续接数据

Reasoning 有三个层次：模型内部计算过程、API 暴露的摘要或续接信息、用户看到的解释。用户可见解释可以用于理解答案依据，但不是执行日志；一段“我已经测试了”不能代替测试进程的退出码与输出。

不同模型的推理接口不同。OpenAI 可能返回需要原样保存的推理条目或加密续接内容；DeepSeek 当前工具推理规则要求回传特定历史字段；Claude 的手动与自适应 thinking 选项存在模型代际差异。适配器不应对所有模型强行添加同一个 `budget_tokens`，也不应出于“节省上下文”擅自删除原生续接字段。[[11]](../references.md#source-openai-reasoning) [[6]](../references.md#source-deepseek-thinking) [[12]](../references.md#source-claude-thinking-legacy)

应用可记录所选推理等级、使用量、耗时和外部决策结果，而不要求拿到内部完整推理。审计动作依据更适合保存目标版本、证据引用、调用参数、授权决定和实际回执。它们能与外部事实核对，思考文本通常不能。

推理预算的调优必须对任务分层。格式抽取可能不需要高预算，约束密集的规划可能受益；结论应来自同任务集、同工具和同验收的实验。不能仅凭生成的解释更长判断质量更好。

## 工具调用与执行回执

调用 ID 是一次协议关联身份。结果回填要引用该 ID；重试时供应商可能生成新 ID，但业务操作可能仍是同一项。因而它既不是授权凭据，也不应无条件充当业务幂等键。

输出层需要检查完整参数和名称；工具层再做 schema、业务对象和权限校验。若执行前有参数改写，最终实际参数必须重新校验。模型返回 `send_email` 只意味着它提议发送，不意味着用户批准、不意味着邮件已经送达。

Pi 的固定源码在响应因 token 上限截断时，不执行其中可能已截断的工具参数，而是把该批次处理为错误。这个选择体现的是执行安全边界，不是模型“不会在截断前给出正确 JSON”的数学断言。自建系统可以更细粒度地使用供应商确认完成的 item，但必须有明确的 item 完成契约，不能从文本表面推断。[[9]](../references.md#source-pi-loop)

## 输出处理中的实际故障与测试

建立一组小型 fixture 往往比手工看十段正常答案更有价值：空 `content` 加工具调用；解释文本加两个调用；重复 ID；未知工具；错误参数 JSON；拒绝；合法 JSON 但业务字段冲突；输出长度耗尽；只到达部分流；用新 provider 重放旧推理条目。

每项测试要指定期望状态和禁止发生的行为。例如“截断”测试不只是返回错误，还应断言工具执行计数为零。“来源伪造”测试不只是语气变谨慎，还应断言结果没有被计入已验证事实。

不要把模型自己给的 confidence 当成校准过的业务概率。它可以成为待评估的特征，但是否足够可靠要独立测量；对高风险动作，应使用确定性证据和权限边界。更多评测设计见 [评测与验证器](../runtime/evaluation.md)。

## 适配实现的边界

最小输出层可返回 `text/calls/native_output/status/usage`，没有必要一开始支持所有多模态事件。不过保留原生数据应有尺寸、保密与生命周期限制，不能为兼容性无限记录秘密。规范化是为了方便程序决策，原生保存是为了正确续接；二者都不应获得比原始请求更高的信任权限。

---

## 响应样例与处理分支

下面的例子把三个容易混淆的响应放在一起。fixture使用内部格式，实际adapter必须先从供应商原生条目保真转换。

```python
fixtures = [
    {"complete": True, "refused": False,
     "text": "", "tool_calls": [{"id": "c1", "name": "read_file"}]},
    {"complete": True, "refused": True,
     "text": "无法执行该请求", "tool_calls": []},
    {"complete": False, "refused": False,
     "text": '{"port": 8080}', "tool_calls": []},
]

def next_action(reply):
    if not reply["complete"]:
        return "report_incomplete"
    if reply["refused"]:
        return "report_refusal"
    if reply["tool_calls"]:
        return "validate_and_dispatch"
    return "validate_answer"
```

第一项空文本不是空响应；第二项拒绝不应进入JSON修复；第三项看起来是合法JSON，但整体响应被标为未完成，保守系统不能把它作为确认结果。输出结构与完成状态一起解释，才不会依赖偶然文本。

如果供应商允许在完整item级别处理早到结果，要把该能力写成adapter profile而不是更改整个系统默认。并发工具、流式断开和晚到拒绝是否允许，必须按具体服务契约验证。

## 结构化输出与业务校验

假设模型返回一个分类字段`action`，取值为`answer`、`lookup`或`needs_input`。下游程序不应执行模型额外塞进的Python代码，也不应对未知值走宽泛默认动作。明确枚举对应处理器：

```python
def route_decision(decision, handlers):
    action = decision["action"]
    if action not in handlers:
        raise ValueError("未知动作类别")
    return handlers[action](decision["payload"])
```

schema可约束枚举和payload结构，但每种handler还要业务校验。例如`lookup`提供订单号，仍需当前用户对该订单的权限；`answer`给出引用，仍需引用来自已提供资料。这样“模型选择下一分支”可以是一个小的受控能力，不必把整个程序交给任意生成代码。

输入缺少必要信息时，`needs_input`应该列出具体字段，而不是生成一个模糊问题再让用户反复猜。输出契约设计良好，能减少后续Loop次数，其收益应在任务级测量。
