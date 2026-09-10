# LLM API 与消息接口

LLM 调用接口把任务指令、消息和能力说明提交给推理服务，再接收包含内容与运行状态的响应。本页集中解释这一次边界。调用服务的 HTTP 请求、模型内部生成步和业务任务不是同一个对象。

## 文本生成的计算过程

以常见的自回归文本模型为背景：输入先被转换为 token 序列，模型根据已有序列生成后续 token。应用提交的消息角色和工具定义通常会经服务端模板转换为模型实际输入，不应假定把同样文本换一个消息位置仍有相同语义。训练改变模型参数；普通 API 调用主要改变本次可见上下文，并不自动把用户资料学进权重。

Prefill 处理已经给定的输入；Decode 逐步生成输出。更长输入会增加输入处理和注意力相关成本，更长输出又增加串行生成时间，但实际延迟还受排队、batch、硬件和服务策略影响。应用工程先记录输入长度、输出长度、首个事件与最终事件时间，再讨论某个请求为什么慢。[[4]](../references.md#source-hf-cache)

采样设置会影响输出选择，不能被当成可靠性证明。低 temperature 不是版本跨越后的逐字重放保证；模型、服务模板、并行数值计算和请求参数变化都可能改变结果。结构化任务应靠模式验证和测试集，不是只把 temperature 设为零。

## 请求的逻辑组成

```python
# 数据示例：内部契约，不是厂商 wire format。
request = {
    "model": "selected-model",
    "instructions": "只依据已提供材料回答，缺少字段时明确说明。",
    "messages": [
        {"role": "user", "content": "解释以下配置：server.port=8080"},
    ],
    "tools": [],                  # 能力描述，不是 Python 函数对象
    "output_contract": None,      # 可选结构化输出要求
    "generation": {"max_output_tokens": 500},
}
```

`instructions` 表达应用要求，`messages` 保留交互内容，`tools` 公布本轮能力，`generation` 是推理控制参数。用户身份、数据库凭据、审批资格不应放在这段模型可见输入中来“实现权限”。它们属于宿主执行状态。

System、Developer、User 不是普通字符串标签。同一段内容通过系统／开发者接口传递，和把“System:”写在用户材料里，不能视为等效。与此同时，供应商对角色位置有不同协议：Claude Messages 使用独立 `system` 字段，OpenAI Responses 可以使用 `instructions` 等入口，DeepSeek 兼容接口又有自己的接受范围。内部抽象要保留语义，而不是机械统一字段名。[[1]](../references.md#source-openai-text) [[5]](../references.md#source-claude-tools) [[6]](../references.md#source-deepseek-thinking)

## 原生 API 与内部适配层

下面是原生单次调用示例。它可以用于生成文本、提取信息或分类，并不需要 Agent 框架。

```python
# 原生 OpenAI SDK 示例；需要 OPENAI_API_KEY、OPENAI_MODEL 和 SDK。
import os
from openai import OpenAI

def generate_explanation(config_text):
    client = OpenAI(timeout=30.0, max_retries=0)
    response = client.responses.create(
        model=os.environ["OPENAI_MODEL"],
        instructions="解释配置字段。不要声称检查了未提供的文件。",
        input=config_text,
        max_output_tokens=800,
    )
    return response       # 不在适配入口就丢掉其他 output 条目
```

代码返回原始响应，而不是立刻 `return response.output_text`，是为了让调用者还能判断是否截断、是否出现工具调用、使用量是否完整。纯文本端点可以再提供一个方便函数，但不能把该函数当成整个模型适配器。[[1]](../references.md#source-openai-text)

自建适配器可以提供一个小的标准化外壳，并保留原生续接数据：

```python
normalized = {
    "text": "...",
    "actions": [],
    "completion": "complete",     # incomplete / refused / error
    "usage": {"input_total": None, "output_total": None},
    "native_output": [],          # 后续请求需要原样续传的完整条目
    "provider_request_id": "...",
}
```

`None` 表示未知，不能替换成零。`native_output` 也不能只保留自己今天认识的字段。遇到新类型应区分“可忽略显示信息”和“执行语义未知”：后者应中止或要求适配升级，不能静默解释成没有工具调用。

## 供应商接口的语义兼容

| 适配维度 | 需要验证的内容 | 只做字段重命名的后果 |
| --- | --- | --- |
| 消息组织 | 指令入口、内容块、调用与结果的位置 | 同样文字被放到不同信任位置 |
| 工具结果 | Call ID、结果类型、错误表示 | 第一轮正常，第二轮因配对失败被拒绝 |
| 推理续接 | 不透明条目、签名、reasoning 字段 | 续接失真或协议校验失败 |
| 输出上限 | 参数含义是否包含推理 token | 看似有预算，实际文本仍被截断 |
| 使用量 | 输入是否含缓存读取、写入、推理输出 | 费用估算重复计数或漏算 |
| 支持能力 | 工具、严格 schema、并行、模型专属选项 | 参数被接受但忽略，产生假兼容 |

DeepSeek 的当前工具推理文档要求带 `tools` 的请求完整回传之前各轮的 `reasoning_content`。这是具体接口契约，不能套用某个旧模型的历史规则。其部分兼容指南也指出，某些中间适配层会丢掉需要的推理字段；问题在序列化链条，不是“换一个 base_url”就结束。[[6]](../references.md#source-deepseek-thinking) [[7]](../references.md#source-deepseek-compat)

工程上用「请求能力清单」而不是只用模型名字符串。可写成 `supports_tools`、`supported_output_schemas`、`continuation_mode`、`usage_mapping_version`。这不是为了建立一个穷尽所有模型的大框架，而是让不支持的组合在发请求前就明确失败。

## 对话状态、模型实例和服务端缓存

应用维护一个会话，不意味着供应商为该用户长期保留一份专属模型。每次请求都需要通过显式历史或服务端会话引用获得上下文。原生会话接口可以省去客户端重复传输部分数据，但历史仍可能计入模型输入和使用量。引用也可能失效，必须记录创建它的 provider、账户范围和模型接口。[[8]](../references.md#source-openai-state)

不要把三个恢复操作混为一谈：重新建立网络连接、继续供应商会话、恢复业务任务。一个会话 ID 能继续对话，并不能告诉你上次工具是否实际发出了邮件。这个区别在 [恢复与对账](../runtime/persistence-recovery.md) 中处理。

Pi 把应用消息变成模型消息的边界单独放在 `transformContext()` 和 `convertToLlm()`：前者处理信息，后者处理表示。这为自建适配提供了一条清晰参考：上下文决策先发生，供应商编码后发生，不让整个业务状态为了适应某个 API 而改变结构。固定提交范围见来源。[[9]](../references.md#source-pi-loop)

## 多轮请求的契约测试

只测试“hello 能返回文本”是适配器的连通性测试，不是可用于 Agent 的验收。建议准备以下固定输入：纯文本；空文本但合法工具调用；两个工具；工具错误结果；混合文本与调用；输出截断；拒绝；原生推理条目续接；模型不支持某选项；使用量缺失。

为每个样例保存原始响应 fixture，离线测试规范化，再做少量真实接口回归。检查点不是“结果大概相似”，而是调用 ID 是否保存、截断是否禁止执行、未知字段是否仍能续传、缓存计数是否按供应商语义换算。涉及原生加密或签名条目时，离线 fixture 只能验证保真序列化，不能代替服务端验签。

调优接口层先测连接复用、首事件延迟、最终完成延迟、解析开销和重试放大倍数。不要在模型适配器、HTTP 客户端和任务系统各自动重试三次；最坏可能把一个逻辑请求放大为多次收费调用。重试责任应有一个清楚的主拥有者。

## 适用边界

本页解释常见的文本及工具交互，不把实时音频、计算机操作、长连接代理服务全部压进相同响应模型。新形态应增加明确能力和事件契约。更换模型前先判断是内容质量变化、API 行为变化还是运行权限变化；这三种问题需要不同的验证方法。

---

## 适配实例：统一请求与原生接口

假设内部任务是“读取项目端口”，第一轮可能产生工具调用。适配器不能只比较返回文字，应逐步检查以下映射：

| 内部对象 | 原生请求需要处理 | 续接时必须保留 |
| --- | --- | --- |
| 应用指令 | 独立system/instructions或支持的角色消息 | 当前接口对继承指令的实际规则 |
| 用户问题 | 普通文本或内容块 | 原始输入及必要附件引用 |
| 工具定义 | `parameters`与`input_schema`等结构差异 | 本轮公布的名称和schema版本 |
| 工具调用 | output item、content block或assistant字段 | 原生调用条目、ID、参数和相关续接信息 |
| 工具结果 | 对应结果条目或user内tool_result | 指向正确call ID，不伪造assistant事实 |

一个实际的兼容测试分三次运行：第一轮只请求调用；第二轮把本地fake工具结果交回；第三轮用户追问此前观察。第一轮正常、第二轮报400通常指向配对或推理字段丢失；前两轮正常、第三轮忘掉规则则需要检查会话指令继承与历史构造。不能因为“模型记忆不好”就反复增强Prompt。

可为adapter建立一个测试表，而不在业务代码里散落供应商条件：

```python
profile = {
    "name": "provider-A-tools",
    "instruction_location": "native",
    "tool_result_encoding": "native",
    "preserve_native_items": True,
    "supported_options": {"output_limit", "tool_choice"},
}
requested_options = {"output_limit"}
unsupported = requested_options - profile["supported_options"]
if unsupported:
    raise ValueError(f"未支持参数：{sorted(unsupported)}")
```

这里不设计统一的wire format生成器，因为原生模式的完整细节更适合由各adapter实现。公共层只检查能力、保存原生内容与统一最小状态。

## 调用质量、能力与保证范围

模型可能知道一个正确答案，却无法按要求返回合法schema；也可能返回完美schema，却填了错误事实；还可能事实和格式都对，但应用无权执行相应动作。分别对应模型内容质量、协议契约和业务执行权限。

这一区分直接决定排障方式。格式问题先看输出约束和adapter；事实问题先看输入证据和模型；权限问题看宿主策略。将三类失败都记成`model_error`，后续数据分析会错误地建议升级模型，而真正的问题可能是工具接入或数据过期。
