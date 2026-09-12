# LLM API 与消息接口

一次模型调用就是三步：程序把「指令 + 内容 + 能力」组织成一个请求，推理服务返回一个「带状态的响应」，程序再解释这个响应。本页给出贯穿「模型交互」四篇的统一最简接口，并说明怎么组织请求、怎么安全读回响应、怎么适配不同供应商。

调用服务的 HTTP 请求、模型内部的生成步、业务任务是三个不同对象；本页讨论中间那一层「一次请求／一次响应」。

---

## 一次调用的最简接口

下面是贯穿「模型交互」四篇的统一教学接口。形状借鉴 OpenAI Responses SDK（`instructions`／`input`／`output`／`usage`），但字段刻意精简，**不等于任何真实 API**，只为把输入与输出的结构讲清楚。[[1]](../references.md#source-openai-text)

```python
# 教学接口：模型交互四篇共用，字段精简，不联网。
from dataclasses import dataclass, field


@dataclass
class Tool:
    name: str
    parameters: dict                 # 参数的 JSON Schema，只是能力描述


@dataclass
class Request:
    model: str
    instructions: str                # System/Developer 指令
    input: str | list                # 用户内容，或多轮消息列表
    tools: list[Tool] = field(default_factory=list)
    max_output_tokens: int = 512


@dataclass
class Text:
    text: str                        # 可显示文本


@dataclass
class ToolCall:
    id: str                          # 一次调用的关联身份
    name: str
    arguments: dict


@dataclass
class Reasoning:
    keep: dict                       # 需原样回传的推理／续接条目


@dataclass
class Response:
    output: list                     # [Text | ToolCall | Reasoning ...]
    status: str                      # "completed" | "incomplete" | "refused"
    usage: dict                      # {"input_tokens": int, "output_tokens": int}

    @property
    def output_text(self) -> str:
        return "".join(p.text for p in self.output if isinstance(p, Text))


def call_model(req: Request) -> Response:
    """表示一次请求边界：真实实现里在这里发 HTTP。本笔记只用它说明形状。"""
    ...
```

要点：`output` 是一个**混合列表**，文本、工具调用、推理条目可以同时出现；`status` 单独表示这次响应是否完整。后三篇都复用这几个类型。

---

## 组织一个请求

以「读取配置里的服务端口」为全篇统一示例。程序声明一个只读工具 `read_file`，模型可能直接回答，也可能先要求读文件。[[2]](../references.md#source-openai-functions)

```python
req = Request(
    model="demo-model",
    instructions="只依据已提供或读到的内容回答；缺少字段就说明。",
    input="配置在 config.yaml，请告诉我服务端口。",
    tools=[Tool(name="read_file",
                parameters={"type": "object",
                            "properties": {"path": {"type": "string"}},
                            "required": ["path"]})],
)
```

- `instructions` 放应用要求，`input` 放用户内容，`tools` 只是能力描述而不是函数本身。
- 用户身份、数据库凭据、审批资格不放进请求；它们属于宿主的执行状态，不是模型输入。

角色也不是普通字符串：把 `System:` 写进用户文本，和通过 `instructions` 传入，语义不同。不同供应商入口也不同——Claude 用独立 `system` 字段，OpenAI 用 `instructions`，DeepSeek 兼容接口另有接受范围；适配层应保留语义，而不是机械统一字段名。[[5]](../references.md#source-claude-tools) [[6]](../references.md#source-deepseek-thinking)

---

## 安全读回一个响应

先看 `status`，再按条目类型处理。拿到一段文本不代表任务完成，也不代表没有待执行的调用。

```python
resp = call_model(req)

if resp.status != "completed":
    handle_incomplete(resp)          # 截断或拒绝：不能当作确定结果
else:
    for item in resp.output:
        if isinstance(item, ToolCall):
            dispatch(item)           # 校验参数后执行该调用
        elif isinstance(item, Text):
            show(item.text)
```

同一响应里可能既有文本又有调用，要逐条处理，不能只取文本就返回。

---

## 适配不同供应商

自建适配层主要做两件事：**发送前检查能力**，**接收后保留原生续接条目**。

```python
# 每个供应商写一份能力档案，不在业务代码里散落 if 供应商。
profile = {"supports_tools": True, "preserve_reasoning": True}


def check(req: Request, profile: dict) -> None:
    if req.tools and not profile["supports_tools"]:
        raise ValueError("该供应商不支持工具调用")   # 发送前就失败，而不是被默默忽略
```

`preserve_reasoning` 决定下一轮是否要把 `Reasoning.keep` 原样回传：DeepSeek 带 `tools` 的请求要求回传上一轮的 `reasoning_content`，OpenAI 也有需原样保存的推理条目。适配层遇到不认识的条目，要区分「可忽略的显示信息」和「执行语义未知」：后者应报错或要求升级，不能静默当成「没有工具调用」。[[6]](../references.md#source-deepseek-thinking) [[7]](../references.md#source-deepseek-compat) [[11]](../references.md#source-openai-reasoning)
