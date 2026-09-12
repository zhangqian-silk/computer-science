# 流式响应与模型适配

流式把一个响应拆成多个事件交付，减少等待、便于显示进展。它改变的是**传输和消费方式**，不改变模型能力，也不保证更快的总完成时间。本页讲怎么把字节和事件安全地拼回 `Response`。

本页沿用 [LLM API](llm-api.md) 的 `Response`／`ToolCall`，新增一个最简事件类型 `StreamEvent`。

---

## 从字节到条目

解析链是分层的，不能对每次网络读取直接 `json.loads`：字节解码 → 组帧（SSE／JSONL）→ 解析事件 → 按 `id` 累积 → 收到权威完成状态。

HTTP 断开不是完成事件；最后一段文本看起来完整，也不能证明后面没有工具调用。**展示可以提前，执行承诺不能提前。**[[13]](../references.md#source-openai-stream)

---

## 拼装工具调用参数

两项调用的片段可以交错到达，必须按 `id` 累积，而不是按「最近工具名」，否则参数会串线。

```python
import json
from dataclasses import dataclass

# 复用 llm-api 的 ToolCall。


@dataclass
class StreamEvent:
    kind: str                 # "call.delta" | "call.done" | "response.done"
    id: str = ""
    name: str = ""
    text: str = ""            # 参数片段


class CallAssembler:
    def __init__(self):
        self.slots: dict[str, dict] = {}

    def feed(self, ev: StreamEvent) -> ToolCall | None:
        if ev.kind == "call.delta":
            slot = self.slots.setdefault(ev.id, {"name": ev.name, "buf": ""})
            if slot["name"] != ev.name:
                raise ValueError("同一 id 的调用名发生变化")
            slot["buf"] += ev.text
            return None
        if ev.kind == "call.done":
            slot = self.slots[ev.id]
            return ToolCall(id=ev.id, name=slot["name"],
                            arguments=json.loads(slot["buf"]))
```

`call.done` 只表示参数拼装完成，不代表立即执行。

---

## 早派发的权衡

保守策略是等 `response.done` 再统一执行；这样最简单，恢复也最容易。

若按条目完成即派发（只读工具可以更早，例如本篇的 `read_file`），就必须保存每个已完成调用的真实回执，把「整体响应异常」和「已完成动作」分开，断线后不能盲目重放。

---

## 背压

慢速消费者不应阻塞模型流。网络读取与 UI 之间用一个有界队列：

- 高频文本增量可以合并成最新快照；
- 工具完成回执必须可靠写入，不能为了不卡 UI 而丢弃。
