# 模型输出与结构化结果

一个响应是「一组带类型的条目 + 一个完成状态」，不是一段必然可信的文字。本页讲怎么先看完成状态、再把响应条目分类，并对结构化输出补上业务校验。

本页沿用 [LLM API](llm-api.md) 的 `Response`／`Text`／`ToolCall`／`Reasoning`。

---

## 先看完成状态，再看内容

一个响应可以先解释意图、再提出工具调用，还携带推理条目。所以不能用一个枚举描述整段响应，也不能见到文字就返回用户。至少分开两个轴：

| 轴 | 取值 | 用途 |
| --- | --- | --- |
| 完成状态 | completed／incomplete／refused | 决定哪些内容能当作确定结果 |
| 条目类型 | Text／ToolCall／Reasoning | 分别进入显示、执行、续接 |

例如文本是「正在读取配置」但同一响应含 `read_file`：这句话是过程说明，不是文件已被读取的证据。

---

## 分类响应条目

逐条处理，保留调用身份，遇到未知条目就停下，而不是猜。

```python
def classify(resp: Response):
    texts, calls, keep = [], [], []
    seen = set()
    for item in resp.output:
        if isinstance(item, Text):
            texts.append(item.text)
        elif isinstance(item, ToolCall):
            if item.id in seen:
                raise ValueError("同一响应出现重复调用 ID")
            seen.add(item.id)
            calls.append(item)
        elif isinstance(item, Reasoning):
            keep.append(item.keep)          # 原样保留，供下一轮续接
        else:
            raise ValueError("未知条目，停止自动分派")
    return "".join(texts), calls, keep
```

这一步只解释响应，不执行函数、也不产生「成功」状态。

---

## 结构化输出与业务校验

「请用 JSON 返回」只是提示；JSON mode 约束可解析形式；strict schema 进一步约束字段。**schema 通过只证明结构，不证明事实与权限**。

沿用端口示例：让模型给出 `port`／`source`／`status`，schema 之外再补业务校验。

```python
port_schema = {
    "type": "object",
    "properties": {
        "port": {"type": ["integer", "null"]},
        "source": {"type": ["string", "null"]},
        "status": {"enum": ["observed", "not_found"]},
    },
    "required": ["port", "source", "status"],
    "additionalProperties": False,
}


def validate_port(result: dict, allowed_sources: set) -> None:
    # schema 校验在前；这里补业务语义。
    if result["status"] == "observed":
        if not (isinstance(result["port"], int) and 1 <= result["port"] <= 65535):
            raise ValueError("不是有效端口")
        if result["source"] not in allowed_sources:
            raise ValueError("来源不可追溯")
    elif result["port"] is not None:
        raise ValueError("not_found 不应附带猜测值")
```

拒绝可能不符合 schema，要单独处理，而不是把「解析失败」一律塞回模型无限修复。修正闭环应有限次数；没有证据或无权限时返回相应状态，不强迫模型编一个合法数字。[[10]](../references.md#source-openai-structured)

---

## 推理条目怎么留

Reasoning 有三层：模型内部计算、API 暴露的续接／摘要、用户可见解释。可见解释帮助理解答案，但不是执行日志——一句「我已经测试了」不能代替退出码与输出。

不同模型的推理接口不同（OpenAI 可能返回需原样保存的条目，DeepSeek 要求回传特定历史字段）。适配器应原样保留 `Reasoning.keep`，不为「省上下文」擅自删除。[[11]](../references.md#source-openai-reasoning) [[6]](../references.md#source-deepseek-thinking)
