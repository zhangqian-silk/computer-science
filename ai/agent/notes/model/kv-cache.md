# KV Cache 与请求缓存

缓存复用的是**计算或结果**，不是模型对用户的长期记忆。本页区分四种缓存，并讲怎么让一个请求更可能复用前缀。

本页沿用 [LLM API](llm-api.md) 的 `Request` 表示请求布局。

---

## 四种缓存别混为一谈

| 缓存 | 保存什么 | 失效重点 |
| --- | --- | --- |
| 单次生成 KV | 本次生成的注意力中间状态 | 同一次计算内的合法续接 |
| 跨请求前缀 KV | 已算过的相同前缀 | 模型、token、配置、租户 |
| 供应商 Prompt Cache | 缓存读写的产品计量 | 模型／API 版本、断点、保留期 |
| 应用结果缓存 | 问题或工具请求的业务结果 | 数据新鲜度、身份、授权、失效 |

按「问题字符串」缓存答案不是 KV Cache；把两个用户的「查我的订单」共用一个键会造成权限泄露。

---

## KV Cache 原理与显存

自回归解码里，已生成位置的 Key／Value 可供后续 token 复用，避免每步重算全部历史；但新 Query 仍要读历史，所以「历史长度不再影响计算」是不成立的。前缀能否复用，看模型实际所见的 token 与配置（空格、工具 schema 顺序、系统指令、模型版本都算），不是人眼觉得语义相同。[[4]](../references.md#source-hf-cache) [[16]](../references.md#source-vllm-apc)

对普通 MHA／GQA 解码器，KV 显存可近似为：

$$
M_{\mathrm{KV}} = 2 L B T H_{\mathrm{kv}} D s
$$

`L` 层数，`B` 活跃序列数，`T` 每序列缓存 token 数，`H_kv` KV 头数，`D` 每头维度，`s` 每元素字节数，前面的 2 来自 K 和 V。

```python
layers, batch, tokens = 32, 1, 8192
kv_heads, head_dim, bytes_per_value = 8, 128, 2
gib = 2 * layers * batch * tokens * kv_heads * head_dim * bytes_per_value / 1024**3
print(gib)   # ≈ 1.0，未计分配器等额外开销
```

并发翻四倍、未共享前缀时缓存也约翻四倍，这就是长上下文与高并发争抢显存的原因。

---

## 让请求更可能复用前缀

常见布局是：稳定指令 → 稳定能力 → 可复用资料 → 当前历史 → 动态查询。用统一的 `Request` 表示，就是把 config 当可复用资料放前面，把用户问题放最后。

```python
req = Request(
    model="demo-model",
    instructions="只依据已提供或读到的内容回答。",        # 稳定前缀
    input=[
        {"role": "user", "content": "<config.yaml 内容，可复用>"},
        {"role": "user", "content": "服务端口是多少？"},   # 动态后缀
    ],
)

manifest = {                       # 记录前缀由什么构成，便于解释命中变化
    "prompt_revision": "p-1",
    "tools_revision": "tools-1",
    "docs_snapshot": "2026-09-10",
}
```

不要把时间戳、随机 ID 放进 `instructions` 开头。压缩历史、工具热升级、切换模型都会改变前缀，应作为**显式失效事件**，而不是把命中率下降都归因于服务端波动。[[17]](../references.md#source-claude-cache) [[18]](../references.md#source-openai-cache)

---

## 应用结果缓存

结果缓存的键要反映资源身份和授权范围，而不是只看参数字符串。

```python
def cache_key(tool: str, args: dict, tenant: str) -> tuple:
    # read_file 的结果按 (租户, 工具, 文件) 缓存；新鲜度另行判断。
    return (tenant, tool, args["path"])
```

订单状态变化后，同样的问题应得到新答案；用户第二次请求「创建相同订单」是新业务意图，不能因参数相同直接返回旧结果。结果缓存、去重账本和 KV Cache 即便都用 Redis，也要分开——它们的失效与正确性条件不同。
