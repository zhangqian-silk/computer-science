# 推理 API、可观测性与可靠性

API 是对调用者的可观察承诺：什么时候开始计时、什么表示完成、输出中断后还能否重试，以及计费/用量采用什么 token 口径。可靠性先定义这些边界，再用日志、指标和 trace 证明系统遵守了它们。

---

## API 契约

至少定义：

- model、adapter、输入消息和模板；
- 最大输出、sampling 与 stop 参数；
- streaming 事件格式和最终 usage；
- deadline、客户端取消和服务端最大执行时间；
- validation、overload、runtime、partial stream 错误；
- token 计量与兼容版本。

HTTP 成功状态不保证流式响应完整，更不保证生成内容正确；状态头可能早于生成结束发送。应另检查终止事件、流错误与任务质量。

---

## Admission 与过载

设每秒到达请求 $\lambda$ 长期超过可完成速率 $\mu$，若无拒绝或流量回落，队列就会持续增长；多排一会儿不会凭空增加服务能力。排队超时、过载拒绝和执行失败应分别统计，否则通过丢弃困难请求可能得到更好看的成功请求延迟。

最大输出长度既是生成规则也是资源上界。若不保守预留全部可能 KV，就必须定义逐步增长时容量耗尽如何处理，不能把 admission 成功解释为无限继续生成的保证。

推理请求的资源需求受输入和输出 token、KV dtype、beam 数、adapter 和媒体输入影响。Admission Control 应在接收时估算最坏或受控上界，并限制：

- 等待队列长度或预计等待时间；
- 每租户在途请求/token/KV；
- 最大 prompt 和输出；
- GPU KV block 安全余量；
- 特殊模型能力的独立配额。

过载时快速返回明确错误通常比无限排队或运行中 OOM 更可控。重试策略属于客户端或上层网关，服务端要提供 retryable 分类和必要的 `Retry-After`，避免重试风暴。

---

## 三类信号

Request ID 适合 trace 与日志关联，不适合直接成为每请求一个时序的 metrics label。内容 hash 也并非天然匿名：低熵输入可以被枚举。调试时优先保存长度、版本、错误类型与受控样本，敏感正文需要独立授权和保留规则。

| 信号 | 必要字段 | 用途 |
| --- | --- | --- |
| Metrics | model、worker、阶段、状态、长度 bucket | 趋势、SLO、容量 |
| Trace | request、queue、prefill、decode、stream span | 单请求因果路径 |
| Logs | request/sequence、rank、version、error、resource | 调试与审计 |

关键指标包括 queue time、TTFT、TPOT/ITL、E2E、Goodput、active sequences、scheduled tokens、KV usage、prefix hit、preemption、OOM/reject/cancel 和 worker health。高基数字段不应直接成为 metrics label。

---

## 失败语义

| 失败位置 | 默认行为 |
| --- | --- |
| 验证/Admission 前 | 不分配 GPU 状态，返回明确错误 |
| 排队中取消 | 移出队列并释放配额 |
| 执行中取消 | 在安全边界停止后续 iteration，幂等释放 |
| 首 token 前 worker 失败 | 由上层按幂等与 deadline 决定是否整请求重试 |
| 已流式输出后失败 | 发送流错误或断开，不透明重放已有文本 |
| KV/allocator 不一致 | 隔离 worker，快速失败并保留诊断 |

自动跨 worker 续跑、持久 KV 或 exactly-once streaming 都需要额外协议。没有明确 SLO 和故障证据时，不把它们设为默认要求。

---

## CPU 路线

API、限流、队列、trace、取消、背压与故障注入可全部使用 CPU mock worker 验证。使用确定性 token generator 可以精确断言事件顺序和资源释放，再替换为真实 GPU engine。

通用背景见[负载均衡](../../../system_design/base/high_availability/load_balancing.md)、[限流](../../../system_design/base/high_availability/rate_limiting.md)、[可观测性](../../../system_design/governance/observability.md)和[混沌工程](../../../system_design/governance/chaos_engineering.md)。
