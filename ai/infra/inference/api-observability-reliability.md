# 推理 API、可观测性与可靠性

推理服务的外部契约不止是 JSON 字段，还包括排队、流式、取消、错误、限额和版本语义。可观测性必须能把客户端请求关联到调度 iteration、worker、模型版本和资源释放。

---

## API 契约

至少定义：

- model、adapter、输入消息和模板；
- 最大输出、sampling 与 stop 参数；
- streaming 事件格式和最终 usage；
- deadline、客户端取消和服务端最大执行时间；
- validation、overload、runtime、partial stream 错误；
- token 计量与兼容版本。

HTTP 成功只表示协议请求完成，不表示生成内容正确。模型质量与系统可用性需要独立评估。

---

## Admission 与过载

推理请求的资源需求受输入和输出 token、KV dtype、beam 数、adapter 和媒体输入影响。Admission Control 应在接收时估算最坏或受控上界，并限制：

- 等待队列长度或预计等待时间；
- 每租户在途请求/token/KV；
- 最大 prompt 和输出；
- GPU KV block 安全余量；
- 特殊模型能力的独立配额。

过载时快速返回明确错误通常比无限排队或运行中 OOM 更可控。重试策略属于客户端或上层网关，服务端要提供 retryable 分类和必要的 `Retry-After`，避免重试风暴。

---

## 三类信号

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
