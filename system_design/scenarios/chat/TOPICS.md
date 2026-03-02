# Chat

本目录对应「业务专题层」中的聊天（IM）域，聚焦高并发长连接与消息链路约束下的系统设计。

---

## 内容范围

- 聊天业务目标、核心对象模型与关键链路（单聊、群聊、会话列表）。
- 消息收发、存储、投递确认、离线拉取与多端同步机制。
- 在线状态、会话一致性、消息顺序、重复投递与幂等等关键约束。
- 与通用机制和组件的协同（如限流、熔断、可观测性与发布回滚）。

---

## 当前主题索引

- 聊天系统总体设计（待补充）：`chat_system.md`
- 相关通用机制：
	- [rate_limiting.md](../../base/high_availability/rate_limiting.md)
	- [degradation.md](../../base/high_availability/degradation.md)
	- [circuit_breaker.md](../../base/high_availability/circuit_breaker.md)
- 相关工程治理：
	- [observability.md](../../governance/observability.md)
	- [release_and_rollback.md](../../governance/release_and_rollback.md)

---

## 学习建议

- 先明确消息语义边界（至少一次/至多一次/恰好一次）与顺序口径。
- 再定义写扩散与读扩散策略，评估存储、延迟、成本三者平衡。
- 最后补齐限流、降级、告警与回滚机制，形成可上线的闭环设计。
