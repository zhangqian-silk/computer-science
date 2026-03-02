# Base 知识点目录

本文件用于梳理「通用机制层」需要系统学习和持续补齐的核心知识点。

---

## 状态说明

- `[已覆盖]`：已有独立文档，具备学习入口。
- `[部分覆盖]`：已有内容但体系不完整。
- `[待补充]`：已建占位文档，等待展开。
- `[未建专题]`：尚未建立文档。

---

## 微服务与服务治理

- 服务注册发现：[部分覆盖]
- API 网关治理（鉴权、路由、限流）：[部分覆盖]
- RPC 与服务通信模型：[部分覆盖]
- 服务配置中心与动态配置：[未建专题]
- 多环境隔离与发布编排：[未建专题]
- Service Mesh 与 Sidecar 治理：[部分覆盖]

---

## 高可用与稳定性

- 负载均衡：[已覆盖] `high_availability/load_balancing.md`
- 限流：[已覆盖] `high_availability/rate_limiting.md`
- 降级：[已覆盖] `high_availability/degradation.md`
- 熔断：[已覆盖] `high_availability/circuit_breaker.md`
- 超时与重试预算：[未建专题]
- 隔离与舱壁模式：[未建专题]
- 故障注入与稳定性验证：[未建专题]

---

## 一致性与数据可靠性

- 幂等机制：[已覆盖] `idempotence.md`
- 分布式事务（TCC、SAGA、Outbox）：[待补充] `distributed_transaction.md`
- 缓存一致性与缓存失效策略：[待补充] `cache_consistency.md`
- 消息可靠性（重试、死信、去重）：[待补充] `message_reliability.md`
- 分片与扩容迁移：[待补充] `sharding_and_migration.md`

---

## 安全机制

- 认证（Session/JWT/SSO/OIDC/MFA）：[已覆盖] `security/authentication.md`
- 授权（RBAC/ABAC/ACL/OAuth）：[已覆盖] `security/authorization.md`
- 密钥管理与轮换策略：[未建专题]
- 服务间身份认证（mTLS、SPIFFE）：[未建专题]

---

## 学习优先级

- P0：分布式事务、缓存一致性、消息可靠性。
- P1：超时与重试预算、隔离与舱壁、配置中心。
- P2：服务间身份体系、分片迁移工程实践。
