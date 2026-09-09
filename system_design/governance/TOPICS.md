# Governance 知识点目录

本文件用于梳理「工程治理层」需要建设的治理专题与补齐优先级。

---

## 状态说明

- `[已覆盖]`：已有专题文档。
- `[部分覆盖]`：已有文档但内容不完整。
- `[待补充]`：已建占位文档。
- `[未建专题]`：尚未建文档。

---

## 发布治理

- 灰度发布与回滚：[已覆盖] [`release_and_rollback.md`](./release_and_rollback.md)
- 变更准入与风险评审：[已覆盖] 见[`release_and_rollback`](./release_and_rollback.md)（门禁）
- 配置变更治理：[已覆盖] [`../components/config_center/config_center.md`](../components/config_center/config_center.md)
- 数据库变更治理（expand/contract）：[已覆盖] 见[`release_and_rollback`](./release_and_rollback.md)（expand/contract）

---

## 运行治理

- 可观测性治理：[已覆盖] [`observability.md`](./observability.md)
- 事件响应与故障处置：[已覆盖] [`incident_response.md`](./incident_response.md)
- 故障演练与混沌工程：[已覆盖] [`chaos_engineering.md`](./chaos_engineering.md)
- Runbook 与值班机制：[已覆盖] 见[`incident_response`](./incident_response.md)

---

## 质量治理

- 容量规划与压测：[已覆盖] [`capacity_and_stress_testing.md`](./capacity_and_stress_testing.md)
- SLA/SLO 管理：[已覆盖] [`sla_slo_management.md`](./sla_slo_management.md)
- 稳定性评审与质量门禁：[已覆盖] 见[`release_and_rollback`](./release_and_rollback.md)与[`cross_component`](../components/cross_component/cross_component.md)

---

## 成本与演进治理

- 资源成本治理（算力、存储、网络）：[已覆盖] [`cost_and_evolution.md`](./cost_and_evolution.md)
- 架构技术债盘点与清理机制：[已覆盖] [`cost_and_evolution.md`](./cost_and_evolution.md)
- 治理指标看板与持续改进闭环：[已覆盖] [`cost_and_evolution.md`](./cost_and_evolution.md)

---

## 学习优先级

- P0：容量压测、事件响应、SLA/SLO。
- P1：混沌工程、配置与 DB 变更治理。
- P2：成本治理与技术债治理体系。
