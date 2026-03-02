# Cloud Drive

本目录对应云盘业务专题，聚焦大文件上传下载、元数据一致性与分享协同场景下的系统设计。

---

## 核心知识点清单

- 核心对象：用户、空间配额、文件、分片、版本、分享链接。
- 主链路：上传（分片、断点续传、秒传）、下载、预览、分享、回收站恢复。
- 存储架构：对象存储与元数据服务分离，冷热分层与生命周期管理。
- 一致性设计：上传提交协议、元数据与对象存储可见性窗口、幂等重试。
- 安全设计：鉴权授权、临时凭证、防盗链、内容加密、审计追踪。
- 成本治理：去重压缩、跨区域复制策略、带宽与存储成本优化。
- 指标体系：上传成功率、秒传命中率、下载延迟 P99、数据修复时延。

---

## 当前主题索引

- 云盘系统总体设计（待补充）：`cloud_drive_system.md`
- 相关通用机制：
	- [idempotence.md](../../base/idempotence.md)
	- [message_reliability.md](../../base/message_reliability.md)
	- [sharding_and_migration.md](../../base/sharding_and_migration.md)
- 相关工程治理：
	- [observability.md](../../governance/observability.md)
	- [capacity_and_stress_testing.md](../../governance/capacity_and_stress_testing.md)

