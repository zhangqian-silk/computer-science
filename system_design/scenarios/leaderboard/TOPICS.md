# Leaderboard

本目录对应排行榜业务专题，聚焦高频分数更新、TopN 查询与多维榜单治理场景下的系统设计。

---

## 核心知识点清单

- 核心对象：用户、分数事件、榜单、赛季、分区与分页游标。
- 主链路：分数写入、增量聚合、榜单更新、TopN 查询、邻近排名查询。
- 数据结构：有序集合、分段榜单与冷热榜分层存储策略。
- 一致性设计：实时榜与离线结算榜口径分离，写入幂等与去重。
- 性能治理：写热点打散、批量刷新、查询缓存与回源保护。
- 风控治理：刷分检测、异常行为回滚、赛季封榜与审计追踪。
- 指标体系：更新延迟、查询延迟 P99、榜单准确率、异常率。

---

## 当前主题索引

- 排行榜系统总体设计（待补充）：`leaderboard_system.md`
- 相关通用机制：
	- [cache_consistency.md](../../base/cache_consistency.md)
	- [rate_limiting.md](../../base/high_availability/rate_limiting.md)
- 相关组件能力：
	- [id_generator/TOPICS.md](../../components/id_generator/TOPICS.md)
- 相关工程治理：
	- [observability.md](../../governance/observability.md)
	- [sla_slo_management.md](../../governance/sla_slo_management.md)
