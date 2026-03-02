# Search

本目录对应搜索业务专题，聚焦召回、排序、索引更新与查询性能约束下的系统设计。

---

## 核心知识点清单

- 核心对象：查询词、文档、倒排索引、排序特征、用户行为日志。
- 主链路：请求解析、召回、粗排、精排、重排、结果聚合与返回。
- 索引构建：离线全量构建与在线增量更新协同。
- 一致性与新鲜度：写入延迟、可见性窗口、最终一致性口径。
- 性能治理：高 QPS 读路径、缓存分层、热点查询保护与降级策略。
- 相关性优化：分词、拼写纠错、同义词扩展与学习排序。
- 质量指标：NDCG、CTR、无结果率、查询延迟 P99。

---

## 当前主题索引

- 搜索系统总体设计（待补充）：`search_system.md`
- 相关通用机制：
	- [load_balancing.md](../../base/high_availability/load_balancing.md)
	- [rate_limiting.md](../../base/high_availability/rate_limiting.md)
	- [cache_consistency.md](../../base/cache_consistency.md)
- 相关工程治理：
	- [observability.md](../../governance/observability.md)
	- [capacity_and_stress_testing.md](../../governance/capacity_and_stress_testing.md)

