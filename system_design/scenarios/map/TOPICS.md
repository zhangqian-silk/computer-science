# Map

本目录对应地图业务专题，聚焦空间数据处理、路径规划与实时路况服务场景下的系统设计。

---

## 核心知识点清单

- 核心对象：POI、道路网络、瓦片数据、路径请求、实时路况事件。
- 主链路：请求解析、候选召回、路径规划、ETA 评估、结果渲染与返回。
- 数据管线：多源采集、清洗融合、离线构图、在线增量更新。
- 检索与索引：地理空间索引（Geohash/网格索引）与附近查询优化。
- 性能治理：高并发低延迟查询、热点区域缓存、降级与兜底路线。
- 一致性与新鲜度：路况更新延迟、索引可见性窗口、多源冲突解决。
- 指标体系：路径耗时偏差、ETA 准确率、查询延迟 P99、无结果率。

---

## 当前主题索引

- 地图系统总体设计（待补充）：`map_system.md`
- 相关通用机制：
	- [load_balancing.md](../../base/high_availability/load_balancing.md)
	- [rate_limiting.md](../../base/high_availability/rate_limiting.md)
	- [cache_consistency.md](../../base/cache_consistency.md)
- 相关工程治理：
	- [observability.md](../../governance/observability.md)
	- [chaos_engineering.md](../../governance/chaos_engineering.md)

