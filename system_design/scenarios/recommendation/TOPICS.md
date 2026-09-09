# Recommendation

本目录对应推荐系统专题，聚焦「从海量内容里为用户选出最相关的少量」的召回-排序链路与实时性、冷启动、多样性治理。

> 主文档：[`recommendation_system.md`](./recommendation_system.md)（召回/排序漏斗 + 特征与实时性）。

---

## 核心知识点清单

- 核心对象：用户、物品、行为日志、特征、召回源、排序模型。
- 主链路：召回（多路）→ 粗排 → 精排 → 重排（多样性/业务规则）。
- 召回策略：协同过滤、内容召回、向量召回（ANN）、热门/新品兜底。
- 特征与实时性：离线特征 + 实时特征、特征一致性（训练/服务一致）。
- 冷启动：新用户/新物品的兜底与探索（EE，Exploration/Exploitation）。
- 多样性与体验：去重、打散、疲劳度、内容生态。
- 指标：CTR、CVR、时长、留存、多样性、覆盖率。

---

## 相关文档

- 与[搜索](../search/search_system.md)/[广告](../advertising/advertising_system.md)共享多阶段漏斗与向量召回。
- 读放大靠[缓存](../../components/cache_component/cache_component.md)；日志流靠[消息可靠性](../../base/message_reliability.md)。
