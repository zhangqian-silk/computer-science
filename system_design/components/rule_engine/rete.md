# Rete 在规则引擎中的工程化落地

本文聚焦「Rete 在规则引擎中的实现与治理」。算法原理、网络结构与复杂度分析放在基础层文档：[`fundamentals/algorithms/rete.md`](../../../fundamentals/algorithms/rete.md)。

---

## 目标与边界

- 目标：在规则量较大、条件有重叠时，降低匹配延迟并提升吞吐稳定性。
- 负责：网络编译、增量匹配、激活管理、互斥与优先级语义落地。
- 不负责：业务动作执行、跨系统事务、业务主数据管理。

---

## 与 rule_engine.md 的关系

- `rule_engine.md`：规则引擎整体设计、模块职责、发布治理。
- 本文：Rete 子模块的工程实现细节。
- 基础算法文档：`../../../fundamentals/algorithms/rete.md`。

---

## 实现映射（基于 demo 代码）

参考实现：[`src/rete_engine.go`](./src/rete_engine.go)。

| 工程对象 | 对应实现 | 说明 |
| :--- | :--- | :--- |
| Alpha 节点去重 | `alphaKey` + `reteBuilder.alphaNodes` | 相同叶子条件共享节点，减少重复计算。 |
| 网络编译 | `reteBuilder.buildExpr` | 将条件树编译为 Alpha/Beta/Not/True 节点。 |
| 会话状态 | `reteSession` | 维护 `facts`、`agenda`、`ruleOrder`、节点内存。 |
| 增量插入 | `InsertFact` | 触发 `true/alpha/not` 节点增量传播。 |
| 增量更新 | `UpdateFact` | 先撤回旧事实再插入新事实，保证状态一致。 |
| 增量撤回 | `RemoveFact` | 清理节点记忆和激活记录。 |
| 冲突消解 | `ResultsForFact` | 按规则顺序 + `mutex_group` 输出结果。 |

---

## 关键工程策略

- 编译期共享：对等价叶子条件做 key 化，提升节点复用率。
- 运行期增量：统一维护 Insert/Update/Retract 路径，避免全量重算。
- 状态隔离：并行评估时按分组创建独立会话，避免共享状态污染。
- 执行确定性：规则预排序，保障优先级语义稳定。

---

## 运行与验收建议

- 基线压测：对比朴素遍历与 Rete 在 `规则数`、`重叠率`、`事实变更率` 下的 P95/P99。
- 内存治理：重点监控 Alpha/Beta 记忆体规模，设置告警与上限。
- 正确性验证：覆盖 Insert/Update/Remove 的一致性回归用例，验证撤回语义。
- 变更安全：规则发布走灰度与回滚，避免热更新抖动放大。

---

## 相关文档

- 规则引擎总体设计：[`rule_engine.md`](./rule_engine.md)
- Rete 基础算法：[`fundamentals/algorithms/rete.md`](../../../fundamentals/algorithms/rete.md)
- 示例代码索引：[`src/TOPICS.md`](./src/TOPICS.md)
