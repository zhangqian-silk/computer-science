# Marketing

本目录对应「业务专题层」中的营销域，聚焦营销业务目标与业务约束下的系统设计。

---

## 内容范围

- 营销业务目标、核心对象模型与主链路。
- 预算、频控、互斥、优先级、配额等业务约束的系统化设计。
- 营销专题中的核心能力拆解（如规则引擎、归因、触达、权益）。
- 与通用能力的协同落地（如短链、限流、发布回滚、可观测性）。

---

## 当前主题索引

- 营销系统总体设计：[marketing_system.md](./marketing_system.md)
- 规则引擎组件专题（组件能力层）：
	- [rule_engine.md](../../components/rule_engine/rule_engine.md)
	- [rete.md](../../components/rule_engine/rete.md)
	- 示例代码：[src/TOPICS.md](../../components/rule_engine/src/TOPICS.md)
- 短链组件专题（组件能力层）：
	- [TOPICS.md](../../components/short_link/TOPICS.md)
- 限流机制（通用机制层）：
	- [rate_limiting.md](../../base/high_availability/rate_limiting.md)
- 红包拆分算法（算法层）：
	- [red_packet_split.md](../../../fundamentals/algorithms/red_packet_split.md)
