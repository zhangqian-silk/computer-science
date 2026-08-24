# GPU Fleet、调度与容量

Fleet 层把异构节点分配给模型副本和分布式 worker group。容量规划需要同时满足模型可放置、请求 SLO、节点故障、发布冗余和成本目标。

---

## 两层容量

单副本容量先由[推理基准](../inference/benchmarking-capacity.md)给出：在固定 workload 与 SLO 下，一个 replica group 的 Goodput 和 KV 上限。Fleet 再计算：

$$
N_{groups}\ge
\frac{QPS_{target}}{Goodput_{group}\times utilization_{target}}
$$

$utilization_{target}<1$ 用于吸收流量波动、节点故障和估算误差。还需额外考虑发布时新旧版本并存和多可用区故障目标。

---

## 放置约束

- TP group 尽量保持在目标 NVLink/NVSwitch 拓扑内；
- 多机 group 需要匹配 NIC、网络域与 gang scheduling；
- CPU、主机内存和本地磁盘不能只按 GPU 数量等比例假设；
- 模型 cache 命中可以缩短启动，但不能破坏制品校验；
- 异构 GPU 可以运行不同 replica，但不要在同一同步 group 中无意混用性能差异过大的设备；
- 热点模型、adapter 或长上下文流量需要独立容量桶。

---

## 扩缩容时标

模型下载、加载、分布式初始化、JIT 和预热可能远慢于普通无状态服务启动。只依据瞬时 QPS 扩容常来不及响应。可组合：

- 基于排队和 Goodput 的反应式扩容；
- 基于周期/事件预测的预扩容；
- 保留热备或已加载但不接流量的容量；
- 对突发流量使用 admission 与明确拒绝。

热备会持续占用昂贵 GPU。是否需要由启动时间、突发频率、SLO 和成本共同证明。

---

## 故障与降级

节点失败时最简单的边界是停止向故障 group 路由并让其在途请求失败，上层按请求状态决定重试。跨组迁移 KV、动态缩小 TP 或在另一硬件上继续生成需要复杂协议，不能作为默认能力。

可以预先定义可验证的降级选项：更小模型、缩短最大上下文/输出、降低接纳率或只保留高优先级租户。降级不能静默改变模型质量或 API 契约。

---

## CPU 路线

CPU 集群或本地多进程可以模拟节点标签、gang placement、启动时间、故障、扩缩容和成本模型。用 mock Goodput 校验控制策略，再用真实 GPU benchmark 替换容量参数。

通用调度背景见[分布式调度](../../../distributed/scheduler.md)。
