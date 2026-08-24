# 模型部署、发布与请求路由

模型发布把不可变制品和 Runtime 配置绑定成可识别版本，再逐步把流量迁移到健康副本。路由要平衡的不只是请求数，还有队列、token 工作量、模型/adapter 能力与已有 KV 局部性。

---

## 发布单元

一个可追踪版本至少包含：

- 模型权重 revision 与制品校验和；
- tokenizer、chat template、generation config；
- quantization、KV dtype、context 与并行布局；
- engine 镜像 digest 和启动参数；
- 支持的 adapter/多模态/结构化输出能力；
- 质量与性能验收报告。

模型别名如 `production` 只用于路由，日志和 usage 应保留解析后的不可变版本。

---

## 发布流程

1. 离线验证制品可加载并通过正确性测试；
2. 在目标硬件启动候选副本并完成预热；
3. readiness 通过后接收影子或极小比例流量；
4. 比较错误、TTFT、TPOT、Goodput、输出质量与资源；
5. 分阶段增加权重，并保留旧版本容量；
6. 异常时停止新增流量并回退路由；
7. 在途请求自然结束或按明确策略排空。

大模型加载慢且占用大量显存，滚动发布的 surge/unavailable 参数必须结合真实节点容量。不能假设新旧版本可在同一 GPU 共存。

---

## 路由策略

| 策略 | 优点 | 边界 |
| --- | --- | --- |
| Round Robin | 简单 | 不感知长度和队列 |
| Least Requests | 反映并发 | 长短请求成本差异大 |
| Queue/Token-aware | 更接近实际负载 | 需要及时、稳定的状态 |
| KV/Prefix-aware | 提高缓存命中 | 目录与热点可能失衡 |
| Capability-aware | 匹配模型/adapter/hardware | 路由规则更复杂 |

基础策略应先保证版本和能力匹配，再使用队列或 token 估算。KV-aware 路由只有在重复前缀比例、节省 Prefill 和目录成本有测量证据时引入。

---

## CPU 路线

使用多个 CPU mock engine 即可完成发布、影子流量、权重迁移、排空、路由和回退演练。真实模型加载时间、GPU 共存和 KV 命中收益在目标环境补测。

通用背景见[负载均衡](../../../system_design/base/high_availability/load_balancing.md)。
