# AI 服务平台

服务平台位于推理 engine 之上，负责把模型制品、计算资源和外部流量组织成可发布、可观测、可扩缩的服务。它不应重新实现 engine 的 KV 或调度，而应通过稳定接口管理版本、路由、容量和故障边界。

| 页面 | 核心问题 |
| --- | --- |
| [GPU 容器与 Kubernetes](./gpu-containers-kubernetes.md) | 节点、驱动、设备和容器怎样形成可运行环境 |
| [部署与路由](./deployment-and-routing.md) | 模型版本怎样发布，请求怎样选择副本 |
| [GPU Fleet 与容量](./fleet-and-capacity.md) | 如何放置模型、扩缩容并保留故障余量 |
| [安全与多租户](./security-and-multitenancy.md) | 输入、模型、GPU 和观测数据怎样隔离 |

通用高可用、网络和可观测性沿用仓库系统设计部分；本分区只展开 AI workload 特有的模型制品、GPU、KV 局部性和 token 资源。
