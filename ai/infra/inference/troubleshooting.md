# 推理系统排障

排障是逐层缩小差异的过程。输出错误先比较输入 ID 与同一前缀 logits，速度异常先比较阶段时间与 workload，资源异常先比较所有权和生命周期。一次改动越多，越难识别真正原因。

---

## 症状地图

| 症状 | 首先检查 | 常见方向 |
| --- | --- | --- |
| TTFT 高 | queue、tokenize、prompt length、prefill span | 过载、长 prompt、chunk/prefix 策略 |
| TPOT 高 | active batch、iteration、KV length、TP collective | Decode 带宽、kernel、通信、CPU launch |
| 吞吐低但延迟正常 | batch/token budget、请求到达 | 流量不足、调度预算过小 |
| GPU 有空洞 | CPU timeline、同步、数据准备 | launch、锁、tokenize、输出背压 |
| OOM | 权重、KV、workspace、graph、allocator | admission 失配、碎片、峰值 |
| 输出错误 | tokenizer/template、model revision、dtype、sampling | 制品不匹配、kernel/量化误差 |
| 多卡挂起 | 最早错误 rank、collective sequence、网络 | rank 分叉、进程退出、拓扑 |
| 尾延迟抖动 | 长度分布、最慢 worker、GC/热降频 | Straggler、burst、慢客户端 |

---

## 标准流程

1. 保存可复现请求与完整版本；
2. 确定影响范围：单请求、单 worker、单模型还是全局；
3. 用高精度或已知稳定 Runtime 建立输出基线；
4. 分解 queue、prefill、decode、sample、stream 时间；
5. 检查最慢 worker/rank 和资源时间线；
6. 缩小到算子、collective 或控制面；
7. 修改单个变量并重复相同 workload；
8. 验证性能、正确性、尾部和失败率；
9. 记录根因、适用边界与回退。

若问题无法稳定复现，先增加低开销观测并保存触发条件，不应凭一次 profile 直接引入复杂恢复机制。

---

## OOM 的分层定位

区分进程启动权重 OOM、Prefill workspace/激活 OOM、KV admission OOM、CUDA Graph pool OOM、通信 buffer OOM 和 allocator 碎片。记录失败申请大小、当时 active sequence/token、allocated/reserved 和设备总占用。

简单地降低 `gpu_memory_utilization` 可能避免一次 OOM，却不能证明容量模型正确。应修正预算、设置安全余量，并让超容量请求在进入执行前被拒绝。

---

## 正确性差异

假设完整重算正常、cached Decode 在跨块后错误，应先固定模型、输入和 greedy 设置，在每步比较 logits；再比较该位置每层 K/V 与 block table。若错误恰在位置 16 或 32 后出现，可能与边界有关，但只有地址映射或最小复现才能确认，不能仅凭规律下结论。

如果只在量化后生成文本不同，先比较首个分叉前同一前缀上的 logits 误差；一旦采样分叉，后续输入已不同，逐步文本差异不再是同一数值计算的直接比较。

性能修复也应有回归条件。例如排队时间降低但 TPOT 变高，可能只是把更多请求放进活跃批次，并未改善用户 SLO。结论应同时保留收益与代价，而不是只截取变好的曲线。

按以下顺序排除：

- 原始文本是否产生相同 token ID；
- 模型、adapter、位置与模板是否相同；
- greedy/固定 RNG 下 logits 从哪一层开始分叉；
- 是否只在特定 dtype、shape、paged/flash kernel 出现；
- TP rank 的切分和 collective 是否一致；
- 量化误差是否超出已定义任务边界。

自回归文本很容易在早期微小 logits 差异后完全分叉，因此应比较首个差异 token 前的 logits 和中间状态，而不是只比较最终文本。

---

## CPU 路线

CPU mock engine 可构造 queue、取消、OOM 和慢客户端；CPU 小模型可定位 tokenizer、sampling、KV 与量化正确性。多进程 CPU 可复现 collective 顺序错误。GPU kernel、显存碎片、NCCL 和热降频需要目标设备证据。
