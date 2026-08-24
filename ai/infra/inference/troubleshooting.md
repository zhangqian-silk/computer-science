# 推理系统排障

推理排障从用户可见症状开始，沿请求状态、Runtime、设备和分布式边界逐层缩小。先建立可重复 workload 和正确性基线，再改变配置；同时改 batch、量化和并行度会让证据失去因果性。

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
