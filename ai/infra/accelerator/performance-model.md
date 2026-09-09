# 性能模型：计算、带宽与延迟

性能模型回答「时间至少花在哪里」以及「改哪一个资源才可能有效」。它不是 benchmark 的替代品：理论下界允许排除不可能的加速主张，真实 profile 才能确定额外开销。先读[GPU 执行层级](./gpu-architecture.md)，本页不要求掌握 CUDA 编程。

<RooflineExplorer />

---

## 四类主要限制

| 限制 | 典型信号 | 常见方向 |
| --- | --- | --- |
| 计算受限 | 计算单元利用率高，数据复用充分 | 更低精度、减少 FLOPs、使用 Tensor Core |
| 内存带宽受限 | 大量 HBM 读写，计算单元等待数据 | 融合、量化、增加 batch、减少中间 Tensor |
| Launch/CPU 受限 | 大量短 kernel、GPU 间隙明显 | 融合、CUDA Graph、异步调度 |
| 通信受限 | Collective 或传输占关键路径 | 拓扑优化、切分调整、计算通信重叠 |

端到端系统还可能受 Tokenization、队列、网络、存储和锁竞争限制。GPU 利用率低并不自动等于需要优化 GPU kernel。

---

## Roofline

算术强度定义为：

$$
I=\frac{\text{FLOPs}}{\text{Bytes transferred}}
$$

若设备峰值计算为 $P_{peak}$，内存带宽为 $BW$，理想性能上界为：

$$
P\le\min(P_{peak},I\times BW)
$$

交点 $P_{peak}/BW$ 称为 ridge point。低于交点的 workload 更可能受带宽限制，高于交点更可能受计算限制。实际 Bytes 必须说明统计的是 HBM、L2 还是主机传输；缓存命中会改变观察层级。

---

## GEMM 与 Decode

::: info 符号与约定
$m,k,n$ 为矩阵维度，$b$ 为每元素字节数，$F$ 为 FLOPs，$R$ 为某个存储层搬运字节。下式假设输入各读一次、输出写一次，是理想流量下界，不包含缓存失效与重复读取。
:::

下界还假设输入初始位于所选存储边界之外、结果需要写回；若输入已驻留更近缓存，必须重新计算该边界的 R，不能固定套用 HBM 字节数。

对于 $[m,k][k,n]\rightarrow[m,n]$：

$$
F\approx2mkn,\qquad R\ge b(mk+kn+mn),\qquad
t\ge\max(F/P_{\text{peak}},R/BW)
$$

当 $m=1$ 且权重 $kn$ 主导流量时，算术强度约为 $2/b$；若相同权重服务 $m$ 个 token，强度可以接近 $2m/b$。这就是增大 batch 能提高权重复用的原因，而非 batch 改变了模型公式。

以假设算子为例，工作量 100 GFLOPs、HBM 流量 20 GB，强度为 5 FLOP/byte；假设设备峰值 100 TFLOP/s、带宽 2 TB/s，则算力时间下界为 1 ms，带宽时间下界为 10 ms。峰值算力再翻倍仍无法突破带宽下界。单位全部是十进制；这不是实际硬件的测量报告。

矩阵乘 $[M,K]\times[K,N]$ 约执行 $2MKN$ FLOPs。大 $M,N,K$ 可以充分复用权重与激活；Decode 时 $M$ 可能只是当前批次的 token 数，权重被读取后只服务少量 token，算术强度下降。

Continuous Batching 的一个重要作用是把不同请求的当前 token 合并，增大有效 $M$。但 batch 增大也会增加排队、KV 容量和单轮执行时间，不能只追求峰值吞吐。

---

## 延迟不能简单相加平均

若某算子只占端到端时间的比例 $f$，将其加速 $s$ 倍，且不改变其余开销，则整体加速比为：

$$
S_{\text{overall}}=\frac{1}{(1-f)+f/s}
$$

例如占比 10% 的算子即便无限加速，总体上限也只有 $1/0.9\approx1.11$ 倍。优化引入额外转换或同步时，收益还会更低。这是从 microbenchmark 返回请求级验证的必要性。

对将时间归入互不重叠阶段的简化请求路径：

$$
T_{e2e}=T_{queue}+T_{tokenize}+T_{prefill}+T_{decode}+T_{network}
$$

若流式网络发送与 Decode 重叠，应沿关键路径计时，不可把包含重叠的各 span 直接相加。

Decode 又包含多个串行 iteration。平均 kernel 时间无法直接给出 p99 请求延迟，因为队列、长度分布、抢占和同步会放大尾部。

服务评测至少同时记录：TTFT、TPOT 或 ITL、E2E latency、token throughput、request throughput 和 Goodput。

---

## CPU 路线

CPU 同样可构建 Roofline：使用对应 CPU 的峰值向量计算能力和内存带宽，配合 BLAS、perf、线程数与 NUMA 绑定测量。CPU 实验可以验证 batch、blocking、量化和内存带宽趋势，但不能把绝对吞吐外推到 GPU。

---

## 参考文献

- Williams, S. et al. (2009). *Roofline: An Insightful Visual Performance Model for Multicore Architectures*.
- Patterson, D. and Hennessy, J. *Computer Organization and Design*.
