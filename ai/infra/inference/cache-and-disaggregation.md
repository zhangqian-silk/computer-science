# 前缀缓存、KV 复用与 Prefill/Decode 分离

缓存和阶段分离都在请求之间引入新的状态关系。它们只有在重复前缀、阶段资源冲突或独立扩缩容确有证据时才值得增加；否则会带来目录、失效、传输与故障恢复成本。

---

## 前缀缓存

若两个请求拥有完全相同的 token 前缀、模型和执行语义，可以复用此前缀的 KV blocks，省去重复 Prefill。缓存 key 至少应绑定：

- model/weight version；
- tokenizer 与 chat template；
- 完整 token IDs；
- adapter、position、RoPE/长上下文配置；
- 影响 KV 的 multimodal encoder 输入；
- KV dtype、并行布局和必要的 kernel 版本。

只按原始文本 hash 不足以证明 KV 可复用，因为模板、tokenizer 或模型版本可能不同。

---

## Radix/Trie 索引

Radix tree 或 trie 可以按 token 序列寻找最长缓存前缀，并让多个请求共享 block。索引节点需要管理 block 引用与淘汰优先级。命中率应按「节省的 Prefill token」而非仅按请求数统计。

缓存占用会与新请求 KV 争夺同一容量。应比较保留一个冷前缀的机会成本：它是否比用这些 block 接纳新请求产生更多 Goodput。

---

## PD 分离

Prefill worker 计算 prompt 并产生 KV，Decode worker 接收 KV 后继续生成。数据面需要传输每层 KV，控制面需要路由请求、确认接收并开始 Decode。

端到端首 token近似增加：

$$
T_{TTFT}=T_{queue,p}+T_{prefill}+T_{KV-transfer}+T_{queue,d}+T_{first-decode}
$$

只有独立扩缩容、减少阶段互相干扰或使用差异化硬件带来的收益大于传输与额外排队，方案才有效。

---

## 失败与一致性边界

- Prefix cache 可定义为 best-effort：丢失只导致重算，不影响正确性；
- KV 传输失败时，若未向客户端承诺输出，可重新 Prefill 或失败；
- 已进入 Decode 后避免同时让两个 worker 继续同一序列；
- 模型版本不一致必须拒绝接收 KV；
- 远程 cache 目录若过期，应以安全 miss 而不是错误 hit 结束。

若没有跨故障恢复 SLO，不必把 KV 作为持久数据；把它视为可重算的易失缓存可以显著简化协议。

---

## CPU 路线

CPU 可以实现 token trie、引用计数、LRU/成本感知淘汰和 KV 传输协议模拟。用可配置网络延迟与带宽评估 PD 分离盈亏点，并注入丢包、worker 退出和版本不匹配。真实 KV 大小和 GPU Direct 传输需在目标环境校准。

## 参考资料

- Zheng, L. et al. (2023). *SGLang: Efficient Execution of Structured Language Model Programs*.
- Zhong, Y. et al. (2024). *DistServe: Disaggregating Prefill and Decoding for Goodput-optimized Large Language Model Serving*.
