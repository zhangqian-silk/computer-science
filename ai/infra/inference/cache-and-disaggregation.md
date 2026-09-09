# 前缀缓存、KV 复用与 Prefill/Decode 分离

前缀缓存复用「同一计算已经完成」的结果，Prefill/Decode 分离改变「计算在哪台设备完成」。前者需要精确命中语义，后者需要状态传输与所有权交接；二者都不改变语言模型概率定义，也都可能因额外状态成本而不划算。

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

若请求 A 的 token 为 `[a,b,c,d]`，B 为 `[a,b,c,e]`，最多复用前三个 token 的相应状态；不能仅因原始字符串开头相同，就忽略模板和分词后的差异。完整块共享还可能使实际复用长度向块边界取整。

一个命中 10 token 的短请求与命中 10,000 token 的长请求，在「请求命中率」里各算一次，实际节约却不同。因此既要报告命中请求比例，也要报告复用 token 数、节约时间和保留缓存占用。

Radix tree 或 trie 可以按 token 序列寻找最长缓存前缀，并让多个请求共享 block。索引节点需要管理 block 引用与淘汰优先级。命中率应按「节省的 Prefill token」而非仅按请求数统计。

缓存占用会与新请求 KV 争夺同一容量。应比较保留一个冷前缀的机会成本：它是否比用这些 block 接纳新请求产生更多 Goodput。

---

## PD 分离

Prefill worker 计算 prompt 并产生 KV，Decode worker 接收 KV 后继续生成。数据面需要传输每层 KV，控制面需要路由请求、确认接收并开始 Decode。

若首 token 在 Decode 侧交付，跨池路径可写为：

$$
T_{TTFT}=T_{queue,p}+T_{prefill}+T_{KV-transfer}+T_{queue,d}+T_{first-decode}
$$

这里 $T_{first-decode}$ 表示该服务协议在 Decode 侧产生/发送首个可见输出所需工作；若 Prefill 已采样并交付首 token，用户 TTFT 不必等待上述整条路径，但随后 token 的间隔仍可能受传输影响。指标边界必须与实现对应。

传输 $M$ 字节 KV 的理想带宽时间至少为 $M/BW$，另加启动、排队和布局转换。只有独立扩缩容、减少阶段干扰等收益大于新增成本，方案才成立。DistServe 研究的是受 SLO 约束的 Goodput，而非只看单 kernel 是否更快。

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

---

## 参考资料

- Zheng, L. et al. (2023). [*SGLang: Efficient Execution of Structured Language Model Programs*](https://arxiv.org/abs/2312.07104).
- Zhong, Y. et al. (2024). [*DistServe: Disaggregating Prefill and Decoding for Goodput-optimized Large Language Model Serving*](https://arxiv.org/abs/2401.09670).
