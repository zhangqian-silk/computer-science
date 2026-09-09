# LoRA、MoE、多模态与结构化输出服务

LoRA、MoE、多模态与结构约束分别改变参数、条件计算、输入接口和可选输出集合。它们不仅是 API 上的开关，还会改变哪些请求能共同执行、哪些缓存可以复用，以及失败时要释放哪些状态。

---

## Multi-LoRA Serving

同一基座的两个请求分别使用 $\Delta W_A$ 和 $\Delta W_B$，底层矩阵相同也不意味着整段 KV 可共享：适配器改变隐藏状态，历史 K/V 可能随之不同。缓存 key 需要绑定适配版本。动态 batch 还需让每行 token 使用正确增量，不能把 batch 中第一个 adapter 误用到全部行。

基础权重常驻设备，多个 LoRA adapter 按请求选择并应用低秩增量。系统需要管理：

- adapter 版本、rank、target module 与兼容基础模型；
- CPU/磁盘到 GPU 的加载和缓存；
- 同一 batch 中多个 adapter 的 grouped kernel；
- adapter miss 时的排队或拒绝；
- adapter 卸载与在途请求引用。

adapter cache 可以是 best-effort，驱逐只导致重新加载。若加载延迟会破坏 SLO，应通过预热、容量或 admission 解决，而不是默认无限保留。

---

## MoE Serving

一次 Top-2 路由会使每 token 产生两份专家分配。Dispatch 后按专家分组，执行，再按原 token ID 和门权重合并；任一重排索引错误都可能形状正确却把别人的结果加回来。先在 CPU 上验证「路由—逆路由」的身份守恒，再测 AllToAll 和 grouped GEMM。

MoE 每个 token 只激活部分专家，但专家权重总量大。Expert Parallel 要把 token 路由到专家设备，再合并输出。需要观察：

- router top-k 与 capacity；
- 每轮专家 token 分布；
- AllToAll 和 token reorder；
- 热门专家造成的 straggler；
- Decode 小 token batch 下 grouped GEMM 效率。

MoE 的「激活参数少」不等于服务内存或通信少，未激活专家权重仍需放置或按需加载。

---

## 多模态

图片尺寸不直接等于语言 token 数。Patch、裁剪、多分辨率策略与连接器压缩共同决定视觉状态大小；同样一张图片在不同 processor 配置下可能产生不同接口。预处理失败应在模型执行前显式结束，不能用空图状态继续并宣称已理解输入。

图片、音频或视频先经 encoder/processor 变为模型可消费表示。资源预算除了文本 token，还要包含：

- 下载、解码、尺寸和安全校验；
- CPU/GPU 预处理；
- encoder 计算与 cache；
- multimodal token 或 cross-attention 状态；
- 与文本模板和位置对齐。

同一媒体输入可复用 encoder 输出时，缓存 key 必须绑定 processor、encoder 和模型版本。外部 URL 获取还涉及网络与安全边界，不应由 model runner 隐式完成。

---

## 结构化输出

语法合法只保证字符串属于约定语言。`{"amount": -3}` 可以是合法 JSON，却违反业务范围；工具调用的目标还可能无权限。因此 grammar、schema、业务校验和授权是四层检查，不应让模型输出越过后两层。

JSON Schema、regex 或 grammar 解码把约束编译成状态机，每步根据当前状态 mask 非法 token。其 CPU 成本和词表 mask 传输可能成为短 Decode 的一部分。

引擎通常只支持约束语言或 Schema 的特定子集；递归 grammar 也不等同于有限状态自动机。不能假定所有 Schema/业务语义均已在 token 选择时保证，仍须按契约验证完整输出。

Batch 中各序列状态不同，引擎需要维护每序列约束状态。Speculative decoding、beam 和回滚必须同步状态，否则候选 token 与 grammar 状态会分叉。

---

## CPU 路线

Adapter 管理、MoE 路由、媒体预处理和 grammar 状态机都可用 CPU/mock model 验证。小模型还可真实执行 LoRA 与 MoE。GPU 环境用于验证 grouped GEMM、AllToAll 和多模态 encoder 的性能。

---

## 参考资料

- Hu, E. et al. (2021). [*LoRA: Low-Rank Adaptation of Large Language Models*](https://arxiv.org/abs/2106.09685).
- Shazeer, N. et al. (2017). [*Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer*](https://arxiv.org/abs/1701.06538).
