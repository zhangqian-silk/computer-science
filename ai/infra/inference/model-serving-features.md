# LoRA、MoE、多模态与结构化输出服务

推理引擎的基础接口是 token 序列，但实际模型可能引入 adapter、专家路由、图像/音频编码器或约束解码。这些特性会改变模型制品、batch 兼容性、缓存 key 和调度预算。

---

## Multi-LoRA Serving

基础权重常驻设备，多个 LoRA adapter 按请求选择并应用低秩增量。系统需要管理：

- adapter 版本、rank、target module 与兼容基础模型；
- CPU/磁盘到 GPU 的加载和缓存；
- 同一 batch 中多个 adapter 的 grouped kernel；
- adapter miss 时的排队或拒绝；
- adapter 卸载与在途请求引用。

adapter cache 可以是 best-effort，驱逐只导致重新加载。若加载延迟会破坏 SLO，应通过预热、容量或 admission 解决，而不是默认无限保留。

---

## MoE Serving

MoE 每个 token 只激活部分专家，但专家权重总量大。Expert Parallel 要把 token 路由到专家设备，再合并输出。需要观察：

- router top-k 与 capacity；
- 每轮专家 token 分布；
- AllToAll 和 token reorder；
- 热门专家造成的 straggler；
- Decode 小 token batch 下 grouped GEMM 效率。

MoE 的「激活参数少」不等于服务内存或通信少，未激活专家权重仍需放置或按需加载。

---

## 多模态

图片、音频或视频先经 encoder/processor 变为模型可消费表示。资源预算除了文本 token，还要包含：

- 下载、解码、尺寸和安全校验；
- CPU/GPU 预处理；
- encoder 计算与 cache；
- multimodal token 或 cross-attention 状态；
- 与文本模板和位置对齐。

同一媒体输入可复用 encoder 输出时，缓存 key 必须绑定 processor、encoder 和模型版本。外部 URL 获取还涉及网络与安全边界，不应由 model runner 隐式完成。

---

## 结构化输出

JSON Schema、regex 或 grammar 解码把约束编译成状态机，每步根据当前状态 mask 非法 token。其 CPU 成本和词表 mask 传输可能成为短 Decode 的一部分。

Batch 中各序列状态不同，引擎需要维护每序列约束状态。Speculative decoding、beam 和回滚必须同步状态，否则候选 token 与 grammar 状态会分叉。

---

## CPU 路线

Adapter 管理、MoE 路由、媒体预处理和 grammar 状态机都可用 CPU/mock model 验证。小模型还可真实执行 LoRA 与 MoE。GPU 环境用于验证 grouped GEMM、AllToAll 和多模态 encoder 的性能。

## 参考资料

- Hu, E. et al. (2021). *LoRA: Low-Rank Adaptation of Large Language Models*.
- Shazeer, N. et al. (2017). *Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer*.
