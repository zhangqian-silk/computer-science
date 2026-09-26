# 模型生成原理：采样、上下文与缓存

模型生成是依据当前输入的条件概率分布逐步选择输出 Token 的过程。本页以**自回归、因果注意力语言模型**为默认对象，解释交互接口背后的生成、资源与状态复用机制；应用侧的消息、工具和多轮协议见[模型交互](./model.md)。多模态编码器、非自回归生成和特定模型的推理阶段可能采用不同实现，不能直接套用全部推导。

::: info 符号与约定
$c$ 是当前请求经模板与分词形成的输入序列，$x_t$ 是第 $t$ 个输出 Token，$\mathcal{V}$ 是词表；$z_i$ 是候选 $i$ 的 logit，$\tau$ 是温度，$p$ 是 top-p 阈值。KV 推导中的 $K,V$ 分别为注意力的 Key、Value，$Q$ 为 Query，$d_k$ 为单头 Key 维度；上标 $\ell$ 表示层，下标 $t$ 表示位置。
:::

---

## 条件生成

聊天消息、角色和内容边界由模型对应的聊天模板序列化，再转换为 Token。模板中的角色标记与特殊结束标记同样参与生成；文字长度不等于 Token 数。多模态输入还需经过相应编码，不能仅按可见文本估算其容量。分词和聊天模板的细节见[Tokenization](../representation/tokenization.md)。

自回归模型把输出序列的联合概率分解为逐位置的条件概率：

$$
P_{\theta}(x_{1:m}\mid c)
=\prod_{t=1}^{m}P_{\theta}(x_t\mid c,x_{<t})
$$

每一步从已经可见的前缀计算候选分数，再选择一个 Token 并把它纳入下一步条件。请求中的任务指令、示例与历史改变的是条件 $c$，不更新参数 $\theta$。模型可输出与任务有关的文本、特殊标记或结构化调用条目；外部工具的实际执行不属于这条概率分解。

典型实现先对输入进行 **Prefill**，建立各位置的中间状态，并从末位置分布选择首个输出 Token；随后 **Decode** 逐步处理新 Token、更新状态并产生后续分布。首 Token 已由 Prefill 的末位置产生，不应再把它算作一次完整的 Decode。执行形状、调度及延迟口径见[Prefill 与 Decode](../infra/inference/prefill-decode.md)。

---

## 候选采样

### Temperature

令 $z_i$ 为当前位置词表候选 $i$ 的 logit，正温度 $\tau>0$ 对 softmax 分布的作用为：

$$
q_i(\tau)
=\frac{\exp(z_i/\tau)}{\sum_{j\in\mathcal{V}}\exp(z_j/\tau)}
$$

两候选的概率之比给出更直接的判据：

$$
\log\frac{q_i(\tau)}{q_j(\tau)}
=\frac{z_i-z_j}{\tau}
$$

因此，当 $z_i>z_j$ 时，降低 $\tau$ 会放大高分候选相对低分候选的优势；提高 $\tau$ 则缩小相对差异。温度只改变既有分数的选择分布，不会为模型增加事实依据。当 $\tau\to0^+$ 且最大 logit 唯一时，概率集中到最高分候选；直接选择最大分数的贪心解码应与 $\tau>0$ 的采样区分。具体服务对 `temperature=0`、并列最大值与其他解码约束的处理须看接口约定，低温也不保证跨硬件或版本逐字复现。

### Top-p（nucleus sampling）

Top-p 不预设固定候选数。将温度处理后的概率按降序排列为 $q_{(1)},q_{(2)},\ldots$，保留累计质量首次达到阈值 $p$ 的最小前缀：

$$
k_p=\min\left\{k:\sum_{j=1}^{k}q_{(j)}\ge p\right\},
\qquad S_p=\{(1),\ldots,(k_p)\}
$$

在保留集合内重新归一化并采样：

$$
\widetilde q_i
=\frac{q_i\,\mathbf{1}[i\in S_p]}{\sum_{j\in S_p}q_j}
$$

例如三个候选概率依次为 $0.6,0.3,0.1$，取 $p=0.8$ 时，第一项未达阈值，第二项使累计值达到 $0.9$；保留前两项后，采样概率分别为 $2/3$ 和 $1/3$。阈值对应的是**累计概率质量**，不是固定保留 80% 的词表；跨过阈值的候选也应保留。[1](https://arxiv.org/abs/1904.09751)

Temperature 调整整个分布的相对概率，top-p 在所得分布上截取候选并重新归一化。如果同时启用，top-p 的保留集合会随温度变化。不同接口也可能提供 top-k、重复惩罚或其他约束；它们改变候选分布的方式各异，不能把参数值当作跨模型可移植的质量刻度。

### 终止与输出预算

生成在模型给出结束标记、命中约定的停止条件，或达到输出上限等条件下结束。结束标记表示模型停止，输出上限仅表示资源边界；被截断的文本即使在语法上像完整句，也不等于任务完成。支持推理阶段的服务可能把非可见推理 Token 计入输出预算，因而可见回答长度不能由预算值直接推断。

---

## 因果注意力与 KV Cache

因果注意力中，新位置只读取当前及先前位置，旧位置不会因后续 Token 到来而重新读取未来信息。对固定参数、模板和位置编码约定，已经计算的历史 Key 与 Value 因此可复用。其注意力计算基础见[Self-Attention](../mechanism/self-attention.md)。[2](https://arxiv.org/abs/1706.03762)

对第 $\ell$ 层的新位置 $t$，忽略多头拆分及位置变换后的布局差异，只需计算新增位置的投影：

$$
k_t^{(\ell)}=W_K^{(\ell)}h_t^{(\ell)},\quad
v_t^{(\ell)}=W_V^{(\ell)}h_t^{(\ell)}
$$

将它们追加到缓存后，当前 Query 读取包含新增位置的累计状态 $K_{\le t}^{(\ell)},V_{\le t}^{(\ell)}$：

$$
o_t^{(\ell)}
=\operatorname{softmax}
\left(\frac{q_t^{(\ell)}(K_{\le t}^{(\ell)})^{\mathsf T}}{\sqrt{d_k}}\right)
V_{\le t}^{(\ell)}
$$

分号表示按位置追加。实际模型可能先对 $q_t,k_t$ 应用位置变换，并使用多头映射或注意力 mask；这里强调的不变量是**只新增当前位置的 KV，当前 Query 仍读取所允许的历史位置**。如果改变已缓存前缀、模型权重、位置约定或可见性规则，原缓存不能无条件复用。

不缓存时，每次为逐渐增长的前缀重新前向计算，会重复生成历史位置的投影与注意力状态；缓存后 Decode 只计算新位置的投影，同时仍需让新 Query 与历史 KV 交互。因此缓存避免的是**历史状态的重复计算**，不是把历史读取成本降为常数。在标准密集因果注意力下，单头新位置的注意力连接数仍随当前长度 $t$ 增长。KV 的存储空间也随层数、序列长度、KV 头数和单头维度增长；容量公式、分页、共享与回收见[KV Cache 管理](../infra/inference/kv-cache-management.md)。

双向 Encoder 的旧位置通常会受新输入影响，不能直接按上述因果递推复用 KV；训练时一次处理完整序列，也不等于逐 Token Decode。KV Cache 是推理中间状态，不是持久记忆，不替代对话历史的保存。[3](https://huggingface.co/docs/transformers/cache_explanation)

---

## 上下文容量与前缀复用 {#capacity}

上下文窗口约束一次生成能处理的输入与生成量。设模型或服务允许的窗口为 $W$、本次输入为 $T_{\mathrm{in}}$、生成总量为 $T_{\mathrm{gen}}$，则在该口径下必须满足：

$$
T_{\mathrm{in}}+T_{\mathrm{gen}}\le W
$$

输入可包含角色边界、工具定义、历史、工具观察及多模态内容；生成可能包括不可见的推理 Token。实际还可能有单独的输出上限或服务预留，不能仅凭剩余窗口推断可生成的最大长度。输入过长时，是拒绝、截断还是由服务端压缩历史，取决于服务策略；静默移除关键资料会改变生成条件。

单次 Decode 的 KV Cache 与跨请求的前缀缓存相关但不相同：前者用于当前生成继续前进，后者在条件满足时让后续请求复用相同前缀已计算的状态。前缀复用通常要求相同的模型配置、输入 Token 及可复用的缓存状态；语义近似不等于缓存命中。历史中间部分一旦改变，其后的状态通常需要重算。还有一种结果缓存直接复用完整回答，命中时可能不再调用模型，不能与前两者混为一谈。

缓存节省计算，但不会改变窗口内**实际可见的输入内容**，也不会扩大 $W$ 或让模型跨对话自然记住事实。应用对历史进行选择、压缩和恢复时，必须同时考虑信息保真与缓存失效，详见[上下文压缩](./context-compression.md)。前缀缓存的服务规则以实际提供方为准。[4](https://developers.openai.com/api/docs/guides/prompt-caching)

---

## 用量与延迟

一次请求通常分别计量输入与生成用量；缓存输入、非可见推理生成可能作为两类总量中的细项报告。例如输入 1,000 Token 中有 600 命中前缀缓存，输出 200 Token 中有 80 是推理 Token，则总量仍是 $1{,}000+200=1{,}200$，不再额外加上 600 和 80。字段名称与计费类别由服务提供方定义，不能把示意数字写成跨厂商规则。

令 $T_{\mathrm{pre}}$ 表示排队、传输与预处理时间，$T_{\mathrm{prefill}}$ 表示前缀计算，$T_{\mathrm{decode},t}$ 表示后续第 $t$ 步的时间。响应完成时间可按组成近似理解为：

$$
T_{\mathrm{total}}\approx T_{\mathrm{pre}}+T_{\mathrm{prefill}}
+\sum_t T_{\mathrm{decode},t}+T_{\mathrm{post}}
$$

输入长度与前缀复用主要关联 Prefill 工作；输出长度与历史 KV 的读取关联 Decode 工作。流式交付可能缩短用户看到首段内容的时间，不必然减少总生成量或总耗时。实际瓶颈还受模型架构、输入／输出长度、批量与并发、精度、硬件和调度实现共同影响，不能由该式直接推断某个服务的性能。

---

## 参考资料

- [1] Holtzman, A. et al. (2020). [*The Curious Case of Neural Text Degeneration*](https://arxiv.org/abs/1904.09751)：nucleus sampling 的原始论文。
- [2] Vaswani, A. et al. (2017). [*Attention Is All You Need*](https://arxiv.org/abs/1706.03762)：缩放点积注意力。
- [3] Hugging Face. [Caching](https://huggingface.co/docs/transformers/cache_explanation)：自回归推理中缓存的输入与更新语义。
- [4] OpenAI. [Prompt caching](https://developers.openai.com/api/docs/guides/prompt-caching)：跨请求前缀复用的服务实例。
- OpenAI. [Reasoning models](https://developers.openai.com/api/docs/guides/reasoning)：推理 Token 与输出用量的接口实例。

在线服务文档核验于 2026-09-26；概率、注意力与缓存递推为指定模型假设下的机制，不是所有服务的统一实现。
