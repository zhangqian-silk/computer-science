# 检索评估：分开验证召回、排序与系统成本

检索评价有两个基础选择：什么算相关，以及什么算一次成功。找到任意一个答案证据、找全多份资料、把最相关结果排在首位，是不同任务；Recall、Hit、RR 与 NDCG 因而不能互换。系统多阶段执行还需要记录文档在哪一层丢失。

<RetrievalMetricsExplorer />

::: info 符号与约定
沿用[数学与符号约定](../foundations/math-notation.md)。$q$ 是单个查询，$\mathcal{Q}$ 是评测查询集合，$\mathcal{R}_q$ 是查询 $q$ 的相关文档集合，$\operatorname{TopK}(q)$ 是系统返回的前 $K$ 个结果；$\operatorname{rank}_q$ 是首个相关结果的排名，$\operatorname{rel}_i$ 是第 $i$ 位结果的相关等级，$\operatorname{DCG}$ 与 $\operatorname{IDCG}$ 分别是实际和理想折损累积增益。
:::

---

## 相关性集合

同一个文档被切成多个高度重叠 chunk 后，若把每个 chunk 都当独立相关项，可能夸大召回数量。应先约定评价单元及去重规则：要求文档覆盖就按文档聚合，要求证据片段定位则按片段标注。候选库更新后，相关性分母也可能变化。

对查询 $q$，设标注相关文档集合为 $\mathcal{R}_q$，系统前 $K$ 个结果为 $\operatorname{TopK}(q)$。评测首先依赖「什么算相关」：

- 二元相关还是分级相关；
- 一个查询有一个答案还是多个可接受证据；
- 文档、段落或 chunk 哪一级作为判断单元；
- 未标注文档是负例还是未知。

不完整标注会把新发现的相关文档误判为错误。候选池构造、标注者一致性和时间范围必须随指标一起报告。

---

## Recall@K 与 Precision@K

$$
\operatorname{Recall@K}(q)
=
\frac{|\mathcal{R}_q\cap\operatorname{TopK}(q)|}{|\mathcal{R}_q|}
$$

Recall@K 适合评估 ANN、BM25 或双塔召回。若相关文档没有进入候选池，重排器和生成器无法恢复它。

例如 $\mathcal{R}_q=\{d_2,d_5\}$，系统前三项为 $(d_1,d_2,d_3)$：

$$
\operatorname{Recall@3}(q)
=
\frac{|\{d_2,d_5\}\cap\{d_1,d_2,d_3\}|}{2}
=\frac{1}{2}
$$

这个查询只找回两个相关文档中的一个。若任务只要求任意一个可回答证据，还可以同时报告 Hit@K；它与需要找全证据的 Recall@K 不是同一目标。

本页 Precision@K 固定以 K 为分母；返回不足 K 时，其余槽按非相关计。若另用实际返回数作分母，需单独标注口径：

$$
\operatorname{Precision@K}(q)
=
\frac{|\mathcal{R}_q\cap\operatorname{TopK}(q)|}{K}
$$

上例的 $\operatorname{Precision@3}=1/3$。Recall 关注相关内容找回多少，Precision 关注返回结果中有多少相关。增大 $K$ 往往提高 Recall，却可能降低 Precision，并增加重排或生成阶段需要处理的噪声。

同时应报告 $K$、候选库规模和过滤规则。增大 $K$ 通常提高召回，却把更多延迟和噪声交给下游。

---

## MRR：第一个相关结果有多靠前

交互实验只有一个查询，所以显示的是 RR；跨查询平均后才叫 MRR。若前 K 项无相关结果，截断 RR@K 记为 0。单查询排名变好并不能说明整个查询分布改善，应在冻结查询集上比较差值和失败切片。

若第一个相关结果排名为 $\operatorname{rank}_q$：

$$
\operatorname{RR}(q)=\frac{1}{\operatorname{rank}_q}
$$

$$
\operatorname{MRR}
=
\frac{1}{|\mathcal{Q}|}\sum_{q\in\mathcal{Q}}\operatorname{RR}(q)
$$

MRR 强调第一个正确结果，适合单答案问答或导航检索。它几乎忽略后续相关结果，不适合完整衡量多证据排序。

继续使用上例，第一个相关结果 $d_2$ 位于第 2 名，因此 $\operatorname{RR}(q)=1/2$。即使第 3 名之后还有更多相关文档，这个查询的 RR 也不会改变。

---

## NDCG：分级相关结果是否排对

IDCG 应从该查询完整已标注候选的相关等级中取理想前 K 排序，不是仅把已返回的 K 项排序。否则一个漏掉全部高相关结果的系统也可能获得虚假的满分。下面数值例子假设展示的相关等级已涵盖理想前三项需要的相关内容。

对第 $i$ 位的相关等级 $\operatorname{rel}_i$：

$$
\operatorname{DCG@K}
=
\sum_{i=1}^{K}
\frac{2^{\operatorname{rel}_i}-1}{\log_2(i+1)}
$$

$$
\operatorname{NDCG@K}
=
\frac{\operatorname{DCG@K}}{\operatorname{IDCG@K}}
$$

NDCG 同时奖励高相关结果和靠前位置。相关等级的定义会改变指标含义，必须提供标注准则。

若前三名的相关等级为 $(2,0,1)$：

$$
\operatorname{DCG@3}
=
\frac{2^2-1}{\log_2 2}
+0
+\frac{2^1-1}{\log_2 4}
=3.5
$$

理想顺序为 $(2,1,0)$：

$$
\operatorname{IDCG@3}
=3+\frac{1}{\log_2 3}
\approx3.631
$$

因此 $\operatorname{NDCG@3}\approx0.964$。它小于 1，是因为等级 1 的结果被排到了第 3 位。

没有任何已标注相关文档的查询，需要在协议中明确排除、记零还是单独作为「无答案查询」评估。不同处理会显著改变平均指标。

---

## 多阶段归因

可做 oracle 对照：将人工确认的正确文档直接送入重排或生成器，看后续是否仍失败。若仍失败，召回不是唯一问题；若只有 oracle 文档能修复，才更有理由优先改检索。这样的对照隔离接口，但不代表线上拥有 oracle 能力。

| 阶段 | 应观察 | 常见失败 |
| --- | --- | --- |
| 召回 | Recall@K、ANN recall、候选数 | 相关文档根本没进入 |
| 融合 | 各通道覆盖、重复率 | 稀疏/稠密结果相互淹没 |
| 重排 | MRR、NDCG、pairwise accuracy | 候选相关但顺序错误 |
| 下游生成 | 答案正确与引用忠实性 | 有证据但模型未使用 |

RAG 最终回答改善并不能证明 embedding 更好；可能是重排器、提示或生成器变化。应固定其他层做消融，并记录每个查询的候选流转。

---

## 质量与成本一起报告

离线质量之外，真实系统至少需要：

- 编码延迟、查询 P50/P95/P99；
- 索引内存、磁盘与构建时间；
- 增量写入和删除可见时间；
- 过滤后的候选不足率；
- 不同语言、领域、查询长度与新鲜度分层。

ANN 参数通常形成 Recall–Latency 曲线，不应只选单个最好看的点。系统实现见[向量检索](../representation/vector-retrieval.md)，生成结果见[生成评估](./generation-evaluation.md)。

平均指标之外，应报告 query 级分布和不确定性。对同一查询集比较两个系统时，可对查询做配对 bootstrap，观察 Recall@K、MRR 或 NDCG 差值的置信区间。大量来自同一模板或同一用户的近重复查询不应被当作完全独立证据。

离线集合还要覆盖无结果查询、精确编号、罕见实体、否定条件、多语言、时间敏感和权限过滤等切片。平均分提高但某个关键切片显著下降，不能直接得出系统整体更好的结论。

---

## 参考文献

- Manning, C. D., Raghavan, P., and Schütze, H. (2008). [*Introduction to Information Retrieval*](https://nlp.stanford.edu/IR-book/).
- Järvelin, K., and Kekäläinen, J. (2002). *Cumulated Gain-based Evaluation of IR Techniques*.
