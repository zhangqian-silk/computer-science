# Memory 记忆系统

Memory 保存独立于单次调用、绑定特定主体、并随交互沉淀或失效的信息。模型调用本身无状态，memory 的作用不是扩展模型权重，而是让后续调用能够复用这个主体的偏好、事实、经历与约定。它的正确性判据是「这条信息现在是否仍然成立」。

Memory 与相邻对象的边界只取结论：context 是本轮实际发送的输入序列，memory 是 context 之外按主体隔离的持久状态，knowledge 是不绑定主体、按权限共享的资料，tool observation 是本轮从业务系统实时读取的观察。四者在一次调用中汇合，但维护方式不同。Memory 一侧需要独立讨论作用域、类型、数据模型、写入、召回、失效与删除。

会话历史不是 memory。原始消息只有经过抽取、确认或显式写入后，才成为独立的记忆条目；订单、余额、库存等实时 observation 默认不写入 memory。

记忆按生命周期分三层：工作记忆、会话记忆和长期记忆。工作记忆就是本轮调用的 context，由本轮信号与会话、长期两层记忆的投影拼接而成；会话记忆是历史的工作记忆，其中有长期价值的部分再固化为长期记忆。

<div class="mem-svg-scroll mem-svg-scroll--wide">
<svg class="mem-svg" viewBox="0 0 1000 524" role="img" aria-label="工作记忆是一次 context，拼接会话与长期记忆投影；会话记忆是历史工作记忆，长期记忆由固化产生">
<defs>
<marker id="s-down" viewBox="0 0 10 10" refX="5" refY="8.5" markerWidth="6.5" markerHeight="6.5" orient="auto"><path d="M0,0 L10,0 L5,10 z" fill="var(--cs-color-text-subtle)"/></marker>
<marker id="s-up" viewBox="0 0 10 10" refX="5" refY="1.5" markerWidth="7" markerHeight="7" orient="auto"><path d="M0,10 L10,10 L5,0 z" fill="var(--cs-color-brand)"/></marker>
</defs>
<rect x="0" y="20" width="1000" height="150" rx="12" fill="var(--cs-color-bg)" stroke="var(--cs-color-border)" stroke-width="1"/><rect x="0" y="20" width="4" height="150" rx="2" fill="var(--cs-color-info)"/><text x="28" y="50" font-family="var(--vp-font-family-base)" font-size="14.5" font-weight="700" fill="var(--cs-color-text)">工作记忆</text><text x="28" y="70" font-family="var(--vp-font-family-mono)" font-size="9.5" letter-spacing="0.5" fill="var(--cs-color-text-subtle)">SINGLE TURN</text><text x="28" y="88" font-family="var(--vp-font-family-base)" font-size="10" fill="var(--cs-color-text-subtle)">一次 context · 每轮装配 · 用完即弃</text>
<rect x="170" y="54" width="148" height="56" rx="8" fill="var(--cs-color-bg)" stroke="var(--cs-color-border-strong)" stroke-width="1"/><text x="244" y="75" text-anchor="middle" font-family="var(--vp-font-family-base)" font-size="12" font-weight="600" fill="var(--cs-color-text)">系统指令/工具</text><text x="244" y="95" text-anchor="middle" font-size="9.5" fill="var(--cs-color-text-subtle)">每轮装配拼入</text>
<rect x="329" y="54" width="148" height="56" rx="8" fill="var(--cs-color-bg)" stroke="var(--cs-color-border-strong)" stroke-width="1"/><text x="403" y="75" text-anchor="middle" font-family="var(--vp-font-family-base)" font-size="12" font-weight="600" fill="var(--cs-color-text)">本轮用户输入</text><text x="403" y="95" text-anchor="middle" font-size="9.5" fill="var(--cs-color-text-subtle)">本轮新信号</text>
<rect x="488" y="54" width="148" height="56" rx="8" fill="var(--cs-color-bg)" stroke="var(--cs-color-border-strong)" stroke-width="1"/><text x="562" y="75" text-anchor="middle" font-family="var(--vp-font-family-base)" font-size="12" font-weight="600" fill="var(--cs-color-text)">Scratchpad</text><text x="562" y="95" text-anchor="middle" font-size="9.5" fill="var(--cs-color-text-subtle)">推理/工具返回</text>
<rect x="647" y="54" width="148" height="56" rx="8" fill="var(--cs-color-brand-soft)" stroke="var(--cs-color-brand)" stroke-width="1.2"/><text x="721" y="75" text-anchor="middle" font-family="var(--vp-font-family-base)" font-size="12" font-weight="600" fill="var(--cs-color-text)">会话记忆投影</text><text x="721" y="95" text-anchor="middle" font-size="9.5" fill="var(--cs-color-text-subtle)">历史/摘要/笔记</text>
<rect x="806" y="54" width="148" height="56" rx="8" fill="var(--cs-color-info-soft)" stroke="var(--cs-color-info)" stroke-width="1.2"/><text x="880" y="75" text-anchor="middle" font-family="var(--vp-font-family-base)" font-size="12" font-weight="600" fill="var(--cs-color-text)">长期记忆投影</text><text x="880" y="95" text-anchor="middle" font-size="9.5" fill="var(--cs-color-text-subtle)">按需召回</text>
<line x1="420" y1="176" x2="420" y2="200" stroke="var(--cs-color-text-subtle)" stroke-width="1.4" marker-end="url(#s-down)"/>
<text x="432" y="194" font-family="var(--vp-font-family-base)" font-size="10.5" fill="var(--cs-color-text-muted)">回合结束 · 沉淀为会话历史</text>
<rect x="0" y="206" width="1000" height="120" rx="12" fill="var(--cs-color-bg)" stroke="var(--cs-color-border)" stroke-width="1"/><rect x="0" y="206" width="4" height="120" rx="2" fill="var(--cs-color-text-subtle)"/><text x="28" y="236" font-family="var(--vp-font-family-base)" font-size="14.5" font-weight="700" fill="var(--cs-color-text)">会话记忆</text><text x="28" y="256" font-family="var(--vp-font-family-mono)" font-size="9.5" letter-spacing="0.5" fill="var(--cs-color-text-subtle)">ONE SESSION</text><text x="28" y="274" font-family="var(--vp-font-family-base)" font-size="10" fill="var(--cs-color-text-subtle)">历史的工作记忆</text>
<rect x="170" y="240" width="190" height="56" rx="8" fill="var(--cs-color-bg)" stroke="var(--cs-color-border-strong)" stroke-width="1"/><text x="265" y="261" text-anchor="middle" font-family="var(--vp-font-family-base)" font-size="12" font-weight="600" fill="var(--cs-color-text)">完整消息历史</text><text x="265" y="281" text-anchor="middle" font-size="9.5" fill="var(--cs-color-text-subtle)">append-only 日志</text>
<rect x="370" y="240" width="190" height="56" rx="8" fill="var(--cs-color-bg)" stroke="var(--cs-color-border-strong)" stroke-width="1"/><text x="465" y="261" text-anchor="middle" font-family="var(--vp-font-family-base)" font-size="12" font-weight="600" fill="var(--cs-color-text)">滑窗 + 摘要</text><text x="465" y="281" text-anchor="middle" font-size="9.5" fill="var(--cs-color-text-subtle)">compaction</text>
<rect x="570" y="240" width="190" height="56" rx="8" fill="var(--cs-color-bg)" stroke="var(--cs-color-border-strong)" stroke-width="1"/><text x="665" y="261" text-anchor="middle" font-family="var(--vp-font-family-base)" font-size="12" font-weight="600" fill="var(--cs-color-text)">会话状态</text><text x="665" y="281" text-anchor="middle" font-size="9.5" fill="var(--cs-color-text-subtle)">checkpoint</text>
<rect x="770" y="240" width="190" height="56" rx="8" fill="var(--cs-color-bg)" stroke="var(--cs-color-border-strong)" stroke-width="1"/><text x="865" y="261" text-anchor="middle" font-family="var(--vp-font-family-base)" font-size="12" font-weight="600" fill="var(--cs-color-text)">结构化笔记</text><text x="865" y="281" text-anchor="middle" font-size="9.5" fill="var(--cs-color-text-subtle)">NOTES / todo</text>
<line x1="420" y1="332" x2="420" y2="356" stroke="var(--cs-color-text-subtle)" stroke-width="1.4" marker-end="url(#s-down)"/>
<text x="432" y="350" font-family="var(--vp-font-family-base)" font-size="10.5" fill="var(--cs-color-text-muted)">固化 · 抽取为跨会话记忆</text>
<rect x="0" y="362" width="1000" height="142" rx="12" fill="var(--cs-color-bg)" stroke="var(--cs-color-border)" stroke-width="1"/><rect x="0" y="362" width="4" height="142" rx="2" fill="var(--cs-color-brand)"/><text x="28" y="392" font-family="var(--vp-font-family-base)" font-size="14.5" font-weight="700" fill="var(--cs-color-text)">长期记忆</text><text x="28" y="412" font-family="var(--vp-font-family-mono)" font-size="9.5" letter-spacing="0.5" fill="var(--cs-color-text-subtle)">CROSS-SESSION</text><text x="28" y="430" font-family="var(--vp-font-family-base)" font-size="10" fill="var(--cs-color-text-subtle)">固化沉淀 · 独立于单次会话</text>
<rect x="170" y="392" width="124" height="38" rx="8" fill="var(--cs-color-bg)" stroke="var(--cs-color-border-strong)" stroke-width="1"/><text x="232" y="415" text-anchor="middle" font-family="var(--vp-font-family-base)" font-size="11.5" font-weight="600" fill="var(--cs-color-text)">事实/语义</text>
<rect x="305" y="392" width="124" height="38" rx="8" fill="var(--cs-color-bg)" stroke="var(--cs-color-border-strong)" stroke-width="1"/><text x="367" y="415" text-anchor="middle" font-family="var(--vp-font-family-base)" font-size="11.5" font-weight="600" fill="var(--cs-color-text)">偏好</text>
<rect x="440" y="392" width="124" height="38" rx="8" fill="var(--cs-color-bg)" stroke="var(--cs-color-border-strong)" stroke-width="1"/><text x="502" y="415" text-anchor="middle" font-family="var(--vp-font-family-base)" font-size="11.5" font-weight="600" fill="var(--cs-color-text)">经历/情景</text>
<rect x="575" y="392" width="124" height="38" rx="8" fill="var(--cs-color-bg)" stroke="var(--cs-color-border-strong)" stroke-width="1"/><text x="637" y="415" text-anchor="middle" font-family="var(--vp-font-family-base)" font-size="11.5" font-weight="600" fill="var(--cs-color-text)">程序性</text>
<rect x="710" y="392" width="124" height="38" rx="8" fill="var(--cs-color-bg)" stroke="var(--cs-color-border-strong)" stroke-width="1"/><text x="772" y="415" text-anchor="middle" font-family="var(--vp-font-family-base)" font-size="11.5" font-weight="600" fill="var(--cs-color-text)">实体/图</text>
<rect x="845" y="392" width="124" height="38" rx="8" fill="var(--cs-color-bg)" stroke="var(--cs-color-border-strong)" stroke-width="1"/><text x="907" y="415" text-anchor="middle" font-family="var(--vp-font-family-base)" font-size="11.5" font-weight="600" fill="var(--cs-color-text)">前瞻/待办</text>
<rect x="170" y="440" width="396" height="34" rx="8" fill="var(--cs-color-bg)" stroke="var(--cs-color-border-strong)" stroke-width="1" stroke-dasharray="3 3"/><text x="368" y="461" text-anchor="middle" font-family="var(--vp-font-family-base)" font-size="11.5" font-weight="600" fill="var(--cs-color-text)">集体记忆（多 Agent 共享）</text>
<rect x="577" y="440" width="392" height="34" rx="8" fill="var(--cs-color-bg)" stroke="var(--cs-color-border-strong)" stroke-width="1" stroke-dasharray="3 3"/><text x="773" y="461" text-anchor="middle" font-family="var(--vp-font-family-base)" font-size="11.5" font-weight="600" fill="var(--cs-color-text)">感知记忆（多模态片段）</text>
<line x1="942" y1="200" x2="942" y2="178" stroke="var(--cs-color-brand)" stroke-width="1.6" marker-end="url(#s-up)"/>
<line x1="942" y1="356" x2="942" y2="334" stroke="var(--cs-color-brand)" stroke-width="1.6" marker-end="url(#s-up)"/>
<text x="934" y="172" text-anchor="end" font-family="var(--vp-font-family-base)" font-size="10" font-weight="600" fill="var(--cs-color-brand)">检索注入</text>
</svg>
</div>


这三层在一次调用中协同：会话层和长期层是本体所在，工作层只持有它们本轮的投影；可重建的投影用完即弃，唯一副本在工作层的本轮推理与未落盘输入则在压缩和固化时需要优先保护。

---

## 作用域与主体

Memory 的第一条约束是主体归属。一条记忆必须能回答「它属于谁、在什么范围内生效」。作用域决定写入时的归属键，也决定召回时的强制过滤条件。

| 作用域 | 主体 | 典型内容 | 生效范围 |
| --- | --- | --- | --- |
| `user` | 终端用户 | 语言、沟通偏好、长期目标、稳定事实 | 该用户与其授权的 Agent |
| `agent` | 某个 Agent | 该 Agent 的工作方式、自沉淀的操作经验 | 仅该 Agent，跨用户时须谨慎 |
| `thread` | 单次会话 / 任务 | 本轮任务约束、待续状态 | 仅当前线程，结束即应清理 |
| `team` | 团队 / 项目 | 编码规范、发布流程、共享处置经验 | 团队成员与其 Agent |
| `org` | 组织 | 组织级制度、合规要求 | 全组织，写入门槛最高 |

`thread` 作用域处理任务连续性，生命周期应与会话对齐；真正跨会话的偏好与事实才进入 `user` 或 `team`。`agent` 作用域不能存放本应属于用户的事实，否则同一用户更换 Agent 后状态会丢失或错配。

主体过滤必须发生在检索层：查询携带作用域键，索引或数据库在候选生成阶段就排除越权记录，而不是取回后再依赖提示词忽略。漏加作用域条件是记忆系统的红线事故，其后果不会随下一轮调用消失，而会在后续会话中持续泄露。

<div class="mem-callout mem-callout--danger">
<span class="mem-callout__tag">隔离红线</span>
<p><strong>跨主体记忆泄露是安全事故，不是排序误差。</strong>须用自动化测试验证漏写作用域条件时查询返回空集，而不能只靠提示词约束。</p>
</div>

---

---

## 记忆类型

记忆按内容的更新语义和召回方式分类，而不是按存储介质分类。不同类型的覆盖、追加、失效规则不同，混在同一种记录上用统一的 upsert 处理，是记忆库退化的主要来源。

<div class="mem-svg-scroll">
<svg class="mem-svg" viewBox="0 0 760 442" role="img" aria-label="长期记忆分陈述性、非陈述性、工程扩展三组，并沿可变性、时效性、作用域取值">
<defs>
<marker id="mt2" viewBox="0 0 10 10" refX="7" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" fill="var(--cs-color-border-strong)"/></marker>
</defs>
<rect x="300" y="14" width="160" height="48" rx="8" fill="var(--cs-color-text)"/>
<text x="380" y="36" text-anchor="middle" font-family="var(--vp-font-family-base)" font-size="14" font-weight="700" fill="var(--cs-color-bg)">Agent 长期记忆</text>
<text x="380" y="53" text-anchor="middle" font-family="var(--vp-font-family-mono)" font-size="8.5" letter-spacing="1.2" fill="var(--cs-color-bg-soft)">LONG-TERM MEMORY</text>
<path d="M380 62 L380 80 M150 80 L610 80 M150 80 L150 96 M380 80 L380 96 M610 80 L610 96" fill="none" stroke="var(--cs-color-border-strong)" stroke-width="1.3" marker-end="url(#mt2)"/>
<rect x="40" y="96" width="220" height="34" rx="6" fill="var(--cs-color-brand-soft)" stroke="var(--cs-color-brand)"/>
<text x="150" y="118" text-anchor="middle" font-family="var(--vp-font-family-base)" font-size="12.5" font-weight="700" fill="var(--cs-color-text)">陈述性 · 知道什么</text>
<rect x="290" y="96" width="180" height="34" rx="6" fill="var(--cs-color-warning-soft)" stroke="var(--cs-color-warning)"/>
<text x="380" y="118" text-anchor="middle" font-family="var(--vp-font-family-base)" font-size="12.5" font-weight="700" fill="var(--cs-color-text)">非陈述性 · 知道怎么做</text>
<rect x="500" y="96" width="220" height="34" rx="6" fill="var(--cs-color-info-soft)" stroke="var(--cs-color-info)"/>
<text x="610" y="118" text-anchor="middle" font-family="var(--vp-font-family-base)" font-size="12.5" font-weight="700" fill="var(--cs-color-text)">工程扩展类</text>
<g font-family="var(--vp-font-family-base)">
<rect x="40" y="142" width="68" height="64" rx="6" fill="var(--cs-color-bg-elevated)" stroke="var(--cs-color-brand)" stroke-width="1.4"/>
<text x="74" y="164" text-anchor="middle" font-size="11.5" font-weight="700" fill="var(--cs-color-text)">事实</text>
<text x="74" y="180" text-anchor="middle" font-family="var(--vp-font-family-mono)" font-size="8.5" fill="var(--cs-color-text-subtle)">semantic</text>
<text x="74" y="197" text-anchor="middle" font-size="9.5" fill="var(--cs-color-text-muted)">向量 + KV</text>
<rect x="116" y="142" width="68" height="64" rx="6" fill="var(--cs-color-bg-elevated)" stroke="var(--cs-color-brand)" stroke-width="1.4"/>
<text x="150" y="164" text-anchor="middle" font-size="11.5" font-weight="700" fill="var(--cs-color-text)">偏好</text>
<text x="150" y="180" text-anchor="middle" font-family="var(--vp-font-family-mono)" font-size="8.5" fill="var(--cs-color-text-subtle)">preference</text>
<text x="150" y="197" text-anchor="middle" font-size="9.5" fill="var(--cs-color-text-muted)">结构化直取</text>
<rect x="192" y="142" width="68" height="64" rx="6" fill="var(--cs-color-bg-elevated)" stroke="var(--cs-color-brand)" stroke-width="1.4"/>
<text x="226" y="164" text-anchor="middle" font-size="11.5" font-weight="700" fill="var(--cs-color-text)">经历</text>
<text x="226" y="180" text-anchor="middle" font-family="var(--vp-font-family-mono)" font-size="8.5" fill="var(--cs-color-text-subtle)">episodic</text>
<text x="226" y="197" text-anchor="middle" font-size="9.5" fill="var(--cs-color-text-muted)">时序追加</text>
<rect x="306" y="142" width="74" height="64" rx="6" fill="var(--cs-color-bg-elevated)" stroke="var(--cs-color-warning)" stroke-width="1.4"/>
<text x="343" y="164" text-anchor="middle" font-size="11.5" font-weight="700" fill="var(--cs-color-text)">程序性</text>
<text x="343" y="180" text-anchor="middle" font-family="var(--vp-font-family-mono)" font-size="8.5" fill="var(--cs-color-text-subtle)">procedural</text>
<text x="343" y="197" text-anchor="middle" font-size="9.5" fill="var(--cs-color-text-muted)">规则触发</text>
<rect x="388" y="142" width="74" height="64" rx="6" fill="var(--cs-color-bg-elevated)" stroke="var(--cs-color-warning)" stroke-width="1.4"/>
<text x="425" y="164" text-anchor="middle" font-size="11.5" font-weight="700" fill="var(--cs-color-text)">感知</text>
<text x="425" y="180" text-anchor="middle" font-family="var(--vp-font-family-mono)" font-size="8.5" fill="var(--cs-color-text-subtle)">sensory</text>
<text x="425" y="197" text-anchor="middle" font-size="9.5" fill="var(--cs-color-text-muted)">多模态</text>
<rect x="500" y="142" width="68" height="64" rx="6" fill="var(--cs-color-bg-elevated)" stroke="var(--cs-color-info)" stroke-width="1.4"/>
<text x="534" y="164" text-anchor="middle" font-size="11.5" font-weight="700" fill="var(--cs-color-text)">实体/图</text>
<text x="534" y="180" text-anchor="middle" font-family="var(--vp-font-family-mono)" font-size="8.5" fill="var(--cs-color-text-subtle)">entity</text>
<text x="534" y="197" text-anchor="middle" font-size="9.5" fill="var(--cs-color-text-muted)">多跳遍历</text>
<rect x="576" y="142" width="68" height="64" rx="6" fill="var(--cs-color-bg-elevated)" stroke="var(--cs-color-info)" stroke-width="1.4"/>
<text x="610" y="164" text-anchor="middle" font-size="11.5" font-weight="700" fill="var(--cs-color-text)">前瞻</text>
<text x="610" y="180" text-anchor="middle" font-family="var(--vp-font-family-mono)" font-size="8.5" fill="var(--cs-color-text-subtle)">prospective</text>
<text x="610" y="197" text-anchor="middle" font-size="9.5" fill="var(--cs-color-text-muted)">条件触发</text>
<rect x="652" y="142" width="68" height="64" rx="6" fill="var(--cs-color-bg-elevated)" stroke="var(--cs-color-info)" stroke-width="1.4"/>
<text x="686" y="164" text-anchor="middle" font-size="11.5" font-weight="700" fill="var(--cs-color-text)">集体</text>
<text x="686" y="180" text-anchor="middle" font-family="var(--vp-font-family-mono)" font-size="8.5" fill="var(--cs-color-text-subtle)">collective</text>
<text x="686" y="197" text-anchor="middle" font-size="9.5" fill="var(--cs-color-text-muted)">共享 + ACL</text>
</g>
<line x1="24" y1="232" x2="736" y2="232" stroke="var(--cs-color-border)"/>
<text x="24" y="252" font-family="var(--vp-font-family-mono)" font-size="9.5" letter-spacing="1" fill="var(--cs-color-text-subtle)">ORTHOGONAL AXES · 三条正交维度</text>
<g font-family="var(--vp-font-family-base)">
<rect x="24" y="264" width="226" height="104" rx="8" fill="var(--cs-color-bg-soft)" stroke="var(--cs-color-border)"/>
<text x="42" y="289" font-size="12.5" font-weight="700" fill="var(--cs-color-text)">① 可变性 Mutability</text>
<text x="42" y="312" font-size="10.5" fill="var(--cs-color-text-muted)">可覆盖：偏好、当前状态事实</text>
<text x="42" y="331" font-size="10.5" fill="var(--cs-color-text-muted)">仅追加：经历、审计日志</text>
<text x="42" y="355" font-size="10.5" fill="var(--cs-color-warning)">→ 决定 UPDATE 还是 append</text>
<rect x="267" y="264" width="226" height="104" rx="8" fill="var(--cs-color-bg-soft)" stroke="var(--cs-color-border)"/>
<text x="285" y="289" font-size="12.5" font-weight="700" fill="var(--cs-color-text)">② 时效性 Volatility</text>
<text x="285" y="312" font-size="10.5" fill="var(--cs-color-text-muted)">永久：出生年月、母语</text>
<text x="285" y="331" font-size="10.5" fill="var(--cs-color-text-muted)">缓变：职位、城市　易变：情绪</text>
<text x="285" y="355" font-size="10.5" fill="var(--cs-color-warning)">→ 决定 TTL 与衰减半衰期</text>
<rect x="510" y="264" width="226" height="104" rx="8" fill="var(--cs-color-bg-soft)" stroke="var(--cs-color-border)"/>
<text x="528" y="289" font-size="12.5" font-weight="700" fill="var(--cs-color-text)">③ 作用域 Scope</text>
<text x="528" y="312" font-size="10.5" fill="var(--cs-color-text-muted)">用户 / 会话 / Agent 级</text>
<text x="528" y="331" font-size="10.5" fill="var(--cs-color-text-muted)">组织（团队共享）/ 全局级</text>
<text x="528" y="355" font-size="10.5" fill="var(--cs-color-warning)">→ 决定隔离键与 ACL</text>
</g>
<rect x="24" y="384" width="712" height="42" rx="8" fill="var(--cs-color-brand-soft)"/>
<text x="42" y="403" font-family="var(--vp-font-family-base)" font-size="11" font-weight="700" fill="var(--cs-color-text)">关键洞察：每条记忆都是「一个类型 × 三条维度取值」的组合。</text>
<text x="42" y="420" font-family="var(--vp-font-family-base)" font-size="10.5" fill="var(--cs-color-text-muted)">数据模型把这四维显式落成字段（type / mutability / ttl / scope），而不是在检索时临时推断。</text>
</svg>
</div>

八个工程类别归入陈述性、非陈述性与工程扩展三组；每条记忆都是「一个类型 × 可变性 / 时效性 / 作用域三维取值」的组合，数据模型需把这四维显式落成字段。

#### 陈述性记忆

陈述性记忆保存「知道什么」，内容可判断真假或表达主观倾向，以条目形式存储并按需召回。

### 事实（semantic fact）

关于主体、项目或环境的、可判断真假的原子断言，如「服务使用 PostgreSQL」「用户负责结算模块」。一条记录只表达一个断言。

- **示例**：「用户担任后端工程师」「项目支付走 PostgreSQL」「该服务部署在华东区」。
- **存储**：一条断言一条记录，落关系库结构化字段（主体、槽位、取值、有效期、来源），并建向量与全文索引；不把多个事实塞进同一段自由文本。
- **召回**：语义向量与字面全文并行，元数据按作用域和有效期过滤；人名、编号、型号等专名必须有 BM25 一类字面匹配兜底。
- **更新**：写入前语义判重；同一槽位变化时给旧记录关闭 `valid_to` 并追加新记录，保留历史，不就地覆盖。
- **易错点**：粒度过粗导致无法单条失效；把余额、订单状态等可从权威源实时查询的可变数据写成事实，必然陈旧；只存原话不存断言，会出现检索命中但仍需二次推理。

### 偏好（preference）

主体对回答或行动方式的稳定倾向。它主观、可变，没有客观真假，只有「当前是否成立」。

- **示例**：「回答用中文、代码注释用英文」「先给结论再展开」「涉及预算先确认」「不吃海鲜（过敏）」。
- **存储**：可枚举的偏好落结构化字段（如 `lang=zh`、`verbosity=concise`），不可枚举的落短文本；高命中偏好放常驻块。
- **召回**：数量少且几乎每轮相关，宜按主体直取或全量常驻，不依赖语义 Top-K；漏召回一条偏好的代价通常大于多带少量 token。
- **更新**：典型可覆盖型，新偏好覆盖同 key 旧值并留变更日志；带稳定原因的偏好不参与自动衰减，一次性指令不进入长期记忆。
- **易错点**：把「这次简短点」写成长期偏好，造成持续的过度个性化；冲突偏好无优先级时输出随机选择；不提供查看与撤销入口会让错误无法自愈。

### 事件经历（episodic event）

带时间、情境与结果的一次性记录。它回答「那次发生了什么」，不声称「一直如此」。

- **示例**：「方案 A 因成本被否决」「该工具超时三次后减小批量成功」「用户在上次工单中要求保留审计日志」。
- **存储**：append-only 时序日志加向量索引；`event_time`（事件发生时间）与 `ingest_time`（系统写入时间）必须分开，并记录参与者、结果与重要性。
- **召回**：时间范围过滤、语义相似与重要性加权三者结合，支持「上次」「上个月」类查询，是相关性 × 时效 × 重要性排序最适用的类型。
- **更新**：只追加、永不覆盖；治理靠反思归并（多条经历提炼为高层洞察，原始条目降权而非删除）与冷数据归档。
- **易错点**：只存写入时间就答不了以事件时间为准的问题；用 UPDATE 抹平历史；全量存每一轮对话导致噪声淹没信号，应只存「有结果、有决策、有异常」的回合。

#### 非陈述性记忆

非陈述性记忆保存「知道怎么做」以及如何感知原始材料，通常不按语义相似度召回，而靠触发、常驻或专用索引。

### 程序性约定（procedural）

在某主体或项目中「这件事应该怎么做」的规则。它以指令形式生效，而不是作为事实被引用。

- **示例**：「本仓库测试用 `pnpm test:unit`」「退款先查订单再查退款状态」「调用该工具前先确认参数 schema」。
- **存储**：按稳定性递增有三类载体——可编辑的规则片段 / 规则库、项目约定文件（如 AGENTS.md / NOTES.md）、模型权重（微调）；前两类可运行时修改，权重最难改。
- **召回**：按任务类型触发注入而非语义相似，给每条规则标 trigger（意图、工具名、文件路径），命中即注入；通用规则常驻，专用规则按需。
- **更新**：版本化加反馈驱动，来源是失败复盘——失败归因后生成或修订规则，并观察后续成功率；每条规则记录命中率与成功率，长期不命中或负相关的下线，必须支持回滚。
- **易错点**：规则只加不删导致互相矛盾、持续占用固定预算；把一次偶发失败固化成永久规则；规则写得过细，无法泛化到相邻场景。

### 感知记忆（sensory memory）

图片、音频、录屏、扫描件等原始多模态片段及其派生描述的存储。工程上借用「感知」一词指代原始模态数据层。

- **示例**：用户上传的票据照片、会议录音与转写、操作录屏关键帧、产品图片的视觉特征。
- **存储**：分两层——原始文件放对象存储只留 URI，文本化描述加多模态向量放检索层；绝不把二进制内容放进 context。
- **召回**：以文本描述的语义检索为主入口（便宜、可解释），必要时用多模态嵌入跨模态检索；命中后按需回取原件交给多模态模型。
- **更新**：原始件只追加不修改，描述与向量可随模型升级重新生成并 reindex；生命周期通常短于文本，存储成本高，需设 TTL 与冷归档。
- **易错点**：只存原图不存描述会使检索失效；图片音频常含人脸、声纹、证件等敏感信息，删除要求更严；多模态嵌入与存储成本显著高于文本，容易失控。

#### 工程扩展类

工程扩展类不是新的内容语义，而是事实或意图在特定索引、作用域或触发方式上的扩展。

### 实体关系（entity / graph）

以实体为中心、显式建模实体间关系的记忆，是事实在多跳查询上的另一种索引组织方式，而不是新的内容类型。

- **示例**：「（用户）—任职于→（A 公司）」「（A 公司）—收购→（B 公司，2025-06 生效）」「（小雨）—是→（用户的女儿）」。
- **存储**：图数据库中的节点与边，边上带来源与有效期属性，可记录系统层创建/失效时间与世界层生效/失效时间。
- **召回**：先用向量或全文定位入口实体，再做 1~2 跳邻域遍历；这是回答多跳关系问题的可靠路径，纯向量检索在多跳上系统性失败。
- **更新**：新关系与旧边冲突时让旧边失效而不删除，并记录由哪次事件导致，从而支持时间点查询（如「上一份工作是什么」）。
- **易错点**：实体消歧没做好会让同一实体拆成多个节点，这是图记忆的头号工程难点；每条消息都跑实体关系抽取，写入延迟与成本远高于纯向量；简单场景上图不划算。

### 待办意图（prospective intention）

关于未来要在某时间点或某条件满足时执行某事的记忆，对应「记得去做」，与「记得发生过」正交。

- **示例**：「下周一提醒提交周报」「PR 合并后继续跑集成测试」「下次涉及预算先询问」。
- **存储**：不放向量库，放调度表或任务队列，记录触发类型、触发值、意图内容、状态与过期时间；触发分时间型与条件型。
- **召回**：反向匹配——不是用户查它，而是系统每轮或每个时钟周期检查是否有条件被满足的待办；时间型定时扫描，条件型逐轮比对状态。
- **更新**：严格状态机 `pending → fired → done / cancelled / expired`，完成即落终态防止重复提醒，并定期清理过期意图。
- **易错点**：靠语义检索「碰巧想起」不可靠、漏触发率高；没有过期机制会让陈旧提醒在很久后突然出现；误触发率需要和召回率一起度量，过度提醒比不提醒更烦人。

### 集体记忆（collective memory）

多个 Agent、用户或会话共享读写的记忆池。它带来的主要是分布式系统问题，而不只是存储问题。

- **示例**：团队共享的代码约定、客服团队沉淀的疑难工单解法、主 Agent 写下供子 Agent 读取的任务计划、某工具当前不可用的全局状态。
- **存储**：共享存储加显式 ACL 与作用域标签；写入带作者标识与置信度，供下游判断可信度。
- **召回**：按作用域分层检索（私有 → 团队 → 全局），合并时私有优先级更高；ACL 必须在检索层强制过滤，不能靠提示词约束。
- **更新**：需要并发协议，简单可靠的做法是串行化写入后广播，更复杂的用乐观锁加版本号并在冲突时交裁决；多 Agent 并发下「总能看到最新状态」不再成立。
- **易错点**：一个 Agent 写入的错误结论被全体继承放大（错误传染），因此共享记忆写入门槛高于私有记忆，需验证或评审；无来源标注无法定位污染源；用户级数据误写进组织作用域即构成隐私事故。

微调改变模型权重，可承载稳定的领域能力与风格；个人化事实需要查看、撤销和按人删除，不适合写入权重，参数化记忆与条目化记忆应分开。

---

---

## 记忆条目模型

记忆库的主记录是唯一事实源，向量、全文和图索引都是它的召回投影。只写向量不写主记录，就无法做时间过滤、冲突消解和级联删除。

| 字段 | 含义 | 治理作用 |
| --- | --- | --- |
| `id` | 条目主键 | 删除、审计与索引对齐 |
| `type` | 事实 / 偏好 / 程序 / 事件 / 待办 | 决定写入与召回策略 |
| `scope` + `subject_id` | 作用域与主体键 | 检索层强制隔离 |
| `slot` | 原子槽位，如 `residence_city` | 判重与同槽位冲突检测 |
| `content` | 断言或规则内容 | 注入模型的正文 |
| `confidence` | 置信度 | 排序与是否以确定语气注入 |
| `source_ref` / `provenance` | 来源消息、工具结果或人工确认 | 可解释、可纠错、可按来源回滚 |
| `event_time` | 事件在现实中发生的时间 | 「上个月」「上次」类时间查询 |
| `ingest_time` | 系统写入该条目的时间 | 审计与「何时知道」 |
| `valid_from` / `valid_to` | 有效期区间 | 当前事实过滤与历史查询 |
| `status` | active / tentative / invalid / deleted | 试用期、失效与软删除 |

按需扩展 `importance`、命中次数、最后命中时间、`embedding_ref`、`supersedes` 与标签。`importance` 表示长期重要性，与表示新鲜度的时间信号是两个独立维度。

### 双时间轴

事实需要区分两个时间：`event_time` 表示事情在现实中何时发生、何时有效，`ingest_time` 表示系统何时知道。「现在住哪」过滤 <code>valid_to IS NULL</code>，「去年住哪」查有效期区间，「我们什么时候知道他搬家」查写入时间。一个 <code>updated_at</code> 无法同时表达这三类问题。

---

## 维护

维护覆盖一条信息从被识别、写入、更新到删除的全过程。门控决定哪些信息值得进入记忆库；把全部对话直接持久化，噪声会随交互线性累积，召回预算最终被一次性指令、工具流水和未确认推断占满。

### 准入与门控

<div class="mem-svg-scroll">
    <svg class="mem-svg" viewBox="0 0 920 566" role="img" aria-label="记忆写入链路决策流程图">
      <defs>
        <marker id="wr" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6.5" markerHeight="6.5" orient="auto-start-reverse">
          <path d="M0,0 L10,5 L0,10 z" fill="var(--cs-color-border-strong)"/>
        </marker>
        <marker id="wr-g" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
          <path d="M0,0 L10,5 L0,10 z" fill="var(--cs-color-text-subtle)"/>
        </marker>
      </defs>
      <rect x="330" y="8" width="260" height="38" rx="6" fill="var(--cs-color-brand)"/>
      <text x="460" y="32" text-anchor="middle" font-family="var(--vp-font-family-base)" font-size="12.5" font-weight="700" fill="var(--cs-color-on-brand)">本回合对话 + 工具轨迹</text>
      <path d="M460 46 L460 64" stroke="var(--cs-color-warning)" stroke-width="1.6" marker-end="url(#wr)"/>
      <!-- 门控 -->
      <rect x="300" y="66" width="320" height="42" rx="6" fill="var(--cs-color-brand-soft)" stroke="var(--cs-color-brand)" stroke-width="1.4"/>
      <text x="460" y="84" text-anchor="middle" font-family="var(--vp-font-family-base)" font-size="12.5" font-weight="700" fill="var(--cs-color-brand)">① LLM 门控抽取</text>
      <text x="460" y="100" text-anchor="middle" font-family="var(--vp-font-family-base)" font-size="10.5" fill="var(--cs-color-text-muted)">输出结构化候选列表，非摘要</text>
      <text x="636" y="90" font-family="var(--vp-font-family-base)" font-size="10.5" fill="var(--cs-color-danger)">大多数回合应输出空列表</text>
      <path d="M460 108 L460 126" stroke="var(--cs-color-warning)" stroke-width="1.6" marker-end="url(#wr)"/>
      <!-- 候选 -->
      <rect x="300" y="128" width="320" height="36" rx="6" fill="var(--cs-color-bg-elevated)" stroke="var(--cs-color-border-strong)"/>
      <text x="460" y="151" text-anchor="middle" font-family="var(--vp-font-family-base)" font-size="12" fill="var(--cs-color-text-muted)">候选记忆：type / content / time / confidence</text>
      <path d="M460 164 L460 182" stroke="var(--cs-color-warning)" stroke-width="1.6" marker-end="url(#wr)"/>
      <!-- 检索相近 -->
      <rect x="300" y="184" width="320" height="42" fill="var(--cs-color-bg-soft)" stroke="var(--cs-color-border-strong)"/>
      <text x="460" y="202" text-anchor="middle" font-family="var(--vp-font-family-base)" font-size="12.5" font-weight="700" fill="var(--cs-color-text)">② 检索语义相近的既有记忆</text>
      <text x="460" y="218" text-anchor="middle" font-family="var(--vp-font-family-base)" font-size="10.5" fill="var(--cs-color-border-strong)">同主体 + 同类型，取语义最相近的若干条</text>
      <path d="M460 226 L460 244" stroke="var(--cs-color-warning)" stroke-width="1.6" marker-end="url(#wr)"/>
      <!-- 决策菱形 -->
      <path d="M460 246 L620 288 L460 330 L300 288 Z" fill="var(--cs-color-bg-soft)" stroke="var(--cs-color-brand)" stroke-width="1.4"/>
      <text x="460" y="283" text-anchor="middle" font-family="var(--vp-font-family-base)" font-size="12.5" font-weight="700" fill="var(--cs-color-brand)">③ 操作决策</text>
      <text x="460" y="300" text-anchor="middle" font-family="var(--vp-font-family-base)" font-size="10.5" fill="var(--cs-color-brand)">LLM tool call 判定</text>
      <!-- 四分支 -->
      <line x1="140" y1="288" x2="300" y2="288" stroke="var(--cs-color-border-strong)" stroke-width="1.3"/>
      <line x1="140" y1="288" x2="140" y2="360" stroke="var(--cs-color-border-strong)" stroke-width="1.3" marker-end="url(#wr)"/>
      <line x1="620" y1="288" x2="784" y2="288" stroke="var(--cs-color-border-strong)" stroke-width="1.3"/>
      <line x1="784" y1="288" x2="784" y2="360" stroke="var(--cs-color-border-strong)" stroke-width="1.3" marker-end="url(#wr)"/>
      <line x1="352" y1="330" x2="352" y2="360" stroke="var(--cs-color-border-strong)" stroke-width="1.3" marker-end="url(#wr)"/>
      <line x1="568" y1="330" x2="568" y2="360" stroke="var(--cs-color-border-strong)" stroke-width="1.3" marker-end="url(#wr)"/>
      <g font-family="var(--vp-font-family-base)">
        <rect x="76" y="362" width="128" height="88" rx="7" fill="var(--cs-color-success-soft)" stroke="var(--cs-color-success)" stroke-width="1.2"/>
        <text x="140" y="382" text-anchor="middle" font-size="12.5" font-weight="700" fill="var(--cs-color-success)">ADD</text>
        <text x="140" y="399" text-anchor="middle" font-size="10.5" fill="var(--cs-color-border-strong)">无等价记忆</text>
        <line x1="90" y1="407" x2="190" y2="407" stroke="var(--cs-color-neutral-soft)"/>
        <text x="140" y="422" text-anchor="middle" font-size="10" fill="var(--cs-color-text-subtle)">新建条目</text>
        <text x="140" y="437" text-anchor="middle" font-size="10" fill="var(--cs-color-text-subtle)">valid_from = now</text>
        <rect x="288" y="362" width="128" height="88" rx="7" fill="var(--cs-color-info-soft)" stroke="var(--cs-color-info)" stroke-width="1.2"/>
        <text x="352" y="382" text-anchor="middle" font-size="12.5" font-weight="700" fill="var(--cs-color-info)">UPDATE</text>
        <text x="352" y="399" text-anchor="middle" font-size="10.5" fill="var(--cs-color-border-strong)">互补信息增强</text>
        <line x1="302" y1="407" x2="402" y2="407" stroke="var(--cs-color-neutral-soft)"/>
        <text x="352" y="422" text-anchor="middle" font-size="10" fill="var(--cs-color-text-subtle)">合并内容</text>
        <text x="352" y="437" text-anchor="middle" font-size="10" fill="var(--cs-color-text-subtle)">保留变更日志</text>
        <rect x="504" y="362" width="128" height="88" rx="7" fill="var(--cs-color-danger-soft)" stroke="var(--cs-color-danger)" stroke-width="1.2"/>
        <text x="568" y="382" text-anchor="middle" font-size="12.5" font-weight="700" fill="var(--cs-color-danger)">DELETE</text>
        <text x="568" y="399" text-anchor="middle" font-size="10.5" fill="var(--cs-color-border-strong)">被新信息矛盾</text>
        <line x1="518" y1="407" x2="618" y2="407" stroke="var(--cs-color-neutral-soft)"/>
        <text x="568" y="422" text-anchor="middle" font-size="10" fill="var(--cs-color-danger)">软删除</text>
        <text x="568" y="437" text-anchor="middle" font-size="10" fill="var(--cs-color-danger)">valid_to = now</text>
        <rect x="720" y="362" width="128" height="88" fill="var(--cs-color-bg-elevated)" stroke="var(--cs-color-text-subtle)" stroke-width="1.5"/>
        <text x="784" y="382" text-anchor="middle" font-size="12.5" font-weight="700" fill="var(--cs-color-border-strong)">NOOP</text>
        <text x="784" y="399" text-anchor="middle" font-size="10.5" fill="var(--cs-color-border-strong)">已存在 / 不值得</text>
        <line x1="734" y1="407" x2="834" y2="407" stroke="var(--cs-color-neutral-soft)"/>
        <text x="784" y="422" text-anchor="middle" font-size="10" fill="var(--cs-color-text-subtle)">不改库</text>
        <text x="784" y="437" text-anchor="middle" font-size="10" fill="var(--cs-color-text-subtle)">最高频的结果</text>
      </g>
      <!-- 汇聚到审计 -->
      <path d="M140 450 L140 476 L460 476" stroke="var(--cs-color-text-subtle)" stroke-width="1.2" fill="none" stroke-dasharray="4 3"/>
      <path d="M352 450 L352 476" stroke="var(--cs-color-text-subtle)" stroke-width="1.2" stroke-dasharray="4 3"/>
      <path d="M568 450 L568 476" stroke="var(--cs-color-text-subtle)" stroke-width="1.2" stroke-dasharray="4 3"/>
      <path d="M784 450 L784 476 L460 476" stroke="var(--cs-color-text-subtle)" stroke-width="1.2" fill="none" stroke-dasharray="4 3"/>
      <path d="M460 476 L460 494" stroke="var(--cs-color-text-subtle)" stroke-width="1.4" marker-end="url(#wr-g)"/>
      <rect x="286" y="496" width="348" height="42" rx="6" fill="var(--cs-color-brand)"/>
      <text x="460" y="514" text-anchor="middle" font-family="var(--vp-font-family-base)" font-size="12.5" font-weight="700" fill="var(--cs-color-on-brand)">④ 写审计日志（不可变）</text>
      <text x="460" y="530" text-anchor="middle" font-family="var(--vp-font-family-base)" font-size="10.5" fill="color-mix(in srgb, var(--cs-color-on-brand) 80%, transparent)">谁在何时因何证据改了哪条 → 可回滚、可解释、可合规举证</text>
      <!-- 侧注 -->
      <rect x="646" y="184" width="258" height="42" fill="var(--cs-color-bg)" stroke="var(--cs-color-danger)" stroke-dasharray="3 2"/>
      <text x="660" y="202" font-family="var(--vp-font-family-base)" font-size="10.5" fill="var(--cs-color-danger)">注意：DELETE 永远是软删除。</text>
      <text x="660" y="217" font-family="var(--vp-font-family-base)" font-size="10.5" fill="var(--cs-color-danger)">物理删除只由合规流程触发。</text>
      <text x="14" y="120" font-family="var(--vp-font-family-base)" font-size="10.5" fill="var(--cs-color-text-subtle)">写链路应异步执行，</text>
      <text x="14" y="135" font-family="var(--vp-font-family-base)" font-size="10.5" fill="var(--cs-color-text-subtle)">因此可用更强模型、</text>
      <text x="14" y="150" font-family="var(--vp-font-family-base)" font-size="10.5" fill="var(--cs-color-text-subtle)">更长 prompt 与重试。</text>
    </svg>
</div>

一条信息进入长期记忆须同时满足：跨会话仍有效；会改变后续回答或行动；无法从权威源更便宜地获得；敏感度可接受。用户明确要求记住、带稳定原因的安全约束走快速通道；健康、财务、宗教等敏感类别默认不自动写入，需显式同意。一次性指令、实时状态、敏感凭证、未确认推断和纯临时上下文在门控处终止。

### 写入时机

同一条信息在不同时机写入，延迟与质量不同。写入分三个触发点：

- **同步写入**：用户显式要求记住，或支付、权限、身份等高影响约束需要立即生效时，在主链路内完成，代价是增加本轮延迟。
- **回合结束异步抽取**：偏好与经历通常在回合结束后离线抽取，不阻塞回答，代价是同会话后续轮次暂时读不到刚产生的记忆。
- **后台归并**：画像重写、冲突清理、相似条目反思归并放后台周期执行，可用更长上下文和更强模型，但结果生效滞后。

异步链路必须可重试、可观测，并监控最终一致；否则用户「记住」的承诺无法兑现。门控在三个时机保持同一套准入标准，不能因为是后台执行就放宽。

### 写操作

<div class="mem-ops">
<div class="mem-op mem-op--add"><b>ADD</b><p>不存在等价条目，新增原子记录，设置 <code>valid_from</code> 与来源。</p></div>
<div class="mem-op mem-op--update"><b>UPDATE</b><p>补充同一记录的非冲突属性，或在确认槽位变化时以新值取代旧值。</p></div>
<div class="mem-op mem-op--delete"><b>DELETE</b><p>旧条目被证伪、用户要求删除或不再允许保留；先软删除并关闭有效期。</p></div>
<div class="mem-op mem-op--noop"><b>NOOP</b><p>信息重复、一次性、可权威查询或证据不足，不改变记忆库。通常是最高频结果。</p></div>
</div>

候选通过门控、判重与冲突检测后，最终落到四种操作。它们的区别不只在「改不改库」，而在对历史和事实源的处理方式：

- **ADD**：不存在等价条目时新建原子记录，设置 `valid_from` 与来源。每条记录只表达一个断言。
- **UPDATE**：存在等价条目时使用。补充非冲突属性用增量更新；同一槽位确认变化时，事实类不是覆盖旧值，而是关闭旧记录 `valid_to` 后追加新记录，偏好类可在同 key 上覆盖但保留变更日志。
- **DELETE**：条目被新信息矛盾、被用户要求删除，或不再允许保留时触发。默认是软删除并关闭有效期，使检索立即排除但保留审计轨迹。
- **NOOP**：信息重复、属于一次性指令、可从权威源实时获得，或证据不足时，不改变记忆库。严格门控下 NOOP 应是最高频结果。

删除要区分两种语义：用户说「别再记这个」是产品行为，软删加排除检索即可，审计记录保留；法务的被遗忘权请求是合规行为，必须物理清除所有派生副本并留存删除凭证，二者的实现路径与时限不同。

### 更新与删除的级联

更新和删除都不是只改一行主记录。主记录是事实源，向量、全文、图、常驻画像和派生素描都是它的投影；投影不随主记录自动更新，必须显式级联，否则会出现「主记录已改、召回仍旧」的不一致。

**更新的级联**面向「值变了但条目还在」，目标是让所有投影与新值一致：

1. 关闭旧记录 `valid_to` 并追加新记录（事实类），或在原记录上覆盖（偏好类），写入审计；
2. 重建或替换该条目的向量与全文投影，删除旧 embedding，避免新旧值同时被召回；
3. 失效引用旧值的图边并重建关系，更新由该条目参与聚合的常驻画像与摘要；
4. 清除含旧值的读取缓存，使下一轮召回读到新值。

**删除的级联**面向「信息不再允许保留」，目标是让信息从所有可读与可重建位置消失。一条信息通常已扩散到主记录之外的多处：向量与全文索引、由它归并出的摘要与洞察、图边、常驻画像、缓存与备份。

1. 主记录软删除并标记 `status = deleted`，检索层立即排除；
2. 同步删除向量、全文等召回投影，使其无法通过相似搜索重新命中；
3. 沿来源链定位并清除派生摘要、洞察、图边与常驻画像中的对应内容；
4. 异步物理清理并逐副本验证；备份与审计日志在法规保留期内单独管控，到期销毁。

软删除只保证检索与产品视图立即不可见，向量残片和摘要仍可能重建信息，因此必须再有一次物理清理；删除原始对话也不等于删除由对话抽取出的记忆。派生素描能否被定位，依赖写入时保留的来源链。

---

## 召回与注入

召回决定取回哪些记忆，注入决定它们以什么身份和顺序进入 context。Memory recall 与文档检索共享 embedding、ANN、BM25 和 rerank 设施，但额外承担主体隔离、类型路由、有效期过滤和错误个性化防护。

<div class="mem-svg-scroll">
    <svg class="mem-svg" viewBox="0 0 920 610" role="img" aria-label="记忆读取链路六阶段流程图">
      <defs>
        <marker id="rd" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6.5" markerHeight="6.5" orient="auto-start-reverse">
          <path d="M0,0 L10,5 L0,10 z" fill="var(--cs-color-brand)"/>
        </marker>
        <marker id="rd-r" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
          <path d="M0,0 L10,5 L0,10 z" fill="var(--cs-color-danger)"/>
        </marker>
      </defs>
      <!-- 阶段 0 入口 -->
      <rect x="330" y="8" width="260" height="40" rx="6" fill="var(--cs-color-brand)"/>
      <text x="460" y="27" text-anchor="middle" font-family="var(--vp-font-family-base)" font-size="13" font-weight="700" fill="var(--cs-color-on-brand)">用户本轮输入 + 近期对话</text>
      <text x="460" y="41" text-anchor="middle" font-family="var(--vp-font-family-mono)" font-size="9.5" fill="color-mix(in srgb, var(--cs-color-on-brand) 70%, transparent)" letter-spacing="1.5">QUERY IN</text>
      <path d="M460 48 L460 68" stroke="var(--cs-color-brand)" stroke-width="1.6" marker-end="url(#rd)"/>
      <!-- 阶段 1 判断是否需要检索 -->
      <rect x="286" y="70" width="348" height="46" rx="6" fill="var(--cs-color-brand-soft)" stroke="var(--cs-color-brand)" stroke-width="1.4"/>
      <text x="460" y="90" text-anchor="middle" font-family="var(--vp-font-family-base)" font-size="12.5" font-weight="700" fill="var(--cs-color-brand)">① 门控：本轮是否需要检索记忆？</text>
      <text x="460" y="107" text-anchor="middle" font-family="var(--vp-font-family-base)" font-size="10.5" fill="var(--cs-color-text-muted)">闲聊 / 纯计算 / 通用问答 → 跳过，省一次检索的延迟与噪声</text>
      <text x="648" y="96" font-family="var(--vp-font-family-mono)" font-size="10" fill="var(--cs-color-text-subtle)">规则即时 / 小模型分类耗时低</text>
      <path d="M460 116 L460 136" stroke="var(--cs-color-brand)" stroke-width="1.6" marker-end="url(#rd)"/>
      <!-- 阶段 2 查询改写 -->
      <rect x="286" y="138" width="348" height="60" fill="var(--cs-color-bg-elevated)" stroke="var(--cs-color-brand)" stroke-width="1.5"/>
      <text x="460" y="158" text-anchor="middle" font-family="var(--vp-font-family-base)" font-size="12.5" font-weight="700" fill="var(--cs-color-brand)">② 查询改写与扩写</text>
      <text x="460" y="174" text-anchor="middle" font-family="var(--vp-font-family-base)" font-size="10.5" fill="var(--cs-color-border-strong)">指代消解（「那个方案」→「方案 A」）· 时间词归一（「上个月」→ 日期区间）</text>
      <text x="460" y="190" text-anchor="middle" font-family="var(--vp-font-family-base)" font-size="10.5" fill="var(--cs-color-border-strong)">多查询生成 · 抽取过滤条件</text>
      <text x="648" y="168" font-family="var(--vp-font-family-mono)" font-size="10" fill="var(--cs-color-text-subtle)">改写耗时随子查询数增加</text>
      <text x="648" y="182" font-family="var(--vp-font-family-mono)" font-size="10" fill="var(--cs-color-danger)">省掉它 → 多轮指代全挂</text>
      <path d="M460 198 L460 218" stroke="var(--cs-color-brand)" stroke-width="1.6" marker-end="url(#rd)"/>
      <!-- 阶段 3 分类型路由 + 并行检索 -->
      <rect x="120" y="220" width="680" height="132" fill="var(--cs-color-bg-soft)" stroke="var(--cs-color-border-strong)"/>
      <text x="136" y="240" font-family="var(--vp-font-family-base)" font-size="12.5" font-weight="700" fill="var(--cs-color-text)">③ 按类型路由 + 多路并行检索</text>
      <text x="136" y="256" font-family="var(--vp-font-family-base)" font-size="10.5" fill="var(--cs-color-border-strong)">不同类型走不同索引；并行发起，总延迟取最慢一路</text>
      <g font-family="var(--vp-font-family-base)" font-size="10.5">
        <rect x="136" y="266" width="148" height="72" fill="var(--cs-color-bg-elevated)" stroke="var(--cs-color-brand)"/>
        <text x="210" y="284" text-anchor="middle" font-size="11.5" font-weight="700" fill="var(--cs-color-brand)">向量检索</text>
        <text x="210" y="300" text-anchor="middle" fill="var(--cs-color-border-strong)">语义相近</text>
        <text x="210" y="315" text-anchor="middle" fill="var(--cs-color-text-subtle)" font-size="10">事实 / 经历</text>
        <text x="210" y="330" text-anchor="middle" font-family="var(--vp-font-family-mono)" font-size="9.5" fill="var(--cs-color-text-subtle)">多路召回候选集</text>
        <rect x="296" y="266" width="148" height="72" fill="var(--cs-color-bg-elevated)" stroke="var(--cs-color-brand)"/>
        <text x="370" y="284" text-anchor="middle" font-size="11.5" font-weight="700" fill="var(--cs-color-brand)">BM25 / 全文</text>
        <text x="370" y="300" text-anchor="middle" fill="var(--cs-color-border-strong)">字面精确</text>
        <text x="370" y="315" text-anchor="middle" fill="var(--cs-color-text-subtle)" font-size="10">人名 / 型号 / ID</text>
        <text x="370" y="330" text-anchor="middle" font-family="var(--vp-font-family-mono)" font-size="9.5" fill="var(--cs-color-text-subtle)">多路召回候选集</text>
        <rect x="456" y="266" width="148" height="72" fill="var(--cs-color-bg-elevated)" stroke="var(--cs-color-brand)"/>
        <text x="530" y="284" text-anchor="middle" font-size="11.5" font-weight="700" fill="var(--cs-color-brand)">图遍历</text>
        <text x="530" y="300" text-anchor="middle" fill="var(--cs-color-border-strong)">1~2 跳邻域</text>
        <text x="530" y="315" text-anchor="middle" fill="var(--cs-color-text-subtle)" font-size="10">多跳关系问题</text>
        <text x="530" y="330" text-anchor="middle" font-family="var(--vp-font-family-mono)" font-size="9.5" fill="var(--cs-color-text-subtle)">按度数限流</text>
        <rect x="616" y="266" width="168" height="72" rx="7" fill="var(--cs-color-info-soft)" stroke="var(--cs-color-info)" stroke-width="1.2"/>
        <text x="700" y="284" text-anchor="middle" font-size="11.5" font-weight="700" fill="var(--cs-color-info)">直取（不检索）</text>
        <text x="700" y="300" text-anchor="middle" fill="var(--cs-color-border-strong)">偏好 / 常驻块</text>
        <text x="700" y="315" text-anchor="middle" fill="var(--cs-color-border-strong)">程序性规则（按 trigger）</text>
        <text x="700" y="330" text-anchor="middle" font-family="var(--vp-font-family-mono)" font-size="9.5" fill="var(--cs-color-info)">全量注入</text>
      </g>
      <text x="810" y="290" font-family="var(--vp-font-family-mono)" font-size="10" fill="var(--cs-color-text-subtle)">元数据过滤</text>
      <text x="810" y="304" font-family="var(--vp-font-family-mono)" font-size="10" fill="var(--cs-color-text-subtle)">全程强制：</text>
      <text x="810" y="318" font-family="var(--vp-font-family-mono)" font-size="10" fill="var(--cs-color-text-muted)">user_id</text>
      <text x="810" y="332" font-family="var(--vp-font-family-mono)" font-size="10" fill="var(--cs-color-text-muted)">valid_to IS NULL</text>
      <path d="M460 352 L460 372" stroke="var(--cs-color-brand)" stroke-width="1.6" marker-end="url(#rd)"/>
      <!-- 阶段 4 融合 -->
      <rect x="286" y="374" width="348" height="60" fill="var(--cs-color-bg-elevated)" stroke="var(--cs-color-brand)" stroke-width="1.5"/>
      <text x="460" y="394" text-anchor="middle" font-family="var(--vp-font-family-base)" font-size="12.5" font-weight="700" fill="var(--cs-color-brand)">④ 融合 Fusion</text>
      <text x="460" y="410" text-anchor="middle" font-family="var(--vp-font-family-base)" font-size="10.5" fill="var(--cs-color-border-strong)">RRF 按排名融合，免去跨检索器分数归一化</text>
<text x="460" y="426" text-anchor="middle" font-family="var(--vp-font-family-mono)" font-size="10.5" fill="var(--cs-color-text-muted)">score(d) = Σ 1 / (k + rank_i(d))，k 为融合常数</text>
      <text x="648" y="404" font-family="var(--vp-font-family-mono)" font-size="10" fill="var(--cs-color-text-subtle)">纯计算，耗时可忽略</text>
      <path d="M460 434 L460 454" stroke="var(--cs-color-brand)" stroke-width="1.6" marker-end="url(#rd)"/>
      <!-- 阶段 5 重排+去冗 -->
      <rect x="286" y="456" width="348" height="74" rx="6" fill="var(--cs-color-bg-soft)" stroke="var(--cs-color-brand)" stroke-width="1.3"/>
      <text x="460" y="476" text-anchor="middle" font-family="var(--vp-font-family-base)" font-size="12.5" font-weight="700" fill="var(--cs-color-brand)">⑤ 重排 + 综合打分 + 去冗</text>
      <text x="460" y="492" text-anchor="middle" font-family="var(--vp-font-family-base)" font-size="10.5" fill="var(--cs-color-brand)">cross-encoder rerank → relevance × recency × importance 加权</text>
      <text x="460" y="507" text-anchor="middle" font-family="var(--vp-font-family-base)" font-size="10.5" fill="var(--cs-color-brand)">MMR 去重（同一事实的多个版本只留最新那条）</text>
      <text x="460" y="522" text-anchor="middle" font-family="var(--vp-font-family-mono)" font-size="10" fill="var(--cs-color-brand)">候选集收敛到个位数</text>
      <text x="648" y="486" font-family="var(--vp-font-family-mono)" font-size="10" fill="var(--cs-color-text-subtle)">耗时取决于 rerank 模型</text>
      <text x="648" y="500" font-family="var(--vp-font-family-mono)" font-size="10" fill="var(--cs-color-text-subtle)">（rerank 模型）</text>
      <path d="M460 530 L460 550" stroke="var(--cs-color-brand)" stroke-width="1.6" marker-end="url(#rd)"/>
      <!-- 阶段 6 注入 -->
      <rect x="286" y="552" width="348" height="46" rx="6" fill="var(--cs-color-brand)"/>
      <text x="460" y="572" text-anchor="middle" font-family="var(--vp-font-family-base)" font-size="12.5" font-weight="700" fill="var(--cs-color-on-brand)">⑥ 预算内注入 context</text>
      <text x="460" y="589" text-anchor="middle" font-family="var(--vp-font-family-base)" font-size="10.5" fill="color-mix(in srgb, var(--cs-color-on-brand) 80%, transparent)">按类型分块 + 标注时间与来源 + 强约束置于首尾</text>
      <!-- 右侧回退分支 -->
      <path d="M634 490 L700 490 L700 570 L638 570" stroke="var(--cs-color-danger)" stroke-width="1.1" fill="none" stroke-dasharray="4 3" marker-end="url(#rd-r)"/>
      <text x="708" y="530" font-family="var(--vp-font-family-base)" font-size="10.5" fill="var(--cs-color-danger)">召回为空 →</text>
      <text x="708" y="545" font-family="var(--vp-font-family-base)" font-size="10.5" fill="var(--cs-color-danger)">显式告知无记忆</text>
      <text x="708" y="560" font-family="var(--vp-font-family-base)" font-size="10.5" fill="var(--cs-color-danger)">禁止编造</text>
      <!-- 左侧预算标注 -->
      <rect x="14" y="374" width="96" height="156" rx="8" fill="var(--cs-color-danger-soft)" stroke="var(--cs-color-danger)" stroke-width="1" stroke-dasharray="3 2"/>
      <text x="62" y="394" text-anchor="middle" font-family="var(--vp-font-family-base)" font-size="11" font-weight="700" fill="var(--cs-color-danger)">预算红线</text>
      <text x="62" y="414" text-anchor="middle" font-family="var(--vp-font-family-mono)" font-size="9.5" fill="var(--cs-color-danger)">注入 token 有上限</text>
      <text x="62" y="430" text-anchor="middle" font-family="var(--vp-font-family-mono)" font-size="9.5" fill="var(--cs-color-danger)">条数保持个位数</text>
      <text x="62" y="446" text-anchor="middle" font-family="var(--vp-font-family-mono)" font-size="9.5" fill="var(--cs-color-danger)">读链路 P95 设预算</text>
      <line x1="26" y1="458" x2="98" y2="458" stroke="var(--cs-color-danger-soft)"/>
      <text x="62" y="476" text-anchor="middle" font-family="var(--vp-font-family-base)" font-size="10" fill="var(--cs-color-danger)">超预算时</text>
      <text x="62" y="491" text-anchor="middle" font-family="var(--vp-font-family-base)" font-size="10" fill="var(--cs-color-danger)">按类型优先级</text>
      <text x="62" y="506" text-anchor="middle" font-family="var(--vp-font-family-base)" font-size="10" fill="var(--cs-color-danger)">截断而非</text>
      <text x="62" y="521" text-anchor="middle" font-family="var(--vp-font-family-base)" font-size="10" fill="var(--cs-color-danger)">均匀裁剪</text>
    </svg>
</div>

排序因子中，相关性回答「像不像这个查询」，时效回答「是否还新鲜」，重要性回答「错过的代价」，置信度回答「证据是否充分」。各项归一化后再加权，权重需按场景用离线集与线上实验校准，不存在通用最优值。

记忆检索与普通文档检索有三处实质差别，图中的每个分支对应一个工程决策：

- **按类型分流，而不是一条向量通路走到底**。事实与事件走向量加全文，实体关系走图库有限跳遍历，偏好和程序性规则靠常驻或 trigger 直取。多路并行发起，总延迟取决于最慢一路。
- **必须先做主体与有效期硬过滤**。`scope`、`subject_id`、权限和 `valid_to IS NULL` 作为查询约束在候选生成阶段生效，不能取回后再筛。
- **多路结果用排名融合而非分数相加**。向量分与全文分的尺度不同，直接加权没有意义；用 RRF 一类按排名融合的方式免去跨检索器归一化，再由 rerank 模型结合查询重排，并用 MMR 对同一事实的多个版本去重。

记忆召回宜少而精。文档漏召回通常使答案不完整；错误召回一条过期或不属于本主体的记忆，模型会把它当成既定事实自信使用，产生跨会话累积的错误个性化。因此记忆侧需要相似度下限、主体硬过滤和「宁可空召回」的降级策略，空召回时禁止编造画像。注入还要受预算约束：按类型分配 token 与条数上限，强约束置于首尾，超预算时按类型优先级截断而不是均匀裁剪。

### 注入标注

记忆进入 context 时必须以数据身份分区呈现，标注主体、类型、来源与时间，并与权威知识、实时 observation 分离：

<div class="mem-tag">[Memory · user · preference · 2026-08-12 用户确认]
偏好中文技术文档，示例默认使用 Go。</div>

召回不等于该用。注入前要过一道规则即可完成的轻量校验：作用域是否严格匹配、`valid_to` 是否为空且未软删、召回集合内部有无同槽位冲突、置信度是否达到阈值（不达阈值则不注入或降级为不确定表述）、内容中的指令性文本是否已剥离。时间标注让模型和用户都能判断依据的新旧。

注入内容中的命令式文本不获得系统指令权限，出现「忽略之前的指令」也按数据处理。记忆、权威知识、实时观察与当轮指令冲突时，裁决顺序应为：当轮显式指令与安全约束优先，其次是实时权威状态，再次是共享知识，记忆只在有效期内作为主体倾向参与；任何记忆都不能扩大工具权限。

---

## 有效性

记忆一旦写入，正确性不是静态的：事实会过期、证据会变化、新旧记忆会冲突。有效性判定决定一条记忆能否被召回、以什么语气注入，遗忘则让陈旧低价值条目退出召回。

### 有效性判定

一条记忆能不能用，不能只看「是否过期」，而要沿五个维度联合判断：

- **时效性**：`valid_to IS NULL` 且未过 `expire_at`，同时看事实的固有易变性。出生日期永久有效，当前职位半年后可能失效，临时情绪次日即失效；易变性应在写入时就估计并落成 TTL，而不是检索时临时判断。
- **来源与出处**：能否追溯到具体的原始消息、文档或工具返回。用户显式陈述高于行为观察，行为观察高于模型推断，外部文档抽取最低；无出处的记忆应视为不可信，可能是幻觉写入或投毒。
- **置信度**：由抽取时的模型确定性、独立确认次数和是否被否认共同决定，随证据动态升降——再次提及则升，被纠正则骤降并标记。低置信记忆可以召回，但注入时要带不确定措辞，而不是作为既定事实断言。
- **一致性**：与库内其他记忆、常驻画像和当前对话是否矛盾。写入时对同槽位做矛盾检测，注入前再做一次轻量检查；召回集合内部自相矛盾时，保留新且高置信的一条，并把矛盾记为待澄清事件。
- **可验证性**：能否用外部权威源核实。可实时校验的状态（订单、余额、文件是否存在）不长期存结论，只存指针、用时实时查；职位、公司等半可校验信息存记忆但设较短 TTL 并在关键决策前确认；喜好、感受等不可校验信息只能靠一致性与用户反馈维护。

### 冲突消解

冲突处理先确认两条记录是否描述同一主体的同一槽位，再按顺序比较证据：

1. 显式陈述高于隐式行为推断；
2. 已验证权威来源高于模型猜测和外部文档；
3. 多次独立确认高于单次弱信号；
4. 新事实只在同一维度上取代旧事实，更换居住城市不改变学历；
5. 证据强度相同且确实矛盾时，两条都保留并降低置信度，在后续相关对话中澄清。

有些差异可以并存：用户可以在不同项目使用不同语言，也可以在不同场景偏好不同详略。把可并存差异判成互斥冲突并互相覆盖，是偏好记忆的高频错误。归并多条事件得到的洞察必须保留 provenance，指回原始事件；原始事件只降权不立即删除，否则一次错误归纳无法追溯修正。

<div class="mem-callout mem-callout--warning">
<span class="mem-callout__tag">一次性指令不是偏好</span>
<p>「这次简短点」「先改这个文件」作用域是当前产物，不进入长期记忆。判定稳定偏好看范围词（以后 / 每次 / 默认）、稳定原因、跨会话重复次数和可迁移性。单次信号可作为 tentative 条目，经多次未纠正再升为 active，被反向纠正立即失效。</p>
</div>

### 遗忘机制

遗忘不是删除历史，而是让陈旧、低价值的记忆退出召回，控制库的无界增长。手段按强度从弱到强：

- **时间衰减**：降低旧条目的排序权重而非删除。半衰期按类型设置，经验上临时状态、事件经历、长期偏好可依次按天、周、月的量级起步，再依数据分布校准；过敏、无障碍需求、法律与安全禁忌不参与自动衰减。
- **TTL 与状态失效**：对易变和半可校验信息设有效期，到期自动置为失效；一次性、试用期未确认的条目到期清理。
- **容量淘汰**：每个主体和类型设容量上限，长期未命中的条目按 LRU 一类策略降权或转入冷存储，而非无限保留。
- **归并降权**：多条同类经历经反思提炼为高层洞察后，原始条目降权保留以备追溯，不与洞察重复竞争召回。
- **软删除与归档**：被否认或失效的记忆软删除并立即退出检索，需要时再异步物理清理（级联见前）。

---

## 安全

记忆是少数会跨会话持续生效的注入面，安全模型与普通提示词注入不同。

普通 prompt injection 通常随会话结束失效；投毒内容一旦被写为长期记忆，会在后续无关会话中反复生效。防御沿写入链路分三层：

- **来源分级**：用户显式陈述与人工确认的可信度最高，行为观察次之，外部网页、文档、工单、仓库和工具返回默认低可信。低可信来源只允许写入非敏感、可验证的类型，不能直接产生程序性规则或安全相关记忆。
- **写入确认与 schema 约束**：写入只接受预定义字段与取值范围，自由文本中的指令在入库前剥离；支付、权限、身份等高影响记忆不自动生效，需要显式确认或二次校验。
- **可回滚**：每条记忆保留来源与写入批次，发现某一来源或时间段的批量污染时，可按来源、批次或类型快速停用并回滚，而不必逐条人工删除。

常驻注入的程序性记忆是投毒影响最大的位置：一旦规则文件被写入恶意指令，会在每次会话生效。规则类记忆的写入门槛应高于事实和偏好，团队规则还需走评审。

---

---

## 冷启动

新主体没有记忆时，召回结果天然为空，这不是故障而是正常状态。冷启动阶段的风险来自两点：把群体统计当成个人事实，以及为了「显得了解用户」而编造画像。

- **空召回按无记忆处理**：不注入任何 `user` 作用域条目，回答退回到当轮输入、knowledge 与实时观察，语气保持中性，不预设偏好。
- **群体先验不是个人记忆**：可以用团队或同类型用户的通用约定（`team` / `org` 作用域）作为缺省行为，但必须与个人记忆分开存储和标注；当用户出现明确的个人信号时，个人记忆覆盖群体先验。
- **试用期写入**：冷启动期采集到的少量信号以 tentative 状态保存，经过跨会话确认再升为 active，避免一次性行为被固化为长期画像。

随着有效记忆累积，门控仍应保持同样的准入标准；记忆数量本身不是目标，越过合适数量后噪声会反向降低召回质量。

---

---

## 工程落地

### 存储选型

记忆系统通常组合多种存储，没有单一介质能同时满足事务、语义检索、多跳关系与大对象存储。主记录与各类索引按下表分工：

| 存储 | 承载 | 不承载 / 边界 |
| --- | --- | --- |
| 关系库 | 主记录、作用域与槽位、有效期、置信度、审计；提供事务与时间查询 | 不直接做语义相似排序 |
| 向量索引 | 内容的 embedding，承担语义候选召回 | 不是事实源；无主记录则无法判重、失效与删除 |
| 全文索引 | 人名、编号、型号、关键字的字面匹配 | 不单独表达有效期与作用域，需与主记录关联 |
| 图数据库 | 实体关系与有限跳遍历，边带有效期与来源 | 简单事实查询不必引入；实体消歧成本高 |
| KV / 常驻缓存 | 偏好、硬约束等几乎每轮使用的小量条目 | 只存高频直取内容，容量需硬上限 |
| 对象存储 | 图片、音频、录屏等原始对象 | 检索层只存 URI、描述与多模态向量 |

---

### 用户控制与审计

记忆含个人数据，用户对自身状态的可见与可控是基本能力，而不是附属设置。系统至少要提供四类入口：

- **查看**：能按条目看到内容、类型、来源和时间，而不只是一个总开关；
- **编辑与撤销**：能修正错误条目、关闭某条记忆或整体停用记忆；
- **删除与导出**：能删除单条、清空某类记忆并导出数据；
- **审计**：系统侧保留不可变变更日志，记录谁在何时因何证据改变了哪条记录，以支持解释、回滚与合规举证。

这些操作在系统内部会级联传播到索引与派生物，而不只是修改主记录一行。

---

### 失败模式与取舍

| 失败模式 | 后果 | 对应机制 |
| --- | --- | --- |
| 全量对话入记忆 | 噪声淹没召回，库体积无界增长 | 写入门控与 NOOP |
| 实时业务状态入记忆 | 订单、余额类陈旧事实被反复引用 | 权威源实时读取，记忆只存指针 |
| 召回后才做主体过滤 | 越权数据可能已进入候选或提示 | 检索层作用域硬过滤 |
| 只写向量不写主记录 | 无法判重、失效与级联删除 | 主记录为事实源，索引为投影 |
| 覆盖写事件经历 | 历史决策依据丢失，错误无法回溯 | 事件只追加，双时间轴保留历史 |
| 未确认推断自动生效 | 幻觉写入并跨会话传播 | tentative 状态、置信度与确认升级 |
| 记忆被当作指令 | 投毒内容改变权限或行为 | 来源分级、指令剥离、按批次回滚 |
| 用户无法查看删除 | 错误无法纠正，不满足合规 | 条目级控制与级联删除 |

工程取舍集中在三处：

- **关系库还是向量库**：主记录与事务语义落在关系库，向量只承担语义候选，多跳关系成为明确需求前不引入图库，避免用存储复杂度掩盖建模缺陷（分工见存储选型）。
- **写入覆盖率还是精确率**：写入侧优先精确率而非覆盖率，一条误写会长期污染后续召回；在精确率不足时扩大写入规模，只会放大噪声。
- **同步还是异步写入**：只有需要立即生效的显式记忆和安全约束走同步，其余异步处理；异步链路必须可重试、可观测，否则「记住」的承诺无法兑现。

---

---

## 评测

记忆系统需沿写入、召回、使用、治理四层分别度量，单一最终答案无法定位故障环节。

| 层次 | 指标 |
| --- | --- |
| 写入 | 候选接受率、误写率、字段完整率、来源覆盖率 |
| 召回 | Precision@K、Recall@K、空召回率、跨主体召回次数 |
| 使用 | 答案正确率、时间推理、知识更新、无证据拒答、错误个性化率 |
| 治理 | 陈旧引用率、冲突残留、僵尸记忆比例、用户纠错率、删除完成率 |

公开基准（如 LongMemEval）覆盖信息抽取、多会话推理、时间推理、知识更新与拒答等能力切片，可作为回归参照；但厂商或论文中的分数受检索预算、裁判模型和评测设置影响，不构成选型依据。业务记忆应建立自有 golden set，按场景离线回归，并在线上观察重复提问率、一次解决率、任务完成率、人工转接率和用户纠错。

评测还必须覆盖记忆造成的反向伤害：旧偏好压过当前意图的过度个性化、过期事实的陈旧引用、在不合时宜场景提起私人信息的越界、低置信记忆导致的强行作答，以及跨主体召回。记忆数量与个性化效果呈倒 U 型关系，越过拐点后噪声和陈旧内容反而降低质量。

---

---

## 参考资料

1. Sumers, T. R. et al. (2023). [Cognitive Architectures for Language Agents](https://arxiv.org/abs/2309.02427)。
2. Park, J. S. et al. (2023). [Generative Agents: Interactive Simulacra of Human Behavior](https://arxiv.org/abs/2304.03442)。
3. Wu, D. et al. (2025). [LongMemEval: Benchmarking Chat Assistants on Long-term Interactive Memory](https://arxiv.org/abs/2410.10813)。
4. Chandar, P. et al. (2025). [Building Production-Ready AI Agents with Scalable Long-Term Memory](https://arxiv.org/abs/2504.19413)。
5. Zep (2025). [A Temporal Knowledge Graph Architecture for Agent Memory](https://arxiv.org/abs/2501.13956)。
6. Anthropic. [Effective context engineering for AI agents](https://www.anthropic.com/engineering/effective-context-engineering-for-ai-agents)。
