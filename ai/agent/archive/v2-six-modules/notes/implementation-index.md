# 实现案例阅读索引

实现案例用于观察通用概念怎样落到具体系统，而不是替代概念主文。阅读时先确定输入、状态、动作和结果，再检查该版本如何处理边界。

下表指向既有核验记录；固定提交和历史文档不标成最新实现。新版正文不依赖 SDK 初始化或真实服务联调。

---

## 一、按知识主题寻找案例

| 学习主题 | 案例来源 | 阅读问题 |
| --- | --- | --- |
| 工具执行 | [Codex 工具运行记录](references.md#source-codex-tools) | 调用如何校验、调度并返回 |
| 历史压缩 | [Pi 压缩记录](references.md#source-pi-compact) | 历史、摘要和保留区间如何关联 |
| Skills | [格式规范](references.md#source-skills-spec) | 入口元数据与正文怎样组织 |
| 扩展 | [Pi 扩展记录](references.md#source-pi-ext-doc) | 注册、事件与释放由谁控制 |
| Workflow 与恢复 | [LangGraph 阅读记录](references.md#source-langgraph-functional) | 重执行位置与结果复用有什么约束 |
| 人工交互 | [App Server 阅读记录](references.md#source-openai-app-article) | 交互事件、审批与运行对象如何关联 |
| 长任务 | [Harness 实践](references.md#source-anthropic-long-harness) | 阶段产物怎样支持继续工作 |
| 调度 | [Symphony 规格记录](references.md#source-symphony) | 任务资格、工作区与执行者怎样联系 |
| 多 Agent | [Cursor 经验](references.md#source-cursor-scaling)、[Cognition 经验](references.md#source-cognition-context) | 收益依赖怎样的任务与上下文边界 |
| 动态编排 | [Cloudflare 案例](references.md#source-cloudflare-code) | 组合动作减少了哪些往返，又增加哪些执行约束 |

表中的问题是阅读方向，不声称每个项目采用同一答案。不能用某个函数的名称推断它已实现完整取消、安全或任务验收。

---

## 二、一次源码或文档阅读的记录

可以围绕一个机制记录：

1. 对象和版本：研究哪个项目、提交或文档快照。
2. 输入条件：已有状态、权限、可用能力。
3. 关键过程：实际控制和数据如何移动。
4. 结果含义：成功、部分完成、未知如何表示。
5. 边界：文档没有保证什么，代码未覆盖什么。
6. 与通用概念的对应：哪些是普遍问题，哪些是产品选择。

这种记录避免把一个具体字段升级为通用定义，也避免把概念正确直接当作产品实现已经正确。

---

## 三、证据类型

论文用于方法定义与有条件的实验结果；规范用于协议契约；固定源码用于所读范围内的实际逻辑；作者工程文章用于理解具体经验。社区讨论可以提示问题，但仍需维护者资料、代码或复现支持结论。

访问不到正文的链接只作为线索。没有运行验证时，不宣称已复现性能、通过安全测试或完成真实联调。

旧版的更细产品对照及代码材料保留在[归档实现索引](../archive/notes/implementation-index.md)，其中的版本、日期和访问范围按原记录解释。
