# 贡献指南

写作、展示、编辑范围与验证要求统一见仓库根 `AGENTS.md`。编辑前再读取目标目录适用的领域补充；本文件只维护贡献与提交流程。

## 贡献流程

1. 确定变更范围，新文件命名沿用所在目录的习惯。
2. 按根规范编辑并完成相应验证，提交说明中注明变更与未完成的检查。
3. 提交前审查差异，确认没有混入无关内容。AI 辅助贡献遵循同一流程，不维护另一套格式规范。

## 提交规范

使用 [Conventional Commits](https://www.conventionalcommits.org/) 格式：

```
<type>(<scope>): <description>

[optional body]

[optional footer]
```

常用类型：`docs`（文档）、`feat`（功能）、`fix`（修复）、`refactor`（重构）、`style`（格式）、`test`（测试）、`chore`（构建与工具）。

### 示例

```
docs(marketing): 补充规则引擎安全与可观测性内容

- 新增开源方案对比
- 新增集成边界说明
- 新增监控指标定义
```
