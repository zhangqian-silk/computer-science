# AI 编辑规范

本文件是 AI 工具编辑本仓库的统一规则源文件。

## 格式规范

| 规则 | 说明 |
|------|------|
| 中文引号 | 中文文档使用「」或""，不转换为英文引号 |
| 英文引号 | 英文文档使用 "" |
| 代码字符串 | 统一使用英文引号 |
| 中文标点 | ，。！？：； |
| 英文标点 | ,.!?:; |

## 编辑规范

- **最小化变更**：只修改必要的部分，不格式化整个文件
- **保持原文风格**：不改整体风格和术语
- **保留历史痕迹**：不删除作者、日期等 front matter

## Markdown 规范

- 缩进使用**制表符**
- 无序列表使用 `-` 而不是 `*`
- 标题和其他元素间保持空行
- 主要章节（一、二级标题）之间使用水平分割线 `---`
- 数学公式统一使用 `$...$`（行内）或 `$$...$$`（块级），禁止使用 `\(...\)` 与 `\[...\]`
- 语言风格偏向专业、学术

## VitePress 文档规范

- 面向 `github.io` / VitePress 渲染的文档，涉及机制、架构、数据流、层级关系时，**优先提供结构图或流程图**
- 结构图首先服务于“帮助读者快速建立整体理解”，避免只堆砌公式或长段文字
- 需要展示数据流动、层间关系、可见性变化、矩阵变换等内容时，**优先使用 VitePress 支持的交互组件（Vue 组件）**
- 交互组件优先用于以下场景：Attention 数据流、Transformer block 结构、Mask / Position 对比、多头注意力、复杂度权衡等
- Mermaid 仅作为补充方案，适用于**简短、低密度、概览式**结构图；过长、过高、信息过密的 Mermaid 图应避免
- 不再优先使用超长 ASCII 图、过度拥挤的 Mermaid 图来表达复杂结构
- 文档中的交互组件应优先满足以下目标：层级清晰、数据流向明确、移动端可读、在 GitHub Pages 上稳定渲染
- 若新增 VitePress 交互组件，放入 `.vitepress/theme/components/`（该目录下的 `.vue` 由 `index.ts` 自动注册，无需手写注册代码）
- 新增结构图或交互组件时，优先复用现有视觉风格，避免同一篇文档中出现多套割裂的颜色、卡片、标签体系
- 对于复杂主题，推荐采用“总览路线图 + 核心机制交互图 + 局部对比图”的组合，而不是单张巨型结构图
- 修改 VitePress 文档、结构图或交互组件后，应尽量执行 `npm run docs:build` 验证渲染与构建是否正常

---

## 前端设计系统规范

完整说明见 `docs/design-system.md`；以下是编辑代码时必须遵守的硬约束。

- 样式按 **色板 → 语义 token → 主题 → 布局 → 组件** 分层，位于 `.vitepress/theme/styles/`，入口是 `styles/index.css`，**import 顺序即层级顺序，不要随意调整**
- 组件（含 `.vue` 的 scoped 样式与 SVG 的 `fill` / `stroke`）**只能消费语义 token** `--cs-color-*`、`--cs-text-*`、`--cs-space-*`、`--cs-radius-*` 等
- **禁止**在组件里写字面量色值（hex、`rgb()`、具名色）、直接引用原始色板 `--cs-<色相>-<台阶>`、或直接引用框架变量 `--vp-c-*`；`--vp-*` 仅由 `styles/themes.css` 单向桥接
- 需要「第 N 条数据用第 N 个颜色」时，用 `series-palette.ts` 的 `seriesColor(n)`，不要在 JS 里维护 hex 数组
- 字号、间距、圆角取 token 台阶，不要新增 `.82rem`、`.55rem` 这类魔法数
- 新增组件前先查 `docs/design-system.md` 的复用类清单（`.cs-state--*`、`.cs-legend-row`、`.cs-split`、`.cs-toggle`、`.infra-*` 等），能复用就不要新写 scoped 样式
- 主题状态的唯一来源是 `theme-registry.ts`，组件不得自行读写 `data-cs-theme` 或主题 localStorage 键
- 新增主题只需改 `styles/themes.css` + `theme-registry.ts` 两处，**不得为此改动任何组件**；每套主题必须同时提供浅色与深色取值，且必需 token 不可缺项
- 可点击元素保持 `min-height: var(--cs-tap-target)`，焦点态复用 `controls.css` 的统一焦点环，不要 `outline: none`
- 改动样式、主题或组件后执行 `npm run check`（token 一致性 + 8 种主题组合的 WCAG 对比度），再执行 `npm run docs:build`

## 代码规范

- 代码实现默认使用 **Golang**

## 伪代码规范

使用《算法导论》风格：

```
ALGORITHM-NAME(arg1, arg2, ...)
    // 注释说明
    if condition then
        statement1
        statement2
    else
        statement3
    end if
    
    for i ← 1 to n do
        statement
    end for
    
    while condition do
        statement
    end while
    
    return result
```

**关键符号**：
- 赋值：`←`
- 注释：`//` 单行
- 块结束：`end if` / `end for` / `end while` / `end function`
- 数组索引：`A[i]` 或 `A[1..n]`
- 数学运算：`⌊⌋` `⌈⌉` `mod` `and` `or` `not`

## 权限规范

- 涉及提权执行时，应优先提交覆盖面适中、语义清晰、可复用的 `prefix_rule`
- `prefix_rule` 应尽量保持“短而稳”，优先使用稳定命令前缀，而不是把整条命令原样提交为长期批准规则
- 适合长期批准的前缀，应限定在低风险、常复用、边界清晰的命令范围内，例如 `git status`、`git diff`、`gh pr view`、`npm run build`
- 不应为高风险或破坏性操作申请宽泛前缀，例如 `git push`、`git reset`、`npm publish`、删除类命令
- 若某条命令包含临时参数、具体路径、一次性上下文或 heredoc，通常不应直接作为 `prefix_rule` 持久批准

## 提交规范

Conventional Commits: `<type>(<scope>): <description>`

类型: `feat` `fix` `docs` `refactor` `style` `test` `chore`
