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
- 若新增 VitePress 交互组件，应放入 `.vitepress/theme/components/`，并在 `.vitepress/theme/index.ts` 中注册
- 新增结构图或交互组件时，优先复用现有视觉风格，避免同一篇文档中出现多套割裂的颜色、卡片、标签体系
- 对于复杂主题，推荐采用“总览路线图 + 核心机制交互图 + 局部对比图”的组合，而不是单张巨型结构图
- 修改 VitePress 文档、结构图或交互组件后，应尽量执行 `npm run docs:build` 验证渲染与构建是否正常

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

## 提交规范

Conventional Commits: `<type>(<scope>): <description>`

类型: `feat` `fix` `docs` `refactor` `style` `test` `chore`
