# 前端设计系统

本文件是仓库根 `AGENTS.md` 的样式实现参考，不另设写作或组件选型规则。以下路径均相对于 `.vitepress/theme/`。

## 样式入口

- `styles/index.css`：样式加载顺序。
- `styles/palette.css`、`styles/themes.css`：原始色板、主题取值与 VitePress 颜色桥接。
- `styles/tokens.css`：字号、间距、圆角等尺度；具体数值以源码为准。
- `styles/base.css`：正文、表格、公式与图示的公共排版。
- `styles/layout.css`、`styles/components/data-display.css`：可直接用于静态 HTML 的布局与展示类。

## 颜色与尺度

文档图示使用语义变量，由主题决定具体颜色：

- 正文、次要说明与弱化标签：`--cs-color-text`、`--cs-color-text-muted`、`--cs-color-text-subtle`。
- 背景与边框：`--cs-color-bg`、`--cs-color-bg-soft`、`--cs-color-bg-elevated`、`--cs-color-border`、`--cs-color-border-strong`。
- 强调：`--cs-color-brand`、`--cs-color-brand-soft`；品牌填充块上的文字使用 `--cs-color-on-brand`。
- 状态：`--cs-color-success`、`--cs-color-warning`、`--cs-color-danger`、`--cs-color-info`，各有对应的 `-soft` 背景。
- 数据系列：`--cs-series-1` 至 `--cs-series-8`。

图示不直接使用原始色板或框架颜色变量。`themes.css` 负责颜色映射；`tokens.css` 中 `--cs-font-mono` 对框架字体变量的引用属于字体复用，不受颜色分层限制。

字号、行高、间距与圆角优先选用 `--cs-text-*`、`--cs-leading-*`、`--cs-space-*`、`--cs-radius-*`。SVG 坐标、图形比例等内容几何按实际关系设置，不必强套排版台阶；不要通过缩小字号解决溢出。

```css
.diagram-note {
	padding: var(--cs-space-3);
	color: var(--cs-color-text-muted);
	background: var(--cs-color-bg-soft);
}

.diagram-series-a {
	fill: var(--cs-series-1);
}
```

## 静态布局

按需要复用现有类，不要求每页使用容器或卡片：

- 纵向堆叠、横向排列：`.cs-stack`、`.cs-row`。
- 图示与数据并列：`.cs-split`、`.cs-split__figure`、`.cs-split__data`。
- 图例与静态比例条：`.cs-legend-row` 系列、`.infra-bar > i`。
- 状态块：`.cs-state` 与 `.cs-state--pass`、`.cs-state--warn`、`.cs-state--fail`。
- 数字与等宽文本：`.cs-num`、`.cs-mono`。
- 原生横线表格：`.cs-line-table`，窄标签列使用 `.cs-line-table__label`。

HTML 表格的结构与阅读形式按根规范选择。`base.css` 提供单元格间距和顶对齐；VitePress 的默认边框、背景与窄屏行为仍可能影响最终呈现，必要时用局部类调整。局部样式限定在当前展示范围内，避免影响整页其他表格或标题。

内联 SVG 的 `fill`、`stroke` 可直接引用语义变量：

```html
<circle stroke="var(--cs-color-border)" fill="var(--cs-color-bg-soft)" />
<text fill="var(--cs-color-text)">节点</text>
```

## 主题维护

主题由 `theme-registry.ts` 统一登记，选择结果写入 `<html data-cs-theme="...">`，浅深色由 `.dark` 区分。站点切换器和首屏初始化脚本共用注册表；文档无需读写主题状态。

新增主题时：

1. 在 `styles/themes.css` 增加浅色与深色取值，所需变量见 `scripts/check-design-tokens.mjs`（仓库根目录）。
2. 在 `theme-registry.ts` 更新 `CsThemeId` 和 `csThemes`，保持预览色 `swatch` 与该主题浅色品牌色一致。
3. 运行下述检查并验证受影响展示。

历史文档组件位于 `components/`，站点外壳位于 `system/`。`series-palette.ts`、`styles/components/controls.css` 等继续服务既有实现，不作为新文档添加交互的理由。

## 校验范围

```bash
npm run check          # 主题变量与对比度检查
npm run docs:build     # 文档编译
npm run docs:dev       # 本地渲染检查
```

`check:tokens` 扫描主题目录中的 Vue／CSS，检查硬编码颜色、越层引用和文档组件直接操作主题状态；另核对主题变量与注册表预览色。色值定义允许出现在 `palette.css`、`themes.css`，注册表中的预览色单独核验。它不扫描 Markdown 内嵌 HTML／SVG 或独立 SVG，也不检查所有字号、布局与变量用途。

`check:contrast` 对注册主题的浅深色取值进行静态计算：

- 正文、次要文字和弱化文字均按普通文字至少 4.5:1 检查；不能因为信息次要就降低门槛。图形颜色按 3:1 检查，若用于普通文字仍需单独满足 4.5:1（WCAG 2.2 SC 1.4.3、1.4.11）。
- 品牌与状态色使用 ΔE ≥ 25、数据系列色使用 ΔE ≥ 15 作为仓库内的差异检查阈值；这不代表色觉无障碍保证，颜色仍需文字或形状辅助。
- 检查范围是脚本列出的前景／背景组合，不是所有页面的实际配色。半透明前景会先与对应背景合成。

修改文档内的 HTML／SVG 时，另查样式是否局部生效、颜色是否随主题变化，以及窄屏是否溢出或文字过小。涉及已有交互时再检查焦点与键盘操作；触控尺寸可复用 `--cs-tap-target`，这是仓库选用的 44px 目标，不是所有控件的通用合规结论。构建和静态检查通过不代替这些渲染验证。
