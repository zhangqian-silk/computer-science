# 前端设计系统

本站前端样式按「主题 / 配色 / 布局 / 组件」四类关注点分层实现。核心约束只有一条：

> **组件只消费语义 token，不接触具体颜色。**

正因如此，切换主题不需要改动任何组件，新增文件也自动继承同一套视觉风格。

---

## 一、目录结构

```
.vitepress/theme/
├── index.ts                    主题入口：注册组件、挂载切换器
├── theme-registry.ts           主题注册表（切换器 / 防闪脚本 / 校验脚本共用）
├── series-palette.ts           数据系列色助手
├── styles/
│   ├── index.css               样式入口，import 顺序即层级顺序
│   ├── palette.css       ①    原始色板：只有 themes.css 可引用
│   ├── tokens.css        ②    语义 token：排版、间距、圆角、动效
│   ├── themes.css        ③    主题配色集 + 桥接到 --vp-*
│   ├── base.css          ④    基础排版：公式、表格、结构图
│   ├── layout.css        ⑤    布局原语：只管排布，不管颜色
│   └── components/       ⑥    组件样式，按关注点拆分
│       ├── lab.css             Lab 容器：卡片、标题、设问、答案区
│       ├── controls.css        表单控件：输入、选择、按钮、开关
│       └── data-display.css    数据展示：结果卡、条形图、芯片、状态色
├── components/                 Markdown 可直接书写的交互组件
└── system/                     站点外壳组件（不注册给 Markdown）
    └── ThemeSwitch.vue         配色切换器
```

分层的单向依赖关系：

```
palette.css  ──只被──▶  themes.css
                            │ 覆盖
                            ▼
                       tokens.css  ──被──▶  layout / components / *.vue
                            │ 桥接
                            ▼
                     VitePress --vp-*（站点外壳）
```

---

## 二、三层颜色模型

| 层 | 命名形态 | 例子 | 谁可以引用 |
|------|------|------|------|
| ① 原始色板 | `--cs-<色相>-<台阶>` | `--cs-blue-600` | 仅 `themes.css` |
| ② 语义 token | `--cs-color-<用途>` | `--cs-color-brand` | 组件、布局、样式文件 |
| ③ 框架变量 | `--vp-*` | `--vp-c-brand-1` | 仅 `themes.css` 桥接使用 |

**为什么组件不能直连色板**：`--cs-blue-600` 写死了「蓝」，主题一换就失真；`--cs-color-brand` 描述的是「品牌色」这一用途，四套主题各自赋值，组件无需知情。

### 语义 token 清单

**品牌**

| token | 用途 |
|------|------|
| `--cs-color-brand` | 主强调色：激活态、关键数值、图表主色 |
| `--cs-color-brand-hover` / `-active` | 悬停 / 按下态 |
| `--cs-color-brand-soft` | 强调色的浅底，用于选中背景 |
| `--cs-color-on-brand` | 覆盖在品牌色填充块之上的文字色 |

**表面与文字**

| token | 用途 |
|------|------|
| `--cs-color-bg` / `-bg-soft` / `-bg-elevated` | 基础底色 / 次级底色 / 浮层底色 |
| `--cs-color-text` / `-text-muted` / `-text-subtle` | 正文 / 次要说明 / 弱化标签 |
| `--cs-color-border` / `-border-strong` | 常规描边 / 强调描边 |
| `--cs-color-shadow` | 阴影颜色（几何部分见 `--cs-shadow-*`） |

**状态**

| token | 语义 |
|------|------|
| `--cs-color-success` / `-success-soft` | 通过、命中、健康 |
| `--cs-color-warning` / `-warning-soft` | 降级、补偿、逼近阈值 |
| `--cs-color-danger` / `-danger-soft` | 失败、拒绝、溢出 |
| `--cs-color-info` / `-info-soft` | 中性提示 |
| `--cs-color-neutral-soft` | 无语义的浅底 |

**数据系列**：`--cs-series-1` 至 `--cs-series-8`，用于「第 N 条数据用第 N 个颜色」的场景。**不要在 JS 里写 hex 数组**，改用助手函数：

```ts
import { seriesColor } from '../series-palette'

const segments = computed(() => [
	{ name: '时间戳', bits: tsBits.value, color: seriesColor(0) },
	{ name: '机房',   bits: dcBits.value, color: seriesColor(1) }
])
```

`seriesColor(n)` 返回 `var(--cs-series-N)`，自动取模，随主题变化，无需组件重渲染。

---

## 三、尺度 token

尺度不随主题变化，用于消除 `.82rem`、`.55rem` 这类近似魔法数。

**排版**：`--cs-text-3xs`(0.68) `2xs`(0.72) `xs`(0.78) `sm`(0.82) `base`(0.88) `md`(0.95) `lg`(1.05) `xl`(1.15)

**行高**：`--cs-leading-tight`(1.35) `-normal`(1.6) `-relaxed`(1.75)

**间距**（4px 基准）：`--cs-space-1`(0.25rem) 至 `--cs-space-10`(2.5rem)。`-9`(2rem) 与 `-10`(2.5rem) 为正文级留白，供段间距、章节标题上方与分隔线使用——正文段间距须**大于**行高，否则段落边界会被行内行距抹平。

**圆角**：`--cs-radius-xs`(4px) `sm`(6px) `md`(8px) `lg`(12px) `pill`(999px) `circle`(50%)

**触控与焦点**：`--cs-tap-target`(44px，WCAG 2.5.5 下限) `--cs-tap-target-lg`(48px) `--cs-focus-ring-width` `--cs-focus-ring-offset`

**动效**：`--cs-duration-fast` `-base` `--cs-ease-standard` `--cs-transition-colors`

---

## 四、一键切换主题

### 运行机制

主题由 `<html data-cs-theme="...">` 选中，与 VitePress 自带的浅色 / 深色**正交**——两者可任意组合，共 4 × 2 = 8 种组合：

```
data-cs-theme="ocean"          → [data-cs-theme='ocean'] 生效（浅色）
data-cs-theme="ocean" + .dark  → .dark[data-cs-theme='ocean'] 生效（深色）
```

切换器位于导航栏右侧（`nav-bar-content-after` 插槽），选择结果写入 `localStorage`。首屏由 `config.mts` 注入的内联脚本同步写入属性，因此不会出现「先按默认主题绘制一帧再切换」的闪烁。

### 内置主题

| id | 名称 | 强调色 | 适用场景 |
|------|------|------|------|
| `blueprint` | 蓝图 | 蓝 | 默认，贴近 VitePress 原生观感 |
| `ocean` | 深海 | 青 | 长时间阅读，视觉刺激更低 |
| `graphite` | 石墨 | 中性深灰 | 读代码与公式 |
| `sunset` | 暮色 | 紫 | 图示与数据系列对比更强 |

> `graphite` 的强调色刻意用中性灰而非绿色：绿色已承担「成功 / 通过」语义，若品牌色也用绿，同一组件里「激活」与「通过」两行会无法区分。同理，`ocean` 深色下提亮了品牌青、压深了成功绿。这类冲突由校验脚本强制拦截。

### 新增一套主题

只需两步，**不涉及任何组件改动**：

1. 在 `styles/themes.css` 追加两个块（浅色 + 深色），提供全部必需 token：

	```css
	[data-cs-theme='forest'] {
		--cs-color-brand: var(--cs-green-600);
		/* …… 其余必需 token */
	}

	.dark[data-cs-theme='forest'] {
		--cs-color-brand: var(--cs-green-400);
		/* …… */
	}
	```

2. 在 `theme-registry.ts` 的 `csThemes` 登记：

	```ts
	{ id: 'forest', label: '林间', description: '……', swatch: '#059669' }
	```

漏写 token 会静默回退到默认值，且配色冲突肉眼难辨，因此**务必运行 `npm run check:contrast`**：缺项、对比度不足与语义色撞色都会被逐条列出。

---

## 五、新增组件的约定

新组件想自动获得统一风格，遵循以下顺序即可。

### 1. 优先复用现成类，而不是新写 scoped 样式

| 需求 | 用这个类 | 出自 |
|------|------|------|
| 组件外壳（标题 + 设问 + 答案） | `<LearningLab>` 组件 | `lab.css` |
| 控件网格 | `.infra-controls` + `.infra-control` | `layout.css` / `controls.css` |
| 结果卡片 | `.infra-results` + `.infra-result` | `layout.css` / `data-display.css` |
| 按钮组 / 页签 | `.infra-tabs`、`.infra-actions` | `layout.css` / `controls.css` |
| 勾选开关 | `.cs-toggle` | `controls.css` |
| 进度条 / 负载条 | `.infra-bar` > `i` | `data-display.css` |
| 图示 + 数据并排 | `.cs-split` + `__figure` + `__data` | `layout.css` |
| 图例行（点 + 名 + 条 + 值） | `.cs-legend-row` 系列 | `data-display.css` |
| 状态块（通过 / 警告 / 失败） | `.cs-state` + `.cs-state--pass\|warn\|fail` | `data-display.css` |
| 仅左侧状态色条 | `.cs-state-bar` + 同上修饰类 | `data-display.css` |
| 芯片 | `.token-chip` + `.is-selected\|is-muted` | `data-display.css` |
| 徽标 | `.cs-badge` | `data-display.css` |
| 纵向堆叠 / 横向排列 | `.cs-stack`、`.cs-row` | `layout.css` |
| 等宽数字 / 等宽字体 | `.cs-num`、`.cs-mono` | `data-display.css` |

### 2. 确实需要 scoped 样式时，只写「本组件独有」的部分

颜色一律取语义 token，尺寸一律取台阶：

```css
/* 正确 */
.my-cell {
	padding: var(--cs-space-3);
	border-radius: var(--cs-radius-sm);
	font-size: var(--cs-text-sm);
	color: var(--cs-color-text-muted);
	background: var(--cs-color-bg-soft);
}

/* 错误：写死颜色，主题切换失效 */
.my-cell { color: #64748b; background: #f8fafc; }

/* 错误：越层引用原始色板 */
.my-cell { color: var(--cs-slate-500); }

/* 错误：直连框架变量，绕过语义层 */
.my-cell { color: var(--vp-c-text-2); }
```

### 3. SVG 图示同样走 token

`fill` / `stroke` 直接写 `var(--cs-color-*)` 或 `seriesColor(n)`：

```html
<circle stroke="var(--cs-color-border)" fill="var(--cs-color-bg-soft)" />
<text fill="var(--cs-color-on-brand)">12</text>
```

### 4. 可访问性下限

- 可点击元素 `min-height: var(--cs-tap-target)`
- 焦点态用 `controls.css` 里的统一焦点环，不要 `outline: none`
- 颜色不作为唯一信息载体，状态同时用文字或图形表达

---

## 六、校验

两个脚本都不依赖浏览器，可直接在 CI 运行：

```bash
npm run check          # 下面两项一起跑
npm run check:tokens   # 分层与 token 一致性
npm run check:contrast # 主题可读性
```

`check:tokens` 拦截四类问题：

1. **字面量色值** —— 组件与样式文件出现 hex / `rgb()` / 具名色（`palette.css` 除外）
2. **越层引用色板** —— 组件直接用 `--cs-blue-600` 等原始变量
3. **主题 token 缺失** —— 某套主题的浅色或深色块漏写必需 token
4. **绕过注册表** —— 组件自行读写 `data-cs-theme` 或主题 localStorage 键

`check:contrast` 自行解析 token 引用链（`--cs-color-brand → var(--cs-teal-700) → #0f766e`），
对 4 套主题 × 浅/深色共 8 种组合做三类检查：

1. **WCAG 对比度** —— 正文类 ≥ 4.5:1，弱化文字与图形类 ≥ 3:1；半透明前景会先与底色合成再计算
2. **语义色可辨性** —— 品牌 / 成功 / 警告 / 失败两两 ΔE ≥ 25，避免两种语义被误读为同一种
3. **数据系列色可辨性** —— 8 个系列色两两 ΔE ≥ 15

新增或调整主题后务必跑一次：漏写 token 会静默回退，对比度与语义冲突则肉眼很难发现。

再配合渲染验证：

```bash
npm run docs:build    # 构建通过
npm run docs:dev      # 逐一切换四套主题 × 浅/深色，抽查组件与图示
```

---

## 七、迁移速查

历史代码里的旧写法与新写法对照：

| 旧 | 新 |
|------|------|
| `var(--vp-c-brand-1)` | `var(--cs-color-brand)` |
| `var(--vp-c-divider)` | `var(--cs-color-border)` |
| `var(--vp-c-text-2)` / `-3` | `var(--cs-color-text-muted)` / `-subtle` |
| `var(--vp-c-bg-soft)` | `var(--cs-color-bg-soft)` |
| `var(--vp-c-green-1)` / `danger` / `warning` | `var(--cs-color-success)` / `-danger` / `-warning` |
| `var(--vp-font-family-mono)` | `var(--cs-font-mono)` |
| `#fff`（品牌块上的文字） | `var(--cs-color-on-brand)` |
| JS 里的 hex 数组 | `seriesColor(n)` |
| `font-size: .82rem` | `font-size: var(--cs-text-sm)` |
| `border-radius: 6px` | `border-radius: var(--cs-radius-sm)` |
| `gap: .5rem` | `gap: var(--cs-space-3)` |
