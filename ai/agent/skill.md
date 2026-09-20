# Agent Skills：按需加载的过程性知识包

Agent Skills（以下简称 Skill）是给 Agent 用的「入职手册」：把一个领域里反复用到的流程、规范与脚本，打包成能进版本库、可被模型按任务发现的能力包，而不是更长的 prompt。

<style>
/* Skill 文档组件：颜色全部映射到站点语义 token，随白色/深色背景与配色主题自适应 */
.skill-cards,
.skill-levels,
.skill-seq,
.skill-note,
.skill-chartbox,
.skill-bare,
.skill-tw {
	--sk-serif: "Noto Serif SC", Songti SC, SimSun, serif;
	--sk-ink: #20281f;
	--sk-ink-2: #3d4438;
	--sk-ink-3: #5c6357;
	--sk-paper: #f0ecdf;
	--sk-card: #f8f5ed;
	--sk-warn: #faf0e9;
	--sk-brass: #8f6418;
	--sk-brass-block: #a97c28;
	--sk-red: #93351f;
	--sk-steel: #6f7268;
	--sk-rule: rgba(94, 76, 36, .18);
	--sk-rule-strong: rgba(94, 76, 36, .32);
	--sk-rule-soft: rgba(94, 76, 36, .1);
	--sk-code-bg: rgba(120, 95, 40, .06);
	--sk-zebra: rgba(120, 95, 40, .05);
	--sk-ring: #ffffff;
	--sk-solid: #20281f;
	--sk-solid-2: #424a3f;
	--sk-on-solid: #f5f1e3;
	--sk-on-solid-soft: rgba(245, 241, 227, .72);
	--sk-slot: #e9e3d2;
	--sk-slot-text: #20281f;
	--sk-block-2: #4d5449;
	--sk-block-3: #d9d3c3;
	--sk-thead-bg: #20281f;
	--sk-thead-text: #f5f1e3;
	--sk-rail-bg: #20281f;
	--sk-rail-text: #c99b45;
}

.dark .dark .skill-cards,
.dark .skill-levels,
.dark .skill-seq,
.dark .dark .skill-note,
.dark .skill-chartbox,
.dark .skill-bare,
.dark .skill-tw {
	--sk-ink: #e9e7dd;
	--sk-ink-2: #c9c6b9;
	--sk-ink-3: #a8a597;
	--sk-paper: #2e2c25;
	--sk-card: #262520;
	--sk-warn: rgba(147, 53, 31, .2);
	--sk-brass: #c9a256;
	--sk-brass-block: #c9a256;
	--sk-red: #d98a76;
	--sk-steel: #9a978a;
	--sk-rule: rgba(233, 231, 221, .13);
	--sk-rule-strong: rgba(233, 231, 221, .26);
	--sk-rule-soft: rgba(233, 231, 221, .08);
	--sk-code-bg: rgba(233, 225, 200, .07);
	--sk-zebra: rgba(233, 225, 200, .045);
	--sk-ring: #1b1b1f;
	--sk-solid: #36332a;
	--sk-solid-2: #4a4639;
	--sk-on-solid: #efe9d8;
	--sk-on-solid-soft: rgba(239, 233, 216, .72);
	--sk-slot: #4c4838;
	--sk-slot-text: #f0ead8;
	--sk-block-2: #4a4639;
	--sk-block-3: #3b3830;
	--sk-thead-bg: #c9a256;
	--sk-thead-text: #201d12;
	--sk-rail-bg: #322f24;
	--sk-rail-text: #c9a256;
}

/* 卡片 */
.skill-cards { display: grid; gap: 14px; margin: 24px 0; }
.skill-cards--2 { grid-template-columns: repeat(2, minmax(0, 1fr)); }
.skill-cards--3 { grid-template-columns: repeat(3, minmax(0, 1fr)); }
.skill-card {
	background: var(--sk-card);
	border: 1px solid var(--sk-rule);
	border-radius: 2px;
	padding: 18px 19px;
}
.skill-card__t {
	font-family: var(--sk-serif);
	font-weight: 700;
	font-size: 16px;
	margin-bottom: 8px;
	display: flex;
	gap: 9px;
	align-items: baseline;
	color: var(--sk-ink);
}
.skill-card__t .i {
	font-family: var(--cs-font-mono);
	font-size: 11px;
	color: var(--sk-brass);
	font-weight: 700;
	flex: none;
}
.skill-card p { margin: 0; font-size: 14.2px; line-height: 1.68; color: var(--sk-ink-2); }

/* 三级加载：三列并列卡片 */
.skill-levels {
	display: grid;
	grid-template-columns: repeat(3, 1fr);
	gap: 12px;
	margin: 18px 0 14px;
}
.skill-level {
	background: var(--sk-card);
	border: 1px solid var(--sk-rule);
	border-top: 3px solid var(--sk-brass);
	border-radius: 4px;
	padding: 14px 16px 16px;
}
.skill-level__h {
	display: flex;
	flex-wrap: wrap;
	align-items: baseline;
	gap: 8px;
	font-family: var(--sk-serif);
	font-weight: 700;
	font-size: 15px;
	color: var(--sk-ink);
	line-height: 1.4;
}
.skill-level__n {
	font-family: var(--cs-font-mono);
	font-size: 12px;
	font-weight: 700;
	color: var(--sk-brass);
	letter-spacing: .04em;
}
.skill-level__cost {
	font-family: var(--cs-font-mono);
	font-size: 10.5px;
	color: var(--sk-ink-3);
	background: var(--sk-code-bg);
	padding: 2px 7px;
	border-radius: 999px;
	font-weight: 500;
}
.skill-level p { margin: 8px 0 0; font-size: 13px; line-height: 1.62; color: var(--sk-ink-2); }

/* 时间线：左标签 + 轴线，右侧标题/精简说明/上下文窗 */
.skill-seq { display: grid; grid-template-columns: 78px 22px 1fr; margin: 22px 0; }
.skill-seq__st {
	font-family: var(--cs-font-mono);
	font-size: 11.5px;
	color: var(--sk-steel);
	padding: 2px 0 16px;
	text-align: right;
	line-height: 1.5;
}
.skill-seq__ax { position: relative; justify-self: center; width: 22px; }
.skill-seq__ax::before {
	content: "";
	position: absolute;
	left: calc(50% - 1px);
	top: 0;
	bottom: 0;
	width: 2px;
	background: var(--sk-rule);
}
.skill-seq__dot {
	position: absolute;
	top: 7px;
	left: 50%;
	transform: translateX(-50%);
	width: 9px;
	height: 9px;
	border-radius: 50%;
	background: var(--sk-brass);
	box-shadow: 0 0 0 3px var(--sk-ring);
}
.skill-seq__ct { padding: 0 0 16px 4px; }
.skill-seq__h { font-weight: 700; font-size: 14.5px; color: var(--sk-ink); line-height: 1.5; }
.skill-seq__win {
	margin-top: 7px;
	font-family: var(--cs-font-mono);
	font-size: 11.5px;
	color: var(--sk-steel);
	background: var(--sk-code-bg);
	border: 1px solid var(--sk-rule-soft);
	padding: 6px 10px;
	line-height: 1.6;
	border-radius: 3px;
	overflow-x: auto;
	white-space: nowrap;
}

/* 提示块 */
.skill-note {
	background: var(--sk-card);
	border: 1px solid var(--sk-rule);
	border-left: 3px solid var(--sk-brass);
	padding: 18px 20px;
	margin: 22px 0;
	font-size: 15px;
	line-height: 1.72;
	border-radius: 2px;
	color: var(--sk-ink-2);
}
.skill-note__tag {
	font-family: var(--cs-font-mono);
	font-size: 10.5px;
	letter-spacing: .14em;
	color: var(--sk-brass);
	display: block;
	margin-bottom: 8px;
}
.skill-note--warn { background: var(--sk-warn); border-left-color: var(--sk-red); }
.skill-note--warn .skill-note__tag { color: var(--sk-red); }

/* 图表盒 */
.skill-chartbox {
	background: var(--sk-card);
	border: 1px solid var(--sk-rule);
	padding: 20px 20px 14px;
	margin: 26px 0;
	border-radius: 2px;
}
.skill-chartbox__h { font-weight: 700; font-size: 15px; margin-bottom: 3px; color: var(--sk-ink); }
.skill-chartbox__s {
	font-size: 12.8px;
	color: var(--sk-steel);
	line-height: 1.6;
	margin-bottom: 14px;
}
.skill-chartbox svg { width: 100%; height: auto; display: block; }
.skill-bare svg { width: 100%; height: auto; display: block; }
.skill-bare { margin: 18px 0; }
.skill-chartbox__f {
	font-size: 12px;
	color: var(--sk-steel);
	line-height: 1.6;
	margin-top: 10px;
	padding-top: 10px;
	border-top: 1px solid var(--sk-rule-soft);
}
.skill-chartbox__f b { color: var(--sk-ink-2); }

/* 表格：深色表头条 + 斑马纹 */
.skill-tw { overflow-x: auto; -webkit-overflow-scrolling: touch; margin: 22px 0; }
.vp-doc .skill-tw table {
	margin: 0;
	min-width: 620px;
	width: 100%;
	border-collapse: collapse;
	font-size: 14.2px;
	line-height: 1.6;
}
.vp-doc .skill-tw table th,
.vp-doc .skill-tw table td {
	padding: 11px 13px;
	text-align: left;
	vertical-align: top;
	border: none;
	line-height: 1.6;
}
.vp-doc .skill-tw table thead th {
	background: var(--sk-thead-bg);
	color: var(--sk-thead-text);
	font-family: var(--vp-font-family-base);
	font-weight: 600;
	font-size: 12.5px;
	letter-spacing: .03em;
	white-space: normal;
	border-bottom: none;
}
.vp-doc .skill-tw table tbody td {
	border-bottom: 1px solid var(--sk-rule);
	color: var(--sk-ink-2);
}
.vp-doc .skill-tw table tbody tr:nth-child(even) { background: var(--sk-zebra); }

/* 六产品对照表：百分比列宽，始终填满内容区，窄屏横向滚动 */
.vp-doc .skill-tw--products table { width: 100%; min-width: 640px; table-layout: fixed; }
.vp-doc .skill-tw--products th:nth-child(1),
.vp-doc .skill-tw--products td:nth-child(1) { width: 12%; }
.vp-doc .skill-tw--products th:nth-child(2),
.vp-doc .skill-tw--products td:nth-child(2) { width: 24%; }
.vp-doc .skill-tw--products th:nth-child(3),
.vp-doc .skill-tw--products td:nth-child(3) { width: 20%; }
.vp-doc .skill-tw--products th:nth-child(4),
.vp-doc .skill-tw--products td:nth-child(4) { width: 21%; }
.vp-doc .skill-tw--products th:nth-child(5),
.vp-doc .skill-tw--products td:nth-child(5) { width: 23%; }
.vp-doc .skill-tw--products td { word-break: break-word; overflow-wrap: anywhere; line-height: 1.7; }
.vp-doc .skill-tw--products td:nth-child(1) { word-break: keep-all; overflow-wrap: normal; }


/* 章节编号角标 */
.vp-doc h2 .skill-sec-n {
	display: inline-block;
	font-family: var(--cs-font-mono);
	font-size: 12px;
	font-weight: 700;
	color: #a97c28;
	border: 1px solid #a97c28;
	padding: 4px 8px;
	border-radius: 2px;
	line-height: 1;
	margin-right: 10px;
	letter-spacing: .1em;
}
.dark .vp-doc h2 .skill-sec-n { color: #c99b45; border-color: #c99b45; }

@media (max-width: 640px) {
	.skill-cards--2,
	.skill-cards--3 { grid-template-columns: 1fr; }
	.skill-levels { grid-template-columns: 1fr; gap: 10px; }
	.skill-seq { grid-template-columns: 62px 20px 1fr; }
	.skill-seq__ct code { white-space: normal; }
}
</style>

---

## <span class="skill-sec-n">01</span> Skill 的定位

模型能力越强，越缺的不是智力，而是组织的过程性知识：一件事分几步做、先查什么、用哪个模板、什么情况要停下来问人。先看一个最小的 Skill 长什么样。

### 最小形态：一个 Markdown 文件

一个合法的 Skill 只有一个 Markdown 文件：开头的 name、description，加上正文。

```markdown
---
name: pdf-processing
description: 处理 PDF 文件。当用户要提取字段、填表或合并 PDF 时使用。
---

# PDF 处理

1. 先跑 scripts/extract_fields.py 拿到结构化字段，不要凭模型读 PDF
2. 填表用 assets/ 里的模板
3. 金额、期限等字段必须回读原文核对
```

放进 `~/.agents/skills/pdf-processing/SKILL.md` 就生效了，不需要注册、不需要写代码。

### 触发与加载

关键不在文件本身，而在于 Agent 对它的感知分为两个阶段：

- **平时**：Agent 的系统提示里只有全部 Skill 的 name + description（每个约 100 tokens），正文并不在上下文里。description 是一句路由说明，模型据此判断「这个任务和我有没有关」。
- **命中时**：用户提出「帮我把这几个 PDF 的金额抽出来」，模型读到 description 与任务相关，才自己发起一次文件读取，把上面这份正文读进上下文，然后按步骤执行。

概括地说，**正文不常驻上下文，仅在 description 触发后才加载**：未命中的 Skill 只消耗一行索引的开销，命中后才读入完整流程；脚本与长文档等更大的资源遵循同一规则，由正文显式引用后才读取或执行（详见第 03 节）。

### 与其他载体的对比

同样一段流程知识，放在别处都会缺一块。下表从提供什么、是否常驻、能否含代码等维度把六种载体并排。

<div class="skill-tw">

<table>
<thead>
<tr><th></th><th>Skill</th><th>Prompt</th><th>Knowledge</th><th>Subagent</th><th>Tool / MCP</th></tr>
</thead>
<tbody>
<tr><td>提供</td><td>过程性知识</td><td>此刻的指令</td><td>背景事实</td><td>任务委派</td><td>连接与动作</td></tr>
<tr><td>持久性</td><td>跨会话</td><td>单次会话</td><td>项目内</td><td>跨会话</td><td>持续连接</td></tr>
<tr><td>内容</td><td>指令+代码+资源</td><td>自然语言</td><td>文档+上下文</td><td>完整 agent 逻辑</td><td>工具定义</td></tr>
<tr><td>加载方式</td><td>显式指定，按需加载</td><td>每一轮</td><td>向量召回</td><td>被调用时</td><td>始终可用</td></tr>
<tr><td>能含代码</td><td>能</td><td>不能</td><td>不能</td><td>能</td><td>能</td></tr>
<tr><td>最适合</td><td>可复用的流程</td><td>临时请求</td><td>事实与背景问答</td><td>隔离型子任务</td><td>数据访问</td></tr>
</tbody>
</table>

</div>

Skill 与 Knowledge 容易被混淆，但二者并非替代关系，而是按知识类型分工：**Knowledge 承载陈述性知识（What），Skill 承载过程性知识（How）**。区别在取数方式：Knowledge 采用**召回策略**，由模型根据语义相似度在向量库中检索，能否命中、命中哪一段都存在不确定性，适合回答事实性问题；Skill 采用**显式寻址**，正文指定读取哪个 reference 即读取哪个，路径确定、可 diff、可 review，适合要求每次都读到同一份正确文档的流程性内容。二者也可配合使用：Skill 正文可以指示「先检索最新口径，再按以下步骤处理」，流程依据 Skill，事实依据 Knowledge。

Skill 也不是第五种平行载体，而是把这些属性拆给同一个包里的不同部分：跨会话复用来自版本库里的 SKILL.md；不常驻来自 description 触发的按需加载；确定性来自显式文件寻址；可执行来自随包脚本。

### 选型路径

<div class="skill-chartbox">
<svg viewBox="0 0 860 400" role="img" aria-label="选型决策流程图：从需求出发依次判断是否一次性、缺接口还是缺流程、中间产物是否很多，最终落到 Skill">
<defs>
<marker id="sk-ar" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto">
<path d="M0,0 L10,5 L0,10 z" fill="var(--sk-steel)"/>
</marker>
<style>
.sk-bx { fill: var(--sk-card); stroke: var(--sk-rule-strong); stroke-width: 1.2; }
.sk-bxk { fill: var(--sk-solid); stroke: var(--sk-solid); }
.sk-bxa { fill: var(--sk-card); stroke: var(--sk-brass); stroke-width: 1.6; }
.sk-tx { font-family: var(--vp-font-family-base); font-size: 13px; fill: var(--sk-ink); }
.sk-tw2 { font-family: var(--vp-font-family-base); font-size: 13px; fill: var(--sk-on-solid); font-weight: 500; }
.sk-ta { font-family: var(--vp-font-family-base); font-size: 13px; fill: var(--sk-brass); font-weight: 700; }
.sk-tl2 { font-family: var(--cs-font-mono); font-size: 10.5px; fill: var(--sk-steel); }
.sk-ln2 { stroke: var(--sk-steel); stroke-width: 1.1; fill: none; marker-end: url(#sk-ar); }
</style>
</defs>
<rect class="sk-bxk" x="8" y="14" width="150" height="40" rx="2"/>
<text class="sk-tw2" x="83" y="39" text-anchor="middle">出现一个需求</text>

<path class="sk-ln2" d="M83 54 L83 84"/>
<rect class="sk-bx" x="8" y="84" width="150" height="46" rx="2"/>
<text class="sk-tx" x="83" y="104" text-anchor="middle">只做这一次？</text>
<text class="sk-tl2" x="83" y="121" text-anchor="middle">是否会反复出现</text>

<path class="sk-ln2" d="M158 107 L262 107"/>
<text class="sk-tl2" x="205" y="100" text-anchor="middle">是</text>
<rect class="sk-bx" x="262" y="86" width="130" height="42" rx="2"/>
<text class="sk-tx" x="327" y="112" text-anchor="middle">写 Prompt</text>

<path class="sk-ln2" d="M83 130 L83 162"/>
<text class="sk-tl2" x="98" y="150">否</text>
<rect class="sk-bx" x="8" y="162" width="150" height="52" rx="2"/>
<text class="sk-tx" x="83" y="182" text-anchor="middle">缺的是「接口」</text>
<text class="sk-tx" x="83" y="200" text-anchor="middle">还是「流程」？</text>

<path class="sk-ln2" d="M158 188 L262 188"/>
<text class="sk-tl2" x="205" y="181" text-anchor="middle">接口</text>
<rect class="sk-bx" x="262" y="167" width="180" height="42" rx="2"/>
<text class="sk-tx" x="352" y="193" text-anchor="middle">接 MCP / 注册 Tool</text>
<path class="sk-ln2" d="M442 188 L520 188"/>
<text class="sk-tl2" x="481" y="181" text-anchor="middle">之后</text>

<path class="sk-ln2" d="M83 214 L83 248"/>
<text class="sk-tl2" x="98" y="236">流程</text>
<rect class="sk-bx" x="8" y="248" width="150" height="52" rx="2"/>
<text class="sk-tx" x="83" y="268" text-anchor="middle">中间产物很多、</text>
<text class="sk-tx" x="83" y="286" text-anchor="middle">你不会再看？</text>

<path class="sk-ln2" d="M158 274 L262 274"/>
<text class="sk-tl2" x="205" y="267" text-anchor="middle">是</text>
<rect class="sk-bx" x="262" y="253" width="180" height="42" rx="2"/>
<text class="sk-tx" x="352" y="279" text-anchor="middle">交给 Subagent</text>
<text class="sk-tl2" x="352" y="311" text-anchor="middle">独立上下文，只回摘要</text>

<path class="sk-ln2" d="M83 300 L83 336"/>
<text class="sk-tl2" x="98" y="322">否</text>
<rect class="sk-bxa" x="8" y="336" width="150" height="46" rx="2"/>
<text class="sk-ta" x="83" y="365" text-anchor="middle">做成 Skill</text>

<path d="M520 188 L520 336 L158 336" fill="none" stroke="var(--sk-steel)" stroke-width="1.1"/>
<rect class="sk-bxa" x="520" y="300" width="322" height="82" rx="2" opacity=".55"/>
<text class="sk-tx" x="536" y="326">Skill 编排这些工具：写清什么场景调哪个、</text>
<text class="sk-tx" x="536" y="346">参数怎么填、失败怎么退、产物交给谁。</text>
<text class="sk-tx" x="536" y="366">工具是能力，Skill 是使用说明书。</text>

<line x1="470" y1="14" x2="470" y2="120" stroke="var(--sk-rule)" stroke-opacity=".14" stroke-width="1"/>
<text class="sk-tl2" x="520" y="30">判据 · Skill 还是 Subagent</text>
<text class="sk-tx" x="520" y="54" font-size="12.5">噪音输出 &gt; 1k tokens、需要工具权限隔离、</text>
<text class="sk-tx" x="520" y="72" font-size="12.5">需要并行 → Subagent</text>
<text class="sk-tx" x="520" y="96" font-size="12.5">需要共享主对话上下文、只是流程与规范</text>
<text class="sk-tx" x="520" y="114" font-size="12.5">→ Skill（Subagent 内部也能用 Skill）</text>
</svg>
</div>

四种选择并不互斥：接入 MCP 之后仍需编写流程说明，Subagent 内部同样可以加载 Skill。Skill 补足的是其余三者都未覆盖的部分——可复用、不常驻、且需要共享主对话上下文的流程；真正全局、每轮都需要的约束应留在规则文件，一次性要求则直接以口头提示给出。

### 它们之间的关系

上表按维度做了静态对比，这里再补两点它没表达的关系：

- **Skill 与 MCP / tool 处于正交的两层**：工具提供原子动作（发消息、读数据库），Skill 提供「何种场景、按何种顺序」的编排规则。仅有 MCP 时，Agent 缺乏工具组合的依据；仅有 Skill 而无 MCP 时，则只能操作文件系统与 shell。前者决定「能做什么」，后者决定「怎么做」，通常配合使用。
- **Skill 与 Subagent 处于嵌套的两层**：Subagent 在独立上下文中执行并只回传摘要，Skill 在当前上下文中替换指令集；Subagent 内部同样可以加载 Skill，从而让同一份专长被多个 Agent 复用。


---

## <span class="skill-sec-n">02</span> 格式规范

上一节的最小文件即可运行，但 Skill 规范刻意保持极简，以保证可移植性。本节说明字段约束，以及规模扩大后三个约定目录的组织方式。其中仅 `name` 与 `description` 为必填字段。

### 目录结构

```text
# 最小形态：只要一个 SKILL.md，就是一个合法 Skill
pdf-processing/
└── SKILL.md              # 必需：YAML frontmatter + Markdown 正文

# 完整形态：三个约定目录，按用途分开
pdf-processing/
├── SKILL.md              # 路由 + 主干工作流 + 硬约束
├── scripts/              # 可执行代码：确定性、重复性操作
│   ├── extract_fields.py
│   └── validate.js
├── references/           # 按需读入上下文的文档
│   ├── REFERENCE.md
│   └── forms.md
└── assets/               # 进入产物的资源：模板、字体、图标、schema
    └── template.docx
```

三个目录是约定而非强制。区分它们的意义在于让 Agent 一眼知道：这个文件是拿去执行的、拿去读的，还是放进产物的。

### frontmatter 字段约束

<div class="skill-tw">

<table>
<thead>
<tr><th>字段</th><th>必填</th><th>约束</th><th>要点</th></tr>
</thead>
<tbody>
<tr><td><code>name</code></td><td>是</td><td>1–64 字符；仅小写字母、数字、连字符；不能以连字符开头或结尾；不能出现连续连字符；必须与父目录同名</td><td>用动名词式命名：processing-pdfs、analyzing-spreadsheets；避免 helper、utils 这类无信息量的名字</td></tr>
<tr><td><code>description</code></td><td>是</td><td>1–1024 字符，非空</td><td>决定触发精度的唯一字段：做什么 + 什么时候用 + 触发词/别名 + 排除边界</td></tr>
<tr><td><code>license</code></td><td>否</td><td>许可证名，或指向捆绑的许可证文件</td><td>对外分发时补上</td></tr>
<tr><td><code>compatibility</code></td><td>否</td><td>≤ 500 字符</td><td>只在真有环境依赖时写：目标产品、系统包、网络需求。多数 Skill 不需要</td></tr>
<tr><td><code>metadata</code></td><td>否</td><td>字符串键值映射</td><td>客户端自定义扩展位，常放 author、version、tags；键名取得独特些以免冲突</td></tr>
<tr><td><code>allowed-tools</code></td><td>否</td><td>空格分隔的工具名（实验性）</td><td>预批准该 Skill 可用的工具，各实现支持度不一致，别当成安全边界依赖</td></tr>
</tbody>
</table>

</div>

```yaml
name: pdf-processing
description: Extracts text and tables from PDF files, fills PDF forms,
  and merges multiple PDFs. Use when working with PDF documents or when
  the user mentions PDFs, forms, or document extraction.
license: Apache-2.0
compatibility: Requires Python 3.12+, pdfplumber and pypdf
metadata:
  author: platform-infra
  version: "1.2"
allowed-tools: Bash(git:*) Bash(jq:*) Read
```

非法 `name` 示例：`PDF-Processing`（含大写）、`-pdf`（首字符为连字符）、`pdf--processing`（含连续连字符）。失效的 `description` 示例：`"Helps with PDFs."`——该触发时无法触发。

### 正文与引用

- 正文激活后整个读入上下文，长度就是成本，建议 < 500 行 / < 5k tokens。
- 以命令式撰写分步指令、输入输出示例与边界情况，并说明原因；模型理解理由后才能正确处理未显式覆盖的情况。
- 引用用相对于 Skill 根的路径，只保持一层深；避免 A 引 B、B 引 C 的深层链条。
- 超过 300 行的 reference 文件在开头加目录，方便 Agent 只读需要的段落。

### 作用域与加载位置（以 Codex 为例）

<div class="skill-tw">

<table>
<thead>
<tr><th>作用域</th><th>路径</th><th>生效范围</th></tr>
</thead>
<tbody>
<tr><td>系统级</td><td><code>/etc/codex/skills/&lt;name&gt;/SKILL.md</code></td><td>该机器所有用户，由管理员下发</td></tr>
<tr><td>个人级</td><td><code>~/.agents/skills/&lt;name&gt;/SKILL.md</code></td><td>本机所有项目，随个人习惯</td></tr>
<tr><td>项目级</td><td><code>&lt;repo&gt;/.agents/skills/&lt;name&gt;/SKILL.md</code></td><td>该仓库；提交进 git 团队共享，Codex 还会向上遍历父目录</td></tr>
</tbody>
</table>

</div>

<div class="skill-note">
<span class="skill-note__tag">易错点</span>
目录名即 Skill 名，与 frontmatter 的 <code>name</code> 必须一致。Codex 遇到不同来源的同名 Skill 不会合并，它们会一起出现在选择器里，由人或模型分辨，不要依赖同名覆盖来做「重载」。
</div>

---

## <span class="skill-sec-n">03</span> 渐进式披露

渐进式披露是 Skill 的核心机制：Agent 启动时只把每个 Skill 的 name + description 放进系统提示，判定与当前任务相关后才读正文，正文点到的参考文件与脚本再按需加载。同一份知识因此分成常驻元数据、触发加载、按需读取三层。

### 三级加载

<div class="skill-levels">

<div class="skill-level">
<div class="skill-level__h"><span class="skill-level__n">L1</span>元数据<span class="skill-level__cost">~100 tokens / Skill</span></div>
<p>会话启动时无条件预载全部 name + description，相当于一份能力清单：Agent 知道自身具备哪些能力，但这些元数据不占用工作记忆。</p>
</div>

<div class="skill-level">
<div class="skill-level__h"><span class="skill-level__n">L2</span>SKILL.md 正文<span class="skill-level__cost">&lt; 5k tokens，≤ 500 行</span></div>
<p>语义匹配命中或用户显式指定时，模型自己读取正文，相当于翻到手册的那一章。</p>
</div>

<div class="skill-level">
<div class="skill-level__h"><span class="skill-level__n">L3</span>捆绑资源<span class="skill-level__cost">无上限</span></div>
<p>正文点名的 references 按需读；scripts 直接执行、不读入上下文，大文件不进预算，只有 stdout 回来。</p>
</div>

</div>

这里不存在隐藏的检索层：L1 是系统提示中的一段文本，L2/L3 则是模型主动发起的普通文件读取或命令执行。由此引出两个前提：其一，该机制依赖 Agent 具备文件系统访问与代码执行能力，在纯对话产品中 Skill 会退化为「长 prompt」；其二，触发依赖模型基于 description 的软路由，存在漏触发与误触发，需要通过评估集加以控制。

### 上下文窗口的实际演进

<div class="skill-seq">

<div class="skill-seq__st">T0 · 启动</div>
<div class="skill-seq__ax"><span class="skill-seq__dot"></span></div>
<div class="skill-seq__ct">
<div class="skill-seq__h">系统提示 + 全部 Skill 元数据 + 用户消息</div>
<div class="skill-seq__win">context = core prompt + N × (name+description) + user msg</div>
</div>

<div class="skill-seq__st">T1 · 触发</div>
<div class="skill-seq__ax"><span class="skill-seq__dot"></span></div>
<div class="skill-seq__ct">
<div class="skill-seq__h">模型调用 Bash 读取 pdf/SKILL.md</div>
<div class="skill-seq__win">context += SKILL.md body (~3–5k)</div>
</div>

<div class="skill-seq__st">T2 · 下钻</div>
<div class="skill-seq__ax"><span class="skill-seq__dot"></span></div>
<div class="skill-seq__ct">
<div class="skill-seq__h">按正文指引读 forms.md</div>
<div class="skill-seq__win">context += forms.md（其余 reference 不产生开销）</div>
</div>

<div class="skill-seq__st">T3 · 执行</div>
<div class="skill-seq__ax"><span class="skill-seq__dot"></span></div>
<div class="skill-seq__ct">
<div class="skill-seq__h">运行 scripts/extract_fields.py，继续任务</div>
<div class="skill-seq__win">context += stdout only</div>
</div>

</div>

### 上下文开销：全量预置 vs 渐进式披露

横轴为已安装 Skill 数量。全量预置把每个 Skill 的完整内容写进系统提示；渐进式披露只常驻元数据，实际只加载本次用到的正文与参考文件。

<div class="skill-bare">
<svg viewBox="0 0 860 360" role="img" aria-label="上下文开销对比折线图：全量预置与渐进式披露在 5 到 80 个 Skill 时的 token 开销，线性数轴">
<defs>
<style>
.sk-t { font-family: var(--vp-font-family-base); font-size: 11.5px; fill: var(--sk-steel); }
.sk-tl { font-family: var(--cs-font-mono); font-size: 10.5px; fill: var(--sk-steel); }
.sk-gl { stroke: var(--sk-rule); stroke-width: 1; }
.sk-ln-r { stroke: var(--sk-red); stroke-width: 2.4; fill: none; }
.sk-ln-b { stroke: var(--sk-brass); stroke-width: 2.4; fill: none; }
.sk-pt-r { fill: var(--sk-red); }
.sk-pt-b { fill: var(--sk-brass); }
.sk-lb-r { font-family: var(--cs-font-mono); font-size: 10.5px; fill: var(--sk-red); }
.sk-lb-b { font-family: var(--cs-font-mono); font-size: 10.5px; fill: var(--sk-brass); }
.sk-ref { stroke: var(--sk-steel); stroke-width: 1; stroke-dasharray: 5 4; }
.sk-area { fill: var(--sk-brass); opacity: .08; }
</style>
</defs>
<!-- grid: 0 / 200k / 400k / 600k / 800k / 1000k（线性） -->
<line class="sk-gl" x1="60" y1="320" x2="820" y2="320"/>
<line class="sk-gl" x1="60" y1="264" x2="820" y2="264"/>
<line class="sk-gl" x1="60" y1="208" x2="820" y2="208"/>
<line class="sk-gl" x1="60" y1="152" x2="820" y2="152"/>
<line class="sk-gl" x1="60" y1="96" x2="820" y2="96"/>
<line class="sk-gl" x1="60" y1="40" x2="820" y2="40"/>
<text class="sk-tl" x="56" y="324" text-anchor="end">0</text>
<text class="sk-tl" x="56" y="268" text-anchor="end">200k</text>
<text class="sk-tl" x="56" y="212" text-anchor="end">400k</text>
<text class="sk-tl" x="56" y="156" text-anchor="end">600k</text>
<text class="sk-tl" x="56" y="100" text-anchor="end">800k</text>
<text class="sk-tl" x="56" y="44" text-anchor="end">1000k</text>
<!-- 200k reference line -->
<line class="sk-ref" x1="60" y1="264" x2="820" y2="264"/>
<text class="sk-t" x="816" y="258" text-anchor="end">200k 上下文窗口</text>
<!-- x labels -->
<text class="sk-t" x="60" y="342" text-anchor="middle">5 个</text>
<text class="sk-t" x="250" y="342" text-anchor="middle">10 个</text>
<text class="sk-t" x="440" y="342" text-anchor="middle">20 个</text>
<text class="sk-t" x="630" y="342" text-anchor="middle">40 个</text>
<text class="sk-t" x="820" y="342" text-anchor="middle">80 个</text>
<!-- full preload: 63k,126k,252k,504k,1008k -->
<path class="sk-ln-r" d="M60,302.4 L250,284.7 L440,249.4 L630,178.9 L820,37.8"/>
<circle class="sk-pt-r" cx="60" cy="302.4" r="4.5"/>
<circle class="sk-pt-r" cx="250" cy="284.7" r="4.5"/>
<circle class="sk-pt-r" cx="440" cy="249.4" r="4.5"/>
<circle class="sk-pt-r" cx="630" cy="178.9" r="4.5"/>
<circle class="sk-pt-r" cx="820" cy="37.8" r="4.5"/>
<text class="sk-lb-r" x="60" y="292.4" text-anchor="middle">63k</text>
<text class="sk-lb-r" x="250" y="274.7" text-anchor="middle">126k</text>
<text class="sk-lb-r" x="440" y="239.4" text-anchor="middle">252k</text>
<text class="sk-lb-r" x="630" y="168.9" text-anchor="middle">504k</text>
<text class="sk-lb-r" x="820" y="27.8" text-anchor="middle">1008k</text>
<!-- progressive: 8.0k,8.5k,9.5k,11.5k,15.5k -->
<path class="sk-area" d="M60,317.8 L250,317.6 L440,317.3 L630,316.8 L820,315.7 L820,320 L60,320 Z"/>
<path class="sk-ln-b" d="M60,317.8 L250,317.6 L440,317.3 L630,316.8 L820,315.7"/>
<circle class="sk-pt-b" cx="60" cy="317.8" r="4.5"/>
<circle class="sk-pt-b" cx="250" cy="317.6" r="4.5"/>
<circle class="sk-pt-b" cx="440" cy="317.3" r="4.5"/>
<circle class="sk-pt-b" cx="630" cy="316.8" r="4.5"/>
<circle class="sk-pt-b" cx="820" cy="315.7" r="4.5"/>
<text class="sk-lb-b" x="60" y="309.8" text-anchor="middle">8.0k</text>
<text class="sk-lb-b" x="250" y="309.6" text-anchor="middle">8.5k</text>
<text class="sk-lb-b" x="440" y="309.3" text-anchor="middle">9.5k</text>
<text class="sk-lb-b" x="630" y="308.8" text-anchor="middle">11.5k</text>
<text class="sk-lb-b" x="820" y="307.7" text-anchor="middle">15.5k</text>
<!-- legend -->
<line x1="60" y1="14" x2="80" y2="14" stroke="var(--sk-red)" stroke-width="2.4"/>
<circle cx="70" cy="14" r="3.5" fill="var(--sk-red)"/>
<text class="sk-t" x="86" y="18">全量预置进 system prompt</text>
<line x1="250" y1="14" x2="270" y2="14" stroke="var(--sk-brass)" stroke-width="2.4"/>
<circle cx="260" cy="14" r="3.5" fill="var(--sk-brass)"/>
<text class="sk-t" x="276" y="18">渐进式披露（元数据常驻 + 本次实读）</text>
</svg>
</div>

**口径（示意模型，非实测）**：单个 Skill 完整包按 12,600 tokens 估算（元数据 100 + 正文 4,500 + 参考资料 8,000）；渐进式披露 = Skill 数 × 100 + 本次实际读取 7,500。上下文窗口按 200k 画参考线。按此口径，全量预置在第 16 个 Skill 就已撑满 200k 窗口；80 个 Skill 时两种方式差约 65 倍。

### 代价

这套机制也有四个代价。其一，软路由依赖模型判断，天然存在漏触发与误触发，需要评估集持续度量。其二，它依赖运行环境提供文件系统访问与代码执行能力；在纯对话产品中，Skill 只能作为一段常驻提示存在，退化为「长 prompt」。其三，写入的内容会随模型换代而过期，针对旧缺陷编写的规则可能不再有效，需要定期重审。其四，Skill 以宿主 Agent 的全部权限执行，指令注入与恶意脚本都在其攻击面内，必须由独立的沙箱与审批机制约束。

---


## <span class="skill-sec-n">04</span> Skill 的编写

构建 Skill 的正确起点不是写文档，而是先让 Agent 跑一批代表性任务，观察它在哪里受阻、缺什么上下文，再针对缺口增量补充。Skill 正文应承载四类内容，其余尽量移出：

- **编排顺序与判断条件**：先执行什么、按什么结果分支、失败时如何处理。工具本身（读文件、发请求）是通用的，但「在这个业务里按什么顺序、什么条件使用它们」是 Skill 的核心，即使所有工具都是通用的，这部分仍需写明。
- **仓库与团队的私有信息**：目录结构、部署顺序、内部系统的操作方式、团队选定的规范。这些信息不在训练数据中，模型无法自行获得。
- **输入输出约定与边界情况**：产物必须包含哪些字段、金额等关键字段必须回读原文、超出适用范围时如何处理。
- **确定性操作的调用方式**：排序、抽取、校验、数值计算等步骤不在正文中描述实现细节，而是指明调用哪个脚本；正文只保留调用顺序与对结果的判断。

两类内容应尽量避免：一是针对当前模型具体缺陷的补丁式规则，这类缺陷可能随模型换代修复，规则会随之失效，需要定期重审；二是与业务无关的通用方法教学（例如一般性的代码审查流程），模型通常已掌握，除非团队在多种做法中明确选定了一种，此时记录的也是选择本身。

正文应保持简短并突出可执行性：31 行与 291 行的 Skill 在对照实验中效果相同，但前者 token 消耗更低。具体的组织方式——分层路由、脚本分工、正文骨架——见以下各节。

### 官方四条原则

- **P1 从评估出发**：先让 Agent 执行一批代表性任务，观察其受阻位置与上下文缺口，再针对缺口增量构建 Skill，避免脱离实际地预先编写大量文档。
- **P2 为规模而拆分**：正文臃肿时应拆分为独立文件并在正文中引用；互斥或极少同时使用的路径应分离存放，以实际降低 token 消耗。
- **P3 从模型视角审视**：观察真实执行轨迹中是否出现预期外路径、是否过度依赖某段上下文；尤其关注决定触发的 name 与 description。
- **P4 与模型协同迭代**：让 Agent 将成功做法与失败教训写回 Skill，在偏离时自行复盘，从而识别其真正需要的上下文。

### 分层架构：正文退化成路由表

复杂 Skill 的正文只负责判定当前情况所属的分支，再指向对应的 reference 或 script，从而将「已读但未使用」的 token 降至最低。

```text
contract-review/
├── SKILL.md                  ① 路由层：判定合同类型 → 指向分支；写死硬约束与红线
├── references/
│   ├── INDEX.md              ② 大目录：>300 行的参考文件先给目录
│   ├── nda.md                ③ 互斥分支：一次任务通常只读其中一个
│   ├── saas-msa.md
│   ├── procurement.md
│   └── clause-library.md     ④ 长尾条款库：命中特定条款时才读
├── scripts/
│   ├── extract_clauses.py    ⑤ 确定性抽取：不读入上下文，只跑，回 JSON
│   ├── diff_against_std.py   ⑥ 与标准模板比对：结构化差异，不靠模型逐字比
│   └── validate_output.py    ⑦ 自检：交付前跑，会失败的动作才是有效验证
├── assets/
│   ├── report-template.docx  ⑧ 进入产物的模板
│   └── risk-matrix.json      ⑨ 查找表：脚本读，不进上下文
└── evals/
    └── cases.yaml            ⑩ 评估集：应触发 / 不应触发 / 端到端结果断言
```

### SKILL.md 正文的推荐骨架

```markdown
---
frontmatter：name / description 决定「能不能被用上」
---

# 合同审查

## 适用范围
能做什么、不能做什么。超出范围就如实告知，不假装实现。

## 关键步骤            ← 主干工作流，命令式，带 why
1. 先跑 scripts/extract_clauses.py 拿到结构化条款
2. 按合同类型读对应 reference：NDA → references/nda.md
   为什么：三类合同的风险点互斥，混读会稀释注意力

## 硬约束              ← 最容易被忽略，所以要前置且可核对
- 金额、期限、责任上限必须逐条回读原文，禁止凭印象
- 未能判定的条款标注「待人工确认」，禁止编造结论

## 交付前自检          ← 用不同于生成路径的方式回读
python3 <SKILL_DIR>/scripts/validate_output.py <报告路径>

## 扩展参考            ← 路由指引，写清「什么时候读哪个」
- 遇到数据跨境 / 竞业限制条款时读 references/clause-library.md
```

### 指令与脚本的分工

<div class="skill-tw">

<table>
<thead>
<tr><th>操作特征</th><th>交给谁</th><th>原因</th></tr>
</thead>
<tbody>
<tr><td>排序、统计、格式转换、字段抽取、校验</td><td>脚本</td><td>用 token 生成既贵又不稳；代码可复现、可测试</td></tr>
<tr><td>合计 / 换算 / 比例</td><td>脚本或计算工具</td><td>数字必须来自可复现计算，不能目测</td></tr>
<tr><td>判断属于哪类、如何取舍、语气与措辞</td><td>模型</td><td>需要语义理解与权衡，写死会失去适应性</td></tr>
<tr><td>模板套用 + 少量填空</td><td>assets 模板 + 模型</td><td>结构固定的部分不该每次重新生成</td></tr>
</tbody>
</table>

</div>

<div class="skill-note">
<span class="skill-note__tag">要点</span>
正文必须写清一个脚本是直接跑还是读进来当参考。含糊不清时模型会去读脚本源码，白花 token。
</div>

### description 的写法

该字段决定是否触发，所有「何时使用」的信息都应写在此处；正文只有在触发后才会被读取。

```yaml
# ✗ 弱：太泛，该触发时不触发
description: "Helps with PDFs."

# ✓ 强结构：做什么 + 何时用（含触发词与别名）+ 排除边界
description: 分析 Figma 设计文件并生成开发交付文档。当用户上传 .fig 文件、
  要求「设计规范」「组件文档」「设计稿还原」「design-to-code 交接」时使用。
  即使用户没有明确说「交付文档」，只要意图是把设计稿转成开发可用说明，
  也应使用本 Skill。不处理运行时前端工程、构建与部署。
```

- 补足触发词与同义表达；写明「不处理 X」压掉误触发。
- 对抗欠触发：模型倾向保守，必要时明写「即使用户没明确要求 X，只要涉及 Y 也应使用」。
- 动名词式命名：processing-pdfs 比 pdf-helper 更利于语义匹配。

### 评估集：把 Skill 当代码测

<div class="skill-tw">

<table>
<thead>
<tr><th>测试类型</th><th>建议数量</th><th>检验什么</th></tr>
</thead>
<tbody>
<tr><td>应触发（正例）</td><td>10 条量级</td><td>漏触发率（False Negative），覆盖不同措辞、口语化说法、隐含意图</td></tr>
<tr><td>不应触发（负例）</td><td>10 条量级</td><td>误触发率（False Positive），放相邻领域任务检查边界</td></tr>
<tr><td>端到端结果断言</td><td>每条正例配 1 组</td><td>产物结构、必含字段、数值口径、格式规范</td></tr>
<tr><td>留出验证集</td><td>约总量 40%</td><td>修完之后是否泛化，而不是只修好调参用的那几条</td></tr>
</tbody>
</table>

</div>

断言应客观、可验证，并使用描述性名称（例如「报告包含执行摘要」优于「结果正确」）。迭代时应优先处理漏触发：它比误触发更隐蔽，因为未触发的 Skill 不会留下任何可见迹象。

官方仅提供评估方法论与用例数据格式，不提供执行器，需自行实现回归。触发类断言是二元判定，适合作自动门禁；开放式质量断言主观性强，更适合保留留出集定期人工或模型评审。

### 反模式红线与常见故障速查

<div class="skill-tw">

<table>
<thead>
<tr><th>反模式</th><th>表现</th><th>修法</th></tr>
</thead>
<tbody>
<tr><td>Prompt 搬家</td><td>聊天记录原样粘进 SKILL.md，通篇「请你帮我……」</td><td>重写成分步指令 + 模板 + 边界情况，命令式，带 why</td></tr>
<tr><td>万能 Skill</td><td>一个 Skill 包办所有办公任务，正文 2000 行</td><td>按任务族拆分，共用部分抽成 reference</td></tr>
<tr><td>只写 what 不写 when</td><td>description 空泛，缺少触发场景与关键词</td><td>补 Use when 段落、同义触发词、排除边界</td></tr>
<tr><td>只写 when 不写 how</td><td>触发很准，正文全是原则口号</td><td>补主干工作流、输入输出示例、可核对检查项</td></tr>
<tr><td>深层引用链</td><td>SKILL.md 引 A、A 引 B、B 引 C</td><td>引用保持一层深；大文件配目录</td></tr>
<tr><td>该跑的却让它读</td><td>长脚本源码写进正文，或没说清执行还是参考</td><td>明确运行命令；脚本自带清晰报错</td></tr>
<tr><td>无关 Skill 全挂上</td><td>元数据区噪音大，增加误触发</td><td>按场景裁剪已启用集合，作用域分层</td></tr>
<tr><td>写完即完工</td><td>没有评估集、版本、归属人</td><td>进版本库、配 evals、定期回归</td></tr>
</tbody>
</table>

</div>

<div class="skill-tw">

<table>
<thead>
<tr><th>现象</th><th>常见原因</th><th>处理</th></tr>
</thead>
<tbody>
<tr><td>从来不触发</td><td>description 太泛或缺触发语；name 无信息量</td><td>按强结构重写；反向问模型「什么情况下你会用这个 Skill」</td></tr>
<tr><td>乱触发</td><td>范围过大、边界未声明</td><td>收窄描述、显式排除项、补负例</td></tr>
<tr><td>触发了但不照做</td><td>正文过长、要点被埋、措辞含糊</td><td>缩短正文、约束前置、关键步骤改由脚本保证</td></tr>
<tr><td>工具调用失败</td><td>MCP / 工具本身连接问题</td><td>先脱离 Skill 单测工具链路，再核对工具名与文档</td></tr>
<tr><td>产出不稳定</td><td>确定性步骤交给了模型</td><td>抽成脚本，模型只做判断部分</td></tr>
</tbody>
</table>

</div>


### 维护：防止内容过期

Skill 纳入版本库并不意味着一劳永逸；模型换代后，针对旧缺陷编写的规则可能反过来影响表现，需要持续维护：

- **定期重审**：模型大版本更新后，逐条核对规则所针对的缺陷是否仍然存在；跨越两个以上大版本的补丁类规则可优先删除。
- **优先清理公开获取的通用 Skill**：公开分发包按定义不包含团队私有信息，其内容不是通用实践（可能已被模型内化），就是原作者的个人习惯（对本团队不适用）。
- **为规则标注日期与起因**：新增补丁类规则时附注「为解决 XX 问题添加，YYYY-MM」，否则数月后将无法判断其存废。
- **将长文件改造为渐进式披露**：对于极少触发但每次触发都整篇读入的大文件，应拆分为路由正文与按需读取的 reference。
- **保留回滚路径**：删除前应选取若干可重复执行的典型任务作为基准；条件不具备时，至少应将删除内容单独存档并记录日期，以便出现回退时恢复。有回滚手段才敢于删除。

---

## <span class="skill-sec-n">05</span> 安全模型与执行边界

<div class="skill-note skill-note--warn">
<span class="skill-note__tag">威胁模型</span>
Skill 通过指令与代码两条通道扩展 Agent，并以宿主 Agent 的全部权限执行，其权限范围覆盖 API key、SSH 凭据、浏览器数据与 shell。恶意 Skill 既可能在运行环境中引入漏洞，也可能诱导 Agent 泄露数据或执行非预期操作。
</div>

### 典型攻击面

<div class="skill-tw">

<table>
<thead>
<tr><th>风险类型</th><th>典型行为</th><th>防护</th></tr>
</thead>
<tbody>
<tr><td>Prompt 注入</td><td>SKILL.md 散文段落里藏隐藏指令，劫持 Agent</td><td>安装前逐字通读正文；第三方 Skill 做 diff review</td></tr>
<tr><td>恶意代码</td><td>脚本内含后门、反弹 shell、数据外传</td><td>审计 scripts/ 与依赖清单；沙箱内运行，最小权限</td></tr>
<tr><td>可疑下载</td><td>诱导执行 curl … | bash 拉取不可信二进制</td><td>禁止无审计的远程执行；网络白名单</td></tr>
<tr><td>凭据处理不当</td><td>诱导明文输出 token / key，或硬编码密钥</td><td>密钥扫描；输出侧脱敏；凭据不进上下文</td></tr>
<tr><td>第三方内容注入</td><td>抓取不可信外部内容直接注入上下文</td><td>隔离与标注抓取内容，不把外部文本当指令执行</td></tr>
<tr><td>身份伪造</td><td>冒充知名工具骗取信任</td><td>只从可信来源安装，核对来源与签名</td></tr>
</tbody>
</table>

</div>

实践上：只从可信来源安装，来源不明时先审计全部捆绑文件；执行侧做隔离（只读文件系统、按需关网、drop capabilities、禁提权、限资源与超时）；`allowed-tools` 只是意图声明，不能当权限边界；企业场景把 Skill 纳入供应链治理（来源登记、版本锁定、变更评审、上线前扫描）。

### Skill 不授予权限

Agent 的扩展体系可以分为常驻规则（AGENTS.md 等，每轮加载）、按需能力（SKILL.md）、工具连接（内置工具与 MCP）、执行边界（沙箱与审批）四层。Skill 只属于其中的按需能力层。

由此产生一个关键边界：**Skill 既不授予权限，也不安装依赖**。它只能调用宿主 Agent 已有的工具，能写哪些目录、能否联网，由独立的沙箱与审批策略决定。Skill 正文中写了需要访问网络，并不意味着请求会被放行；「声明需要什么能力」与「运行时是否允许使用该能力」是两件独立的事。理解这一点可以避免把权限配置问题误当作 Skill 编写问题。

### 一次工具调用的判定链

沙箱定义技术执行边界（能写哪里、能不能联网、哪些路径受保护），审批策略决定何时必须停下来问人，两者互补而非替代。

<div class="skill-chartbox">
<svg viewBox="0 0 900 470" role="img" aria-label="一次工具调用的权限判定链流程图：拒绝规则、关键路径熔断、受保护路径、权限模式、审批出口、沙箱执行">
<defs>
<marker id="sk-ar11" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto">
<path d="M0,0 L10,5 L0,10 z" fill="var(--sk-steel)"/>
</marker>
<marker id="sk-ar11r" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto">
<path d="M0,0 L10,5 L0,10 z" fill="var(--sk-red)"/>
</marker>
</defs>
<rect x="300" y="14" width="300" height="42" fill="var(--sk-solid)"/>
<text x="450" y="41" text-anchor="middle" font-family="var(--vp-font-family-base)" font-size="13.5" font-weight="700" fill="var(--sk-on-solid)">模型产出一次工具调用</text>
<line x1="450" y1="56" x2="450" y2="76" stroke="var(--sk-steel)" marker-end="url(#sk-ar11)"/>

<rect x="270" y="78" width="360" height="52" fill="var(--sk-card)" stroke="var(--sk-rule-strong)"/>
<text x="450" y="99" text-anchor="middle" font-family="var(--vp-font-family-base)" font-size="12.8" font-weight="700" fill="var(--sk-ink)">① 前置钩子 / 拒绝规则</text>
<text x="450" y="119" text-anchor="middle" font-family="var(--vp-font-family-base)" font-size="11.5" fill="var(--sk-steel)">deny 规则在所有模式下生效，含 bypassPermissions</text>
<line x1="450" y1="130" x2="450" y2="150" stroke="var(--sk-steel)" marker-end="url(#sk-ar11)"/>
<line x1="630" y1="104" x2="742" y2="104" stroke="var(--sk-red)" marker-end="url(#sk-ar11r)"/>

<rect x="270" y="152" width="360" height="52" fill="var(--sk-card)" stroke="var(--sk-rule-strong)"/>
<text x="450" y="173" text-anchor="middle" font-family="var(--vp-font-family-base)" font-size="12.8" font-weight="700" fill="var(--sk-ink)">② 关键路径熔断</text>
<text x="450" y="193" text-anchor="middle" font-family="var(--vp-font-family-base)" font-size="11.5" fill="var(--sk-steel)">删根目录/家目录/工作目录等，allow 规则与钩子都批不动</text>
<line x1="450" y1="204" x2="450" y2="224" stroke="var(--sk-steel)" marker-end="url(#sk-ar11)"/>
<line x1="630" y1="178" x2="742" y2="178" stroke="var(--sk-red)" marker-end="url(#sk-ar11r)"/>

<rect x="270" y="226" width="360" height="52" fill="var(--sk-card)" stroke="var(--sk-rule-strong)"/>
<text x="450" y="247" text-anchor="middle" font-family="var(--vp-font-family-base)" font-size="12.8" font-weight="700" fill="var(--sk-ink)">③ 受保护路径检查</text>
<text x="450" y="267" text-anchor="middle" font-family="var(--vp-font-family-base)" font-size="11.5" fill="var(--sk-steel)">检查早于 allow 规则求值，写 .git / .claude 等不被预批</text>
<line x1="450" y1="278" x2="450" y2="298" stroke="var(--sk-steel)" marker-end="url(#sk-ar11)"/>

<rect x="270" y="300" width="360" height="52" fill="var(--sk-card)" stroke="var(--sk-brass)" stroke-width="1.6"/>
<text x="450" y="321" text-anchor="middle" font-family="var(--vp-font-family-base)" font-size="12.8" font-weight="700" fill="var(--sk-ink)">④ 权限模式 / 审批策略基线</text>
<text x="450" y="341" text-anchor="middle" font-family="var(--vp-font-family-base)" font-size="11.5" fill="var(--sk-steel)">只读 · 自动改文件 · 计划 · 分类器 · 仅预批 · 全放行</text>
<line x1="450" y1="352" x2="450" y2="372" stroke="var(--sk-steel)" marker-end="url(#sk-ar11)"/>

<rect x="112" y="374" width="240" height="52" fill="var(--sk-card)" stroke="var(--sk-rule-strong)"/>
<text x="232" y="395" text-anchor="middle" font-family="var(--vp-font-family-base)" font-size="12.5" font-weight="700" fill="var(--sk-ink)">出口 A · 问人</text>
<text x="232" y="414" text-anchor="middle" font-family="var(--vp-font-family-base)" font-size="11.3" fill="var(--sk-steel)">单次批准 / 本会话批准该类</text>

<rect x="366" y="374" width="240" height="52" fill="var(--sk-card)" stroke="var(--sk-rule-strong)"/>
<text x="486" y="395" text-anchor="middle" font-family="var(--vp-font-family-base)" font-size="12.5" font-weight="700" fill="var(--sk-ink)">出口 B · 交审核代理</text>
<text x="486" y="414" text-anchor="middle" font-family="var(--vp-font-family-base)" font-size="11.3" fill="var(--sk-steel)">低/中风险可放行，严重风险拒绝</text>

<rect x="620" y="374" width="168" height="52" fill="var(--sk-card)" stroke="var(--sk-rule-strong)"/>
<text x="704" y="395" text-anchor="middle" font-family="var(--vp-font-family-base)" font-size="12.5" font-weight="700" fill="var(--sk-ink)">出口 C · 直接执行</text>
<text x="704" y="414" text-anchor="middle" font-family="var(--vp-font-family-base)" font-size="11.3" fill="var(--sk-steel)">留在沙箱内的动作</text>

<line x1="330" y1="352" x2="240" y2="372" stroke="var(--sk-steel)" marker-end="url(#sk-ar11)"/>
<line x1="570" y1="352" x2="680" y2="372" stroke="var(--sk-steel)" marker-end="url(#sk-ar11)"/>

<rect x="742" y="82" width="146" height="118" fill="var(--sk-warn)" stroke="var(--sk-red)"/>
<text x="815" y="110" text-anchor="middle" font-family="var(--cs-font-mono)" font-size="11" fill="var(--sk-red)">DENIED</text>
<text x="815" y="134" text-anchor="middle" font-family="var(--vp-font-family-base)" font-size="11.5" fill="var(--sk-steel)">无条件拒绝</text>
<text x="815" y="154" text-anchor="middle" font-family="var(--vp-font-family-base)" font-size="11.5" fill="var(--sk-steel)">不受模式影响</text>
<text x="815" y="178" text-anchor="middle" font-family="var(--vp-font-family-base)" font-size="11.5" fill="var(--sk-steel)">这是唯一的硬边界</text>

<text x="24" y="452" font-family="var(--vp-font-family-base)" font-size="11.8" fill="var(--sk-steel)">读法：①②③ 是不可被「放宽模式」绕过的前置检查；④ 才是那个平时被当成「权限设置」的东西。三个出口共享同一个沙箱边界。</text>
</svg>
</div>

①②③ 是不可被「放宽模式」绕过的前置检查：拒绝规则与钩子在 bypassPermissions 下依然生效，受保护路径检查早于 allow 规则求值；④ 才是平时被当成「权限设置」的东西，三个出口共享同一个沙箱边界。

### 上下文压缩后的影响

长会话中上下文占用达到阈值时，运行时会对历史消息进行压缩摘要。压缩后的 Skill 能否继续生效取决于产品是否实现了「重挂」机制：部分产品会在压缩后按一定预算重新注入已调用 Skill 的部分内容，部分产品则要求模型在需要时重新读取。

这一机制对 Skill 编写有两个通用约束：

- **重要指令置于正文前部**。重挂时通常只保留正文开头的一部分，因此适用范围、关键步骤、硬约束等必须最先出现，不能假设模型在压缩后还能看到文件末尾。
- **不依赖一次性读过就始终有效**。关键约束不应只写在深层 reference 里；若某个步骤在任务后期仍需遵守，应在正文或相关文件中可被重新发现，而不是依赖压缩前的上下文残留。

具体的重挂预算（每个 Skill 保留多少、合计上限）与截断方式由各产品决定，应以对应产品的当前文档为准。

### 执行环境与审批

各产品的沙箱边界都包含两个维度：文件系统的可写范围与网络的出站控制，通常通过操作系统级隔离或一次性容器实现，并默认拒绝网络访问。Skill 声明本身不改变这些边界。

工具调用是否需要停下来询问用户由审批策略决定。部分产品提供自动审批机制，对需要审批的动作按风险分级处理：涉及数据外泄、凭据访问、破坏性操作的高风险动作直接拒绝或强制要求人类授权，无法判断时默认拒绝。自动审批只决定「是否还要问人」，不放宽沙箱边界。

无论执行环境如何实现，Agent 的主循环一致：模型推理产出工具调用，在受限环境中执行，观察结果回灌后继续推理。Skill 的脚本与文件读取都发生在这个循环中，其执行边界由宿主而非 Skill 自身决定。

## <span class="skill-sec-n">06</span> 实现分歧

Agent Skills 是开放标准，Codex、Claude Code、Cursor、Copilot、Devin、OpenHands 共用同一份 SKILL.md。标准只规定文件格式，四件事留给了各产品自己决定；这些分歧不是参数差异，而是不同的治理与可靠性取舍，理解这些取舍有助于跨产品落地时规避相应问题。

### 分歧一：元数据清单占多少上下文

软路由要求系统提示里常驻全部 name + description，装得越多清单越大，所以每家都要决定「分多少窗口给清单、超了怎么办」。Claude Code 取窗口 1%、单条上限 1536 字符，超预算时先丢最少调用者的描述；Codex 约 2%（窗口未知时 8000 字符），先缩短描述、再整条省略并告警；多数产品未公开数字。

设计含义：**description 要按会被截断来写**，关键用途和触发词前置；装了用不到的 Skill 会实实在在挤占清单预算、稀释信噪比，而不是零成本。

### 分歧二：触发只靠模型，还是允许确定性路径

基于 description 的软路由是默认方式，但模型可能漏触发或误触发。Cursor、OpenHands 与 Claude Code 均额外提供 paths / glob 路径规则：文件命中匹配模式时即注入，无需模型判断。OpenHands 甚至不向模型暴露路径规则，使基线开销为零。

设计含义：**凡可用文件模式确定描述的场景（修改某类文件即应附带某规范），应优先采用确定性触发**；语义匹配仅用于无法事先限定的开放场景。

### 分歧三：同名 Skill 谁生效

同一个名字出现在企业、个人、项目多个作用域时，四家给出了四种策略，对应四种治理假设。

<div class="skill-chartbox">
<svg viewBox="0 0 900 300" role="img" aria-label="命名冲突消解优先级对照：Claude Code 覆盖式、Codex 并存式、OpenHands 取代式、Devin 磁盘覆盖索引">
<text x="18" y="26" font-family="var(--cs-font-mono)" font-size="11" fill="var(--sk-brass)">命名冲突的四种官方策略</text>
<!-- CC -->
<text x="18" y="60" font-family="var(--vp-font-family-base)" font-size="12.5" font-weight="700" fill="var(--sk-ink)">Claude Code · 覆盖</text>
<rect x="200" y="44" width="150" height="26" fill="var(--sk-solid)"/>
<text x="275" y="62" text-anchor="middle" font-family="var(--vp-font-family-base)" font-size="11.5" fill="var(--sk-on-solid)">enterprise</text>
<text x="362" y="62" font-family="var(--cs-font-mono)" font-size="12" fill="var(--sk-brass)">&gt;</text>
<rect x="380" y="44" width="130" height="26" fill="var(--sk-block-2)"/>
<text x="445" y="62" text-anchor="middle" font-family="var(--vp-font-family-base)" font-size="11.5" fill="var(--sk-on-solid)">personal</text>
<text x="522" y="62" font-family="var(--cs-font-mono)" font-size="12" fill="var(--sk-brass)">&gt;</text>
<rect x="540" y="44" width="120" height="26" fill="var(--sk-block-3)" opacity=".5"/>
<text x="600" y="62" text-anchor="middle" font-family="var(--vp-font-family-base)" font-size="11.5" fill="var(--sk-ink)">project</text>
<rect x="680" y="44" width="202" height="26" fill="none" stroke="var(--sk-rule)" stroke-dasharray="3 3"/>
<text x="781" y="62" text-anchor="middle" font-family="var(--vp-font-family-base)" font-size="11" fill="var(--sk-steel)">plugin 走命名空间，不参与</text>
<!-- Codex -->
<text x="18" y="112" font-family="var(--vp-font-family-base)" font-size="12.5" font-weight="700" fill="var(--sk-ink)">Codex · 并存</text>
<rect x="200" y="96" width="150" height="26" fill="var(--sk-block-2)"/>
<text x="275" y="114" text-anchor="middle" font-family="var(--vp-font-family-base)" font-size="11.5" fill="var(--sk-on-solid)">repo</text>
<rect x="360" y="96" width="150" height="26" fill="var(--sk-block-2)"/>
<text x="435" y="114" text-anchor="middle" font-family="var(--vp-font-family-base)" font-size="11.5" fill="var(--sk-on-solid)">user</text>
<rect x="520" y="96" width="140" height="26" fill="var(--sk-block-2)"/>
<text x="590" y="114" text-anchor="middle" font-family="var(--vp-font-family-base)" font-size="11.5" fill="var(--sk-on-solid)">admin / system</text>
<text x="680" y="114" font-family="var(--vp-font-family-base)" font-size="11" fill="var(--sk-red)">不合并，全部出现在选择器里</text>
<!-- OpenHands -->
<text x="18" y="164" font-family="var(--vp-font-family-base)" font-size="12.5" font-weight="700" fill="var(--sk-ink)">OpenHands · 取代</text>
<rect x="200" y="148" width="150" height="26" fill="var(--sk-solid)"/>
<text x="275" y="166" text-anchor="middle" font-family="var(--vp-font-family-base)" font-size="11.5" fill="var(--sk-on-solid)">project</text>
<text x="362" y="166" font-family="var(--cs-font-mono)" font-size="12" fill="var(--sk-brass)">&gt;</text>
<rect x="380" y="148" width="130" height="26" fill="var(--sk-block-2)"/>
<text x="445" y="166" text-anchor="middle" font-family="var(--vp-font-family-base)" font-size="11.5" fill="var(--sk-on-solid)">user</text>
<text x="522" y="166" font-family="var(--cs-font-mono)" font-size="12" fill="var(--sk-brass)">&gt;</text>
<rect x="540" y="148" width="120" height="26" fill="var(--sk-block-3)" opacity=".5"/>
<text x="600" y="166" text-anchor="middle" font-family="var(--vp-font-family-base)" font-size="11.5" fill="var(--sk-ink)">public</text>
<text x="680" y="166" font-family="var(--vp-font-family-base)" font-size="11" fill="var(--sk-steel)">按优先级取代，不合并正文</text>
<!-- Devin -->
<text x="18" y="216" font-family="var(--vp-font-family-base)" font-size="12.5" font-weight="700" fill="var(--sk-ink)">Devin · 就近</text>
<rect x="200" y="200" width="230" height="26" fill="var(--sk-solid)"/>
<text x="315" y="218" text-anchor="middle" font-family="var(--vp-font-family-base)" font-size="11.5" fill="var(--sk-on-solid)">克隆到机器上的磁盘版本</text>
<text x="442" y="218" font-family="var(--cs-font-mono)" font-size="12" fill="var(--sk-brass)">&gt;</text>
<rect x="460" y="200" width="200" height="26" fill="var(--sk-block-2)"/>
<text x="560" y="218" text-anchor="middle" font-family="var(--vp-font-family-base)" font-size="11.5" fill="var(--sk-on-solid)">后端索引版本</text>
<text x="680" y="218" font-family="var(--vp-font-family-base)" font-size="11" fill="var(--sk-steel)">保证用的是当前分支</text>
<line x1="18" y1="242" x2="882" y2="242" stroke="var(--sk-rule)"/>
<text x="18" y="266" font-family="var(--vp-font-family-base)" font-size="11.8" fill="var(--sk-steel)">覆盖式假设「组织规范应压过个人」；并存式认为同名是作者的事、工具不替你选；取代式让就近的更具体；就近式保证分支上的才是真的。</text>
<text x="18" y="288" font-family="var(--vp-font-family-base)" font-size="11.8" fill="var(--sk-steel)">注意 Claude Code 的顺序是 enterprise &gt; personal &gt; project，与「项目覆盖个人」的直觉相反，容易记错。</text>
</svg>
</div>

跨产品分发 Skill 时不要假设同名覆盖行为，目录命名与命名空间要自己管。

### 分歧四：一个时刻能激活几个 Skill

Claude Code 允许多技能叠加（配套有压缩后 5000/25000 的重挂预算）；Devin 同一时刻只激活一个，调用新技能会替换旧的。设计含义：需要多流程配合时，要么确认运行时支持叠加，要么把它们合并成一个带分支的 Skill。

### 语义冲突的四个解法

同名冲突有规则可依，但**语义冲突**（两个技能都想接同一件事）没有工具能替你判，只能从机制上规避：

- **确定性触发优先**：能用 paths / glob 限定就别靠语义匹配。
- **排除边界写进 description**：写正文里没用（正文命中后才读）；描述可能被截断，关键用途与触发词要前置。
- **让冲突方退出竞争**：几家都提供「只许显式调用」开关，关掉其中一个的自动触发最省事。
- **控制候选集**：默认只启用推荐技能，或把低优先技能降级为「只列名字不给描述」。



### 六种实现对照

<div class="skill-tw skill-tw--products">

<table>
<thead>
<tr><th>产品</th><th>发现路径</th><th>元数据预算</th><th>触发方式</th><th>同名冲突</th></tr>
</thead>
<tbody>
<tr><td>Claude Code</td><td>enterprise、~/.claude/skills、项目 .claude/skills、plugin</td><td>窗口 1%，单条上限 1536 字符；超预算先丢最少调用者</td><td>自动 + /skill-name；可关自动</td><td>enterprise &gt; personal &gt; project；plugin 走命名空间</td></tr>
<tr><td>Codex</td><td>$CWD/.agents/skills（含父目录）、~/.agents/skills、/etc/codex/skills</td><td>约窗口 2%，未知时 8000 字符；先缩短再省略</td><td>隐式 + /skills 或 $；可仅显式</td><td>不合并，全部进选择器</td></tr>
<tr><td>Cursor</td><td>.agents/skills、.cursor/skills；兼容 .claude/.codex</td><td>未公开</td><td>自动 + 显式；paths glob 浮现；可随 Mode 常驻</td><td>嵌套目录限定作用域</td></tr>
<tr><td>Copilot</td><td>项目 .github/.claude/.agents；个人 ~/.copilot、~/.agents</td><td>未公开</td><td>按 prompt 与 description；allowed-tools 可预批</td><td>未给规则</td></tr>
<tr><td>Devin</td><td>扫描 .agents/.devin/.github/.claude/.cursor/.codex 等 9 个目录</td><td>启动即见全部元数据</td><td>自动 + @skills:name（支持参数）</td><td>磁盘版覆盖索引，当前分支优先</td></tr>
<tr><td>OpenHands</td><td>.agents/skills、~/.agents/skills、公共注册表</td><td>未公开</td><td>模型自选 / 关键词 / paths glob</td><td>project &gt; user &gt; public，取代不合并</td></tr>
</tbody>
</table>

</div>

数字随版本变化，以各产品当前官方文档为准；这里保留的是机制形状，不是具体配额。

---

## <span class="skill-sec-n">07</span> 软路由与硬编排

<div class="skill-note">
<span class="skill-note__tag">一句话定性</span>
让 agent 可靠复用一段专业流程有三条路线：① Markdown 软路由（写 SKILL.md，模型看描述自己决定读不读）；② 可组合 harness（工具与中间件是代码，何时调用仍交给模型）；③ 图编排（节点与条件边，路由由代码判定）。选哪个不取决于技术先进性，而取决于你愿意为确定性付多少开发成本。
</div>

<div class="skill-tw">

<table>
<thead>
<tr><th>维度</th><th>Markdown 软路由</th><th>可组合 harness</th><th>图编排（LangGraph）</th></tr>
</thead>
<tbody>
<tr><td>能力载体</td><td>SKILL.md + 同目录脚本/参考，不写代码即可新增能力</td><td>带 name/description/参数 schema 的工具 + 中间件</td><td>图节点（函数）+ 显式状态 schema + 条件边</td></tr>
<tr><td>谁决定何时调用</td><td>模型读 description 软路由</td><td>模型读工具描述软路由</td><td>代码按状态取值走条件边</td></tr>
<tr><td>确定性与代价</td><td>开发成本最低、不确定性最高，靠描述与 evals 压</td><td>工具边界可控，路由仍是概率性的</td><td>确定性最高，但必须先把流程画清楚，迁移等于重写编排层</td></tr>
</tbody>
</table>

</div>

<div class="skill-chartbox">
<svg viewBox="0 0 900 350" role="img" aria-label="三条路线的路由决策对照图：Markdown 软路由、可组合 harness、图编排硬路由">
<defs>
<marker id="sk-ar14" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto">
<path d="M0,0 L10,5 L0,10 z" fill="var(--sk-steel)"/>
</marker>
</defs>
<text x="18" y="24" font-family="var(--cs-font-mono)" font-size="11" fill="var(--sk-brass)">同一个请求，三种路由方式</text>

<!-- Route A -->
<rect x="18" y="40" width="278" height="252" fill="var(--sk-card)" stroke="var(--sk-rule-strong)"/>
<text x="34" y="64" font-family="var(--vp-font-family-base)" font-size="14.5" font-weight="700" fill="var(--sk-ink)">A · Markdown 软路由</text>
<text x="34" y="83" font-family="var(--cs-font-mono)" font-size="10.5" fill="var(--sk-steel)">SKILL.md</text>
<rect x="34" y="94" width="246" height="30" fill="var(--sk-solid)"/>
<text x="157" y="114" text-anchor="middle" font-family="var(--vp-font-family-base)" font-size="11.5" fill="var(--sk-on-solid)">系统提示里只有 name + description</text>
<line x1="157" y1="124" x2="157" y2="142" stroke="var(--sk-steel)" marker-end="url(#sk-ar14)"/>
<rect x="34" y="144" width="246" height="30" fill="var(--sk-warn)" stroke="var(--sk-red)"/>
<text x="157" y="164" text-anchor="middle" font-family="var(--vp-font-family-base)" font-size="11.5" fill="var(--sk-red)">模型判断：要不要读正文？</text>
<line x1="157" y1="174" x2="157" y2="192" stroke="var(--sk-steel)" marker-end="url(#sk-ar14)"/>
<rect x="34" y="194" width="246" height="30" fill="var(--sk-card)" stroke="var(--sk-rule-strong)"/>
<text x="157" y="214" text-anchor="middle" font-family="var(--vp-font-family-base)" font-size="11.5" fill="var(--sk-ink)">读文件 → 按指令执行</text>
<text x="34" y="248" font-family="var(--vp-font-family-base)" font-size="11" fill="var(--sk-red)">失效模式：漏读、误读、多个都想接</text>
<text x="34" y="268" font-family="var(--vp-font-family-base)" font-size="11" fill="var(--sk-steel)">新增能力成本：写一个 Markdown 文件</text>

<!-- Route B -->
<rect x="311" y="40" width="278" height="252" fill="var(--sk-card)" stroke="var(--sk-rule-strong)"/>
<text x="327" y="64" font-family="var(--vp-font-family-base)" font-size="14.5" font-weight="700" fill="var(--sk-ink)">B · 可组合 harness</text>
<text x="327" y="83" font-family="var(--cs-font-mono)" font-size="10.5" fill="var(--sk-steel)">tools + middleware</text>
<rect x="327" y="94" width="246" height="30" fill="var(--sk-solid)"/>
<text x="450" y="114" text-anchor="middle" font-family="var(--vp-font-family-base)" font-size="11.5" fill="var(--sk-on-solid)">工具 schema + 描述常驻</text>
<line x1="450" y1="124" x2="450" y2="142" stroke="var(--sk-steel)" marker-end="url(#sk-ar14)"/>
<rect x="327" y="144" width="246" height="30" fill="var(--sk-warn)" stroke="var(--sk-red)"/>
<text x="450" y="164" text-anchor="middle" font-family="var(--vp-font-family-base)" font-size="11.5" fill="var(--sk-red)">模型判断：调哪个工具？</text>
<line x1="450" y1="174" x2="450" y2="192" stroke="var(--sk-steel)" marker-end="url(#sk-ar14)"/>
<rect x="327" y="194" width="246" height="30" fill="var(--sk-card)" stroke="var(--sk-brass)" stroke-width="1.6"/>
<text x="450" y="214" text-anchor="middle" font-family="var(--vp-font-family-base)" font-size="11.5" fill="var(--sk-ink)">中间件拦一道 → 执行 → 回灌</text>
<text x="327" y="248" font-family="var(--vp-font-family-base)" font-size="11" fill="var(--sk-red)">失效模式：工具选错、参数填错</text>
<text x="327" y="268" font-family="var(--vp-font-family-base)" font-size="11" fill="var(--sk-steel)">新增能力成本：写函数 + 写好描述</text>

<!-- Route C -->
<rect x="604" y="40" width="278" height="252" fill="var(--sk-solid)"/>
<text x="620" y="64" font-family="var(--vp-font-family-base)" font-size="14.5" font-weight="700" fill="var(--sk-on-solid)">C · 图编排硬路由</text>
<text x="620" y="83" font-family="var(--cs-font-mono)" font-size="10.5" fill="var(--sk-rail-text)">nodes + conditional edges</text>
<rect x="620" y="94" width="246" height="30" fill="var(--sk-steel)"/>
<text x="743" y="114" text-anchor="middle" font-family="var(--vp-font-family-base)" font-size="11.5" fill="var(--sk-on-solid)">状态 schema 显式声明</text>
<line x1="743" y1="124" x2="743" y2="142" stroke="var(--sk-brass)" marker-end="url(#sk-ar14)"/>
<rect x="620" y="144" width="246" height="30" fill="var(--sk-slot)"/>
<text x="743" y="164" text-anchor="middle" font-family="var(--vp-font-family-base)" font-size="11.5" fill="var(--sk-slot-text)">代码判断：状态满足哪条边？</text>
<line x1="743" y1="174" x2="743" y2="192" stroke="var(--sk-brass)" marker-end="url(#sk-ar14)"/>
<rect x="620" y="194" width="246" height="30" fill="var(--sk-steel)"/>
<text x="743" y="214" text-anchor="middle" font-family="var(--vp-font-family-base)" font-size="11.5" fill="var(--sk-on-solid)">进入指定节点 → checkpoint 落盘</text>
<text x="620" y="248" font-family="var(--vp-font-family-base)" font-size="11" fill="var(--sk-rail-text)">失效模式：图没覆盖到的情况直接走不通</text>
<text x="620" y="268" font-family="var(--vp-font-family-base)" font-size="11" fill="var(--sk-on-solid-soft)">新增能力成本：改图结构与状态定义</text>

<line x1="18" y1="308" x2="882" y2="308" stroke="var(--sk-rule)"/>
<text x="18" y="332" font-family="var(--vp-font-family-base)" font-size="11.8" fill="var(--sk-steel)">读法：三格里那个红框 / 白框的位置就是「不确定性注入点」。A 与 B 把它放在路由上，C 把它挪进了单个节点内部。</text>
</svg>
</div>

最关键的判据是「由谁决定何时调用」：Skill 的主要脆弱性（反模式、冲突消解）本质上都是对概率性路由的修补；图编排则从根本上消除了该问题，代价是必须预先定义流程结构。流程可预先结构化的场景（审批流、数据管线、固定 SOP）适合图编排；流程取决于用户即时输入的场景（编码 Agent、开放式助手）适合 Skill。两条路线正在融合：LangChain Deep Agents 同时实现了 AGENTS.md 常驻记忆与按需加载的 skills，并明确 memory 始终注入、skills 才采用渐进式披露。

「委派」由此构成一条完整谱系：Skill 是在同一 Agent 中替换指令集；subagent / task 工具是在独立上下文中执行并仅回传结果；handoff（OpenAI Agents SDK）则是移交整段对话的控制权。三者的上下文归属与返回物各不相同。

### 框架侧值得记住的机制细节

**LangGraph**：持久化按 super-step 落 checkpoint（time travel 与恢复的基础）；恢复不是「从那一行继续」而是重放到停止处，所以非确定操作与副作用必须包进幂等 task；耐久度有 exit / async / sync 三档；interrupt() 需配 checkpointer 与 thread，且要放在节点操作之前（恢复会重放该节点）。

**Deep Agents**：大工具结果超约 20000 tokens 即卸载为文件引用；上下文过 85% 先卸载写工具调用、再生成结构化摘要并保留 10% 近期上下文，取不到模型 profile 时退回 170000 / 6 条；重任务委派给独立上下文的 subagent。

**Pi coding agent**：核心极小，权限弹窗、subagent、plan、MCP、后台 bash 全部不做、交给扩展（官方建议在容器里跑）；它不要求 name 与目录同名（多 harness 共用目录时该约束不合适）；命名冲突取先发现者胜；压缩可被扩展接管；会话以树形存储。它反过来说明 subagent、权限门这些机制是产品对工作流的固化，不是模型能力。


---

## 附录 · 最小 Skill 加载器（Python 示意）

以下 Python 代码展示前述机制在加载器内部的最小实现，覆盖扫描目录、校验 frontmatter、消解同名冲突、注入元数据与预算裁剪、软路由判定、L2/L3 按需加载及脚本执行的完整链路。该代码仅用于帮助理解，并非规范要求；其中 `listdir`、`split_frontmatter`、`model_generate` 均为占位函数，错误处理、缓存与并发均已省略。

发现与解析阶段：每个子目录对应一个 Skill，不含 `SKILL.md` 的目录忽略不计；必填字段仅 `name` 与 `description`，且 name 必须与目录名一致。

```python
import re

NAME_RE = re.compile(r"[a-z0-9]+(-[a-z0-9]+)*")  # 无大写、无首尾/连续连字符

def discover_skills(scopes):
    """scopes 按作用域优先级排列，例如 [("project", ".claude/skills"), ...]。"""
    found = []
    for scope, root in scopes:
        for entry in os.listdir(root):
            path = os.path.join(root, entry, "SKILL.md")
            if not os.path.exists(path):
                continue
            meta, body = split_frontmatter(read(path))
            if not is_valid(meta, entry):
                warn("skipped", path)           # 工程上要给出具体错因
                continue
            found.append(Skill(meta["name"], meta["description"],
                               scope, os.path.join(root, entry), body))
    return found                                 # 此刻正文只在内存，一字节都没进上下文

def is_valid(meta, dirname):
    return (
        "name" in meta and "description" in meta
        and bool(NAME_RE.fullmatch(meta["name"]))
        and len(meta["name"]) <= 64
        and 0 < len(meta["description"]) <= 1024
        and meta["name"] == dirname              # 目录名即 Skill 名
    )
```

冲突消解决定同名技能的取舍。以下实现采用「先发现者优先并告警」策略；覆盖式、取代式与并存式的差异仅体现在此循环中。

```python
def resolve_conflicts(found):
    """found 已按作用域优先级排好。"""
    picked = {}
    for skill in found:
        if skill.name in picked:
            warn("duplicate skill name", skill.name, skill.dir)
            continue                             # 覆盖式改为直接写入；并存式则不调用本函数
        picked[skill.name] = skill
    return list(picked.values())
```

元数据注入与预算裁剪：清单仅占上下文窗口的一定比例（Claude Code 约 1%，Codex 约 2%）；超出预算时先缩短描述，再整条省略。

```python
def build_listing(skills, context_window, fraction):
    budget = int(context_window * fraction)
    used = 0
    lines = []
    for s in sorted(skills, key=lambda s: -s.call_count):  # 常用的先占位
        line = f"- {s.name}: {s.description}"
        if used + tokens(line) > budget:
            line = f"- {s.name}: {shorten(s.description)}"  # 降级一：缩短描述
        if used + tokens(line) > budget:
            warn("skill omitted from listing", s.name)       # 降级二：整条省略
            continue
        lines.append(line)
        used += tokens(line)
    return "
".join(lines)
```

需要区分的是，软路由并非加载器内部的一次判定：加载器只将文件读取暴露为工具，是否读取正文由模型决定。若需消除该不确定性，可补充基于路径匹配的确定性路由。

```python
def agent_turn(user_msg, history, tools):
    reply = model_generate(SYSTEM_PROMPT, history + [user_msg], tools)
    for call in reply.tool_calls:
        history.append(run_tool(call))          # 结果回灌，下一轮接着推理
    return reply

def path_triggered(skills, touched_file):
    """对照组：文件命中 glob 即注入，不依赖模型判断。"""
    return [s for s in skills
            if any(fnmatch.fnmatch(touched_file, p) for p in s.paths)]
```

L2 与 L3 共用同一个读取函数，区别仅在调用时机；脚本执行是安全边界的入口，在调用 `shell` 之前必须经过沙箱与审批判定。

```python
def read_file(path):
    text = read(path)
    context.append(text)                        # 只有这一步消耗 token 预算
    return text

read_file("skills/pdf-processing/SKILL.md")             # L2，约 3–5k tokens
read_file("skills/pdf-processing/references/forms.md")  # L3，其余 reference 不花 token
```

```python
def run_script(cmd):
    proc = shell(cmd)                           # 工程上必须在沙箱里跑
    context.append(truncate(proc.stdout))        # 脚本源码、依赖、大文件都不进上下文
    return proc.returncode

run_script("python3 skills/pdf-processing/scripts/extract_fields.py in.pdf")
```

<div class="skill-note">
<span class="skill-note__tag">真实加载器还要补的</span>
token 计数（示意里的 tokens() 不能用字符数代替）、压缩后重挂已调用 Skill 的 5k/25k 预算、沙箱与审批判定链、加载错误的结构化上报、分发来源的版本解析与固定、路径域规则在压缩后的重新加载。
</div>

---

---

## 参考来源

<details>
<summary>主要依据（点击展开）</summary>

- riba2534，《写得越好的提示词，过期得越快》（三类提示词、判别方法与实证）：<a href="https://x.com/riba2534/status/2081112107547017330">x.com/riba2534</a>
- Anthropic Engineering，《Equipping agents for the real world with Agent Skills》（2025-10-16，含 2025-12-18 开放标准更新）：anthropic.com/engineering
- Agent Skills 规范全文（目录结构、frontmatter 约束、三级加载、校验）：agentskills.io/specification
- Claude，《Skills explained: How Skills compares to prompts, Projects, MCP, and subagents》：claude.com/blog/skills-explained
- OWASP Agentic Skills Top 10 — AST01 Malicious Skills：owasp.org/www-project-agentic-skills-top-10
- Claude Code 文档，Agent Skills（清单预算 1%、压缩后重挂 5k·25k、skillOverrides）：code.claude.com/docs/en/skills
- Claude 平台文档，Compaction（compact_20260112、trigger 默认 150k / 最小 50k）：platform.claude.com/docs/compaction
- Claude Code 文档，Permission modes / Headless mode / Plugin marketplaces：code.claude.com/docs/en
- Anthropic 文档，Agent Skills best practices（评估驱动开发、跨 Haiku/Sonnet/Opus 测试、无内置评估执行器）：docs.anthropic.com/agent-skills/best-practices
- OpenAI Codex 文档，Skills 与 Agent approvals & security：developers.openai.com/codex
- AGENTS.md 开放格式与兼容工具清单：agents.md
- Cursor 文档，Skills 与 Rules：cursor.com/docs/skills
- GitHub 文档，Copilot coding agent 与 Agent skills：docs.github.com/copilot/coding-agent
- Devin 文档，Skills：docs.devin.ai/product-guides/skills
- OpenHands 文档，Skills 与架构：docs.openhands.dev/overview/skills
- Gemini CLI 文档，Configuration（model.compressionThreshold 默认 0.5）：geminicli.com/docs/reference/configuration
- Pi coding agent 官方文档（Skills / Settings / Compaction / Extensions）：pi.dev/docs
- LangGraph 文档，Durable execution / Persistence / Interrupts：docs.langchain.com/langgraph/durable-execution
- LangChain 文档，Deep Agents 上下文工程：docs.langchain.com/deepagents/context-engineering
- OpenAI Agents SDK 文档，Agents（manager 与 handoffs、guardrails）：openai.github.io/openai-agents-python/agents

</details>
