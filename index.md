# Computer Science

这是一个以 Markdown 组织的计算机科学知识库，现已支持通过 GitHub Pages 发布为静态站点，并支持以下渲染能力：

- 原生 HTML
- KaTeX 数学公式
- Mermaid 图表
- VitePress 组件扩展

---

## 主要内容

- [AI](/ai/mechanism/attention)
- [基础](/fundamentals/README)
- [语言](/language/golang/README)
- [中间件](/middleware/redis/redis)
- [分布式](/distributed/theory)
- [系统设计](/system_design/README)
- [设计模式](/design_patterns/CreationalPatterns)

---

## 本地预览

```bash
npm install
npm run docs:dev
```

---

## 页面扩展

VitePress 支持在 Markdown 中直接编写 HTML，也支持通过 Vue 组件组织更复杂的交互内容。仓库中已有的数学公式与 Mermaid 图表会在站点中直接渲染。
