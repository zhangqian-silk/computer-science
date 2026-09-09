# 搜索组件示例

本目录提供教学级倒排索引与 TF-IDF 检索，演示建索引、AND/OR 查询与相关性排序。

上层原理文档见 [search_component.md](../search_component.md)。

## 目录结构

- index.go：倒排索引（Add / Search，TF-IDF 打分 + AND/OR）
- main.go：多查询检索与排序示例

## 快速开始

在当前目录执行：

```bash
go run .
```

## 观察点

- 同一查询 AND 比 OR 命中文档更少（精度高、召回低）。
- 含高频且稀有查询词的文档排序更靠前（TF-IDF 效果）。
- 为便于纯空格分词，示例语料使用英文词元；中文需接入分词器。
