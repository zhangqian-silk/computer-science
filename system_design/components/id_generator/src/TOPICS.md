# 分布式 ID 示例

本目录提供教学级 Snowflake 生成器，演示位段拼装、并发唯一性与时钟回拨处理。

上层原理文档见 [id_generator.md](../id_generator.md)。

## 目录结构

- snowflake.go：Snowflake 生成器（位段、序列自增、时钟回拨处理）
- segment.go：号段模式（双 buffer + 阈值预取，mock DB 批发区间）
- main.go：Snowflake 顺序/并发唯一性校验 + 号段跨段连续发号

## 快速开始

在当前目录执行：

```bash
go run .
```

## 观察点

- 顺序生成的 ID 趋势递增。
- 并发生成一万个 ID 无重复，验证同毫秒序列自增 + 加锁的正确性。
- 号段模式跨多个区间连续发号且严格递增，切段无缝。
