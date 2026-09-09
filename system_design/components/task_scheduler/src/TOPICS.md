# 任务调度示例

本目录提供教学级单层时间轮，演示延迟任务如何用「槽位 + rounds」高效触发。

上层原理文档见 [task_scheduler.md](../task_scheduler.md)。

## 目录结构

- wheel.go：单层时间轮（Add / Advance，rounds 承载超过一圈的延迟）
- main.go：注册延迟 3/8/20 的任务并推进指针

## 快速开始

在当前目录执行：

```bash
go run .
```

## 观察点

- 延迟 3 的任务在 tick=3 触发。
- 延迟 8（正好一圈）与延迟 20（两圈多）通过 rounds 计数在正确 tick 触发，验证固定槽位承载任意延迟。
