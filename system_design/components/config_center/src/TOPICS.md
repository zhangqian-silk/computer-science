# 配置中心示例

本目录提供教学级配置中心，演示动态推送（Watch）、版本发布与秒级回滚。

上层原理文档见 [config_center.md](../config_center.md)。

## 目录结构

- center.go：配置中心（Watch / Publish / Rollback + 版本历史）
- main.go：客户端监听 + 发布 + 回滚示例

## 快速开始

在当前目录执行：

```bash
go run .
```

## 观察点

- 客户端通过 Watch 通道实时收到每次配置变更，无需重启。
- 发布错误配置后回滚到历史版本，客户端立即收到修正值。
