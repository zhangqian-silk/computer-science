# 服务注册与发现示例

本目录提供教学级注册中心，演示心跳续约、TTL 摘除与服务发现。

上层原理文档见 [service_registry.md](../service_registry.md)。

## 目录结构

- registry.go：注册中心（Register / Heartbeat / Deregister / Discover）
- main.go：注册、心跳续约、TTL 摘除与优雅下线示例

## 快速开始

在当前目录执行：

```bash
go run .
```

## 观察点

- 持续心跳的实例始终可被发现。
- 停止心跳的实例超过 TTL 后从可用列表自动消失。
- 主动注销后立即从发现结果中移除。
