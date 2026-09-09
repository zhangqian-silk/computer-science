# 分布式锁示例

本目录用带 TTL 的内存 KV 模拟 Redis，演示分布式锁的关键点：唯一 token、锁提前过期、看门狗续约，以及 **fencing token 阻断停顿导致的双写**。

上层原理文档见 [distributed_lock.md](../distributed_lock.md)。

## 目录结构

- store.go：带 TTL 的 KV（SetNX / DelIfMatch / RenewIfMatch）+ 单调 Incr（fencing token）
- lock.go：分布式锁（唯一 token + 看门狗续约 + 获取时返回 fencing token）
- main.go：互斥、锁提前过期、看门狗续约、fencing 阻断双写四个场景

## 快速开始

在当前目录执行：

```bash
go run .
```

## 观察点

- A 持锁期间 B 无法获取，释放后可获取。
- 无看门狗且 TTL 短于业务耗时，锁过期后 D 抢到锁（并发风险），C 因 token 不符解锁失败。
- 开看门狗后，业务运行超过初始 TTL 锁仍有效，F 无法加锁。
- fencing：G 停顿期间锁过期、H 获取并写入后，G 用旧 fence 的迟到写入被资源拒绝，避免双写。
