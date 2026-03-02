# Booking

本目录对应预定业务专题，聚焦库存锁定、超卖防护与订单确认链路约束下的系统设计。

---

## 核心知识点清单

- 核心对象：资源（房间/座位/时段）、库存、预定单、支付单、确认状态。
- 主链路：查询可售、库存预占、支付确认、出票/留房、超时释放。
- 一致性设计：库存扣减原子性、预占令牌、订单状态机与幂等写入。
- 高并发场景：热门资源抢购、排队削峰、限流防刷与库存隔离。
- 异常处理：支付回调乱序、重复请求、取消退订、补偿事务。
- 稳定性治理：分区容灾、故障切流、服务降级与熔断。
- 指标体系：预定成功率、超卖率、取消率、确认时延 P99。

---

## 当前主题索引

- 预定系统总体设计（待补充）：`booking_system.md`
- 相关通用机制：
	- [distributed_transaction.md](../../base/distributed_transaction.md)
	- [idempotence.md](../../base/idempotence.md)
	- [rate_limiting.md](../../base/high_availability/rate_limiting.md)
- 相关组件能力：
	- [task_scheduler/TOPICS.md](../../components/task_scheduler/TOPICS.md)
- 相关工程治理：
	- [release_and_rollback.md](../../governance/release_and_rollback.md)
	- [incident_response.md](../../governance/incident_response.md)

