# 限流（Rate Limiting）

> **摘要**：限流通过限制请求速率或并发量保护系统免于过载。本文覆盖四种算法（固定窗口/滑动窗口/漏桶/令牌桶，含 Redis 伪码）、限流的三个维度（对象/位置/口径）、单机 vs 分布式限流的实现与坑（Redis+Lua 原子）、自适应/并发限流、限流后的响应策略、故障模式与常见误区。

> **前置阅读**：限流与[熔断](./circuit_breaker.md)、[降级](./degradation.md)、[隔离舱壁](./bulkhead_isolation.md)组成立体防护；分布式限流依赖 Redis 原子操作；组件化视角见[限流器组件](../../components/rate_limiter/TOPICS.md)。

限流的主要目标是通过限制请求速率或并发量，保护系统免受过载影响，确保核心服务稳定运行，主要场景如下：

- **负载过高**：如服务器性能一般，机器数量较少，请求耗时长
- **突发流量**：如秒杀活动、热点事件带来的瞬时高并发
- **恶意攻击**：如 DDoS 攻击、爬虫高频请求
- **下游限额**：如下游服务有明确的 QPS 调用要求

下面的交互组件对比「令牌桶」与「漏桶」在同一突发到达序列下的放行/拒绝差异，可先感性理解四种算法要解决的核心问题——**如何对待突发**：

<RateLimiterExplorer />

## 限流的三个维度

设计限流先想清楚三件事，而非一上来就选算法：

- **限流对象（对谁）**：全局、单 IP、单用户、单接口、单租户/渠道、单实例——粒度越细越精准但状态越多。常多维叠加（如「单用户 + 单接口」）。
- **限流位置（在哪）**：接入层/网关（挡住大部分流量，最省资源）、服务层（保护自身）、依赖调用层（遵守下游 QPS 配额）。**越靠前越好**，把无效流量挡在下游之前。
- **限流口径（限什么）**：QPS（速率）还是并发数（in-flight）。速率适合防突发，并发数适合保护慢资源（如 DB 连接）——二者常配合。

## 固定窗口计数器（Fixed Window）

将时间划分为多个时间窗口，统计单位时间（如1秒）内的请求数，超限则拒绝，到达下一个时间窗口时重置计数器。

**优点**

- 实现简单

**缺点**

- 限流不够平滑，请求有可能集中在窗口前一半，导致整体有一半的时间完全不可用
- 无法保障速率，窗口切换时可能双倍流量（如第0.9秒和第1.1秒各100次）

**单机实现**

```python
def allow_request():
    now = current_time()
    if now - window_start > window_sec:  # 窗口过期则重置
        counter = 0
        window_start = now
    if counter >= threshold:
        return False
    counter += 1
    return True
```

**分布式实现**

```python
# Redis key: rate_limit:{service}
def allow_request(key):
    current = redis.incr(key)
    if current == 1:
        redis.expire(key, window_sec)
    return current <= threshold
```

## 滑动窗口计数器（Sliding Window）

将时间分割为多个小窗口（如 1 分钟分为 60 个 1 秒窗口），动态统计最近 N 个窗口的总请求。

**优点**

- 解决固定窗口的临界问题，精度更高

**缺点**

- 内存占用和计算量随子窗口数增加而上升。  

**单机实现**

```python
# slot_size = 1 秒、slot_count = 60
# 表示时间轴按 1 秒分槽，共 60 个槽，覆盖 60 秒的窗口
def allow_request():
    now = current_time()
    slot_idx = (now // slot_size) % slot_count
    if slot_idx != current_slot:  # 新子窗口，重置旧数据
        slots[slot_idx] = 0
        current_slot = slot_idx
    if sum(slots) >= threshold:
        return False
    slots[slot_idx] += 1
    return True
```

**分布式实现**

```python
# Redis key: sliding_window:{service}
def allow_request(key):
    now = current_time()
    redis.ZREMRANGEBYSCORE(key, 0, now - window_sec)
    redis.ZADD(key, {now: now})
    redis.EXPIRE(key, window_sec)
    count = redis.ZCARD(key)
    return count <= threshold:
```

## 漏桶算法（Leaky Bucket）

请求以任意速率进入桶，以固定速率流出，桶满则拒绝。

**优点**

- 严格平滑流量，可以控制限流速率

**缺点**

- 无法应对突发流量，只能以固定速率处理
- 如果处理速度始终小于请求发送速度，在桶满后，大部分新请求会被丢弃，服务可用性下降

**单机实现**

```python
def allow_request():
    now = current_time()
    # 计算漏出水量：时间差 * 流出速率
    leaked = (now - last_leak_time) * leak_rate
    water = max(0, water - leaked)
    last_leak_time = now
    if water >= threshold:
        return False
    water += 1
    return True
```

**分布式实现**

```python
# Redis key: leaky_bucket:{service}
def allow_request(key):
    script = """
    local key = KEYS[1]
    local leak_rate = tonumber(ARGV[1])
    local threshold = tonumber(ARGV[2])
    local now = tonumber(ARGV[3])
    local current_level, last_time = redis.call('HMGET', key, 'current_level', 'last_time' )
    current_level = tonumber(current_level) or 0
    last_time = tonumber(last_time) or now
    local leaked = (now - last_time) * leak_rate
    current_level = math.max(current_level - leaked, 0)
    if current_level + 1 <= threshold then
        redis.call('HMSET', key, 'last_time', now, 'current_level', current_level + 1)
        redis.call('EXPIRE', key, math.ceil(threshold / leak_rate))
        return 1
    else
        return 0
    end
    """

    return redis.EVAL(script, 1, key, leak_rate, threshold, current_time())
```

## 令牌桶算法（Token Bucket）

系统以特定速率生成令牌并放入令牌桶，直至桶满，请求需获取令牌才能被处理。

**优点**

- 允许突发流量（最多取完桶内令牌），兼顾灵活性与保护性
- 可以根据实际负载，动态调整令牌生成速率

**缺点**

- 流出速度不受管控，需要额外考虑令牌产生速率与桶的容量
- 需要维护令牌状态，实现略复杂

**单机实现**

```python
def allow_request():
    now = current_time()
    # 计算新增令牌数：时间差 * 生成速率
    new_tokens = (now - last_refill_time) * refill_rate
    tokens = min(threshold, tokens + new_tokens)
    last_refill_time = now
    if tokens < 1:
        return False
    tokens -= 1
    return True
```

**分布式实现**

```python
# Redis key: token_bucket:{service}
def allow_request(key):
    script = """
    local key = KEYS[1]
    local refill_rate = tonumber(ARGV[1])
    local threshold = tonumber(ARGV[2])
    local now = tonumber(ARGV[3])
    local tokens, last_refill = redis.call('HMGET', key, 'tokens', 'last_refill')
    tokens = tonumber(tokens) or threshold
    last_refill = tonumber(last_refill) or now
    local time_passed = now - last_refill
    if time_passed > 0 then
        local new_tokens = time_passed * refill_rate
        tokens = math.min(tokens + new_tokens, threshold)
        last_refill = now
    end
    if tokens >= 1 then
        tokens = tokens - 1
        redis.call('HMSET', key, 'tokens', tokens, 'last_refill', last_refill)
        redis.call('EXPIRE', key, math.ceil(threshold / refill_rate) * 2)
        return 1
    else
        return 0
    end
    """

    return redis.EVAL(script, 1, key, refill_rate, threshold, current_time())
```

## 单机 vs 分布式限流

- **单机限流**：进程内计数（如 Guava RateLimiter、`golang.org/x/time/rate`），零网络开销、精准，但只管本机——N 台机器实际总量是单机阈值 × N。
- **分布式限流**：多实例共享一个全局配额，用 **Redis + Lua** 把「读计数 + 判断 + 扣减」做成一条原子脚本（避免先读后写的并发超发）。代价是每次限流判定一次 Redis 往返。
- **两级限流（推荐）**：本地先粗过滤（挡住明显超限），再由 Redis 做全局精确配额，兼顾性能与准确。
- **集群配额分配**：把全局配额按实例数**动态分摊**到各实例本地（如全局 1000 QPS、10 实例各 100），实例增减时重分配——减少 Redis 依赖，但分配不均时有误差。

```text
DISTRIBUTED-TOKEN-BUCKET(key, rate, capacity)   // Redis + Lua 原子
    // 一条脚本内完成：按时间补充令牌 → 判断是否有令牌 → 扣减
    // 避免「读令牌数」与「扣减」之间的并发窗口导致超发
```

## 自适应与并发限流

固定阈值难整定且不适应容量变化，进阶做法：

- **并发数限流**：限制同时在处理的请求数（信号量），直接保护慢资源（DB 连接池、线程池），比 QPS 更贴合「资源是否被占满」。
- **自适应限流**：根据系统实时指标（RT、CPU、并发、成功率）动态调阈值——如 Sentinel 的系统自适应、TCP BBR 思路、Netflix concurrency-limits（基于 RTT 梯度自动探测最大并发）。免去人工整定，能随容量波动自动收放。

## 限流后的响应策略

被限流的请求不只是「拒绝」，按业务选择：

- **快速拒绝**：返回 429/友好提示（「系统繁忙，请重试」），最常见。
- **排队等待**：漏桶式排队 + 超时，平滑突发（适合可等待的写）。
- **降级兜底**：转[降级](./degradation.md)返回默认值/缓存。
- **优先级**：高优先级请求优先放行，低优先级先限（如保交易限营销）。

## 故障模式与常见误区

| 问题 | 说明 | 对策 |
| :--- | :--- | :--- |
| 分布式超发 | 先读后写非原子 | Redis + Lua 原子脚本 |
| 固定窗口临界突刺 | 窗口边界两倍流量 | 滑动窗口 |
| Redis 限流单点/抖动 | 强依赖 Redis | 本地兜底 + 降级放行/拒绝策略 |
| 阈值拍脑袋 | 静态阈值难准 | 压测定阈值 + 自适应限流 |
| 限流位置太靠后 | 无效流量已消耗资源 | 尽量前置到网关/接入层 |

- **误区：固定窗口就够。** 窗口边界会放过两倍瞬时流量，突发敏感用滑动窗口/令牌桶。
- **误区：单机限流等于全局限流。** 多实例总量是单机 × N，需分布式限流或配额分摊。
- **误区：限流只能拒绝。** 可排队、降级、按优先级放行。
- **误区：阈值设一次不变。** 应压测定基线并考虑自适应限流随容量调整。

## Ref

- <https://javaguide.cn/high-availability/limit-request.html>
- <https://github.com/2637309949/go-interview/blob/master/docs%2FNetwork%2F%E9%99%90%E6%B5%81%E7%AD%96%E7%95%A5.md>
