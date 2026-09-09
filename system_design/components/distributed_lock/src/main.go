package main

import (
	"fmt"
	"time"
)

func main() {
	store := newKVStore()
	const key = "lock:order:1001"

	// 场景 1：互斥——A 持锁期间 B 无法获取
	a := NewLock(store, key, 2*time.Second)
	fmt.Printf("A 加锁: %v\n", a.TryLock(false))
	b := NewLock(store, key, 2*time.Second)
	fmt.Printf("B 加锁(应失败): %v\n", b.TryLock(false))
	fmt.Printf("A 解锁: %v\n", a.Unlock())
	fmt.Printf("A 释放后 B 加锁: %v\n\n", b.TryLock(false))
	b.Unlock()

	// 场景 2：无看门狗 + TTL 短于业务 → 锁提前过期，B 抢到锁
	c := NewLock(store, key, 1*time.Second)
	c.TryLock(false)
	fmt.Println("C 持锁，TTL=1s，业务需 2s，无看门狗...")
	time.Sleep(1200 * time.Millisecond)
	d := NewLock(store, key, 1*time.Second)
	fmt.Printf("锁过期后 D 抢到锁(并发风险): %v\n", d.TryLock(false))
	fmt.Printf("C 解锁(token 已非自己, 应失败): %v\n\n", c.Unlock())
	d.Unlock()

	// 场景 3：看门狗续约 → 业务超过初始 TTL 锁仍不过期
	e := NewLock(store, key, 1*time.Second)
	e.TryLock(true)
	fmt.Println("E 持锁 TTL=1s 且开看门狗，业务运行 3s...")
	time.Sleep(3 * time.Second)
	f := NewLock(store, key, 1*time.Second)
	fmt.Printf("业务期间 F 加锁(应失败): %v\n", f.TryLock(false))
	fmt.Printf("E 解锁: %v\n", e.Unlock())

	// 场景 4：fencing token 阻断 STW 停顿导致的双写
	fmt.Println("\n=== fencing token：阻断停顿导致的双写 ===")
	res := &Resource{}
	const fkey = "lock:account:1"
	g := NewLock(store, fkey, 1*time.Second)
	g.TryLock(false)
	fmt.Printf("G 获取锁，fence=%d；随后 G 发生长时间 STW 停顿，锁过期...\n", g.Fence)
	time.Sleep(1100 * time.Millisecond) // 模拟 G 停顿期间锁过期
	h := NewLock(store, fkey, 5*time.Second)
	h.TryLock(false)
	fmt.Printf("H 获取锁，fence=%d\n", h.Fence)
	fmt.Printf("H 写入(fence=%d): %v\n", h.Fence, res.Write(h.Fence, "H-data"))
	fmt.Printf("G 停顿结束，用旧 fence=%d 迟到写入: %v（被拒绝，避免双写）\n", g.Fence, res.Write(g.Fence, "G-data"))
	fmt.Printf("资源最终值: %q（最大 fence=%d）\n", res.value, res.maxFence)
}

// Resource 是受保护的外部资源：只接受 fence 不小于已见最大值的写入。
type Resource struct {
	maxFence int64
	value    string
}

func (r *Resource) Write(fence int64, data string) bool {
	if fence < r.maxFence {
		return false // 旧令牌，拒绝，防止停顿的旧持有者覆盖
	}
	r.maxFence = fence
	r.value = data
	return true
}
