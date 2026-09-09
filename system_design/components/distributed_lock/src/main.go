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
}
