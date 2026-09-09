package main

import (
	"fmt"
	"sort"
	"time"
)

func show(r *Registry, service string) {
	addrs := r.Discover(service)
	sort.Strings(addrs)
	fmt.Printf("  发现 %s 的健康实例: %v\n", service, addrs)
}

func main() {
	r := NewRegistry(2 * time.Second) // TTL=2s
	const svc = "order-service"

	r.Register(svc, "i-1", "10.0.0.1:8080")
	r.Register(svc, "i-2", "10.0.0.2:8080")
	fmt.Println("两实例注册后：")
	show(r, svc)

	// i-2 停止心跳，i-1 持续续约
	fmt.Println("\ni-2 停止心跳，i-1 持续续约，等待 TTL 过期...")
	for i := 0; i < 3; i++ {
		time.Sleep(900 * time.Millisecond)
		r.Heartbeat(svc, "i-1")
	}
	show(r, svc) // i-2 因超过 TTL 被自动摘除

	// i-1 优雅下线
	r.Deregister(svc, "i-1")
	fmt.Println("\ni-1 优雅下线后：")
	show(r, svc)
}
