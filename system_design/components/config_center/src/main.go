package main

import (
	"fmt"
	"sync"
)

func main() {
	cc := NewConfigCenter("rate_limit=100")

	// 一个客户端监听配置变更，动态更新本地内存值（无需重启）
	var wg sync.WaitGroup
	wg.Add(1)
	ch := cc.Watch()
	go func() {
		defer wg.Done()
		for i := 0; i < 4; i++ { // 初始值 + 3 次变更
			v := <-ch
			fmt.Printf("  客户端收到配置 v%d: %s\n", v.ver, v.value)
		}
	}()

	fmt.Println("发布与回滚：")
	fmt.Printf("发布 v%d\n", cc.Publish("rate_limit=200"))
	fmt.Printf("发布 v%d（错误配置）\n", cc.Publish("rate_limit=0"))
	fmt.Printf("回滚到 v2: %v\n", cc.Rollback(2)) // 秒级回滚到正确版本

	wg.Wait()
}
