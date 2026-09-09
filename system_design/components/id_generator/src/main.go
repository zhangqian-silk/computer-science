package main

import (
	"fmt"
	"sync"
)

func main() {
	sf, err := NewSnowflake(1)
	if err != nil {
		panic(err)
	}

	// 顺序生成，观察趋势递增
	fmt.Println("顺序生成 5 个 ID：")
	for i := 0; i < 5; i++ {
		id, _ := sf.NextID()
		fmt.Printf("  %d\n", id)
	}

	// 并发生成 10000 个，校验全局唯一
	const n = 10000
	var wg sync.WaitGroup
	var mu sync.Mutex
	seen := make(map[int64]struct{}, n)
	dup := 0
	for i := 0; i < n; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			id, err := sf.NextID()
			if err != nil {
				return
			}
			mu.Lock()
			if _, ok := seen[id]; ok {
				dup++
			}
			seen[id] = struct{}{}
			mu.Unlock()
		}()
	}
	wg.Wait()
	fmt.Printf("\n并发生成 %d 个 ID，唯一 %d 个，重复 %d 个\n", n, len(seen), dup)
}
