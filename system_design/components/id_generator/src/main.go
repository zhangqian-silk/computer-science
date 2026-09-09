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

	// 号段模式：跨段连续发号
	fmt.Println("\n号段模式（step=1000）：")
	seg := NewSegmentID(&mockDB{maxID: 1000}, 1000)
	segIDs := make([]int64, 0, 2500)
	for i := 0; i < 2500; i++ { // 跨越约 3 个号段
		segIDs = append(segIDs, seg.Next())
	}
	fmt.Printf("  生成 %d 个，首=%d 末=%d，连续递增=%v\n",
		len(segIDs), segIDs[0], segIDs[len(segIDs)-1], isStrictlyIncreasing(segIDs))
}

func isStrictlyIncreasing(a []int64) bool {
	for i := 1; i < len(a); i++ {
		if a[i] != a[i-1]+1 {
			return false
		}
	}
	return true
}
