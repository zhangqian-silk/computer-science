package main

import "fmt"

func main() {
	const slots = 8
	w := NewTimingWheel(slots)

	fire := func(id string, tick int) {
		fmt.Printf("  tick=%2d 触发任务 %s\n", tick, id)
	}

	// 注册不同延迟的任务：3、8（正好一圈）、20（超过两圈）
	w.Add("close-order-A", 3, fire)
	w.Add("close-order-B", 8, fire)
	w.Add("close-order-C", 20, fire)
	fmt.Printf("时间轮槽位=%d，注册延迟 3/8/20 的任务，开始推进...\n", slots)

	for i := 0; i < 21; i++ {
		w.Advance()
	}

	// 最小堆调度器：同一批延迟任务，触发时刻应一致
	fmt.Println("\n最小堆调度器，注册延迟 3/8/20 的任务：")
	hs := NewHeapScheduler()
	hs.Add("close-order-A", 3, fire)
	hs.Add("close-order-B", 8, fire)
	hs.Add("close-order-C", 20, fire)
	for i := 0; i < 21; i++ {
		hs.Advance()
	}
	fmt.Printf("剩余待触发：%d\n", hs.Pending())
}
