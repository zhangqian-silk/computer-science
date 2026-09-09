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
}
