package main

import "container/heap"

// delayTask 是堆中的延迟任务，dueTick 为到期的逻辑时刻。
type delayTask struct {
	id      string
	dueTick int
	run     func(id string, tick int)
}

// taskHeap 是按 dueTick 排序的最小堆。
type taskHeap []*delayTask

func (h taskHeap) Len() int            { return len(h) }
func (h taskHeap) Less(i, j int) bool  { return h[i].dueTick < h[j].dueTick }
func (h taskHeap) Swap(i, j int)       { h[i], h[j] = h[j], h[i] }
func (h *taskHeap) Push(x any)         { *h = append(*h, x.(*delayTask)) }
func (h *taskHeap) Pop() any {
	old := *h
	n := len(old)
	t := old[n-1]
	*h = old[:n-1]
	return t
}

// HeapScheduler 用最小堆管理延迟任务：O(log n) 入队，堆顶即最早到期。
type HeapScheduler struct {
	h    taskHeap
	tick int
}

func NewHeapScheduler() *HeapScheduler {
	s := &HeapScheduler{}
	heap.Init(&s.h)
	return s
}

func (s *HeapScheduler) Add(id string, delay int, run func(id string, tick int)) {
	heap.Push(&s.h, &delayTask{id: id, dueTick: s.tick + delay, run: run})
}

// Advance 前进一个 tick，触发所有已到期任务。
func (s *HeapScheduler) Advance() {
	s.tick++
	for s.h.Len() > 0 && s.h[0].dueTick <= s.tick {
		t := heap.Pop(&s.h).(*delayTask)
		t.run(t.id, s.tick)
	}
}

func (s *HeapScheduler) Pending() int { return s.h.Len() }
