package main

import "sync"

// mockDB 模拟号段表：allocate 相当于 UPDATE max_id = max_id + step 并返回新区间 [start, end)。
type mockDB struct {
	mu    sync.Mutex
	maxID int64
}

func (d *mockDB) allocate(step int64) (start, end int64) {
	d.mu.Lock()
	defer d.mu.Unlock()
	start = d.maxID
	d.maxID += step
	return start, d.maxID
}

type buffer struct{ cur, max int64 }

// SegmentID 是号段模式 ID 生成器：双 buffer + 阈值预取。
type SegmentID struct {
	db   *mockDB
	step int64
	mu   sync.Mutex
	cur  *buffer
	next *buffer // 备用 buffer，消耗到阈值时提前填充
}

func NewSegmentID(db *mockDB, step int64) *SegmentID {
	s, e := db.allocate(step)
	return &SegmentID{db: db, step: step, cur: &buffer{cur: s, max: e}}
}

func (g *SegmentID) prefetchLocked() {
	if g.next == nil {
		s, e := g.db.allocate(g.step)
		g.next = &buffer{cur: s, max: e}
	}
}

// Next 返回下一个 ID；消耗到 90% 时预取备用段，当前段用尽时无缝切换。
func (g *SegmentID) Next() int64 {
	g.mu.Lock()
	defer g.mu.Unlock()
	if float64(g.cur.max-g.cur.cur) <= float64(g.step)*0.1 {
		g.prefetchLocked() // 生产中此步应异步，示例为可读性同步执行
	}
	if g.cur.cur >= g.cur.max {
		g.prefetchLocked()
		g.cur = g.next
		g.next = nil
	}
	id := g.cur.cur
	g.cur.cur++
	return id
}
