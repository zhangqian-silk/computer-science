package main

// task 是时间轮中的一个延迟任务；rounds 表示还需转多少整圈才触发。
type task struct {
	id     string
	rounds int
	run    func(id string, tick int)
}

// TimingWheel 是单层时间轮：每 tick 前进一格，槽位数为 slots。
type TimingWheel struct {
	slots   int
	buckets [][]*task
	pos     int // 当前指针位置
	tick    int // 已走过的总 tick 数（逻辑时间）
}

func NewTimingWheel(slots int) *TimingWheel {
	return &TimingWheel{slots: slots, buckets: make([][]*task, slots)}
}

// Add 注册一个延迟 delay 个 tick 后触发的任务。
func (w *TimingWheel) Add(id string, delay int, run func(id string, tick int)) {
	if delay < 0 {
		delay = 0
	}
	slot := (w.pos + delay) % w.slots
	rounds := delay / w.slots
	w.buckets[slot] = append(w.buckets[slot], &task{id: id, rounds: rounds, run: run})
}

// Advance 前进一个 tick：触发当前槽内 rounds==0 的任务，其余 rounds 减一。
func (w *TimingWheel) Advance() {
	w.tick++
	w.pos = (w.pos + 1) % w.slots
	bucket := w.buckets[w.pos]
	kept := bucket[:0]
	for _, t := range bucket {
		if t.rounds == 0 {
			t.run(t.id, w.tick)
			continue
		}
		t.rounds--
		kept = append(kept, t)
	}
	w.buckets[w.pos] = kept
}
