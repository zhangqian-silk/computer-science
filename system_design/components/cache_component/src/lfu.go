package main

// LFUCache 是教学级 LFU：按访问频率淘汰，平局淘汰同频中最早进入者。
// 用 freq -> 有序 key 列表 + 记录 minFreq 实现近似 O(1)。
type lfuEntry struct {
	value string
	freq  int
	seq   int // 同频内的插入顺序，用于平局淘汰
}

type LFUCache struct {
	cap     int
	items   map[string]*lfuEntry
	freqs   map[int][]string // freq -> 该频率下的 key（按插入顺序）
	minFreq int
	seq     int
	Hits    int
	Miss    int
}

func NewLFUCache(capacity int) *LFUCache {
	return &LFUCache{cap: capacity, items: make(map[string]*lfuEntry), freqs: make(map[int][]string)}
}

func removeKey(list []string, key string) []string {
	for i, k := range list {
		if k == key {
			return append(list[:i], list[i+1:]...)
		}
	}
	return list
}

func (c *LFUCache) touch(key string, e *lfuEntry) {
	c.freqs[e.freq] = removeKey(c.freqs[e.freq], key)
	if len(c.freqs[e.freq]) == 0 {
		delete(c.freqs, e.freq)
		if c.minFreq == e.freq {
			c.minFreq++
		}
	}
	e.freq++
	c.freqs[e.freq] = append(c.freqs[e.freq], key)
}

func (c *LFUCache) Get(key string) (string, bool) {
	if e, ok := c.items[key]; ok {
		c.touch(key, e)
		c.Hits++
		return e.value, true
	}
	c.Miss++
	return "", false
}

func (c *LFUCache) Put(key, value string) {
	if c.cap == 0 {
		return
	}
	if e, ok := c.items[key]; ok {
		e.value = value
		c.touch(key, e)
		return
	}
	if len(c.items) >= c.cap { // 淘汰最低频、同频最早
		victim := c.freqs[c.minFreq][0]
		c.freqs[c.minFreq] = c.freqs[c.minFreq][1:]
		if len(c.freqs[c.minFreq]) == 0 {
			delete(c.freqs, c.minFreq)
		}
		delete(c.items, victim)
	}
	c.seq++
	c.items[key] = &lfuEntry{value: value, freq: 1, seq: c.seq}
	c.freqs[1] = append(c.freqs[1], key)
	c.minFreq = 1
}
