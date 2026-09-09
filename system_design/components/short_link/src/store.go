package main

import "sync"

// Shortener 是一个教学级短链服务：发号器（自增 ID）+ 62 进制编码 + 内存存储。
type Shortener struct {
	mu      sync.RWMutex
	seq     uint64            // 发号器
	longToC map[string]string // 长链 -> 短码（复用，避免重复发号）
	cToLong map[string]string // 短码 -> 长链
}

func NewShortener(startID uint64) *Shortener {
	return &Shortener{
		seq:     startID,
		longToC: make(map[string]string),
		cToLong: make(map[string]string),
	}
}

// Shorten 返回长链对应的短码；相同长链复用同一短码。
func (s *Shortener) Shorten(long string) string {
	s.mu.Lock()
	defer s.mu.Unlock()
	if c, ok := s.longToC[long]; ok {
		return c
	}
	s.seq++
	code := Encode(s.seq)
	s.longToC[long] = code
	s.cToLong[code] = long
	return code
}

// Resolve 根据短码解析出长链。
func (s *Shortener) Resolve(code string) (string, bool) {
	s.mu.RLock()
	defer s.mu.RUnlock()
	long, ok := s.cToLong[code]
	return long, ok
}
