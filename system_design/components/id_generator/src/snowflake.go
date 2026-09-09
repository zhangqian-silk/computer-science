package main

import (
	"errors"
	"sync"
	"time"
)

// Snowflake 位段：1 符号位 + 41 时间戳 + 10 机器 + 12 序列。
const (
	epoch         int64 = 1704067200000 // 自定义起点：2024-01-01 UTC，单位毫秒
	workerBits    uint  = 10
	seqBits       uint  = 12
	workerMax     int64 = -1 ^ (-1 << workerBits) // 1023
	seqMask       int64 = -1 ^ (-1 << seqBits)    // 4095
	workerShift         = seqBits
	tsShift             = seqBits + workerBits
	maxBackwardMs int64 = 5 // 允许等待的最大时钟回拨
)

// Snowflake 是并发安全的 ID 生成器。
type Snowflake struct {
	mu       sync.Mutex
	lastTs   int64
	seq      int64
	workerID int64
}

func NewSnowflake(workerID int64) (*Snowflake, error) {
	if workerID < 0 || workerID > workerMax {
		return nil, errors.New("workerID out of range [0, 1023]")
	}
	return &Snowflake{workerID: workerID, lastTs: -1}, nil
}

func (s *Snowflake) NextID() (int64, error) {
	s.mu.Lock()
	defer s.mu.Unlock()

	now := time.Now().UnixMilli()
	if now < s.lastTs { // 时钟回拨
		if s.lastTs-now <= maxBackwardMs {
			for now < s.lastTs { // 小幅回拨：自旋等待时钟追上
				now = time.Now().UnixMilli()
			}
		} else {
			return 0, errors.New("clock moved backwards, refusing to generate id")
		}
	}

	if now == s.lastTs {
		s.seq = (s.seq + 1) & seqMask
		if s.seq == 0 { // 当前毫秒序列用尽，等待下一毫秒
			for now <= s.lastTs {
				now = time.Now().UnixMilli()
			}
		}
	} else {
		s.seq = 0
	}

	s.lastTs = now
	return ((now - epoch) << tsShift) | (s.workerID << workerShift) | s.seq, nil
}
