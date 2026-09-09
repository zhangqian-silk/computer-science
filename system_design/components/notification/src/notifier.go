package main

import (
	"fmt"
	"sync"
)

// SendResult 表示一次发送的处理结果。
type SendResult string

const (
	Delivered   SendResult = "已送达"
	Duplicated  SendResult = "去重跳过"
	RateLimited SendResult = "频控拦截"
)

// Notifier 是教学级通知中心：内置去重与单用户频控。
type Notifier struct {
	mu         sync.Mutex
	dedup      map[string]bool // 幂等键 -> 是否已发送
	dailyLimit int             // 单用户每渠道上限
	counter    map[string]int  // 用户#渠道 -> 已发送数
}

func NewNotifier(dailyLimit int) *Notifier {
	return &Notifier{
		dedup:      make(map[string]bool),
		dailyLimit: dailyLimit,
		counter:    make(map[string]int),
	}
}

// Send 处理一条通知：先去重，再频控，通过后送达。
func (n *Notifier) Send(eventID, user, channel, content string) SendResult {
	n.mu.Lock()
	defer n.mu.Unlock()

	key := eventID + "#" + user + "#" + channel // 业务幂等键
	if n.dedup[key] {
		return Duplicated
	}
	ck := user + "#" + channel
	if n.counter[ck] >= n.dailyLimit {
		return RateLimited
	}
	n.dedup[key] = true
	n.counter[ck]++
	_ = content // 实际这里会走渠道路由与重试
	return Delivered
}

func (n *Notifier) String() string {
	return fmt.Sprintf("dailyLimit=%d", n.dailyLimit)
}
