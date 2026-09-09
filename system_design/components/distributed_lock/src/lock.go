package main

import (
	"crypto/rand"
	"encoding/hex"
	"time"
)

// Lock 是基于 kvStore 的分布式锁，演示 token 唯一性与看门狗续约。
type Lock struct {
	store *kvStore
	key   string
	token string
	ttl   time.Duration
	stop  chan struct{}
}

func newToken() string {
	b := make([]byte, 16)
	_, _ = rand.Read(b)
	return hex.EncodeToString(b)
}

func NewLock(store *kvStore, key string, ttl time.Duration) *Lock {
	return &Lock{store: store, key: key, ttl: ttl}
}

// TryLock 尝试加锁；withWatchdog 为 true 时启动后台续约。
func (l *Lock) TryLock(withWatchdog bool) bool {
	token := newToken()
	if !l.store.SetNX(l.key, token, l.ttl) {
		return false
	}
	l.token = token
	if withWatchdog {
		l.stop = make(chan struct{})
		go l.watchdog()
	}
	return true
}

func (l *Lock) watchdog() {
	ticker := time.NewTicker(l.ttl / 3)
	defer ticker.Stop()
	for {
		select {
		case <-l.stop:
			return
		case <-ticker.C:
			if !l.store.RenewIfMatch(l.key, l.token, l.ttl) {
				return // 锁已不属于自己，停止续约
			}
		}
	}
}

// Unlock 停止续约并校验 token 后释放。
func (l *Lock) Unlock() bool {
	if l.stop != nil {
		close(l.stop)
		l.stop = nil
	}
	return l.store.DelIfMatch(l.key, l.token)
}
