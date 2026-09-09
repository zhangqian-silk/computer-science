package main

import (
	"sync"
	"time"
)

// kvStore 模拟带 TTL 的 Redis：仅实现锁需要的 SET NX EX、带 token 校验的 DEL、以及续约。
type kvStore struct {
	mu     sync.Mutex
	val    map[string]string
	expire map[string]time.Time
}

func newKVStore() *kvStore {
	return &kvStore{val: make(map[string]string), expire: make(map[string]time.Time)}
}

func (s *kvStore) evictLocked(key string) {
	if exp, ok := s.expire[key]; ok && time.Now().After(exp) {
		delete(s.val, key)
		delete(s.expire, key)
	}
}

// SetNX 对应 SET key token NX EX ttl：不存在（或已过期）才写入。
func (s *kvStore) SetNX(key, token string, ttl time.Duration) bool {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.evictLocked(key)
	if _, ok := s.val[key]; ok {
		return false
	}
	s.val[key] = token
	s.expire[key] = time.Now().Add(ttl)
	return true
}

// DelIfMatch 对应解锁 Lua：仅当 value == token 才删除。
func (s *kvStore) DelIfMatch(key, token string) bool {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.evictLocked(key)
	if s.val[key] != token {
		return false
	}
	delete(s.val, key)
	delete(s.expire, key)
	return true
}

// RenewIfMatch 看门狗续约：仅当仍是自己的锁才重置 TTL。
func (s *kvStore) RenewIfMatch(key, token string, ttl time.Duration) bool {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.evictLocked(key)
	if s.val[key] != token {
		return false
	}
	s.expire[key] = time.Now().Add(ttl)
	return true
}
