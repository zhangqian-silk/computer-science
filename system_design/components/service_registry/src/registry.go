package main

import (
	"sync"
	"time"
)

// Instance 是一个服务实例。
type Instance struct {
	ID       string
	Addr     string
	lastBeat time.Time
}

// Registry 是带心跳 TTL 的教学级注册中心。
type Registry struct {
	mu       sync.RWMutex
	ttl      time.Duration
	services map[string]map[string]*Instance // 服务名 -> 实例ID -> 实例
}

func NewRegistry(ttl time.Duration) *Registry {
	return &Registry{ttl: ttl, services: make(map[string]map[string]*Instance)}
}

// Register 注册实例并记录首个心跳时间。
func (r *Registry) Register(service, id, addr string) {
	r.mu.Lock()
	defer r.mu.Unlock()
	if r.services[service] == nil {
		r.services[service] = make(map[string]*Instance)
	}
	r.services[service][id] = &Instance{ID: id, Addr: addr, lastBeat: time.Now()}
}

// Heartbeat 续约：刷新实例的最近心跳时间。
func (r *Registry) Heartbeat(service, id string) bool {
	r.mu.Lock()
	defer r.mu.Unlock()
	if insts, ok := r.services[service]; ok {
		if inst, ok := insts[id]; ok {
			inst.lastBeat = time.Now()
			return true
		}
	}
	return false
}

// Deregister 主动注销。
func (r *Registry) Deregister(service, id string) {
	r.mu.Lock()
	defer r.mu.Unlock()
	if insts, ok := r.services[service]; ok {
		delete(insts, id)
	}
}

// Discover 返回 TTL 内仍健康的实例地址（自动跳过过期实例）。
func (r *Registry) Discover(service string) []string {
	r.mu.RLock()
	defer r.mu.RUnlock()
	var out []string
	now := time.Now()
	for _, inst := range r.services[service] {
		if now.Sub(inst.lastBeat) <= r.ttl {
			out = append(out, inst.Addr)
		}
	}
	return out
}
