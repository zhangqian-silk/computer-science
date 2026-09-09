package main

import "sync"

// version 记录一次配置发布。
type version struct {
	ver   int
	value string
}

// ConfigCenter 是教学级配置中心：支持发布、版本回滚与变更监听（Watch）。
type ConfigCenter struct {
	mu        sync.Mutex
	current   version
	history   []version
	listeners []chan version // 订阅者，模拟长轮询/Watch 推送
}

func NewConfigCenter(initial string) *ConfigCenter {
	v := version{ver: 1, value: initial}
	return &ConfigCenter{current: v, history: []version{v}}
}

// Watch 注册一个监听通道，配置变更时收到最新版本（动态生效，无需重启）。
func (c *ConfigCenter) Watch() <-chan version {
	c.mu.Lock()
	defer c.mu.Unlock()
	ch := make(chan version, 8)
	ch <- c.current // 首次同步当前值
	c.listeners = append(c.listeners, ch)
	return ch
}

func (c *ConfigCenter) notify(v version) {
	for _, ch := range c.listeners {
		ch <- v
	}
}

// Publish 发布新配置，产生新版本并推送给所有监听者。
func (c *ConfigCenter) Publish(value string) int {
	c.mu.Lock()
	defer c.mu.Unlock()
	v := version{ver: c.current.ver + 1, value: value}
	c.current = v
	c.history = append(c.history, v)
	c.notify(v)
	return v.ver
}

// Rollback 回滚到指定历史版本（作为一次新发布推送出去）。
func (c *ConfigCenter) Rollback(target int) bool {
	c.mu.Lock()
	defer c.mu.Unlock()
	for _, h := range c.history {
		if h.ver == target {
			v := version{ver: c.current.ver + 1, value: h.value}
			c.current = v
			c.history = append(c.history, v)
			c.notify(v)
			return true
		}
	}
	return false
}
