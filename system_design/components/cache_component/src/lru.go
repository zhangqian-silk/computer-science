package main

import "container/list"

// entry 是双向链表节点承载的键值。
type entry struct {
	key   string
	value string
}

// LRUCache 用「哈希表 + 双向链表」实现 O(1) 的 Get/Put。
// 链表头为最近使用，尾为最久未使用（淘汰端）。
type LRUCache struct {
	cap   int
	ll    *list.List
	items map[string]*list.Element
	Hits  int
	Miss  int
}

func NewLRUCache(capacity int) *LRUCache {
	return &LRUCache{cap: capacity, ll: list.New(), items: make(map[string]*list.Element)}
}

// Get 命中则把节点移到表头并返回值。
func (c *LRUCache) Get(key string) (string, bool) {
	if el, ok := c.items[key]; ok {
		c.ll.MoveToFront(el)
		c.Hits++
		return el.Value.(*entry).value, true
	}
	c.Miss++
	return "", false
}

// Put 写入或更新；超容量则淘汰链表尾部（最久未使用）。
func (c *LRUCache) Put(key, value string) {
	if el, ok := c.items[key]; ok {
		el.Value.(*entry).value = value
		c.ll.MoveToFront(el)
		return
	}
	el := c.ll.PushFront(&entry{key: key, value: value})
	c.items[key] = el
	if c.ll.Len() > c.cap {
		c.evict()
	}
}

func (c *LRUCache) evict() {
	el := c.ll.Back()
	if el == nil {
		return
	}
	c.ll.Remove(el)
	delete(c.items, el.Value.(*entry).key)
}

// Keys 从最近到最久返回当前缓存的 key，便于观察淘汰顺序。
func (c *LRUCache) Keys() []string {
	out := make([]string, 0, c.ll.Len())
	for el := c.ll.Front(); el != nil; el = el.Next() {
		out = append(out, el.Value.(*entry).key)
	}
	return out
}
