package main

import (
	"hash/crc32"
	"sort"
	"strconv"
)

// Ring 是一个带虚拟节点的一致性哈希环（教学级实现）。
// 设计要点：
//   - 每个物理节点在环上放置 replicas 个虚拟节点，缓解数据倾斜。
//   - sortedHashes 保持有序，查询用二分找顺时针后继，复杂度 O(log M)。
type Ring struct {
	replicas     int               // 每个物理节点的虚拟节点数
	sortedHashes []uint32          // 升序排列的虚拟节点哈希值
	ringMap      map[uint32]string // 虚拟节点哈希 -> 物理节点名
	nodes        map[string]bool   // 已加入的物理节点集合
}

// NewRing 创建一个环，replicas 为每个物理节点的虚拟节点数。
func NewRing(replicas int) *Ring {
	return &Ring{
		replicas: replicas,
		ringMap:  make(map[uint32]string),
		nodes:    make(map[string]bool),
	}
}

func (r *Ring) hash(key string) uint32 {
	return crc32.ChecksumIEEE([]byte(key))
}

// Add 把一个物理节点加入环，展开成 replicas 个虚拟节点。
func (r *Ring) Add(node string) {
	if r.nodes[node] {
		return
	}
	r.nodes[node] = true
	for v := 0; v < r.replicas; v++ {
		h := r.hash(node + "#" + strconv.Itoa(v))
		r.ringMap[h] = node
		r.sortedHashes = append(r.sortedHashes, h)
	}
	sort.Slice(r.sortedHashes, func(i, j int) bool { return r.sortedHashes[i] < r.sortedHashes[j] })
}

// Remove 从环上移除一个物理节点及其全部虚拟节点。
func (r *Ring) Remove(node string) {
	if !r.nodes[node] {
		return
	}
	delete(r.nodes, node)
	kept := r.sortedHashes[:0]
	for _, h := range r.sortedHashes {
		if r.ringMap[h] == node {
			delete(r.ringMap, h)
			continue
		}
		kept = append(kept, h)
	}
	r.sortedHashes = kept
}

// Get 返回 key 顺时针方向遇到的第一个物理节点。
func (r *Ring) Get(key string) string {
	if len(r.sortedHashes) == 0 {
		return ""
	}
	h := r.hash(key)
	i := sort.Search(len(r.sortedHashes), func(i int) bool {
		return r.sortedHashes[i] >= h
	})
	if i == len(r.sortedHashes) { // 越过环尾则回到环首
		i = 0
	}
	return r.ringMap[r.sortedHashes[i]]
}
