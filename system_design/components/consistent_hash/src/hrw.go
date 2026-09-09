package main

import "hash/crc32"

// HRW（Highest Random Weight，又称 Rendezvous Hashing）：
// 对每个 (key, node) 计算一个权重，取权重最大的 node。
// 特点：无需环、无需虚拟节点即可较均匀；增删节点时只有「归属被删节点」的 key 迁移，
// 天然满足单调性。代价是查询 O(n)（需遍历所有节点），节点数很大时不划算。
func HRW(key string, nodes []string) string {
	var best string
	var bestWeight uint32
	for _, node := range nodes {
		w := crc32.ChecksumIEEE([]byte(key + "@" + node))
		if best == "" || w > bestWeight {
			bestWeight = w
			best = node
		}
	}
	return best
}
