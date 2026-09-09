package main

import (
	"fmt"
	"hash/crc32"
	"math"
	"sort"
)

const keyCount = 100000

func keyOf(i int) string { return fmt.Sprintf("key-%d", i) }

// stdevPct 返回一组计数相对均值的相对标准差（%），衡量负载均衡程度，越小越均衡。
func stdevPct(counts []int) float64 {
	if len(counts) == 0 {
		return 0
	}
	var sum float64
	for _, c := range counts {
		sum += float64(c)
	}
	mean := sum / float64(len(counts))
	var variance float64
	for _, c := range counts {
		d := float64(c) - mean
		variance += d * d
	}
	return math.Sqrt(variance/float64(len(counts))) / mean * 100
}

// --- 方案一：一致性哈希环 + 虚拟节点 ---
func evalRing() {
	fmt.Println("== 方案一：一致性哈希环 + 虚拟节点 ==")
	physical := []string{"A", "B", "C", "D"}
	for _, v := range []int{1, 20, 200} {
		r := NewRing(v)
		for _, n := range physical {
			r.Add(n)
		}
		cnt := map[string]int{}
		for i := 0; i < keyCount; i++ {
			cnt[r.Get(keyOf(i))]++
		}
		counts := make([]int, 0, len(physical))
		for _, n := range physical {
			counts = append(counts, cnt[n])
		}
		fmt.Printf("  虚拟节点=%-3d  负载相对标准差=%.1f%%\n", v, stdevPct(counts))
	}
	// 迁移比例：加一个节点
	r := NewRing(200)
	for _, n := range physical {
		r.Add(n)
	}
	before := make([]string, keyCount)
	for i := 0; i < keyCount; i++ {
		before[i] = r.Get(keyOf(i))
	}
	r.Add("E")
	migrated := 0
	for i := 0; i < keyCount; i++ {
		if r.Get(keyOf(i)) != before[i] {
			migrated++
		}
	}
	fmt.Printf("  4→5 节点迁移比例=%.1f%%（理想 1/(N+1)≈%.1f%%）\n\n",
		float64(migrated)/keyCount*100, 100.0/5)
}

// --- 方案二：跳跃一致性哈希 ---
func evalJump() {
	fmt.Println("== 方案二：跳跃一致性哈希（Jump） ==")
	for _, n := range []int{4, 8} {
		counts := make([]int, n)
		for i := 0; i < keyCount; i++ {
			counts[JumpConsistentHash(uint64(crc32.ChecksumIEEE([]byte(keyOf(i)))), n)]++
		}
		fmt.Printf("  桶数=%d  负载相对标准差=%.1f%%\n", n, stdevPct(counts))
	}
	migrated := 0
	for i := 0; i < keyCount; i++ {
		h := uint64(crc32.ChecksumIEEE([]byte(keyOf(i))))
		if JumpConsistentHash(h, 4) != JumpConsistentHash(h, 5) {
			migrated++
		}
	}
	fmt.Printf("  4→5 桶迁移比例=%.1f%%（理想 1/(N+1)≈%.1f%%）\n\n",
		float64(migrated)/keyCount*100, 100.0/5)
}

// --- 方案三：HRW / Rendezvous ---
func evalHRW() {
	fmt.Println("== 方案三：HRW（Rendezvous） ==")
	nodes := []string{"A", "B", "C", "D"}
	cnt := map[string]int{}
	before := make([]string, keyCount)
	for i := 0; i < keyCount; i++ {
		owner := HRW(keyOf(i), nodes)
		before[i] = owner
		cnt[owner]++
	}
	counts := make([]int, 0, len(nodes))
	for _, n := range nodes {
		counts = append(counts, cnt[n])
	}
	fmt.Printf("  节点=%d  负载相对标准差=%.1f%%\n", len(nodes), stdevPct(counts))
	nodes2 := append([]string{}, nodes...)
	nodes2 = append(nodes2, "E")
	migrated := 0
	for i := 0; i < keyCount; i++ {
		if HRW(keyOf(i), nodes2) != before[i] {
			migrated++
		}
	}
	sort.Strings(nodes2)
	fmt.Printf("  4→5 节点迁移比例=%.1f%%（理想 1/(N+1)≈%.1f%%）\n", float64(migrated)/keyCount*100, 100.0/5)
}

func main() {
	fmt.Printf("样本 key 数=%d\n\n", keyCount)
	evalRing()
	evalJump()
	evalHRW()
	evalReplication()
}

// --- 副本放置：沿环取 N 个不同物理节点作为偏好列表 ---
func evalReplication() {
	fmt.Println("\n== 副本放置（偏好列表，N=3）==")
	r := NewRing(150)
	for _, n := range []string{"A", "B", "C", "D", "E"} {
		r.Add(n)
	}
	for _, k := range []string{"user-1", "user-2", "order-9"} {
		fmt.Printf("  %-8s -> %v\n", k, r.GetN(k, 3))
	}
}
