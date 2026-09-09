package main

import (
	"fmt"
	"sort"
)

// distribution 统计 keyCount 个 key 在环上的物理节点分布。
func distribution(r *Ring, keyCount int) map[string]int {
	counts := make(map[string]int)
	for i := 0; i < keyCount; i++ {
		counts[r.Get(fmt.Sprintf("key-%d", i))]++
	}
	return counts
}

// owners 记录每个 key 当前归属，用于计算迁移比例。
func owners(r *Ring, keyCount int) []string {
	out := make([]string, keyCount)
	for i := 0; i < keyCount; i++ {
		out[i] = r.Get(fmt.Sprintf("key-%d", i))
	}
	return out
}

func printDist(title string, dist map[string]int) {
	fmt.Println(title)
	names := make([]string, 0, len(dist))
	for n := range dist {
		names = append(names, n)
	}
	sort.Strings(names)
	for _, n := range names {
		fmt.Printf("  %-8s %d\n", n, dist[n])
	}
}

func main() {
	const keyCount = 10000
	physical := []string{"node-A", "node-B", "node-C", "node-D"}

	for _, replicas := range []int{1, 50, 200} {
		r := NewRing(replicas)
		for _, n := range physical {
			r.Add(n)
		}
		printDist(fmt.Sprintf("[虚拟节点=%d] %d 个 key 的分布：", replicas, keyCount), distribution(r, keyCount))
	}

	// 迁移比例：移除一个节点后有多少 key 改变归属
	r := NewRing(200)
	for _, n := range physical {
		r.Add(n)
	}
	before := owners(r, keyCount)
	r.Remove("node-D")
	after := owners(r, keyCount)
	migrated := 0
	for i := range before {
		if before[i] != after[i] {
			migrated++
		}
	}
	fmt.Printf("\n移除 node-D 后迁移比例：%.2f%%（理想 1/N ≈ %.2f%%）\n",
		float64(migrated)/float64(keyCount)*100, 100.0/float64(len(physical)))
}
