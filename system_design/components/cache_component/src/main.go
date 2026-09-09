package main

import "fmt"

func main() {
	cache := NewLRUCache(3)
	accesses := []string{"A", "B", "C", "A", "D", "A", "B", "E", "A", "C"}

	fmt.Println("LRU 容量=3，访问序列 A B C A D A B E A C：")
	for _, k := range accesses {
		if _, ok := cache.Get(k); ok {
			fmt.Printf("  访问 %s 命中   缓存: %v\n", k, cache.Keys())
		} else {
			cache.Put(k, "v-"+k)
			fmt.Printf("  访问 %s 未命中 缓存: %v\n", k, cache.Keys())
		}
	}
	total := cache.Hits + cache.Miss
	fmt.Printf("\n命中 %d，未命中 %d，命中率 %.0f%%\n",
		cache.Hits, cache.Miss, float64(cache.Hits)/float64(total)*100)

	// LFU 对比：同一访问序列，观察频率淘汰与 LRU 的差异
	lfu := NewLFUCache(3)
	for _, k := range accesses {
		if _, ok := lfu.Get(k); !ok {
			lfu.Put(k, "v-"+k)
		}
	}
	lt := lfu.Hits + lfu.Miss
	fmt.Printf("LFU  命中 %d，未命中 %d，命中率 %.0f%%\n",
		lfu.Hits, lfu.Miss, float64(lfu.Hits)/float64(lt)*100)
}
