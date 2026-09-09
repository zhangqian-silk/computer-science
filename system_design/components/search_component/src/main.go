package main

import "fmt"

func main() {
	ix := NewInvertedIndex()
	ix.Add(1, "distributed system design cache consistency")
	ix.Add(2, "cache breakdown penetration avalanche cache")
	ix.Add(3, "distributed transaction consistency compensation distributed")
	ix.Add(4, "search inverted index distributed system")

	for _, q := range []struct {
		query      string
		requireAll bool
	}{
		{"distributed cache", false},
		{"distributed cache", true},
		{"consistency", false},
	} {
		mode := "OR"
		if q.requireAll {
			mode = "AND"
		}
		fmt.Printf("查询 %-22q [%s]:\n", q.query, mode)
		res := ix.Search(q.query, q.requireAll)
		if len(res) == 0 {
			fmt.Println("  无命中")
		}
		for _, r := range res {
			fmt.Printf("  D%d  score=%.3f  | %s\n", r.DocID, r.Score, ix.docs[r.DocID])
		}
		fmt.Println()
	}
}
