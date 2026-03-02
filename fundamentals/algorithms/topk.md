# TopK 问题

TopK 问题指在大规模数据中找到「最大（或最小）的 K 个元素」。  
它广泛出现在推荐、搜索、日志分析、风控与监控系统中，是「海量数据下快速筛选重点对象」的基础能力。

---

## 问题定义

给定一个规模为 `N` 的数据集合 `A` 与参数 `K`（通常 `K << N`），求：

- 最大 TopK：值最大的 `K` 个元素。
- 最小 TopK：值最小的 `K` 个元素。

输出形式可能不同：

- 只要元素集合，不要求有序。
- 要求最终按大小有序。
- 要求带上元素原始标识（如 URL、用户 ID、词条）。

---

## 为什么 TopK 需要专门算法

直觉上「先全排序再截断」最容易实现，但在 `N` 很大时会带来不必要开销。  
TopK 的典型特征是：只关心少量头部结果，不关心完整全序关系，因此可利用更低复杂度的方法：

- 时间上，从 `O(N log N)` 降到 `O(N log K)` 或期望 `O(N)`。
- 空间上，只维护 `K` 级别的候选集而不是全部数据。
- 工程上，更适合流式、在线、分布式场景。

---

## 典型问题

- `10^9` 个数中找最大 `10` 个。
- `10^5` 个单词中找词频最高 `10` 个。
- 海量日志中找访问最频繁的 URL TopK。

---

## 常见解法

## 全排序

- 思路：先排序，再截取前 `K`。
- 复杂度：`O(N log N)`。
- 适用：数据规模不大，或后续仍需全序结果。

## 最小堆（求最大 TopK）

- 思路：维护大小为 `K` 的最小堆。
- 复杂度：`O(N log K)`，空间 `O(K)`。
- 适用：`K << N` 的主流场景。

## Quickselect（快速选择）

- 思路：基于分区过程，只递归进入一侧。
- 平均复杂度：`O(N)`，最坏 `O(N^2)`。
- 适用：只关心第 `K` 分位与 TopK 集合，不要求稳定顺序。

## 分布式聚合

- 思路：分片内先算局部 TopK，再做全局归并。
- 适用：海量离线计算（MapReduce/Spark/Flink）。

---

## 最小堆伪代码（工程化版本，求最大 TopK）

```text
TOPK-MAX-BY-MINHEAP(A, K)
	if K <= 0 then
		return empty array
	end if

	if length(A) == 0 then
		return empty array
	end if

	if K >= length(A) then
		B ← COPY(A)
		SORT-DESC(B)
		return B
	end if

	H ← EMPTY-MIN-HEAP()
	for i ← 1 to length(A) do
		x ← A[i]
		if size(H) < K then
			HEAP-PUSH(H, x)
		else if x > HEAP-TOP(H) then
			HEAP-POP(H)
			HEAP-PUSH(H, x)
		end if
	end for

	result ← HEAP-TO-ARRAY(H)
	SORT-DESC(result)
	return result
end function
```

---

## Golang 参考实现（最小堆）

```go
package topk

import (
	"container/heap"
	"sort"
)

type intMinHeap []int

func (h intMinHeap) Len() int           { return len(h) }
func (h intMinHeap) Less(i, j int) bool { return h[i] < h[j] }
func (h intMinHeap) Swap(i, j int)      { h[i], h[j] = h[j], h[i] }

func (h *intMinHeap) Push(x any) {
	*h = append(*h, x.(int))
}

func (h *intMinHeap) Pop() any {
	old := *h
	n := len(old)
	x := old[n-1]
	*h = old[:n-1]
	return x
}

// TopKMaxByMinHeap 返回 nums 中最大的 k 个元素，结果按降序输出。
func TopKMaxByMinHeap(nums []int, k int) []int {
	if k <= 0 || len(nums) == 0 {
		return []int{}
	}
	if k >= len(nums) {
		out := append([]int(nil), nums...)
		sort.Slice(out, func(i, j int) bool { return out[i] > out[j] })
		return out
	}

	h := &intMinHeap{}
	heap.Init(h)

	for _, x := range nums {
		if h.Len() < k {
			heap.Push(h, x)
			continue
		}
		if x > (*h)[0] {
			heap.Pop(h)
			heap.Push(h, x)
		}
	}

	out := make([]int, h.Len())
	copy(out, *h)
	sort.Slice(out, func(i, j int) bool { return out[i] > out[j] })
	return out
}
```

---

## Quickselect 伪代码（工程化版本，求最大 TopK 集合）

```text
TOPK-MAX-BY-QUICKSELECT(A, K)
	if K <= 0 then
		return empty array
	end if

	if length(A) == 0 then
		return empty array
	end if

	if K >= length(A) then
		B ← COPY(A)
		SORT-DESC(B)
		return B
	end if

	B ← COPY(A)
	target ← length(B) - K
	left ← 1
	right ← length(B)

	while left <= right do
		pivot_idx ← PARTITION(B, left, right)
		if pivot_idx == target then
			break
		else if pivot_idx < target then
			left ← pivot_idx + 1
		else
			right ← pivot_idx - 1
		end if
	end while

	result ← B[target..length(B)]
	SORT-DESC(result)
	return result
end function
```

---

## Golang 参考实现（Quickselect）

```go
package topk

import "sort"

// TopKMaxByQuickselect 返回 nums 中最大的 k 个元素，结果按降序输出。
func TopKMaxByQuickselect(nums []int, k int) []int {
	if k <= 0 || len(nums) == 0 {
		return []int{}
	}
	if k >= len(nums) {
		out := append([]int(nil), nums...)
		sort.Slice(out, func(i, j int) bool { return out[i] > out[j] })
		return out
	}

	arr := append([]int(nil), nums...)
	target := len(arr) - k
	left, right := 0, len(arr)-1

	for left <= right {
		p := partition(arr, left, right)
		if p == target {
			break
		}
		if p < target {
			left = p + 1
		} else {
			right = p - 1
		}
	}

	out := append([]int(nil), arr[target:]...)
	sort.Slice(out, func(i, j int) bool { return out[i] > out[j] })
	return out
}

func partition(a []int, left, right int) int {
	pivot := a[right]
	i := left
	for j := left; j < right; j++ {
		if a[j] < pivot {
			a[i], a[j] = a[j], a[i]
			i++
		}
	}
	a[i], a[right] = a[right], a[i]
	return i
}
```

---

## Golang 参考实现（词频 TopK）

```go
package topk

import (
	"container/heap"
	"sort"
)

type WordCount struct {
	Word  string
	Count int
}

type wordCountMinHeap []WordCount

func (h wordCountMinHeap) Len() int { return len(h) }

func (h wordCountMinHeap) Less(i, j int) bool {
	if h[i].Count == h[j].Count {
		return h[i].Word > h[j].Word
	}
	return h[i].Count < h[j].Count
}

func (h wordCountMinHeap) Swap(i, j int) { h[i], h[j] = h[j], h[i] }
func (h *wordCountMinHeap) Push(x any)   { *h = append(*h, x.(WordCount)) }

func (h *wordCountMinHeap) Pop() any {
	old := *h
	n := len(old)
	x := old[n-1]
	*h = old[:n-1]
	return x
}

// TopKWords 返回词频最高的 k 个词，按频次降序、词典序升序输出。
func TopKWords(words []string, k int) []WordCount {
	if k <= 0 || len(words) == 0 {
		return []WordCount{}
	}

	freq := make(map[string]int, len(words))
	for _, w := range words {
		freq[w]++
	}
	if k > len(freq) {
		k = len(freq)
	}

	h := &wordCountMinHeap{}
	heap.Init(h)

	for w, c := range freq {
		item := WordCount{Word: w, Count: c}
		if h.Len() < k {
			heap.Push(h, item)
			continue
		}
		top := (*h)[0]
		if c > top.Count || (c == top.Count && w < top.Word) {
			heap.Pop(h)
			heap.Push(h, item)
		}
	}

	out := make([]WordCount, h.Len())
	copy(out, *h)
	sort.Slice(out, func(i, j int) bool {
		if out[i].Count == out[j].Count {
			return out[i].Word < out[j].Word
		}
		return out[i].Count > out[j].Count
	})
	return out
}
```

---

## 实现校验清单

- 边界输入：`k <= 0`、空数组、`k >= n`。
- 正确性：返回结果长度应为 `min(k, n)`，且每个元素都来自原数组。
- 顺序语义：如果要求有序，需显式排序输出；堆内天然无序。
- Quickselect 注意：分区通常是原地修改，若调用方需要原数组，必须先复制。
- 复杂度：时间 `O(N log K)`，空间 `O(K)`。

---

## 选型建议

- 需要最终全排序：全排序。
- 只要 TopK 集合且 `K` 小：最小堆优先。
- 追求期望线性时间：Quickselect。
- 数据超大且分布式：分片 TopK + 全局归并。
