package main

// JumpConsistentHash 是 Google 的跳跃一致性哈希（Lamping & Veach, 2014）。
// 特点：不存储环、内存 O(1)、计算 O(ln n)、分布极均匀，扩容时迁移量最优。
// 局限：只能返回 [0, numBuckets) 的桶编号，桶必须是「编号连续」的，
// 因此只支持在「末尾」增删节点，不支持任意节点上下线（这类场景仍需环）。
func JumpConsistentHash(key uint64, numBuckets int) int32 {
	var b int64 = -1
	var j int64 = 0
	for j < int64(numBuckets) {
		b = j
		// 线性同余发生器（LCG）推进 key，产生伪随机序列
		key = key*2862933555777941757 + 1
		// 计算下一次「跳跃」到的桶；概率上保证均匀且单调
		j = int64(float64(b+1) * (float64(int64(1) << 31) / float64((key >> 33) + 1)))
	}
	return int32(b)
}
