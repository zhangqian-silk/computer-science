# 红包拆分算法

红包拆分的核心是：在总金额固定、人数固定的约束下，生成一组金额分配结果。  
这不是单纯的随机数问题，而是一个「受约束的随机分配」问题：既要保证资金绝对正确，又要让用户感知到随机性与公平性。

---

## 问题定义

给定：

- 总金额 `M`（建议统一使用「分」做整数计算）。
- 领取人数 `N`。

求一组分配结果 `x1, x2, ..., xN`，满足：

- `x1 + x2 + ... + xN = M`
- `xi >= 1`
- 在满足约束的前提下，分配结果具备一定随机性。

---

## 设计目标与评价维度

红包算法通常要在以下维度做平衡：

- 正确性：金额守恒、无超发、无负数、无遗漏。
- 公平性：不出现长期系统性偏差（如先到先得总是更大）。
- 随机体验：结果有波动，但不过度极端。
- 可解释性：策略能够被业务与风控理解、验证、复盘。
- 工程可落地：在高并发下可稳定执行，并支持对账与补偿。

---

## 业务约束

- 总金额精确守恒（建议统一用「分」做整数运算）。
- 每人至少 `1` 分。
- 份数固定。
- 随机性与公平性可控（避免极端偏斜）。

---

## 常见策略

## 等额拆分

- 每人固定金额。
- 优点：简单、可解释。
- 缺点：缺乏随机体验。

## 二倍均值法（随机红包）

- 当前可分配金额为 `R`，剩余人数为 `N`。
- 当前人可取区间：`[1, floor(2 * R / N)]`。
- 每次随机后更新 `R` 与 `N`，最后一人拿剩余全部金额。

## 线段切分法

- 在区间 `[0, M]` 上随机取 `N-1` 个切点，排序后做差分。
- 可增加最小值约束与权重约束。

---

## 二倍均值法伪代码（工程化版本，单位：分）

```text
RED-PACKET-SPLIT(total_cent, count)
	if count <= 0 then
		return error
	end if

	if total_cent < count then
		return error
	end if

	result ← empty array
	remain_cent ← total_cent
	remain_count ← count

	for i ← 1 to count - 1 do
		// 预留给后续每人的最小金额（每人至少 1 分）
		must_leave ← remain_count - 1
		max_by_mean ← ⌊2 * remain_cent / remain_count⌋
		max_by_budget ← remain_cent - must_leave
		max_i ← MIN(max_by_mean, max_by_budget)

		if max_i < 1 then
			return error
		end if

		x ← RANDOM-INT(1, max_i)
		append(result, x)
		remain_cent ← remain_cent - x
		remain_count ← remain_count - 1
	end for

	append(result, remain_cent)
	return result
end function
```

---

## Golang 参考实现（可测试、可复现）

```go
package redpacket

import (
	"errors"
	"math/rand"
)

var (
	ErrInvalidCount  = errors.New("count must be greater than 0")
	ErrInvalidAmount = errors.New("total amount must be >= count (at least 1 cent per user)")
)

// SplitByDoubleMean 使用二倍均值法拆分红包。
// totalCent 与返回结果单位均为「分」。
// 传入 rng 便于测试时固定 seed 做可复现验证。
func SplitByDoubleMean(totalCent int64, count int, rng *rand.Rand) ([]int64, error) {
	if count <= 0 {
		return nil, ErrInvalidCount
	}
	if totalCent < int64(count) {
		return nil, ErrInvalidAmount
	}
	if rng == nil {
		rng = rand.New(rand.NewSource(1))
	}

	result := make([]int64, 0, count)
	remainCent := totalCent
	remainCount := int64(count)

	for i := 0; i < count-1; i++ {
		mustLeave := remainCount - 1
		maxByMean := (2 * remainCent) / remainCount
		maxByBudget := remainCent - mustLeave
		maxPick := min64(maxByMean, maxByBudget)
		if maxPick < 1 {
			return nil, errors.New("invalid state: maxPick < 1")
		}

		// 随机区间 [1, maxPick]
		pick := 1 + rng.Int63n(maxPick)
		result = append(result, pick)
		remainCent -= pick
		remainCount--
	}

	result = append(result, remainCent)
	return result, nil
}

func min64(a, b int64) int64 {
	if a < b {
		return a
	}
	return b
}
```

---

## 实现校验清单

- 金额守恒：`sum(result) == total_cent`。
- 最小金额：所有元素 `>= 1`。
- 份数正确：`len(result) == count`。
- 错误输入可预期失败：`count <= 0`、`total_cent < count`。
- 固定随机种子时结果可复现，便于单测与回归。

---

## 工程注意事项

- 资金扣减要原子化，避免超发。
- 领取接口要幂等，防止重复领取。
- 高并发场景通常配合库存扣减与分布式锁/乐观锁。
- 需要完整对账与补偿机制，保障资金安全。

---

## 关联专题

- 支付业务专题：`system_design/scenarios/payment/TOPICS.md`
- 营销业务专题：`system_design/scenarios/marketing/TOPICS.md`
