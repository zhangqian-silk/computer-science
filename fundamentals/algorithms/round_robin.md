# Round Robin

## Weighted Round Robin

### 平滑加权轮询算法

通过动态调整当前权重，实现均匀分配。每个节点维护三个参数：

- **固定权重（Weight）**：初始配置的优先级。
- **有效权重（Effective Weight）**：动态调整后的权重（初始等于固定权重）。
- **当前权重（Current Weight）**：实时计算的临时权重。

**步骤说明**

1. **初始化权重**：所有节点的当前权重（Current Weight）初始为 0
2. **选择节点**：每次选择 **当前权重最大** 的节点（若多个节点权重相同，按顺序选择）
3. **调整当前权重**：将选中节点的当前权重减去所有节点的 **总有效权重之和**，其余节点保持当前权重不变
4. **恢复有效权重**：下一轮开始时，所有节点的当前权重增加其有效权重值，重复步骤 2~4

**示例演示**

假设三个节点 **A(5)、B(3)、C(2)**，总有效权重为 \( 5+3+2=10 \)。

| 轮询次数 | 调整前当前权重（A, B, C） | 选中节点 | 调整后当前权重（A, B, C）        |
|----------|---------------------------|----------|----------------------------------|
| 1        | (5, 3, 2) → A=5 最大      | A        | A:5-10=**-5**；B:3；C:2 → (-5,3,2) |
| 2        | (-5+5=0, 3+3=6, 2+2=4)   | B        | B:6-10=**-4**；A:0；C:4 → (0,-4,4)  |
| 3        | (0+5=5, -4+3=-1, 4+2=6)  | C        | C:6-10=**-4**；A:5；B:-1 → (5,-1,-4)|
| 4        | (5+5=10, -1+3=2, -4+2=-2)| A        | A:10-10=**0**；B:2；C:-2 → (0,2,-2) |
| 5        | (0+5=5, 2+3=5, -2+2=0)   | A 或 B   | 选中 A 或 B，权重减 10 → 后续轮询重复 |

**最终轮询顺序**：A → B → C → A → B → A → C → B → A → C  
（符合 5:3:2 比例且分布均匀）

**伪代码实现（工程化版本）**

```text
SWRR-NEXT(nodes)
	if length(nodes) == 0 then
		return error
	end if

	total_weight ← 0
	best_idx ← -1

	for i ← 1 to length(nodes) do
		if nodes[i].effective_weight <= 0 then
			continue
		end if

		nodes[i].current_weight ← nodes[i].current_weight + nodes[i].effective_weight
		total_weight ← total_weight + nodes[i].effective_weight

		if best_idx == -1 or nodes[i].current_weight > nodes[best_idx].current_weight then
			best_idx ← i
		end if
	end for

	if best_idx == -1 then
		return error
	end if

	nodes[best_idx].current_weight ← nodes[best_idx].current_weight - total_weight
	return nodes[best_idx]
end function
```

---

## Golang 参考实现

```go
package balancer

import "errors"

var ErrNoAvailableNode = errors.New("no available node")

type Node struct {
	Name            string
	Weight          int
	EffectiveWeight int
	CurrentWeight   int
}

type SmoothWeightedRR struct {
	nodes []Node
}

func NewSmoothWeightedRR(items []Node) (*SmoothWeightedRR, error) {
	if len(items) == 0 {
		return nil, ErrNoAvailableNode
	}
	nodes := make([]Node, 0, len(items))
	for _, n := range items {
		if n.Weight <= 0 {
			continue
		}
		if n.EffectiveWeight <= 0 {
			n.EffectiveWeight = n.Weight
		}
		nodes = append(nodes, n)
	}
	if len(nodes) == 0 {
		return nil, ErrNoAvailableNode
	}
	return &SmoothWeightedRR{nodes: nodes}, nil
}

// Next 返回本轮应被选中的节点。
func (s *SmoothWeightedRR) Next() (Node, error) {
	best := -1
	total := 0

	for i := range s.nodes {
		n := &s.nodes[i]
		if n.EffectiveWeight <= 0 {
			continue
		}
		n.CurrentWeight += n.EffectiveWeight
		total += n.EffectiveWeight
		if best == -1 || n.CurrentWeight > s.nodes[best].CurrentWeight {
			best = i
		}
	}

	if best == -1 {
		return Node{}, ErrNoAvailableNode
	}

	s.nodes[best].CurrentWeight -= total
	return s.nodes[best], nil
}
```

---

## 实现校验清单

- 边界输入：空节点列表、全零/负权重节点。
- 比例正确性：长序列统计中，命中比例应逼近权重比例。
- 平滑性：避免连续集中命中同一节点（相对普通 WRR）。
- 故障恢复：降低 `EffectiveWeight` 后可逐步恢复，避免流量骤增。
