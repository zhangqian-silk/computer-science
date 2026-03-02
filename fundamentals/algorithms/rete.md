# Rete 算法

Rete 算法用于高效解决「大量规则 + 频繁事实变更」场景下的模式匹配问题。  
它通常用于规则引擎的「条件匹配层」：输入是事实（Fact）流，输出是被命中的规则集合（Activation）。  
核心思想是通过缓存中间结果与条件节点共享，避免每次事实变更都从头全量匹配。

---

## 问题定义

给定：

- 一组规则 `R = {r1, r2, ..., rm}`，每条规则由多个条件组成。
- 一组运行时事实 `F = {f1, f2, ..., fn}`，事实会被持续插入、更新、撤回。

目标：

- 在每次事实变更后，快速得到「哪些规则被命中」。
- 在保证匹配正确性的前提下，把平均延迟与吞吐保持在可控范围。

---

## 为什么朴素匹配代价高

朴素做法通常是「事实一变更，就遍历全部规则并重算全部条件」。  
当规则数、事实数、条件重叠度上升时，会出现明显问题：

- 重复计算多：多个规则共享同一条件，但被重复求值。
- 变更不局部：一个小改动触发全量重算，延迟抖动大。
- 吞吐不稳定：高频更新时 CPU 被无效匹配占满。

Rete 的本质就是把这些重复工作转化为可复用的网络状态与增量传播。

---

## 核心思想

- 空间换时间：维护 Alpha/Beta Memory 缓存中间匹配结果。
- 增量计算：事实变更时，只传播受影响路径，不全量重算。
- 条件共享：多条规则共享公共条件节点，减少重复求值。

---

## 网络结构

- Root：事实入口。
- Alpha 网络：单事实条件过滤（如 `user.level >= 3`）。
- Beta 网络：多事实连接（如 `order.user_id == user.id`）。
- Terminal：规则命中终点，产生活动（Activation）。

---

## 关键数据结构

- AlphaMemory：通过某个 Alpha 条件的事实集合。
- BetaMemory：Join 后的 Token 集合。
- Token：已部分匹配的事实链路状态。
- Agenda：待执行规则集合（通常带优先级与冲突消解策略）。

---

## 核心流程伪代码

```text
RETE-INSERT-FACT(fact)
	for each node in ROOT.children do
		ALPHA-ACTIVATE(node, fact)
	end for
end function

ALPHA-ACTIVATE(node, fact)
	if MATCH(node.condition, fact) == FALSE then
		return
	end if

	INSERT(node.alpha_memory, fact)
	for each child in node.children do
		RIGHT-ACTIVATE(child, fact)
	end for
end function

RIGHT-ACTIVATE(beta, fact)
	INSERT(beta.right_memory, fact)
	for each token in beta.left_memory do
		PERFORM-JOIN(beta, token, fact)
	end for
end function

LEFT-ACTIVATE(beta, token)
	INSERT(beta.left_memory, token)
	for each fact in beta.right_memory do
		PERFORM-JOIN(beta, token, fact)
	end for
end function

PERFORM-JOIN(beta, token, fact)
	if JOIN-MATCH(beta.join_condition, token, fact) == FALSE then
		return
	end if

	new_token ← EXTEND(token, fact)
	INSERT(beta.result_memory, new_token)

	if IS-TERMINAL(beta) == TRUE then
		ADD-TO-AGENDA(beta.rule_id, new_token)
	else
		for each child in beta.children do
			LEFT-ACTIVATE(child, new_token)
		end for
	end if
end function
```

---

## 复杂度与适用性

- 朴素匹配通常接近 `O(规则数 × 事实数)` 的重复求值。
- Rete 更接近「与变更规模相关」的增量开销。
- 适合：规则量大、条件重叠高、在线实时决策。
- 不适合：规则很少、变更极低、内存预算紧张的离线批任务。

---

## 实践注意点

- 需要控制 BetaMemory 膨胀，避免低选择性 Join 造成爆炸。
- 需要完善撤回（Retract）语义，保证更新/删除事实后一致性。
- 需要配合优先级、互斥组、短路策略做冲突消解。

---

## 关联阅读

- 规则引擎工程化落地：`system_design/components/rule_engine/rete.md`
- 规则引擎整体设计：`system_design/components/rule_engine/rule_engine.md`
