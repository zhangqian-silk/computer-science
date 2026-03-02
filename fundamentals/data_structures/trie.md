# Trie（前缀树）

Trie（前缀树）是一种面向字符串集合的树形数据结构，核心思想是「公共前缀复用」。

---

## 结构定义

- 每条边表示一个字符。
- 从根节点到某个节点的路径，表示一个前缀。
- 节点包含：
	- `children`：子节点映射（字符 -> 子节点）。
	- `is_end`：是否为完整单词结尾。
	- `value`（可选）：业务附加值（频次、ID 列表、标签等）。

---

## 核心操作

- 插入字符串：按字符逐层创建节点并下探，末尾标记 `is_end`。
- 查询完整字符串：逐字符下探，最终节点 `is_end = true` 才算存在。
- 查询前缀：逐字符下探，只要路径存在即命中。

---

## 复杂度

- 设字符串长度为 `L`：
	- 插入：`O(L)`
	- 精确查询：`O(L)`
	- 前缀查询：`O(L)`
- 空间复杂度与「去重后前缀总字符数」相关。

---

## 伪代码

```text
TRIE-INSERT(root, word)
	node ← root
	for i ← 1 to length(word) do
		ch ← word[i]
		if ch not in node.children then
			node.children[ch] ← NEW-NODE()
		end if
		node ← node.children[ch]
	end for
	node.is_end ← true
end function
```

```text
TRIE-SEARCH(root, word)
	node ← root
	for i ← 1 to length(word) do
		ch ← word[i]
		if ch not in node.children then
			return false
		end if
		node ← node.children[ch]
	end for
	return node.is_end
end function
```

---

## 常见场景

- 自动补全（输入法、搜索建议）。
- 词典匹配（敏感词、黑白名单）。
- 路由前缀匹配（IP/CIDR 最长前缀匹配，通常用压缩 Trie/Radix Tree）。

---

## 变种

- 压缩 Trie（Radix Tree）：把单链路径压缩，降低节点数。
- 双数组 Trie：提升查询性能，常见于词典和分词系统。
