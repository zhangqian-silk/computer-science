<script setup lang="ts">
import { computed, ref } from "vue"

/**
 * 前缀缓存的成本构成。
 *
 * 组件说明两件事：
 *   1. 一轮请求的输入成本被拆成三段——命中的前缀、新写入的前缀、完全未缓存的部分，
 *      三段的单价差一个量级，所以「输入 token 总量」这个指标本身没有决策价值；
 *   2. 只要有一处动态内容落在前缀里，命中段就整体归零——不是打个折，是没有。
 *
 * 倍率取公开定价文档给出的口径（写入 1.25×、命中 0.1×），
 * 轮数与 token 量是为呈现相对关系而设的教学化取值，不对应任何产品实测。
 */

const WRITE_MULT = 1.25
const READ_MULT = 0.1

/** 会话内不变的部分：工具定义与系统提示 */
const stableTokens = ref(12000)
/** 每轮新增的历史（上一轮的回答加这一轮的提问） */
const perTurnTokens = ref(1500)
const turns = ref(12)
/** 前缀里是否混入了每轮都变的内容 */
const polluted = ref(false)

type Row = {
	turn: number
	read: number
	write: number
	raw: number
	cost: number
}

const rows = computed<Row[]>(() => {
	const out: Row[] = []
	for (let t = 1; t <= turns.value; t++) {
		const total = stableTokens.value + perTurnTokens.value * t
		if (polluted.value) {
			// 前缀里有每轮变化的内容：命中恒为零，每轮把整段前缀重写一遍
			out.push({ turn: t, read: 0, write: total, raw: 0, cost: total * WRITE_MULT })
			continue
		}
		// 干净前缀：上一轮写过的部分这一轮读到，本轮新增的部分写入
		const read = t === 1 ? 0 : stableTokens.value + perTurnTokens.value * (t - 1)
		const write = total - read
		out.push({ turn: t, read, write, raw: 0, cost: read * READ_MULT + write * WRITE_MULT })
	}
	return out
})

/** 完全不用缓存时的基线：每轮全量按原价重算 */
const baseline = computed(() => {
	let sum = 0
	for (let t = 1; t <= turns.value; t++) sum += stableTokens.value + perTurnTokens.value * t
	return sum
})

const totalCost = computed(() => rows.value.reduce((s, r) => s + r.cost, 0))
const ratio = computed(() => totalCost.value / baseline.value)
const maxCost = computed(() => Math.max(...rows.value.map((r) => r.cost), 1))

/** 单条前缀被复用多少次才能抵掉写入溢价 */
const breakEven = computed(() => (WRITE_MULT - 1) / (1 - READ_MULT))

function fmt(n: number) {
	return n >= 1000 ? `${(n / 1000).toFixed(1)}k` : String(Math.round(n))
}
</script>

<template>
	<div class="pce">
		<div class="infra-controls">
			<label class="infra-control">
				<span>会话内不变的前缀（工具定义 + 系统提示）</span>
				<input v-model.number="stableTokens" type="range" min="2000" max="40000" step="1000" />
				<b class="cs-num">{{ fmt(stableTokens) }} token</b>
			</label>
			<label class="infra-control">
				<span>每轮新增历史</span>
				<input v-model.number="perTurnTokens" type="range" min="200" max="6000" step="100" />
				<b class="cs-num">{{ fmt(perTurnTokens) }} token</b>
			</label>
			<label class="infra-control">
				<span>轮数</span>
				<input v-model.number="turns" type="range" min="2" max="30" step="1" />
				<b class="cs-num">{{ turns }} 轮</b>
			</label>
		</div>

		<label class="cs-toggle pce__toggle">
			<input v-model="polluted" type="checkbox" />
			<span>前缀里混入每轮都变的内容（时间戳、随机标识、每轮刷新的环境信息）</span>
		</label>

		<div class="infra-results">
			<div class="infra-result">
				<span>相对全量重算</span>
				<b class="cs-num">{{ (ratio * 100).toFixed(0) }}%</b>
			</div>
			<div class="infra-result">
				<span>命中折算单价</span>
				<b class="cs-num">{{ READ_MULT }}×</b>
			</div>
			<div class="infra-result">
				<span>写入折算单价</span>
				<b class="cs-num">{{ WRITE_MULT }}×</b>
			</div>
			<div class="infra-result">
				<span>写入溢价的回本次数</span>
				<b class="cs-num">{{ breakEven.toFixed(2) }} 次</b>
			</div>
		</div>

		<ul class="pce__bars">
			<li v-for="r in rows" :key="r.turn">
				<span class="pce__turn cs-mono">{{ r.turn }}</span>
				<span class="pce__bar" :style="{ '--w': `${(r.cost / maxCost) * 100}%` }">
					<i
						class="pce__seg pce__seg--read"
						:style="{ width: `${(r.read * READ_MULT / Math.max(r.cost, 1)) * 100}%` }"
					></i>
					<i
						class="pce__seg pce__seg--write"
						:style="{ width: `${(r.write * WRITE_MULT / Math.max(r.cost, 1)) * 100}%` }"
					></i>
				</span>
				<span class="pce__val cs-num">{{ fmt(r.cost) }}</span>
			</li>
		</ul>

		<div class="cs-legend-row pce__legend">
			<span><i class="pce__key pce__key--read"></i>命中的前缀（按 {{ READ_MULT }}× 计）</span>
			<span><i class="pce__key pce__key--write"></i>本轮写入的前缀（按 {{ WRITE_MULT }}× 计）</span>
		</div>

		<p class="pc-note">
			把开关打开再看一次那组柱子，就能理解为什么这件事值得单独讨论：
			<b>命中段不是变小了，而是整段消失</b>。原因在于缓存匹配的是从头开始的连续前缀，
			前面任何一处发生变化，它之后的全部内容都不再是同一个前缀。
			这也解释了为什么工具定义应当排在最前——它体积大且会话内不变，
			放在前面才能让可复用的那一段尽可能长。反过来，把每轮变化的内容混进这一段，
			付出的是每轮 {{ WRITE_MULT }}× 的写入价，换回的是零次命中：
			<b>比不做缓存更贵。</b>
		</p>
	</div>
</template>

<style scoped>
.pce { margin: 1rem 0; }
.pce__toggle { margin: var(--cs-space-4) 0; }
.pce__bars { list-style: none; margin: var(--cs-space-5) 0 var(--cs-space-3); padding: 0; display: grid; gap: var(--cs-space-1); }
.pce__bars li { display: grid; grid-template-columns: 2rem minmax(0, 1fr) 3.5rem; gap: var(--cs-space-3); align-items: center; }
.pce__turn { font-size: var(--cs-text-3xs); color: var(--cs-color-text-subtle); text-align: right; }
.pce__bar { display: flex; width: var(--w); min-width: 2px; height: 14px; border-radius: var(--cs-radius-xs); overflow: hidden; background: var(--cs-color-neutral-soft); }
.pce__seg { display: block; height: 100%; }
.pce__seg--read { background: var(--cs-color-success); }
.pce__seg--write { background: var(--cs-color-warning); }
.pce__val { font-size: var(--cs-text-2xs); color: var(--cs-color-text-muted); }
.pce__legend { margin-bottom: var(--cs-space-4); font-size: var(--cs-text-2xs); color: var(--cs-color-text-muted); display: flex; flex-wrap: wrap; gap: var(--cs-space-2) var(--cs-space-5); }
.pce__key { display: inline-block; width: 10px; height: 10px; border-radius: var(--cs-radius-xs); margin-right: var(--cs-space-2); vertical-align: middle; }
.pce__key--read { background: var(--cs-color-success); }
.pce__key--write { background: var(--cs-color-warning); }
</style>
