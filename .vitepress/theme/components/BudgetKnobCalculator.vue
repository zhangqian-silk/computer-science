<script setup lang="ts">
import { computed, ref } from "vue"
import { seriesColor } from "../series-palette"

/**
 * 预算旋钮：一次长任务的成本由哪几个量决定，各自往哪个方向拨。
 *
 * 这个组件解决的是一类反复出现的误判——把「限制步数」当作唯一的成本闸门。
 * 实际上单步成本随步数线性增长（每一步都要为此前全部前缀重新付费），
 * 所以总成本近似随步数的**平方**增长；而压缩改变的是这条抛物线的斜率，
 * 代价是压缩当步的缓存被击穿、必须按未命中价重建前缀。
 *
 * 计价与令牌数是为呈现量级关系所设的示例值，不代表任何产品的实测数据；
 * 真实数字取决于模型、供应方与当日价目表。
 */

const steps = ref(40)
const base = ref(8000)
const perStep = ref(1200)
const cacheHit = ref(true)
/** 压缩后保留的比例；1 表示不压缩 */
const compactAt = ref(0)

const PRICE_MISS = 3.0
const PRICE_HIT = 0.3
const KEEP_RATIO = 0.4

type Run = { tokens: number; cost: number }

/** 逐步累加，按「新增部分按原价、其余前缀走缓存价」计费 */
function simulate(useCache: boolean, compactStep: number): Run {
	let prefix = base.value
	let tokens = 0
	let cost = 0

	for (let i = 1; i <= steps.value; i++) {
		// 压缩发生在该步之前：前缀被改写，缓存随之失效
		const compactedHere = compactStep > 0 && i === compactStep
		if (compactedHere) prefix = Math.round(prefix * KEEP_RATIO)

		const input = prefix
		tokens += input

		if (!useCache || i === 1 || compactedHere) {
			// 冷启动与压缩当步：整段前缀都按未命中计价
			cost += (input / 1e6) * PRICE_MISS
		} else {
			const fresh = Math.min(perStep.value, input)
			cost += (fresh / 1e6) * PRICE_MISS + ((input - fresh) / 1e6) * PRICE_HIT
		}

		prefix += perStep.value
	}

	return { tokens, cost }
}

const plain = computed(() => simulate(cacheHit.value, 0))
const withCompact = computed(() =>
	compactAt.value > 0 ? simulate(cacheHit.value, compactAt.value) : null
)
const noCache = computed(() => simulate(false, compactAt.value))

const rows = computed(() => {
	const list = [
		{ name: "不启用缓存", cost: noCache.value.cost, color: seriesColor(4) },
		{ name: "命中前缀缓存", cost: simulate(true, 0).cost, color: seriesColor(0) }
	]
	if (withCompact.value) {
		list.push({
			name: `第 ${compactAt.value} 步压缩一次`,
			cost: withCompact.value.cost,
			color: seriesColor(2)
		})
	}
	return list
})

const maxCost = computed(() => Math.max(...rows.value.map((r) => r.cost)))
const finalInput = computed(() => base.value + perStep.value * (steps.value - 1))
const fmt = (n: number) => n.toFixed(2)
const fmtInt = (n: number) => n.toLocaleString("en-US")

/** 压缩是否划算：与同等条件下不压缩相比 */
const verdict = computed(() => {
	if (!withCompact.value) return null
	const baseline = simulate(cacheHit.value, 0).cost
	const delta = baseline - withCompact.value.cost
	if (delta > 0.01) return { state: "pass", text: `净省约 ${fmt(delta)} 美元` }
	if (delta < -0.01) return { state: "fail", text: `净亏约 ${fmt(-delta)} 美元——剩余步数不足以摊回重建前缀的开销` }
	return { state: "warn", text: "基本持平：收益刚好抵消一次缓存重建" }
})
</script>

<template>
	<div class="bkc">
		<div class="infra-controls">
			<div class="infra-control">
				<label for="bkc-steps">步数 <b class="cs-num">{{ steps }}</b></label>
				<input id="bkc-steps" v-model.number="steps" type="range" min="5" max="80" step="5" />
			</div>
			<div class="infra-control">
				<label for="bkc-base">起始上下文 <b class="cs-num">{{ fmtInt(base) }}</b></label>
				<input id="bkc-base" v-model.number="base" type="range" min="2000" max="30000" step="1000" />
			</div>
			<div class="infra-control">
				<label for="bkc-per">每步新增 <b class="cs-num">{{ fmtInt(perStep) }}</b></label>
				<input id="bkc-per" v-model.number="perStep" type="range" min="200" max="4000" step="100" />
			</div>
			<div class="infra-control">
				<label for="bkc-compact">
					压缩时机
					<b class="cs-num">{{ compactAt === 0 ? "不压缩" : `第 ${compactAt} 步` }}</b>
				</label>
				<input id="bkc-compact" v-model.number="compactAt" type="range" min="0" :max="steps" step="1" />
			</div>
		</div>

		<label class="cs-toggle bkc__toggle">
			<input v-model="cacheHit" type="checkbox" />
			<span>前缀缓存命中</span>
		</label>

		<div class="infra-results bkc__results">
			<div class="infra-result">
				<span>末步单次输入</span>
				<strong class="cs-num">{{ fmtInt(finalInput) }}</strong>
			</div>
			<div class="infra-result">
				<span>累计输入令牌</span>
				<strong class="cs-num">{{ fmtInt(plain.tokens) }}</strong>
			</div>
			<div class="infra-result">
				<span>当前设置下的成本</span>
				<strong class="cs-num">{{ fmt((withCompact ?? plain).cost) }} 美元</strong>
			</div>
		</div>

		<div class="cs-stack cs-stack--tight bkc__bars">
			<div v-for="r in rows" :key="r.name" class="bkc__row">
				<span class="bkc__dot" :style="{ background: r.color }"></span>
				<span class="bkc__name">{{ r.name }}</span>
				<span class="infra-bar bkc__bar">
					<i :style="{ width: `${(r.cost / maxCost) * 100}%`, background: r.color }"></i>
				</span>
				<b class="cs-num bkc__val">{{ fmt(r.cost) }}</b>
			</div>
		</div>

		<div v-if="verdict" class="cs-state bkc__verdict" :class="`cs-state--${verdict.state}`">
			这次压缩：{{ verdict.text }}
		</div>

		<p class="pc-note">
			三个可以带走的判断。其一，<b>总成本随步数近似平方增长</b>，
			因为每一步都要为此前的全部前缀重新付费——所以「限制步数」是比「限制单步输出」有效得多的闸门。
			其二，<b>缓存带来的节省达不到单价差的倍数</b>：每步新增的那部分与冷启动的首步都必须按原价计，
			单价差是上限而不是实际收益。其三，<b>压缩不是免费的</b>——
			它改写前缀，因而击穿缓存、需要按未命中价重建一次；
			收益要靠之后每一步更小的计价基数慢慢赚回来。把压缩点往后拖到只剩几步时，它会净亏。
			这也说明触发时机应当由评测集确定，而不是拍一个固定阈值。
		</p>
	</div>
</template>

<style scoped>
.bkc { margin: 1rem 0; }
.bkc .infra-control label b { color: var(--cs-color-brand); }
.bkc__toggle { margin-top: var(--cs-space-4); }
.bkc__results { margin-top: var(--cs-space-5); }
.bkc__bars { margin-top: var(--cs-space-5); }
.bkc__row { display: flex; align-items: center; gap: var(--cs-space-3); font-size: var(--cs-text-sm); }
.bkc__dot { flex: none; width: 10px; height: 10px; border-radius: var(--cs-radius-circle); }
.bkc__name { flex: 0 0 auto; width: 9rem; font-size: var(--cs-text-xs); color: var(--cs-color-text-muted); }
.bkc__bar { flex: 1; margin-top: 0; min-width: 0; }
.bkc__val { flex: 0 0 auto; color: var(--cs-color-brand); }
.bkc__verdict { margin-top: var(--cs-space-4); }
@media (max-width: 640px) {
	.bkc__row { flex-wrap: wrap; }
	.bkc__name { width: auto; flex: 1; }
	.bkc__bar { flex: 1 0 100%; }
}
</style>
