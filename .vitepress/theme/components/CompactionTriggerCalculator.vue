<script setup lang="ts">
import { computed, ref } from "vue"

/**
 * 触发线的两种口径对照：距上限的绝对缓冲 vs 窗口百分比。
 *
 * 结论先行：需要保护的是「摘要输出所需的空间」，而这个空间的大小与窗口大小无关，
 * 所以它是一个绝对量。用百分比会让小窗口留不够、大窗口浪费太多。
 * 拖动窗口容量即可观察两种口径给出的触发线如何分道扬镳。
 */

const WINDOWS = [32_000, 128_000, 200_000, 400_000, 1_000_000]
const idx = ref(2)
const win = computed(() => WINDOWS[idx.value])

/** 摘要输出预算：与窗口大小无关的绝对量 */
const reserve = ref(16_000)
/** 输出上限占位：有效窗口先要扣掉留给本轮回答的空间 */
const maxOut = ref(20_000)
/** 百分比口径：常见取值在 50%–85% 之间 */
const pct = ref(85)

const effective = computed(() => win.value - maxOut.value)
const lineAbs = computed(() => effective.value - reserve.value)
const linePct = computed(() => Math.round(win.value * (pct.value / 100)))

/** 百分比口径下实际留给摘要的空间；小窗口会留不够，大窗口会留过多 */
const pctReserve = computed(() => effective.value - linePct.value)
const verdict = computed(() => {
	const r = pctReserve.value
	// 绝对缓冲口径本身就为负，说明这组参数在这个窗口下根本不可行——先报这一条
	if (lineAbs.value <= 0) {
		return {
			tone: "fail",
			text: `绝对缓冲口径给出的触发线是 ${fmt(lineAbs.value)}，已经是负数——输出上限 ${fmt(maxOut.value)} 加摘要预算 ${fmt(reserve.value)} 超过了整个窗口，这组参数在此窗口下不可行，须先压低输出上限或换更大的窗口`
		}
	}
	if (r <= 0) return { tone: "fail", text: `百分比触发线已经越过有效窗口，留给摘要输出的空间是 ${fmt(r)}——越线时连摘要请求都发不出去` }
	if (r < reserve.value) return { tone: "fail", text: `只剩 ${fmt(r)} 供摘要输出，不足所需的 ${fmt(reserve.value)}——越线后可能连摘要请求都发不出去` }
	if (r > reserve.value * 2.5) return { tone: "warn", text: `留出 ${fmt(r)}，是所需的 ${(r / reserve.value).toFixed(1)} 倍——大量窗口被闲置，压缩触发得过早` }
	return { tone: "pass", text: `留出 ${fmt(r)}，与所需的 ${fmt(reserve.value)} 大致相当——这个窗口下两种口径恰好接近` }
})

const bar = computed(() => {
	const w = win.value
	const safe = Math.max(0, lineAbs.value)
	return {
		safe: (safe / w) * 100,
		buffer: (Math.max(0, effective.value - safe) / w) * 100,
		out: (maxOut.value / w) * 100,
		pctPos: (Math.min(linePct.value, w) / w) * 100,
		absPos: (safe / w) * 100
	}
})

const fmt = (n: number) => {
	const sign = n < 0 ? "−" : ""
	const a = Math.abs(n)
	return a >= 1000 ? `${sign}${Math.round(a / 1000)}K` : `${sign}${a}`
}

/** 同一组参数在五个窗口下的对照表：这才是「为什么不用百分比」的直接证据 */
const table = computed(() =>
	WINDOWS.map((w) => {
		const eff = w - maxOut.value
		const abs = eff - reserve.value
		const p = Math.round(w * (pct.value / 100))
		const left = eff - p
		return {
			w,
			abs,
			// 触发线为负：这个窗口装不下「输出上限 + 摘要预算」，参数组合本身不可行
			infeasible: abs <= 0,
			p,
			left,
			state: left < reserve.value ? "fail" : left > reserve.value * 2.5 ? "warn" : "pass"
		}
	})
)
</script>

<template>
	<div class="ctc">
		<div class="infra-controls ctc__controls">
			<label class="infra-control ctc__ctl">
				<span>窗口容量</span>
				<input v-model.number="idx" type="range" min="0" :max="WINDOWS.length - 1" step="1" />
				<b>{{ fmt(win) }}</b>
			</label>
			<label class="infra-control ctc__ctl">
				<span>摘要输出预算</span>
				<input v-model.number="reserve" type="range" min="4000" max="48000" step="2000" />
				<b>{{ fmt(reserve) }}</b>
			</label>
			<label class="infra-control ctc__ctl">
				<span>本轮输出上限</span>
				<input v-model.number="maxOut" type="range" min="4000" max="64000" step="4000" />
				<b>{{ fmt(maxOut) }}</b>
			</label>
			<label class="infra-control ctc__ctl">
				<span>百分比口径</span>
				<input v-model.number="pct" type="range" min="50" max="95" step="5" />
				<b>{{ pct }}%</b>
			</label>
		</div>

		<div class="ctc__bar" role="img"
			:aria-label="`窗口 ${fmt(win)}：绝对缓冲口径触发线 ${fmt(lineAbs)}，百分比口径触发线 ${fmt(linePct)}`">
			<span class="ctc__zone ctc__zone--safe" :style="{ width: `${bar.safe}%` }">常态工作区</span>
			<span class="ctc__zone ctc__zone--buf" :style="{ width: `${bar.buffer}%` }">摘要预算</span>
			<span class="ctc__zone ctc__zone--out" :style="{ width: `${bar.out}%` }">输出</span>
			<span class="ctc__mark ctc__mark--abs" :style="{ left: `${bar.absPos}%` }">
				<i></i><em>绝对缓冲 {{ fmt(lineAbs) }}</em>
			</span>
			<span class="ctc__mark ctc__mark--pct" :style="{ left: `${bar.pctPos}%` }">
				<i></i><em>{{ pct }}% → {{ fmt(linePct) }}</em>
			</span>
		</div>

		<p class="ctc__verdict" :class="`is-${verdict.tone}`">
			<b>{{ pct }}% 口径在 {{ fmt(win) }} 窗口下</b>：{{ verdict.text }}
		</p>

		<table class="ctc__table">
			<caption>同一组参数（摘要预算 {{ fmt(reserve) }}、输出上限 {{ fmt(maxOut) }}、百分比 {{ pct }}%）在五种窗口下的对照</caption>
			<thead>
				<tr>
					<th>窗口</th>
					<th>绝对缓冲触发线</th>
					<th>百分比触发线</th>
					<th>百分比口径剩余空间</th>
				</tr>
			</thead>
			<tbody>
				<tr v-for="r in table" :key="r.w" :class="{ 'is-cur': r.w === win }">
					<td class="cs-mono">{{ fmt(r.w) }}</td>
					<td class="cs-mono">
						{{ fmt(r.abs) }}
						<em v-if="r.infeasible" class="ctc__flag">窗口装不下</em>
					</td>
					<td class="cs-mono">{{ fmt(r.p) }}</td>
					<td>
						<span class="ctc__pill" :class="`is-${r.state}`">{{ fmt(r.left) }}</span>
					</td>
				</tr>
			</tbody>
		</table>

		<p class="pc-note">
			参数为可调的教学口径，非任何产品的默认值；真实缓冲应取自己实测的摘要输出长度高分位。
			表格最后一列是关键：绝对缓冲口径下这个数恒等于摘要预算，而百分比口径下它随窗口线性变化——
			<b>32K 窗口留不够，1M 窗口浪费上百 K</b>。这就是阈值应当写成绝对量的理由。
		</p>
	</div>
</template>

<style scoped>
.ctc { margin: var(--cs-space-6) 0; background: var(--cs-color-bg); border: var(--cs-border); border-radius: var(--cs-radius-lg); padding: var(--cs-space-5) var(--cs-space-6); }
.ctc__controls { margin-bottom: var(--cs-space-6); }
.ctc__ctl { display: grid; grid-template-columns: auto 1fr auto; align-items: center; gap: var(--cs-space-2); min-height: var(--cs-tap-target); font-size: var(--cs-text-sm); }
.ctc__ctl span { color: var(--cs-color-text-muted); }
.ctc__ctl b { font-family: var(--cs-font-mono); text-align: right; color: var(--cs-color-brand); }
.ctc__bar { position: relative; display: flex; height: var(--cs-lab-bar-height); margin: calc(var(--cs-space-6) + var(--cs-space-4)) 0 var(--cs-space-6); border: var(--cs-border); border-radius: var(--cs-radius-sm); overflow: visible; }
.ctc__zone { display: flex; align-items: center; justify-content: center; font-size: var(--cs-text-3xs); white-space: nowrap; overflow: hidden; transition: width var(--cs-duration-base) var(--cs-ease-standard); }
.ctc__zone--safe { background: var(--cs-color-success-soft); color: var(--cs-color-text-muted); border-radius: var(--cs-radius-sm) 0 0 var(--cs-radius-sm); }
.ctc__zone--buf { background: var(--cs-color-warning-soft); color: var(--cs-color-text-muted); }
.ctc__zone--out { background: var(--cs-color-neutral-soft); color: var(--cs-color-text-subtle); border-radius: 0 var(--cs-radius-sm) 0 0; }
.ctc__mark { position: absolute; top: 0; bottom: 0; transform: translateX(-1px); transition: left var(--cs-duration-base) var(--cs-ease-standard); }
.ctc__mark i { position: absolute; top: calc(-1 * var(--cs-space-2)); bottom: calc(-1 * var(--cs-space-2)); width: var(--cs-border-width-emphasis); }
.ctc__mark em { position: absolute; white-space: nowrap; font-family: var(--cs-font-mono); font-size: var(--cs-text-3xs); font-style: normal; transform: translateX(-50%); }
.ctc__mark--abs i { background: var(--cs-color-brand); }
.ctc__mark--abs em { top: calc(-1 * var(--cs-space-7)); color: var(--cs-color-brand); }
.ctc__mark--pct i { background: var(--cs-color-danger); }
.ctc__mark--pct em { bottom: calc(-1 * var(--cs-space-7)); color: var(--cs-color-danger); }
.ctc__verdict { margin: var(--cs-space-6) 0 var(--cs-space-5); padding: var(--cs-space-3) var(--cs-space-4); border-radius: var(--cs-radius-sm); font-size: var(--cs-text-base); line-height: var(--cs-leading-relaxed); }
.ctc__verdict.is-pass { background: var(--cs-color-success-soft); }
.ctc__verdict.is-warn { background: var(--cs-color-warning-soft); }
.ctc__verdict.is-fail { background: var(--cs-color-danger-soft); }
.ctc__table { width: 100%; margin: 0; font-size: var(--cs-text-sm); }
.ctc__table caption { caption-side: top; text-align: left; font-size: var(--cs-text-2xs); color: var(--cs-color-text-subtle); padding-bottom: var(--cs-space-2); }
.ctc__table th { text-align: left; font-size: var(--cs-text-2xs); color: var(--cs-color-text-muted); font-weight: 600; }
.ctc__table tr.is-cur { background: var(--cs-color-brand-soft); }
.ctc__flag { display: block; font-family: var(--cs-font-mono); font-size: var(--cs-text-3xs); font-style: normal; color: var(--cs-color-danger); }
.ctc__pill { display: inline-block; font-family: var(--cs-font-mono); font-size: var(--cs-text-2xs); border-radius: var(--cs-radius-pill); padding: var(--cs-space-1) var(--cs-space-3); }
.ctc__pill.is-pass { background: var(--cs-color-success-soft); color: var(--cs-color-success); }
.ctc__pill.is-warn { background: var(--cs-color-warning-soft); color: var(--cs-color-warning); }
.ctc__pill.is-fail { background: var(--cs-color-danger-soft); color: var(--cs-color-danger); }
@media (max-width: 720px) {
	.ctc__controls { grid-template-columns: 1fr; }
	.ctc__table { font-size: var(--cs-text-xs); }
}
</style>
