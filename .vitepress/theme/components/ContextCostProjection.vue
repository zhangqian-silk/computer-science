<script setup lang="ts">
import { computed } from "vue"

/**
 * 一条 50 步 Agent 轨迹的累计输入成本推演。
 *
 * 口径全部写死在下面的常量里，图中每个数字都可由这些常量复算：
 *   - 起始上下文 BASE：系统提示 + 工具定义 + 首轮输入
 *   - 每步追加 STEP：一次动作 + 一次观测
 *   - 单价 P_MISS / P_HIT：Claude Sonnet 某一时点的公开报价，仅用于说明量级
 * 这是可复现的估算，不是任何产品的实测账单。
 */
const BASE = 8000
const STEP = 1200
const N = 50
const P_MISS = 3.0
const P_HIT = 0.3
// 第 25 步触发压缩，把已累积的历史压到 40%
const CUT_AT = 25
const KEEP = 0.4

type Track = { key: string; label: string; color: string; note: string; pts: number[] }

/** 逐步累计成本（美元）。mode: miss 全未命中 / hit 前缀命中 / comp 命中且中途压缩 */
function accumulate(mode: "miss" | "hit" | "comp"): number[] {
	const out: number[] = []
	let ctx = BASE
	let acc = 0
	for (let i = 1; i <= N; i++) {
		if (i === 1) {
			// 冷启动：首步没有任何可复用前缀，全量按未命中计价
			acc += (ctx / 1e6) * P_MISS
		} else {
			ctx += STEP
			if (mode === "miss") {
				acc += (ctx / 1e6) * P_MISS
			} else if (mode === "comp" && i === CUT_AT) {
				// 压缩改写了前缀，本步缓存被击穿，需按未命中重建
				ctx = BASE + (ctx - BASE) * KEEP
				acc += (ctx / 1e6) * P_MISS
			} else {
				// 只有新增的 STEP 是未命中的，其余前缀走缓存价
				acc += ((ctx - STEP) / 1e6) * P_HIT + (STEP / 1e6) * P_MISS
			}
		}
		out.push(acc)
	}
	return out
}

const tracks: Track[] = [
	{ key: "miss", label: "未命中缓存", color: "var(--cs-color-danger)", note: "前缀每轮被改动，全量重新预填", pts: accumulate("miss") },
	{ key: "hit", label: "命中缓存", color: "var(--cs-color-success)", note: "前缀稳定、只追加，仅新增部分按原价", pts: accumulate("hit") },
	{ key: "comp", label: "命中缓存 + 第 25 步压缩", color: "var(--cs-color-info)", note: "压缩当步击穿缓存，之后基数更小", pts: accumulate("comp") }
]

const finals = computed(() => tracks.map((t) => ({ key: t.key, label: t.label, color: t.color, note: t.note, value: t.pts[N - 1] })))
const costMiss = computed(() => tracks[0].pts[N - 1])
const costHit = computed(() => tracks[1].pts[N - 1])
const costComp = computed(() => tracks[2].pts[N - 1])
const ratioHit = computed(() => costMiss.value / costHit.value)
const savedComp = computed(() => (1 - costComp.value / costHit.value) * 100)
const totalTokens = computed(() => {
	let sum = 0
	for (let i = 1; i <= N; i++) sum += BASE + (i - 1) * STEP
	return sum
})
const lastCtx = BASE + (N - 1) * STEP

// 对数纵轴：三条曲线相差近一个数量级，线性轴会把两条低线压在底部无法区分
const W = 560
const H = 260
const padL = 52
const padR = 116
const padT = 18
const padB = 40
const plotW = W - padL - padR
const plotH = H - padT - padB
const baseY = padT + plotH
const yMin = 0.02
const yMax = 6
const logMin = Math.log10(yMin)
const logMax = Math.log10(yMax)
const xOf = (step: number) => padL + ((step - 1) / (N - 1)) * plotW
const yOf = (v: number) => {
	const clamped = Math.min(Math.max(v, yMin), yMax)
	return baseY - ((Math.log10(clamped) - logMin) / (logMax - logMin)) * plotH
}
const paths = computed(() =>
	tracks.map((t) => ({
		...t,
		d: t.pts.map((v, i) => `${i === 0 ? "M" : "L"}${xOf(i + 1).toFixed(1)} ${yOf(v).toFixed(1)}`).join(" "),
		endX: xOf(N),
		endY: yOf(t.pts[N - 1])
	}))
)
const yTicks = [0.02, 0.05, 0.1, 0.3, 1, 3, 6]
const xTicks = [1, 10, 20, 30, 40, 50]
const cutX = computed(() => xOf(CUT_AT))
const fmt = (v: number) => v.toFixed(2)

// 末端标签防重叠：命中与压缩两条终值很近，按最小行距把标签上下推开
const endLabels = computed(() => {
	const MIN_GAP = 12
	const rows = tracks
		.map((t) => ({ key: t.key, color: t.color, value: t.pts[N - 1], y: yOf(t.pts[N - 1]) }))
		.sort((a, b) => a.y - b.y)
	for (let i = 1; i < rows.length; i++) {
		if (rows[i].y - rows[i - 1].y < MIN_GAP) rows[i].y = rows[i - 1].y + MIN_GAP
	}
	return rows
})
</script>

<template>
	<figure class="ccp">
		<figcaption class="ccp__ttl pc-serif">50 步轨迹的累计输入成本</figcaption>
		<p class="ccp__sub">起始 {{ BASE.toLocaleString() }} token，每步追加 {{ STEP.toLocaleString() }} token；纵轴为对数刻度，单位美元</p>

		<div class="ccp__scroll">
			<svg :viewBox="`0 0 ${W} ${H}`" class="ccp__svg" role="img"
				aria-label="累计输入成本对比：未命中缓存约 5.61 美元，命中缓存约 0.74 美元，命中并在第 25 步压缩约 0.66 美元">
				<g v-for="t in yTicks" :key="`y${t}`">
					<line :x1="padL" :y1="yOf(t)" :x2="padL + plotW" :y2="yOf(t)" class="ccp__grid" />
					<text :x="padL - 8" :y="yOf(t) + 3.5" class="ccp__tick" text-anchor="end">{{ t < 1 ? t.toFixed(2) : t }}</text>
				</g>
				<line :x1="padL" :y1="baseY" :x2="padL + plotW" :y2="baseY" class="ccp__axis" />
				<line :x1="padL" :y1="padT" :x2="padL" :y2="baseY" class="ccp__axis" />
				<g v-for="t in xTicks" :key="`x${t}`">
					<text :x="xOf(t)" :y="baseY + 15" class="ccp__tick" text-anchor="middle">{{ t }}</text>
				</g>
				<text :x="padL + plotW / 2" :y="H - 4" class="ccp__axlab" text-anchor="middle">步数</text>

				<line :x1="cutX" :y1="padT" :x2="cutX" :y2="baseY" class="ccp__cut" />
				<text :x="cutX + 5" :y="padT + 11" class="ccp__cutlab">第 {{ CUT_AT }} 步压缩</text>

				<path v-for="p in paths" :key="p.key" :d="p.d" class="ccp__line" :style="{ stroke: p.color }" />
				<circle v-for="p in paths" :key="`d-${p.key}`" :cx="p.endX" :cy="p.endY" r="3.5" :style="{ fill: p.color }" />
				<text v-for="l in endLabels" :key="`e-${l.key}`" :x="xOf(N) + 8" :y="l.y + 4" class="ccp__endlab" :style="{ fill: l.color }">${{ fmt(l.value) }}</text>
			</svg>
		</div>

	<div class="ccp__hint">← 左右滑动查看完整曲线 →</div>

		<ul class="ccp__legend">
			<li v-for="f in finals" :key="f.key">
				<span class="ccp__swatch" :style="{ background: f.color }"></span>
				<span class="ccp__lgd-lab">{{ f.label }}</span>
				<span class="ccp__lgd-val">${{ fmt(f.value) }}</span>
				<span class="ccp__lgd-note">{{ f.note }}</span>
			</li>
		</ul>

		<p class="ccp__read">
			50 步累计输入 {{ totalTokens.toLocaleString() }} token，末步单次输入已达 {{ lastCtx.toLocaleString() }} token。
			保住前缀缓存把总成本降到约 <b>1/{{ ratioHit.toFixed(1) }}</b>；在此之上，第 {{ CUT_AT }} 步压缩再省约 <b>{{ savedComp.toFixed(0) }}%</b>。
			注意压缩当步曲线有一段上抬——改写前缀会击穿缓存，压缩不是免费的，它靠之后每步更小的基数把这笔钱赚回来。
		</p>
		<p class="pc-note">
			可复现估算：按上述固定口径与 {{ P_MISS.toFixed(2) }} / {{ P_HIT.toFixed(2) }} 美元每百万 token 的公开报价直接计算，非实测账单。
			单价为特定时点报价，仅说明量级；真实轨迹的每步增量并不均匀，压缩比例与触发时机也需按自己的评测集确定。
		</p>
	</figure>
</template>

<style scoped>
.ccp { margin: 1rem 0; background: var(--cs-color-bg); border: 1px solid var(--cs-color-border); border-radius: 10px; padding: 1rem 1.1rem .9rem; }
.ccp__ttl { font-size: 1rem; font-weight: 700; margin: 0; }
.ccp__sub { font-size: .8rem; color: var(--cs-color-text-subtle); margin: .2rem 0 .7rem; }
.ccp__scroll { overflow-x: auto; }
.ccp__hint { display: none; }
.ccp__svg { display: block; width: 100%; }
.ccp__grid { stroke: var(--cs-color-border); stroke-width: 1; stroke-dasharray: 3 3; }
.ccp__axis { stroke: var(--cs-color-border-strong, var(--cs-color-border)); stroke-width: 1; }
.ccp__tick { font-family: var(--cs-font-mono); font-size: 10px; fill: var(--cs-color-text-subtle); }
.ccp__axlab { font-size: 11px; fill: var(--cs-color-text-muted); }
.ccp__cut { stroke: var(--cs-color-info); stroke-width: 1; stroke-dasharray: 4 3; opacity: .7; }
.ccp__cutlab { font-size: 10px; fill: var(--cs-color-info); }
.ccp__line { fill: none; stroke-width: 2; stroke-linecap: round; stroke-linejoin: round; }
.ccp__endlab { font-family: var(--cs-font-mono); font-size: 11px; font-weight: 700; }
.ccp__legend { list-style: none; margin: .7rem 0 0; padding: 0; display: grid; gap: .4rem; }
.ccp__legend li { display: grid; grid-template-columns: 12px 10.5rem 3.6rem 1fr; gap: .5rem; align-items: baseline; font-size: .82rem; }
.ccp__swatch { width: 12px; height: 12px; border-radius: 3px; align-self: center; }
.ccp__lgd-lab { color: var(--cs-color-text); }
.ccp__lgd-val { font-family: var(--cs-font-mono); font-weight: 700; color: var(--cs-color-text); }
.ccp__lgd-note { color: var(--cs-color-text-muted); }
.ccp__read { margin: .8rem 0 .4rem; font-size: .86rem; line-height: 1.75; color: var(--cs-color-text-muted); }
.ccp__read b { color: var(--cs-color-text); }
@media (max-width: 720px) {
	/* 宽幅折线在窄屏若等比缩放会让刻度与末端标签不可读，改为固定最小宽度 + 横向滑动 */
	.ccp__svg { width: 520px; min-width: 520px; }
	.ccp__hint { display: block; font-size: .72rem; color: var(--cs-color-text-subtle); text-align: center; padding-top: .3rem; }
	.ccp__legend li { grid-template-columns: 12px 1fr auto; row-gap: .1rem; }
	.ccp__lgd-note { grid-column: 2 / -1; font-size: .78rem; }
}
</style>
