<script setup lang="ts">
const W = 260, H = 172, padL = 30, padR = 12, padT = 14, padB = 30
const plotW = W - padL - padR
const plotH = H - padT - padB
const baseY = padT + plotH

const ns = [1, 2, 3, 4, 5, 6, 7, 8]
const maxSq = 64
const slot = plotW / ns.length
const barW = slot * 0.58
const bars = ns.map((n, i) => {
	const cx = padL + (i + 0.5) * slot
	const h = (n * n / maxSq) * plotH
	return { n, x: cx - barW / 2, y: baseY - h, w: barW, h, cx }
})

const rot: { t: number; r: number }[] = []
for (let i = 0; i <= 20; i++) {
	const t = i / 20
	rot.push({ t, r: 1 / (1 + Math.pow(1.7 * t, 2)) })
}
const rotPts = rot.map((p) => `${(padL + p.t * plotW).toFixed(1)},${(baseY - p.r * plotH).toFixed(1)}`).join(" ")
const rotArea = `${padL},${baseY} ${rotPts} ${padL + plotW},${baseY}`
</script>

<template>
	<div class="pc-figs">
		<figure class="pc-fig">
			<figcaption class="pc-fig__ttl pc-serif">注意力开销随长度二次增长</figcaption>
			<p class="pc-fig__sub">token 数 n 与需要建模的两两关系数 n²</p>
			<svg :viewBox="`0 0 ${W} ${H}`" class="pc-fig__svg" role="img" aria-label="n 与 n 平方的柱状对比">
				<line :x1="padL" :y1="baseY" :x2="padL + plotW" :y2="baseY" class="axis" />
				<g v-for="b in bars" :key="b.n">
					<rect :x="b.x" :y="b.y" :width="b.w" :height="b.h" class="bar" rx="1.5" />
					<text :x="b.cx" :y="baseY + 14" class="tick" text-anchor="middle">{{ b.n }}</text>
				</g>
			</svg>
			<p class="pc-note">按 n² 直接计算绘制，说明规模关系，不代表任何模型的实测耗时。</p>
		</figure>

		<figure class="pc-fig">
			<figcaption class="pc-fig__ttl pc-serif">Context Rot：召回随长度递减</figcaption>
			<p class="pc-fig__sub">是性能梯度，而不是到某长度才断崖</p>
			<svg :viewBox="`0 0 ${W} ${H}`" class="pc-fig__svg" role="img" aria-label="召回随输入长度递减的示意曲线">
				<line :x1="padL" :y1="padT" :x2="padL" :y2="baseY" class="axis" />
				<line :x1="padL" :y1="baseY" :x2="padL + plotW" :y2="baseY" class="axis" />
				<polygon :points="rotArea" class="rot-area" />
				<polyline :points="rotPts" class="rot-line" />
				<text :x="padL - 6" :y="padT + 6" class="tick" text-anchor="end">高</text>
				<text :x="padL - 6" :y="baseY" class="tick" text-anchor="end">低</text>
				<text :x="padL + plotW" :y="baseY + 14" class="tick" text-anchor="end">输入变长 →</text>
			</svg>
			<p class="pc-note">示意曲线，用于表达定性趋势；非任何模型的实测数据，不可用于选型。</p>
		</figure>
	</div>
</template>

<style scoped>
.pc-figs { display: grid; grid-template-columns: 1fr 1fr; gap: .8rem; margin: 1rem 0; }
.pc-fig { margin: 0; background: var(--cs-color-bg); border: 1px solid var(--cs-color-border); border-radius: 10px; padding: 1rem 1.1rem .8rem; }
.pc-fig__ttl { font-size: 1rem; font-weight: 700; margin: 0; }
.pc-fig__sub { font-size: .8rem; color: var(--cs-color-text-subtle); margin: .2rem 0 .6rem; }
.pc-fig__svg { width: 100%; height: auto; display: block; }
.axis { stroke: var(--cs-color-border); stroke-width: 1; }
.tick { fill: var(--cs-color-text-subtle); font-size: 9px; font-family: var(--cs-font-mono); }
.bar { fill: var(--cs-color-success); }
.rot-line { fill: none; stroke: var(--cs-color-danger); stroke-width: 2; stroke-linejoin: round; }
.rot-area { fill: color-mix(in srgb, var(--cs-color-danger) 12%, transparent); stroke: none; }
@media (max-width: 640px) { .pc-figs { grid-template-columns: 1fr; } }
</style>
