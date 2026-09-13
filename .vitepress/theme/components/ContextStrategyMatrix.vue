<script setup lang="ts">
const W = 320, H = 240, padL = 42, padR = 18, padT = 18, padB = 36
const plotW = W - padL - padR
const plotH = H - padT - padB
const baseY = padT + plotH

const items = [
	{ key: "压缩", x: 0.74, y: 0.24, color: "var(--cs-color-success)", do: "把历史交给模型总结，保留决策、未解 bug 与实现细节，丢弃冗余工具原文。", fit: "需要大量往返的长对话流" },
	{ key: "结构化笔记", x: 0.5, y: 0.42, color: "var(--cs-color-warning)", do: "把要点写到窗口之外的持久文件（待办、NOTES.md），需要时读回。", fit: "有清晰里程碑的迭代开发" },
	{ key: "子代理", x: 0.64, y: 0.82, color: "var(--cs-color-danger)", do: "主代理持高层计划，子代理以干净窗口深挖后只回传提炼结论。", fit: "可并行拆分的探索与研究" },
	{ key: "即时检索", x: 0.3, y: 0.58, color: "var(--cs-color-info)", do: "只保留轻量标识符（路径、查询、链接），运行时按需加载正文。", fit: "语料庞大、内容多变、依赖元数据线索" }
]
const pts = items.map((it) => {
	const px = padL + it.x * plotW
	const py = padT + (1 - it.y) * plotH
	const end = it.x > 0.58
	return { ...it, px, py, anchor: end ? "end" : "start", dx: end ? -11 : 11 }
})
</script>

<template>
	<div class="pc-matrix">
		<figure class="pc-matrix__fig">
			<svg :viewBox="`0 0 ${W} ${H}`" role="img" aria-label="按时间跨度与并行度定位四种上下文策略">
				<line :x1="padL" :y1="padT + plotH / 2" :x2="padL + plotW" :y2="padT + plotH / 2" class="grid" />
				<line :x1="padL + plotW / 2" :y1="padT" :x2="padL + plotW / 2" :y2="baseY" class="grid" />
				<line :x1="padL" :y1="baseY" :x2="padL + plotW" :y2="baseY" class="axis" />
				<line :x1="padL" :y1="baseY" :x2="padL" :y2="padT" class="axis" />
				<text :x="padL + plotW" :y="baseY + 22" class="lab" text-anchor="end">时间跨度 →</text>
				<text :x="padL - 8" :y="padT + 4" class="lab" text-anchor="end">高</text>
				<text :x="padL - 8" :y="baseY" class="lab" text-anchor="end">低</text>
				<text class="lab" text-anchor="middle" :transform="`translate(14 ${padT + plotH / 2}) rotate(-90)`">可并行度 →</text>
				<g v-for="p in pts" :key="p.key">
					<circle :cx="p.px" :cy="p.py" r="7" :style="{ fill: p.color }" />
					<text :x="p.px + p.dx" :y="p.py + 4" :text-anchor="p.anchor" class="dot-lab pc-serif">{{ p.key }}</text>
				</g>
			</svg>
			<figcaption class="pc-note">坐标为定性示意：横轴任务时间跨度，纵轴可并行拆分程度。四者不互斥，实践中常组合。</figcaption>
		</figure>
		<div class="pc-matrix__legend">
			<div v-for="it in items" :key="it.key" class="pc-matrix__item">
				<span class="pc-matrix__dot" :style="{ background: it.color }" />
				<div>
					<b>{{ it.key }}</b>
					<p>{{ it.do }}</p>
					<span class="pc-matrix__fit">适合：{{ it.fit }}</span>
				</div>
			</div>
		</div>
	</div>
</template>

<style scoped>
.pc-matrix { display: grid; grid-template-columns: 1fr 1fr; gap: 1rem; margin: 1rem 0; align-items: start; }
.pc-matrix__fig { margin: 0; background: var(--cs-color-bg); border: 1px solid var(--cs-color-border); border-radius: 10px; padding: 1rem 1.1rem .8rem; }
.pc-matrix__fig svg { width: 100%; height: auto; display: block; }
.axis { stroke: var(--cs-color-text-muted); stroke-width: 1.2; }
.grid { stroke: var(--cs-color-border); stroke-width: 1; stroke-dasharray: 3 3; }
.lab { fill: var(--cs-color-text-subtle); font-size: 9px; font-family: var(--cs-font-mono); }
.dot-lab { fill: var(--cs-color-text); font-size: 11px; font-weight: 700; }
.pc-matrix__legend { display: flex; flex-direction: column; gap: .6rem; }
.pc-matrix__item { display: grid; grid-template-columns: 14px 1fr; gap: .6rem; align-items: start; }
.pc-matrix__dot { width: 12px; height: 12px; border-radius: 50%; margin-top: .35rem; }
.pc-matrix__item b { font-size: .92rem; }
.pc-matrix__item p { margin: .15rem 0 .2rem; font-size: .83rem; line-height: 1.65; color: var(--cs-color-text-muted); }
.pc-matrix__fit { font-family: var(--cs-font-mono); font-size: .72rem; color: var(--cs-color-text-subtle); }
@media (max-width: 720px) { .pc-matrix { grid-template-columns: 1fr; } }
</style>
