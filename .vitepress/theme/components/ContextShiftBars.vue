<script setup lang="ts">
const shift = [
	{ step: "第 1 步", stat: 62, dyn: 20, acc: 18 },
	{ step: "第 30 步", stat: 12, dyn: 20, acc: 68 }
]
</script>

<template>
	<figure class="pc-shift">
		<figcaption class="pc-shift__ttl pc-serif">同一任务，第 1 步与第 30 步的 token 构成</figcaption>
		<div class="pc-shift__legend" aria-hidden="true">
			<span><i class="s" />静态</span>
			<span><i class="d" />动态</span>
			<span><i class="a" />累积</span>
		</div>
		<div v-for="s in shift" :key="s.step" class="pc-shift__row">
			<span class="pc-shift__step">{{ s.step }}</span>
			<div class="pc-shift__track" role="img" :aria-label="`${s.step}：静态 ${s.stat}%、动态 ${s.dyn}%、累积 ${s.acc}%`">
				<span class="seg s" :style="{ width: s.stat + '%' }">{{ s.stat }}%</span>
				<span class="seg d" :style="{ width: s.dyn + '%' }">{{ s.dyn }}%</span>
				<span class="seg a" :style="{ width: s.acc + '%' }">{{ s.acc }}%</span>
			</div>
		</div>
		<figcaption class="pc-note">比例为定性示意，非任何系统的实测分布。静态部分绝对值不变、占比被稀释，累积部分从边缘变成主体——真正的战场是消息历史与工具返回值。</figcaption>
	</figure>
</template>

<style scoped>
.pc-shift { margin: 1rem 0 0; background: var(--cs-color-bg); border: 1px solid var(--cs-color-border); border-radius: 10px; padding: 1rem 1.1rem .85rem; }
.pc-shift__ttl { font-size: 1rem; font-weight: 700; margin: 0 0 .8rem; }
.pc-shift__legend { display: flex; gap: 1rem; margin-bottom: .55rem; font-size: .76rem; color: var(--cs-color-text-muted); }
.pc-shift__legend span { display: inline-flex; align-items: center; gap: .35rem; }
.pc-shift__legend i { width: 10px; height: 10px; border-radius: 2px; display: inline-block; }
.pc-shift__legend i.s { background: var(--cs-color-info); }
.pc-shift__legend i.d { background: var(--cs-color-success); }
.pc-shift__legend i.a { background: var(--cs-color-warning); }
.pc-shift__row { display: grid; grid-template-columns: 64px 1fr; gap: .7rem; align-items: center; margin-bottom: .5rem; }
.pc-shift__step { font-family: var(--cs-font-mono); font-size: .74rem; color: var(--cs-color-text-muted); text-align: right; white-space: nowrap; }
.pc-shift__track { display: flex; height: 30px; border-radius: var(--cs-radius-sm); overflow: hidden; border: 1px solid var(--cs-color-border); }
.pc-shift__track .seg { display: flex; align-items: center; justify-content: center; font-size: .68rem; font-family: var(--cs-font-mono); font-variant-numeric: tabular-nums; color: var(--cs-color-on-brand); white-space: nowrap; overflow: hidden; }
.pc-shift__track .seg.s { background: var(--cs-color-info); }
.pc-shift__track .seg.d { background: var(--cs-color-success); }
.pc-shift__track .seg.a { background: var(--cs-color-warning); }
@media (max-width: 640px) {
	.pc-shift__row { grid-template-columns: auto 1fr; gap: .55rem; }
	.pc-shift__step { font-size: .68rem; }
	.pc-shift__track { height: 26px; }
}
</style>
