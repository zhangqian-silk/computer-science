<script setup lang="ts">
const modes = [
	{ zh: "上下文中毒", en: "Context Poisoning", why: "幻觉或错误进入上下文后被反复当作事实引用，误差在迭代中放大。", fix: "校验工具输出，区分「已验证事实」与「模型猜测」；裁剪被证伪的信息，但保留失败记录本身作为纠错证据。" },
	{ zh: "上下文干扰", en: "Context Distraction", why: "历史过长时模型过度复述既有轨迹，忽略训练所得的通用判断。", fix: "未满也主动总结与修剪，周期性把目标与约束重述到最近位置。" },
	{ zh: "上下文混淆", en: "Context Confusion", why: "冗余内容被「捡起来用」，最常见是工具过多、功能重叠导致选错。", fix: "裁出最小、不重叠的工具集；人都说不清何时该用的两个工具，应合并或删掉一个。" },
	{ zh: "上下文冲突", en: "Context Clash", why: "新进入的信息或工具与既有内容直接矛盾，早期错误尝试持续拖累后续推理。", fix: "需求尽量一次给全；口径变更时显式声明「以下取代此前的 X」，必要时开新会话而非在污染历史上打补丁。" }
]
</script>

<template>
	<div class="pc-fail">
		<article v-for="(m, i) in modes" :key="m.en" class="pc-fail__card">
			<span class="pc-fail__idx">0{{ i + 1 }}</span>
			<h4 class="pc-serif">{{ m.zh }}</h4>
			<p class="pc-fail__en">{{ m.en }}</p>
			<p class="pc-fail__why">{{ m.why }}</p>
			<div class="pc-fail__fix"><b>应对</b>{{ m.fix }}</div>
		</article>
	</div>
</template>

<style scoped>
.pc-fail { display: grid; grid-template-columns: 1fr 1fr; gap: .8rem; margin: 1rem 0; }
.pc-fail__card { position: relative; background: var(--cs-color-bg); border: 1px solid var(--cs-color-border); border-top: 3px solid var(--cs-color-danger); border-radius: 10px; padding: 1.05rem 1.15rem; overflow: hidden; }
.pc-fail__idx { position: absolute; top: .5rem; right: .8rem; font-family: var(--cs-font-mono); font-size: 2rem; line-height: 1; color: color-mix(in srgb, var(--cs-color-danger) 22%, transparent); }
.pc-fail__card h4 { margin: 0; font-size: 1.02rem; font-weight: 700; color: var(--cs-color-danger); }
.pc-fail__en { margin: .15rem 0 .6rem; font-family: var(--cs-font-mono); font-size: .74rem; letter-spacing: .06em; color: var(--cs-color-text-subtle); }
.pc-fail__why { margin: 0 0 .7rem; font-size: .86rem; line-height: 1.7; color: var(--cs-color-text-muted); }
.pc-fail__fix { border-top: 1px dashed var(--cs-color-border); padding-top: .6rem; font-size: .84rem; line-height: 1.7; color: var(--cs-color-text-muted); }
.pc-fail__fix b { display: block; font-family: var(--cs-font-mono); font-size: .68rem; letter-spacing: .1em; text-transform: uppercase; color: var(--cs-color-success); margin-bottom: .25rem; }
@media (max-width: 640px) { .pc-fail { grid-template-columns: 1fr; } }
</style>
