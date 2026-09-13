<script setup lang="ts">
// 输入侧单价：命中缓存与否的差距（Claude Sonnet 公开报价，Manus 工程博客引用）。
// 数值仅说明量级，条宽按占最大值比例绘制。
const priceMax = 3.0
const prices = [
	{ label: "未命中缓存", value: 3.0, cls: "miss" },
	{ label: "命中缓存", value: 0.3, cls: "hit" }
]

const rules = [
	{ n: "①", t: "前缀必须稳定", d: "自回归下一个 token 的差异，就会让该位置之后的缓存全部失效。最典型的错误是把精确到秒的时间戳放进系统提示开头。" },
	{ n: "②", t: "上下文只追加", d: "不回改此前的动作与观测，并确保 JSON 序列化键序确定——很多库不保证键序稳定，会静默击穿缓存。" },
	{ n: "③", t: "必要时显式标断点", d: "不支持自动增量前缀缓存时手动插入缓存断点，至少覆盖到系统提示末尾；自托管还要开启前缀缓存并用会话 ID 稳定路由。" }
]
</script>

<template>
	<div class="pc-cache">
		<figure class="pc-cache__cost">
			<figcaption class="pc-cache__ttl pc-serif">输入侧单价：命中缓存与否的差距</figcaption>
			<p class="pc-cache__sub">以 Claude Sonnet 公开报价为例，单位：美元 / 百万 token</p>
			<div v-for="p in prices" :key="p.label" class="pc-cache__bar-row">
				<span class="pc-cache__bar-lab">{{ p.label }}</span>
				<div class="pc-cache__bar-track">
					<span class="pc-cache__bar" :class="p.cls" :style="{ width: (p.value / priceMax * 100) + '%' }"></span>
				</div>
				<span class="pc-cache__bar-val">{{ p.value.toFixed(2) }}</span>
			</div>
			<p class="pc-note">输入输出严重不对称（约 100:1），开销几乎全在预填。命中与未命中的输入单价可差约一个数量级——所以 KV-cache 命中率是生产阶段最重要的单一指标之一，它同时决定成本与首字延迟。数值为特定时点报价，仅说明量级。</p>
		</figure>
		<div class="pc-cache__rules">
			<article v-for="r in rules" :key="r.t" class="pc-cache__rule">
				<span class="pc-cache__n">{{ r.n }}</span>
				<div>
					<b class="pc-serif">{{ r.t }}</b>
					<p>{{ r.d }}</p>
				</div>
			</article>
		</div>
		<div class="pc-cache__mask">
			<b>工具集变更：遮蔽，而不是删除</b>
			<p>工具定义通常位于上下文最前部，中途增删会让其后所有动作与观测的缓存失效，还会因历史里引用了已消失的工具而制造冲突。正确做法是保留定义不变，用状态机在解码时遮蔽 logits 控制可用性，并配合统一的工具名前缀（如 <code>browser_</code>、<code>shell_</code>）按组启停。</p>
		</div>
	</div>
</template>

<style scoped>
.pc-cache { margin: 1rem 0; }
.pc-cache__cost { margin: 0; background: var(--cs-color-bg); border: 1px solid var(--cs-color-border); border-radius: 10px; padding: 1rem 1.1rem .85rem; }
.pc-cache__ttl { font-size: 1rem; font-weight: 700; margin: 0; }
.pc-cache__sub { font-size: .8rem; color: var(--cs-color-text-subtle); margin: .2rem 0 .8rem; }
.pc-cache__bar-row { display: grid; grid-template-columns: 92px 1fr 48px; gap: .6rem; align-items: center; margin-bottom: .5rem; }
.pc-cache__bar-lab { font-family: var(--cs-font-mono); font-size: .74rem; color: var(--cs-color-text-muted); text-align: right; }
.pc-cache__bar-track { background: var(--cs-color-bg-soft); border-radius: var(--cs-radius-sm); overflow: hidden; }
.pc-cache__bar { display: block; height: 24px; border-radius: var(--cs-radius-sm); }
.pc-cache__bar.miss { background: var(--cs-color-danger); }
.pc-cache__bar.hit { background: var(--cs-color-success); }
.pc-cache__bar-val { font-family: var(--cs-font-mono); font-size: .82rem; color: var(--cs-color-text); }
.pc-cache__rules { display: grid; grid-template-columns: repeat(3, 1fr); gap: .8rem; margin-top: .8rem; }
.pc-cache__rule { display: grid; grid-template-columns: 24px 1fr; gap: .5rem; background: var(--cs-color-bg); border: 1px solid var(--cs-color-border); border-top: 3px solid var(--cs-color-brand); border-radius: 10px; padding: .9rem 1rem; }
.pc-cache__n { font-family: var(--cs-font-mono); font-size: 1.1rem; color: var(--cs-color-brand); line-height: 1.2; }
.pc-cache__rule b { display: block; font-size: .95rem; margin-bottom: .25rem; }
.pc-cache__rule p { margin: 0; font-size: .82rem; line-height: 1.65; color: var(--cs-color-text-muted); }
.pc-cache__mask { margin-top: .8rem; background: var(--cs-color-bg-soft); border: 1px solid var(--cs-color-border); border-left: 3px solid var(--cs-color-info); border-radius: 10px; padding: .9rem 1.1rem; }
.pc-cache__mask b { display: block; font-family: var(--cs-font-mono); font-size: .72rem; letter-spacing: .06em; text-transform: uppercase; color: var(--cs-color-info); margin-bottom: .35rem; }
.pc-cache__mask p { margin: 0; font-size: .85rem; line-height: 1.7; color: var(--cs-color-text-muted); }
.pc-cache__mask code { font-family: var(--cs-font-mono); font-size: .8rem; }
@media (max-width: 720px) {
	.pc-cache__rules { grid-template-columns: 1fr; }
}
</style>
