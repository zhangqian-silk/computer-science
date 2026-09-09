<script setup lang="ts">
import { computed, ref, useId } from "vue"
import LearningLab from "./LearningLab.vue"
import { useLabReset } from "../use-lab-reset"

const labId = useId()
const fieldId = (name: string) => `${labId}-${name}`

const followers = ref(1000)
const postsPerDay = ref(5)
const readsPerDay = ref(20)

// 写扩散(push)：发一条写入每个粉丝收件箱；读便宜
const pushWrites = computed(() => postsPerDay.value * followers.value)
const pushReads = computed(() => readsPerDay.value) // 直接读自己收件箱
// 读扩散(pull)：只写一份；读时合并所关注作者
const pullWrites = computed(() => postsPerDay.value)
const pullReads = computed(() => readsPerDay.value * followers.value / 50) // 读时聚合关注对象(示意)

const recommend = computed(() =>
	followers.value > 10000
		? "该账号是「大 V」：写扩散会产生百万级写放大，应改用读扩散或推拉结合（大 V 走拉、普通用户走推）。"
		: "普通账号：写扩散（push）更合适，写放大可接受且读极快。"
)

function fmt(n: number): string {
	if (n >= 1e6) return (n / 1e6).toFixed(2) + "M"
	if (n >= 1e3) return (n / 1e3).toFixed(1) + "K"
	return n.toFixed(0)
}

const resetLab = useLabReset(followers, postsPerDay, readsPerDay)
</script>

<template>
	<LearningLab topic="FanoutExplorer" @reset="resetLab">
		<p class="infra-lab__hint">Feed/群消息的核心权衡：写扩散（发时写入每个收件箱）vs 读扩散（读时聚合）。调整粉丝数看写放大。</p>
		<div class="infra-controls">
			<div class="infra-control"><label :for="fieldId('f')">粉丝/群成员数：{{ followers }}</label><input :id="fieldId('f')" v-model.number="followers" type="range" min="10" max="100000" step="10"></div>
			<div class="infra-control"><label :for="fieldId('p')">日发布数：{{ postsPerDay }}</label><input :id="fieldId('p')" v-model.number="postsPerDay" type="range" min="1" max="50"></div>
		</div>
		<div class="fo-cmp">
			<div class="fo-col">
				<h4>写扩散 Push</h4>
				<div class="infra-result"><span>写操作/天</span><strong>{{ fmt(pushWrites) }}</strong></div>
				<div class="infra-result"><span>读成本</span><strong>低（直读收件箱）</strong></div>
			</div>
			<div class="fo-col">
				<h4>读扩散 Pull</h4>
				<div class="infra-result"><span>写操作/天</span><strong>{{ fmt(pullWrites) }}</strong></div>
				<div class="infra-result"><span>读成本</span><strong>高（读时聚合）</strong></div>
			</div>
		</div>
		<p class="infra-note">{{ recommend }} 写扩散读快但写放大随粉丝数线性增长；读扩散写便宜但读要实时聚合。工业界（微博/朋友圈）多用「推拉结合」按账号规模分流。</p>
	</LearningLab>
</template>

<style scoped>
.fo-cmp { display: grid; grid-template-columns: 1fr 1fr; gap: .8rem; margin: .8rem 0; }
.fo-col h4 { margin: 0 0 .4rem; font-size: .85rem; }
.fo-col .infra-result { margin-bottom: .35rem; }
@media (max-width: 560px) { .fo-cmp { grid-template-columns: 1fr; } }
</style>
