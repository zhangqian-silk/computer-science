<script setup lang="ts">
import { computed, ref } from "vue"
import LearningLab from "./LearningLab.vue"
import { useLabReset } from "../use-lab-reset"

type Step = { actor: "W" | "R"; text: string; tone?: string }
type Outcome = { steps: Step[]; cache: string; dirty: boolean; note: string }

const strategy = ref<"write-then-del" | "del-then-write" | "double-del">("write-then-del")
const concurrentRead = ref(true)

const result = computed<Outcome>(() => {
	if (strategy.value === "del-then-write") {
		if (concurrentRead.value) {
			return {
				steps: [
					{ actor: "W", text: "删除缓存", tone: "active" },
					{ actor: "R", text: "读缓存未命中", tone: "neutral" },
					{ actor: "R", text: "回源读到旧值 V0", tone: "warning" },
					{ actor: "R", text: "回填缓存 = V0", tone: "danger" },
					{ actor: "W", text: "写 DB = V1", tone: "active" }
				],
				cache: "V0", dirty: true,
				note: "先删后写：删缓存与写 DB 之间的读会把旧值回填，且不会自愈，故不推荐。"
			}
		}
		return {
			steps: [
				{ actor: "W", text: "删除缓存", tone: "active" },
				{ actor: "W", text: "写 DB = V1", tone: "active" },
				{ actor: "R", text: "读未命中 → 回源 V1 → 回填", tone: "success" }
			],
			cache: "V1", dirty: false,
			note: "无并发读时先删后写也能得到新值，但并发下不安全。"
		}
	}
	if (strategy.value === "write-then-del") {
		if (concurrentRead.value) {
			return {
				steps: [
					{ actor: "R", text: "读缓存未命中", tone: "neutral" },
					{ actor: "R", text: "回源读到旧值 V0", tone: "warning" },
					{ actor: "W", text: "写 DB = V1", tone: "active" },
					{ actor: "W", text: "删除缓存", tone: "active" },
					{ actor: "R", text: "迟到的回填 = V0", tone: "danger" }
				],
				cache: "V0", dirty: true,
				note: "先写后删仅在「读旧值早于写、回填晚于删」这一极窄窗口才脏，概率很低，是主流选择。"
			}
		}
		return {
			steps: [
				{ actor: "W", text: "写 DB = V1", tone: "active" },
				{ actor: "W", text: "删除缓存", tone: "active" },
				{ actor: "R", text: "读未命中 → 回源 V1 → 回填", tone: "success" }
			],
			cache: "V1", dirty: false,
			note: "先写 DB 后删缓存：多数并发下最终为新值。"
		}
	}
	return {
		steps: [
			{ actor: "W", text: "删除缓存", tone: "active" },
			{ actor: "W", text: "写 DB = V1", tone: "active" },
			{ actor: "R", text: "并发读回填了旧值 V0", tone: "warning" },
			{ actor: "W", text: "延迟后再次删除缓存", tone: "active" },
			{ actor: "R", text: "下次读未命中 → 回源 V1 → 回填", tone: "success" }
		],
		cache: "V1", dirty: false,
		note: "延迟双删：第二次删除清掉窗口内被回填的旧值，覆盖主从延迟 / 回填窗口。"
	}
})

const resetLab = useLabReset(strategy, concurrentRead)
</script>

<template>
	<LearningLab topic="CacheAsideRaceExplorer" @reset="resetLab">
		<p class="infra-lab__hint">DB 从旧值 V0 更新为 V1。W 是写线程，R 是发生缓存未命中的并发读线程。切换更新策略，观察最终缓存是新值还是脏值。</p>
		<div class="infra-tabs" role="group" aria-label="更新策略">
			<button type="button" :aria-pressed="strategy === 'write-then-del'" @click="strategy = 'write-then-del'">先写 DB 后删缓存</button>
			<button type="button" :aria-pressed="strategy === 'del-then-write'" @click="strategy = 'del-then-write'">先删缓存后写 DB</button>
			<button type="button" :aria-pressed="strategy === 'double-del'" @click="strategy = 'double-del'">延迟双删</button>
		</div>
		<label class="cache-toggle"><input type="checkbox" v-model="concurrentRead"> 存在并发缓存未命中读（延迟双删默认叠加）</label>
		<ol class="race-track" aria-label="操作时序">
			<li v-for="(s, i) in result.steps" :key="i" :class="`tone-${s.tone ?? 'neutral'}`">
				<span :class="`actor actor-${s.actor}`">{{ s.actor }}</span><strong>{{ s.text }}</strong>
			</li>
		</ol>
		<div class="infra-results">
			<div class="infra-result"><span>最终缓存值</span><strong>{{ result.cache }}</strong></div>
			<div class="infra-result"><span>是否脏读</span><strong>{{ result.dirty ? "脏（读到旧值）" : "干净（最终一致）" }}</strong></div>
		</div>
		<p class="infra-note">{{ result.note }}</p>
	</LearningLab>
</template>

<style scoped>
.cache-toggle { display: flex; align-items: center; gap: .5rem; margin: .6rem 0; font-size: .85rem; color: var(--vp-c-text-2); }
.race-track { list-style: none; padding: 0; margin: 1rem 0; display: flex; flex-direction: column; gap: .4rem; }
.race-track li { display: flex; align-items: center; gap: .6rem; padding: .5rem .6rem; border: 1px solid var(--vp-c-divider); border-left-width: 3px; border-radius: 6px; background: var(--vp-c-bg); }
.race-track li strong { font-size: .85rem; font-weight: 500; }
.actor { display: inline-flex; width: 24px; height: 24px; align-items: center; justify-content: center; border-radius: 50%; font-size: .72rem; font-weight: 700; }
.actor-W { background: var(--vp-c-brand-soft); color: var(--vp-c-brand-1); }
.actor-R { background: var(--vp-c-default-soft); color: var(--vp-c-text-2); }
.tone-warning { border-left-color: var(--vp-c-warning-1); }
.tone-danger { border-left-color: var(--vp-c-danger-1); }
.tone-success { border-left-color: var(--vp-c-green-1); }
.tone-active { border-left-color: var(--vp-c-brand-1); }
</style>
