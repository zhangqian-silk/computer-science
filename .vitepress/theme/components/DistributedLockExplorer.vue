<script setup lang="ts">
import { computed, ref, useId } from "vue"
import LearningLab from "./LearningLab.vue"
import { useLabReset } from "../use-lab-reset"

const labId = useId()
const fieldId = (name: string) => `${labId}-${name}`

const ttl = ref(10)          // 锁 TTL（秒）
const workTime = ref(25)     // 业务实际耗时（秒）
const watchdog = ref(true)   // 是否开启看门狗自动续约
const uniqueToken = ref(true) // 解锁是否校验唯一 token

type Event = { t: number; text: string; tone: string }

const result = computed<{ events: Event[]; safe: boolean; verdict: string }>(() => {
	const events: Event[] = []
	events.push({ t: 0, text: `A 加锁成功，TTL=${ttl.value}s，写入唯一 token`, tone: "active" })

	if (watchdog.value) {
		// 看门狗每 TTL/3 续约，锁在业务期间不会过期
		const step = Math.max(1, Math.round(ttl.value / 3))
		for (let t = step; t < workTime.value; t += step) {
			events.push({ t, text: `看门狗续约，锁 TTL 重置为 ${ttl.value}s`, tone: "neutral" })
		}
		events.push({ t: workTime.value, text: `A 业务完成（${workTime.value}s）`, tone: "success" })
		if (uniqueToken.value) {
			events.push({ t: workTime.value, text: "A 用 token 校验后原子解锁", tone: "success" })
			return { events, safe: true, verdict: "安全：看门狗续约保证锁不早退，唯一 token 保证只删自己的锁。" }
		}
		events.push({ t: workTime.value, text: "A 直接 DEL（未校验 token）", tone: "warning" })
		return { events, safe: true, verdict: "基本安全，但解锁不校验 token 存在误删他人锁的风险。" }
	}

	// 无看门狗：TTL < 业务耗时则锁提前过期
	if (ttl.value < workTime.value) {
		events.push({ t: ttl.value, text: `锁到期自动释放（A 仍在执行！）`, tone: "danger" })
		events.push({ t: ttl.value + 1, text: "B 获取到同一把锁，进入临界区", tone: "danger" })
		events.push({ t: workTime.value, text: "A 业务完成", tone: "neutral" })
		if (uniqueToken.value) {
			events.push({ t: workTime.value, text: "A 解锁校验 token：已非自己的锁，放弃删除", tone: "warning" })
			return { events, safe: false, verdict: "不安全：TTL 小于业务耗时导致锁提前过期、A 与 B 同时进入临界区。token 校验只避免了误删，防不住并发。" }
		}
		events.push({ t: workTime.value, text: "A 直接 DEL，误删了 B 持有的锁", tone: "danger" })
		return { events, safe: false, verdict: "严重不安全：锁提前过期造成并发，且无 token 校验又误删了 B 的锁，连环故障。" }
	}

	events.push({ t: workTime.value, text: `A 业务完成（${workTime.value}s < TTL）`, tone: "success" })
	events.push({ t: workTime.value, text: uniqueToken.value ? "A 校验 token 后解锁" : "A 直接 DEL", tone: uniqueToken.value ? "success" : "warning" })
	return { events, safe: true, verdict: "安全：TTL 大于业务耗时，锁不会提前过期。但 TTL 需预估准确，估短即出问题。" }
})

const resetLab = useLabReset(ttl, workTime, watchdog, uniqueToken)
</script>

<template>
	<LearningLab topic="DistributedLockExplorer" @reset="resetLab">
		<p class="infra-lab__hint">A 持有 Redis 锁执行业务。调整锁 TTL 与业务耗时，看锁是否会在业务完成前过期，导致 B 抢到锁造成并发。</p>
		<div class="infra-controls">
			<div class="infra-control"><label :for="fieldId('ttl')">锁 TTL：{{ ttl }}s</label><input :id="fieldId('ttl')" v-model.number="ttl" type="range" min="3" max="30"></div>
			<div class="infra-control"><label :for="fieldId('work')">业务耗时：{{ workTime }}s</label><input :id="fieldId('work')" v-model.number="workTime" type="range" min="3" max="40"></div>
		</div>
		<div class="dl-toggles">
			<label><input type="checkbox" v-model="watchdog"> 看门狗自动续约</label>
			<label><input type="checkbox" v-model="uniqueToken"> 解锁校验唯一 token</label>
		</div>
		<ol class="dl-track" aria-label="锁事件时序">
			<li v-for="(e, i) in result.events" :key="i" :class="`tone-${e.tone}`">
				<span class="dl-time">{{ e.t }}s</span><strong>{{ e.text }}</strong>
			</li>
		</ol>
		<div class="infra-result" :class="result.safe ? 'dl-ok' : 'dl-bad'"><span>{{ result.safe ? "安全" : "存在互斥破坏" }}</span><strong>{{ result.verdict }}</strong></div>
		<p class="infra-note">两条铁律：锁必须能覆盖业务全程（看门狗续约或 TTL 足够），解锁必须校验唯一 token 原子删除（避免误删他人锁）。</p>
	</LearningLab>
</template>

<style scoped>
.dl-toggles { display: flex; flex-wrap: wrap; gap: .8rem; margin: .6rem 0; font-size: .84rem; color: var(--vp-c-text-2); }
.dl-toggles label { display: flex; align-items: center; gap: .4rem; }
.dl-track { list-style: none; padding: 0; margin: .8rem 0; display: flex; flex-direction: column; gap: .35rem; }
.dl-track li { display: flex; align-items: center; gap: .6rem; padding: .45rem .6rem; border: 1px solid var(--vp-c-divider); border-left-width: 3px; border-radius: 6px; background: var(--vp-c-bg); }
.dl-track li strong { font-size: .82rem; font-weight: 500; }
.dl-time { font-variant-numeric: tabular-nums; font-size: .72rem; color: var(--vp-c-text-3); width: 34px; flex: 0 0 auto; }
.tone-active { border-left-color: var(--vp-c-brand-1); }
.tone-neutral { border-left-color: var(--vp-c-divider); }
.tone-success { border-left-color: var(--vp-c-green-1); }
.tone-warning { border-left-color: var(--vp-c-warning-1); }
.tone-danger { border-left-color: var(--vp-c-danger-1); }
.dl-ok strong { color: var(--vp-c-green-1); }
.dl-bad strong { color: var(--vp-c-danger-1); }
</style>
