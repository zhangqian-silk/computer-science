<script setup lang="ts">
import { computed, ref, useId } from "vue"
import LearningLab from "./LearningLab.vue"
import { useLabReset } from "../use-lab-reset"

const labId = useId()
const fieldId = (name: string) => `${labId}-${name}`

const model = ref<"heartbeat" | "session">("heartbeat")
const heartbeatInterval = ref(5)   // 心跳间隔(s)
const ttl = ref(15)                // 失效判定 TTL(s)
const lastHeartbeatAgo = ref(8)    // 距离上次心跳(s)：模拟实例卡顿/宕机

const healthy = computed(() =>
	model.value === "session" ? lastHeartbeatAgo.value < 30 : lastHeartbeatAgo.value < ttl.value
)

const missed = computed(() => Math.floor(lastHeartbeatAgo.value / heartbeatInterval.value))

const verdict = computed(() => {
	if (model.value === "session") {
		return lastHeartbeatAgo.value < 30
			? "会话保活中：客户端与注册中心的长连接存活，实例视为健康。"
			: "会话超时：长连接断开，注册中心立即摘除该实例（如 ZooKeeper 临时节点消失）。"
	}
	return healthy.value
		? `健康：${ttl.value}s 内收到过心跳（漏跳 ${missed.value} 次仍在容忍范围）。`
		: `已摘除：超过 TTL=${ttl.value}s 未收到心跳，判定下线，从可用列表剔除并通知订阅方。`
})

const resetLab = useLabReset(model, heartbeatInterval, ttl, lastHeartbeatAgo)
</script>

<template>
	<LearningLab topic="ServiceRegistryExplorer" @reset="resetLab">
		<p class="infra-lab__hint">实例向注册中心上报存活。调整「距上次心跳时间」模拟卡顿/宕机，观察实例何时被判定下线。</p>
		<div class="infra-tabs" role="group" aria-label="健康检查模型">
			<button type="button" :aria-pressed="model === 'heartbeat'" @click="model = 'heartbeat'">心跳 + TTL（Eureka/Nacos）</button>
			<button type="button" :aria-pressed="model === 'session'" @click="model = 'session'">会话/临时节点（ZK/etcd）</button>
		</div>
		<div class="infra-controls" v-if="model === 'heartbeat'">
			<div class="infra-control"><label :for="fieldId('hb')">心跳间隔：{{ heartbeatInterval }}s</label><input :id="fieldId('hb')" v-model.number="heartbeatInterval" type="range" min="1" max="10"></div>
			<div class="infra-control"><label :for="fieldId('ttl')">失效 TTL：{{ ttl }}s</label><input :id="fieldId('ttl')" v-model.number="ttl" type="range" min="5" max="45"></div>
		</div>
		<div class="infra-control"><label :for="fieldId('ago')">距上次心跳：{{ lastHeartbeatAgo }}s</label><input :id="fieldId('ago')" v-model.number="lastHeartbeatAgo" type="range" min="0" max="45"></div>
		<div class="sr-status" :class="healthy ? 'sr-up' : 'sr-down'">
			<span class="sr-dot"></span>{{ healthy ? "UP 在线" : "DOWN 已摘除" }}
		</div>
		<div class="infra-result"><span>判定</span><strong>{{ verdict }}</strong></div>
		<p class="infra-note">TTL 要覆盖若干个心跳周期以容忍网络抖动（如 3×间隔），过短会误摘健康实例，过长则宕机实例长期滞留在可用列表。会话型靠长连接断开即时感知，更快但依赖连接稳定。</p>
	</LearningLab>
</template>

<style scoped>
.sr-status { display: inline-flex; align-items: center; gap: .5rem; margin: .8rem 0; padding: .4rem .8rem; border-radius: 999px; font-size: .85rem; font-weight: 600; }
.sr-dot { width: 10px; height: 10px; border-radius: 50%; }
.sr-up { background: var(--vp-c-green-soft); color: var(--vp-c-green-1); }
.sr-up .sr-dot { background: var(--vp-c-green-1); }
.sr-down { background: var(--vp-c-danger-soft); color: var(--vp-c-danger-1); }
.sr-down .sr-dot { background: var(--vp-c-danger-1); }
</style>
