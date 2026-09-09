<script setup lang="ts">
import { computed, ref, useId } from "vue"
import LearningLab from "./LearningLab.vue"
import { useLabReset } from "../use-lab-reset"

const labId = useId()
const fieldId = (name: string) => `${labId}-${name}`

const instances = ref(20)
const grayPercent = ref(10)
const pushMode = ref<"longpoll" | "watch">("longpoll")
const rolledBack = ref(false)

const newCount = computed(() => rolledBack.value ? 0 : Math.round(instances.value * grayPercent.value / 100))
const oldCount = computed(() => instances.value - newCount.value)

const latency = computed(() => pushMode.value === "watch" ? "毫秒级（长连接事件推送）" : "秒级（长轮询 hold 请求，变更即返回）")

const verdict = computed(() => {
	if (rolledBack.value) return "已回滚：所有实例切回旧配置版本 v1，无需重启。"
	if (newCount.value === 0) return "灰度比例为 0：全部实例仍用旧配置 v1。"
	if (newCount.value === instances.value) return "全量发布：所有实例已切到新配置 v2。"
	return `灰度中：${newCount.value} 个实例使用新配置 v2，其余 ${oldCount.value} 个保持 v1，可观察指标后再扩大比例。`
})

const resetLab = useLabReset(instances, grayPercent, pushMode, rolledBack)
</script>

<template>
	<LearningLab topic="ConfigCenterExplorer" @reset="resetLab">
		<p class="infra-lab__hint">配置中心把配置变更动态推送到实例，无需重启。按比例灰度发布新配置 v2，观察实例分布；出问题一键回滚。</p>
		<div class="infra-tabs" role="group" aria-label="推送方式">
			<button type="button" :aria-pressed="pushMode === 'longpoll'" @click="pushMode = 'longpoll'">长轮询（Nacos/Apollo）</button>
			<button type="button" :aria-pressed="pushMode === 'watch'" @click="pushMode = 'watch'">Watch 长连接（etcd/ZK）</button>
		</div>
		<div class="infra-controls">
			<div class="infra-control"><label :for="fieldId('inst')">实例数：{{ instances }}</label><input :id="fieldId('inst')" v-model.number="instances" type="range" min="5" max="60"></div>
			<div class="infra-control"><label :for="fieldId('gray')">灰度比例：{{ grayPercent }}%</label><input :id="fieldId('gray')" v-model.number="grayPercent" type="range" min="0" max="100" step="5" :disabled="rolledBack"></div>
		</div>
		<div class="cc-bar" role="img" aria-label="实例配置版本分布">
			<div class="cc-seg cc-new" :style="{ flexGrow: newCount || 0.0001 }"><span v-if="newCount">v2 · {{ newCount }}</span></div>
			<div class="cc-seg cc-old" :style="{ flexGrow: oldCount || 0.0001 }"><span v-if="oldCount">v1 · {{ oldCount }}</span></div>
		</div>
		<div class="infra-actions">
			<button type="button" class="secondary" :disabled="!rolledBack" @click="rolledBack = false">重新发布 v2</button>
			<button type="button" :disabled="rolledBack || newCount === 0" @click="rolledBack = true">一键回滚 v1</button>
		</div>
		<div class="infra-results">
			<div class="infra-result"><span>推送时延</span><strong>{{ latency }}</strong></div>
			<div class="infra-result"><span>状态</span><strong>{{ verdict }}</strong></div>
		</div>
		<p class="infra-note">动态配置的价值在于「改配置不重启」+「灰度可控」+「秒级回滚」。客户端需缓存本地快照，配置中心宕机时用旧值兜底，避免拉不到配置导致启动失败。</p>
	</LearningLab>
</template>

<style scoped>
.cc-bar { display: flex; width: 100%; height: 40px; border-radius: 6px; overflow: hidden; margin: .8rem 0; }
.cc-seg { display: flex; align-items: center; justify-content: center; color: #fff; font-size: .78rem; min-width: 0; }
.cc-new { background: var(--vp-c-brand-1); }
.cc-old { background: var(--vp-c-default-3, #94a3b8); color: var(--vp-c-text-1); }
</style>
