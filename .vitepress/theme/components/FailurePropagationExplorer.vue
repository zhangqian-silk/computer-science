<script setup lang="ts">
import { computed, ref } from "vue"
import LearningLab from "./LearningLab.vue"
import { useLabReset } from "../use-lab-reset"

// from 调用 to
const edges: [string, string][] = [
	["Gateway", "OrderSvc"],
	["OrderSvc", "Cache"],
	["OrderSvc", "InventorySvc"],
	["OrderSvc", "PaymentSvc"],
	["InventorySvc", "DB"]
]
const layers = [["Gateway"], ["OrderSvc"], ["Cache", "InventorySvc", "PaymentSvc"], ["DB"]]
const allNodes = layers.flat()

const failNode = ref("DB")
const isolation = ref(false)

function callersOf(x: string): string[] {
	return edges.filter(([, to]) => to === x).map(([from]) => from)
}

const status = computed<Record<string, string>>(() => {
	const st: Record<string, string> = {}
	for (const n of allNodes) st[n] = "healthy"
	st[failNode.value] = "failed"
	if (isolation.value) {
		for (const c of callersOf(failNode.value)) {
			if (st[c] === "healthy") st[c] = "degraded" // 直接调用方降级兜底，止血
		}
	} else {
		const queue = [failNode.value] // 向上级联传播
		while (queue.length) {
			const cur = queue.shift() as string
			for (const c of callersOf(cur)) {
				if (st[c] === "healthy") {
					st[c] = "cascade"
					queue.push(c)
				}
			}
		}
	}
	return st
})

const summary = computed(() => {
	const cascade = allNodes.filter(n => status.value[n] === "cascade").length
	const degraded = allNodes.filter(n => status.value[n] === "degraded").length
	return isolation.value
		? `隔离生效：故障被 ${degraded} 个直接调用方以降级兜底止血，上游 ${status.value["Gateway"] === "healthy" ? "保持可用" : "受影响"}。`
		: `无隔离：故障沿调用链向上级联，波及 ${cascade} 个上游服务，最终拖垮入口。`
})

const label: Record<string, string> = { healthy: "正常", failed: "故障源", degraded: "降级兜底", cascade: "级联失败" }
const resetLab = useLabReset(failNode, isolation)
</script>

<template>
	<LearningLab topic="FailurePropagationExplorer" @reset="resetLab">
		<p class="infra-lab__hint">调用链 Gateway → OrderSvc → (Cache / InventorySvc → DB / PaymentSvc)。选择故障节点，开关「隔离」看故障是否被级联放大。</p>
		<div class="infra-controls">
			<div class="infra-control">
				<label for="fp-node">故障节点</label>
				<select id="fp-node" v-model="failNode">
					<option v-for="n in allNodes" :key="n" :value="n">{{ n }}</option>
				</select>
			</div>
		</div>
		<label class="fp-toggle"><input type="checkbox" v-model="isolation"> 开启隔离（熔断 + 舱壁 + 降级兜底）</label>
		<div class="fp-graph">
			<div v-for="(layer, li) in layers" :key="li" class="fp-layer">
				<div v-for="n in layer" :key="n" class="fp-node" :class="`fp-${status[n]}`">
					<strong>{{ n }}</strong><span>{{ label[status[n]] }}</span>
				</div>
			</div>
		</div>
		<div class="infra-result"><span>结果</span><strong>{{ summary }}</strong></div>
		<p class="infra-note">跨组件协同的核心是「不让单点故障级联成全局故障」：熔断快速失败、舱壁隔离资源、降级返回兜底。这也是错误预算与故障演练要联合治理的原因。</p>
	</LearningLab>
</template>

<style scoped>
.fp-toggle { display: flex; align-items: center; gap: .5rem; margin: .6rem 0; font-size: .85rem; color: var(--vp-c-text-2); }
.fp-graph { display: flex; flex-direction: column; gap: .5rem; margin: .8rem 0; }
.fp-layer { display: flex; gap: .5rem; flex-wrap: wrap; justify-content: center; }
.fp-node { flex: 1 1 90px; max-width: 160px; text-align: center; padding: .5rem; border: 1px solid var(--vp-c-divider); border-radius: 8px; background: var(--vp-c-bg); }
.fp-node strong { display: block; font-size: .82rem; }
.fp-node span { font-size: .7rem; color: var(--vp-c-text-3); }
.fp-healthy { border-color: var(--vp-c-green-1); }
.fp-failed { border-color: var(--vp-c-danger-1); background: var(--vp-c-danger-soft); }
.fp-cascade { border-color: var(--vp-c-danger-1); background: var(--vp-c-danger-soft); opacity: .85; }
.fp-degraded { border-color: var(--vp-c-warning-1); background: var(--vp-c-warning-soft); }
</style>
