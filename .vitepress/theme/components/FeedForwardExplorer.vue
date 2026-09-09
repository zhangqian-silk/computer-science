<script setup lang="ts">
import { computed, ref, useId } from "vue"
import LearningLab from "./LearningLab.vue"
import { useLabReset } from "../use-lab-reset"
const id = useId()
const x1 = ref(1)
const x2 = ref(-1)
const mode = ref<"silu" | "sigmoid">("silu")
const gate = computed(() => [x1.value + .5 * x2.value, -.5 * x1.value + x2.value])
const up = computed(() => [x1.value - x2.value, .5 * x1.value + .5 * x2.value])
const activated = computed(() => gate.value.map(value => mode.value === "silu" ? value / (1 + Math.exp(-value)) : 1 / (1 + Math.exp(-value))))
const output = computed(() => activated.value.map((value, i) => value * up.value[i]))
const format = (items: number[]) => `[${items.map(value => value.toFixed(3)).join(", ")}]`
const resetLab = useLabReset(x1, x2, mode)
</script>
<template>
	<LearningLab topic="FeedForwardExplorer" @reset="resetLab">
		<p class="infra-lab__hint">二维算例：Wg=[[1,0.5],[-0.5,1]]，Wu=[[1,-1],[0.5,0.5]]，Wo=单位矩阵，无 bias。权重固定，不进行训练。</p>
		<div class="infra-controls">
			<div class="infra-control"><label :for="`${id}-x1`">输入 x₁：{{ x1.toFixed(1) }}</label><input :id="`${id}-x1`" v-model.number="x1" type="range" min="-3" max="3" step=".1"></div>
			<div class="infra-control"><label :for="`${id}-x2`">输入 x₂：{{ x2.toFixed(1) }}</label><input :id="`${id}-x2`" v-model.number="x2" type="range" min="-3" max="3" step=".1"></div>
		</div>
		<div class="infra-tabs" role="group" aria-label="门函数"><button type="button" :aria-pressed="mode === 'silu'" @click="mode = 'silu'">SiLU 门</button><button type="button" :aria-pressed="mode === 'sigmoid'" @click="mode = 'sigmoid'">Sigmoid 门</button></div>
		<div class="infra-cards">
			<div class="infra-card"><span>门分支：Wg x → 激活</span><code>{{ format(gate) }} → {{ format(activated) }}</code></div>
			<div class="infra-card"><span>值分支：Wu x</span><code>{{ format(up) }}</code></div>
		</div>
		<p class="lab-equation">y = Wo · (φ(Wg x) ⊙ Wu x)</p>
		<div class="infra-result"><span>逐元素乘与输出</span><strong>{{ format(output) }}</strong></div>
		<p class="infra-note">Sigmoid 门在 0 与 1 之间，SiLU 门可以为负或大于 1；两分支不是独立 token，而是同一 token 的不同通道投影。</p>
	</LearningLab>
</template>
