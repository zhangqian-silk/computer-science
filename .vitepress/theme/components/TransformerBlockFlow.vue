<script setup lang="ts">
import { computed, ref, useId } from "vue"
import LearningLab from "./LearningLab.vue"
import { useLabReset } from "../use-lab-reset"
const labId = useId()
const fieldId = (name: string) => `${labId}-${name}`

const stage = ref(0)
const norm = ref("pre")
const stages = computed(() => [
	{ name: "输入残差流", shape: "[B, T, d_model]", note: "保留 X 作为第一条残差来源。位置可加在输入，也可在 Q/K 上旋转，不能混为一步。" },
	{ name: norm.value === "pre" ? "Norm → Attention" : "Attention", shape: "Q/K/V → [B, heads, T, d_head]", note: norm.value === "pre" ? "A = Attention(Norm(X))，读取可见位置后投影回模型宽度。" : "A = Attention(X)，Norm 尚未执行；mask 决定双向或因果。" },
	{ name: norm.value === "pre" ? "第一次残差" : "第一次残差 → Norm", shape: "[B, T, d_model]", note: norm.value === "pre" ? "Y = X + A，保留直接恒等路径。" : "Y = Norm(X + A)，残差梯度也经过外侧 Norm。" },
	{ name: norm.value === "pre" ? "Norm → FFN" : "FFN", shape: "[B, T, d_ff] → [B, T, d_model]", note: norm.value === "pre" ? "F = FFN(Norm(Y))；逐位置共用权重，不再混合 token。" : "F = FFN(Y)；升维、非线性与降维恢复模型宽度。" },
	{ name: norm.value === "pre" ? "第二次残差" : "第二次残差 → Norm", shape: "[B, T, d_model]", note: norm.value === "pre" ? "Z = Y + F。完整 Pre-Norm 主干通常另设最后输出 Norm。" : "Z = Norm(Y + F)，作为下一层输入。" }
])
const current = computed(() => stages.value[stage.value])
const resetLab = useLabReset(stage, norm)
</script>

<template>
	<LearningLab topic="TransformerBlockFlow" @reset="resetLab">
		<p class="infra-lab__hint">先选择子层顺序，再沿残差流检查形状。原始 Transformer 是 Post-Norm，不能用 Pre-Norm 图替代而不注明。</p>
		<div class="infra-tabs" role="group" aria-label="Norm 位置"><button type="button" :aria-pressed="norm === 'pre'" @click="norm = 'pre'">Pre-Norm</button><button type="button" :aria-pressed="norm === 'post'" @click="norm = 'post'">Post-Norm</button></div>
		<div class="infra-tabs" role="group" aria-label="计算步骤"><button v-for="(item, i) in stages" :key="i" type="button" :aria-pressed="stage === i" @click="stage = i">{{ i + 1 }} · {{ item.name }}</button></div>
		<div class="infra-control">
			<label :for="fieldId('block-stage')">阶段：{{ stage + 1 }} / {{ stages.length }}</label>
			<input :id="fieldId('block-stage')" v-model.number="stage" type="range" min="0" :max="stages.length - 1">
		</div>
		<div class="infra-results">
			<div class="infra-result"><span>当前步骤</span><strong>{{ current.name }}</strong></div>
			<div class="infra-result"><span>主要形状</span><strong>{{ current.shape }}</strong></div>
		</div>
		<p class="infra-note">{{ current.note }}</p>
	</LearningLab>
</template>
