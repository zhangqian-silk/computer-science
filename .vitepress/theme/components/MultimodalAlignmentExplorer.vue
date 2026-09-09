<script setup lang="ts">
import { computed, ref, useId } from "vue"
import LearningLab from "./LearningLab.vue"
import { useLabReset } from "../use-lab-reset"
const id = useId()
const scores = [ref(2), ref(0), ref(0), ref(2)]
const wrongLabels = ref(false)
const normalize = (values: number[]) => {
	const maximum = Math.max(...values)
	const valuesExp = values.map(value => Math.exp(value - maximum))
	const sum = valuesExp.reduce((a, b) => a + b, 0)
	return valuesExp.map(value => value / sum)
}
const rows = computed(() => [normalize(scores.slice(0, 2).map(x => x.value)), normalize(scores.slice(2).map(x => x.value))])
const columns = computed(() => [normalize([scores[0].value, scores[2].value]), normalize([scores[1].value, scores[3].value])])
const loss = computed(() => {
	const target = wrongLabels.value ? [1, 0] : [0, 1]
	return -[0, 1].reduce((sum, i) => sum + Math.log(rows.value[i][target[i]]) + Math.log(columns.value[i][target[i]]), 0) / 4
})
const resetLab = useLabReset(scores, wrongLabels)
</script>
<template>
	<LearningLab topic="MultimodalAlignmentExplorer" @reset="resetLab">
		<p class="infra-lab__hint">两张图像与两段文本。输入是已经除以温度的四个 logits，不是训练得到的真实 CLIP 向量。正确配对位于对角线。</p>
		<div class="infra-controls">
			<div v-for="(score, i) in scores" :key="i" class="infra-control"><label :for="`${id}-${i}`">图 {{ Math.floor(i / 2) + 1 }} ↔ 文 {{ i % 2 + 1 }}：{{ score.value.toFixed(1) }}</label><input :id="`${id}-${i}`" v-model.number="score.value" type="range" min="-4" max="4" step=".1"></div>
		</div>
		<table><caption>同一 logits 矩阵，两种归一化方向</caption><thead><tr><th>方向</th><th>候选 1 概率</th><th>候选 2 概率</th></tr></thead><tbody>
			<tr v-for="(row, i) in rows" :key="`row-${i}`"><th>图 {{ i + 1 }} 找文本</th><td>{{ row[0].toFixed(3) }}</td><td>{{ row[1].toFixed(3) }}</td></tr>
			<tr v-for="(column, i) in columns" :key="`col-${i}`"><th>文 {{ i + 1 }} 找图像</th><td>{{ column[0].toFixed(3) }}</td><td>{{ column[1].toFixed(3) }}</td></tr>
		</tbody></table>
		<div class="infra-tabs"><button type="button" :aria-pressed="wrongLabels" @click="wrongLabels = !wrongLabels">{{ wrongLabels ? "使用错误的交叉标签" : "使用正确的对角标签" }}</button></div>
		<div class="infra-result"><span>对称交叉熵（自然对数，四项平均）</span><strong>{{ loss.toFixed(3) }}</strong></div>
		<p class="infra-note">只改变标签而不改变得分，也会改变优化方向。行概率在两个文本中比较，列概率在两个图像中比较，不能当作独立于候选集的置信度。</p>
	</LearningLab>
</template>
