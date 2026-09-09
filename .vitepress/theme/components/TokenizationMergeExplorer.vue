<script setup lang="ts">
import { computed, ref, useId } from "vue"
import LearningLab from "./LearningLab.vue"
import { useLabReset } from "../use-lab-reset"
import { trainBPE, encodeBPE } from "../learning-algorithms"
const labId = useId()
const fieldId = (name: string) => `${labId}-${name}`

const step = ref(0)
const word = ref("lower")
const stages = trainBPE([{ word: "low", count: 5 }, { word: "lower", count: 2 }, { word: "lowest", count: 1 }], 10)
const current = computed(() => stages[step.value])
const encoded = computed(() => encodeBPE(word.value, current.value.rules))
const resetLab = useLabReset(step, word)
</script>

<template>
	<LearningLab topic="TokenizationMergeExplorer" @reset="resetLab">
		<p class="infra-lab__hint">语料为 low × 5、lower × 2、lowest × 1。从字符和词尾标记开始，每一步重新计算相邻对频率；平局按符号对排序。</p>
		<div class="infra-control">
			<label :for="fieldId('token-merge-step')">合并步骤：{{ step }} / {{ stages.length - 1 }}</label>
			<input :id="fieldId('token-merge-step')" v-model.number="step" type="range" min="0" :max="stages.length - 1">
		</div>
		<table>
			<caption>当前语料与候选合并</caption>
			<thead><tr><th>词频</th><th>切分</th></tr></thead>
			<tbody><tr v-for="item in current.words" :key="item.word"><td>{{ item.word }} × {{ item.count }}</td><td>{{ item.tokens.join(" | ") }}</td></tr></tbody>
		</table>
		<div class="infra-control"><label :for="fieldId('encode-word')">用当前已学规则编码</label><select :id="fieldId('encode-word')" v-model="word"><option>low</option><option>lower</option><option>lowest</option><option>slow</option></select></div>
		<div class="token-strip" aria-label="当前子词序列">
			<span v-for="(token, index) in encoded" :key="`${index}-${token}`" class="token-chip is-selected">{{ token }}</span>
		</div>
		<div class="infra-results">
			<div class="infra-result"><span>下一条最高频合并</span><strong>{{ current.candidates[0]?.pair.join(" + ") ?? "没有候选" }}（{{ current.candidates[0]?.count ?? 0 }} 次）</strong></div>
			<div class="infra-result"><span>编码后长度</span><strong>{{ encoded.length }} token</strong></div>
		</div>
		<p class="infra-note">「slow」没有整词出现在训练语料中，但字符都在基础词表中。&lt;/w&gt; 仅是此例词尾标记；本实验不实现 Unicode 正规化和字节回退。</p>
	</LearningLab>
</template>
