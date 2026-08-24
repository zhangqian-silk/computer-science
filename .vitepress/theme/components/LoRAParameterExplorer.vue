<script setup lang="ts">
import { computed, ref } from 'vue'

const inputDimension = ref(4096)
const outputDimension = ref(4096)
const rank = ref(16)
const matrices = ref(64)
const fullParameters = computed(() => inputDimension.value * outputDimension.value * matrices.value)
const loraParameters = computed(() => rank.value * (inputDimension.value + outputDimension.value) * matrices.value)
const ratio = computed(() => loraParameters.value / fullParameters.value * 100)
const trainingMiB = computed(() => loraParameters.value * 2 / 1024 ** 2)
</script>

<template>
	<div class="infra-lab">
		<p class="infra-lab__title">LoRA 参数实验台</p>
		<p class="infra-lab__hint">比较目标矩阵全量更新与低秩增量 A、B 的可训练参数；不包含 optimizer、激活和基础权重。</p>
		<div class="infra-controls">
			<div class="infra-control"><label for="lora-input">输入维度</label><select id="lora-input" v-model.number="inputDimension"><option :value="1024">1024</option><option :value="4096">4096</option><option :value="8192">8192</option></select></div>
			<div class="infra-control"><label for="lora-output">输出维度</label><select id="lora-output" v-model.number="outputDimension"><option :value="1024">1024</option><option :value="4096">4096</option><option :value="11008">11008</option><option :value="28672">28672</option></select></div>
			<div class="infra-control"><label for="lora-rank">Rank：{{ rank }}</label><input id="lora-rank" v-model.number="rank" type="range" min="1" max="256"></div>
			<div class="infra-control"><label for="lora-matrices">目标矩阵数：{{ matrices }}</label><input id="lora-matrices" v-model.number="matrices" type="range" min="1" max="256"></div>
		</div>
		<div class="infra-results">
			<div class="infra-result"><span>全量目标参数</span><strong>{{ (fullParameters / 1e9).toFixed(2) }}B</strong></div>
			<div class="infra-result"><span>LoRA 可训练参数</span><strong>{{ (loraParameters / 1e6).toFixed(2) }}M</strong></div>
			<div class="infra-result"><span>相对目标矩阵</span><strong>{{ ratio.toFixed(3) }}%</strong></div>
			<div class="infra-result"><span>FP16/BF16 参数主体</span><strong>{{ trainingMiB.toFixed(1) }} MiB</strong></div>
		</div>
		<p class="infra-note">更高 rank 提高增量容量，也增加训练、加载和 Multi-LoRA kernel 成本；效果必须结合目标任务验证。</p>
	</div>
</template>
