<script setup lang="ts">
import { computed, ref } from 'vue'

const paramsB = ref(7)
const bits = ref(16)
const groupSize = ref(128)
const scaleBytes = ref(2)
const weightGiB = computed(() => paramsB.value * 1e9 * bits.value / 8 / 1024 ** 3)
const scaleGiB = computed(() => paramsB.value * 1e9 / groupSize.value * scaleBytes.value / 1024 ** 3)
const totalGiB = computed(() => weightGiB.value + (bits.value < 16 ? scaleGiB.value : 0))
const relative = computed(() => 16 / bits.value)
</script>

<template>
	<div class="infra-lab">
		<p class="infra-lab__title">量化容量权衡</p>
		<p class="infra-lab__hint">估算权重主体与 group scale；不含 zero point、padding、runtime buffer 和 KV。</p>
		<div class="infra-controls">
			<div class="infra-control"><label for="quant-params">参数量：{{ paramsB }}B</label><input id="quant-params" v-model.number="paramsB" type="range" min="1" max="100"></div>
			<div class="infra-control"><label for="quant-bits">Weight bits</label><select id="quant-bits" v-model.number="bits"><option :value="16">16</option><option :value="8">8</option><option :value="4">4</option><option :value="2">2</option></select></div>
			<div class="infra-control"><label for="quant-group">Group size</label><select id="quant-group" v-model.number="groupSize"><option :value="32">32</option><option :value="64">64</option><option :value="128">128</option><option :value="256">256</option></select></div>
		</div>
		<div class="infra-results">
			<div class="infra-result"><span>估算权重制品</span><strong>{{ totalGiB.toFixed(2) }} GiB</strong></div>
			<div class="infra-result"><span>相对 FP16 主体缩小</span><strong>{{ relative.toFixed(1) }}×</strong></div>
			<div class="infra-result"><span>Scale 元数据</span><strong>{{ bits < 16 ? scaleGiB.toFixed(3) : '0.000' }} GiB</strong></div>
		</div>
		<p class="infra-note">位宽越低不等于同比提速；质量、反量化、kernel 和硬件支持仍需实测。</p>
	</div>
</template>
