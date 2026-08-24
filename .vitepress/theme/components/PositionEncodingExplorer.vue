<script setup lang="ts">
import { computed, ref } from 'vue'

type PositionMethod = 'sinusoidal' | 'relative' | 'rope' | 'alibi'

const props = withDefaults(defineProps<{ initialMethod?: PositionMethod }>(), {
	initialMethod: 'rope'
})

const method = ref<PositionMethod>(props.initialMethod)
const queryPosition = ref(2)
const keyPosition = ref(6)
const frequency = ref(0.5)
const slope = ref(0.2)
const distance = computed(() => keyPosition.value - queryPosition.value)
const queryAngle = computed(() => queryPosition.value * frequency.value)
const keyAngle = computed(() => keyPosition.value * frequency.value)
const relativeDot = computed(() => Math.cos(distance.value * frequency.value))
const alibiAllowed = computed(() => keyPosition.value <= queryPosition.value)
const alibiBias = computed(() => alibiAllowed.value ? -slope.value * Math.abs(distance.value) : null)
const methodSummary = computed(() => ({
	sinusoidal: '向输入加入多频位置坐标',
	relative: '按位置差查找或计算 score 偏置',
	rope: '旋转 Q/K，使点积依赖相位差',
	alibi: '按距离在线性 score 上施加惩罚'
})[method.value])
</script>

<template>
	<div class="infra-lab">
		<p class="infra-lab__title">位置表示实验台</p>
		<p class="infra-lab__hint">改变两个位置，观察绝对角度、相对位移和 Attention score 修正之间的区别。</p>
		<div class="infra-tabs" role="group" aria-label="位置表示方法">
			<button type="button" :aria-pressed="method === 'sinusoidal'" @click="method = 'sinusoidal'">正弦余弦</button>
			<button type="button" :aria-pressed="method === 'relative'" @click="method = 'relative'">相对偏置</button>
			<button type="button" :aria-pressed="method === 'rope'" @click="method = 'rope'">RoPE</button>
			<button type="button" :aria-pressed="method === 'alibi'" @click="method = 'alibi'">ALiBi</button>
		</div>
		<div class="infra-controls">
			<div class="infra-control"><label for="position-query">Query 位置：{{ queryPosition }}</label><input id="position-query" v-model.number="queryPosition" type="range" min="0" max="16"></div>
			<div class="infra-control"><label for="position-key">Key 位置：{{ keyPosition }}</label><input id="position-key" v-model.number="keyPosition" type="range" min="0" max="16"></div>
			<div v-if="method === 'sinusoidal' || method === 'rope'" class="infra-control"><label for="position-frequency">示例频率：{{ frequency.toFixed(2) }}</label><input id="position-frequency" v-model.number="frequency" type="range" min="0.1" max="1.5" step="0.05"></div>
			<div v-if="method === 'alibi'" class="infra-control"><label for="alibi-slope">斜率：{{ slope.toFixed(2) }}</label><input id="alibi-slope" v-model.number="slope" type="range" min="0.05" max="1" step="0.05"></div>
		</div>
		<div v-if="method === 'sinusoidal' || method === 'rope'" class="angle-stage" aria-label="二维位置相位">
			<div class="angle-vector query-vector" :style="{ transform: `rotate(${queryAngle}rad)` }"><span>Q</span></div>
			<div class="angle-vector key-vector" :style="{ transform: `rotate(${keyAngle}rad)` }"><span>K</span></div>
		</div>
		<div class="infra-results">
			<div class="infra-result"><span>相对位移 K − Q</span><strong>{{ distance }}</strong></div>
			<div class="infra-result"><span>注入方式</span><strong>{{ methodSummary }}</strong></div>
			<div v-if="method === 'rope'" class="infra-result"><span>单位向量旋转后点积</span><strong>cos(Δθ) = {{ relativeDot.toFixed(3) }}</strong></div>
			<div v-if="method === 'sinusoidal'" class="infra-result"><span>当前位置二维坐标</span><strong>[{{ Math.sin(keyAngle).toFixed(2) }}, {{ Math.cos(keyAngle).toFixed(2) }}]</strong></div>
			<div v-if="method === 'relative'" class="infra-result"><span>偏置索引</span><strong>b({{ distance }})</strong></div>
			<div v-if="method === 'alibi'" class="infra-result"><span>Score 偏置</span><strong>{{ alibiBias === null ? '未来 Key：先被因果 mask 屏蔽' : alibiBias.toFixed(2) }}</strong></div>
		</div>
		<p class="infra-note">二维频率只用于展示相位；真实正弦位置和 RoPE 会并行使用多组频率。位置可计算不等于超出训练长度后仍能正确利用。</p>
	</div>
</template>

<style scoped>
.angle-stage { position: relative; width: 150px; height: 150px; margin: 1rem auto; border: 1px solid var(--vp-c-divider); border-radius: 50%; background: radial-gradient(circle, var(--vp-c-bg) 0 8%, transparent 9%), linear-gradient(var(--vp-c-divider), var(--vp-c-divider)) center/1px 100% no-repeat, linear-gradient(90deg, var(--vp-c-divider), var(--vp-c-divider)) center/100% 1px no-repeat; }
.angle-vector { position: absolute; left: 50%; top: calc(50% - 2px); width: 62px; height: 4px; transform-origin: left center; border-radius: 999px; }
.angle-vector::after { content: ''; position: absolute; right: -2px; top: -4px; border-left: 8px solid currentColor; border-top: 6px solid transparent; border-bottom: 6px solid transparent; }
.angle-vector span { position: absolute; right: 4px; bottom: 7px; font-size: .72rem; font-weight: 700; }
.query-vector { background: var(--vp-c-brand-1); color: var(--vp-c-brand-1); }
.key-vector { background: var(--vp-c-green-1); color: var(--vp-c-green-1); }
@media (max-width: 640px) {
	.angle-stage { width: 120px; height: 120px; }
	.angle-vector { width: 48px; }
}
@media (max-width: 340px) {
	.angle-stage { width: 104px; height: 104px; }
	.angle-vector { width: 42px; }
}
</style>
