<script setup lang="ts">
import { computed, ref, useId } from "vue"
import { labLessons } from "../lab-lessons"

const props = defineProps<{ topic: keyof typeof labLessons }>()
const lesson = computed(() => labLessons[props.topic])
const expanded = ref(false)
const emit = defineEmits<{ reset: [] }>()
const id = useId()
function reset() {
	emit("reset")
	expanded.value = false
}
</script>

<template>
	<section class="infra-lab learning-lab" :aria-labelledby="`${id}-title`">
		<header class="learning-lab__header">
			<div>
				<span class="learning-lab__eyebrow">交互推演 · {{ lesson.kind }}</span>
				<h3 :id="`${id}-title`">{{ lesson.title }}</h3>
			</div>
			<button type="button" class="learning-lab__reset" @click="reset">恢复初始值</button>
		</header>
		<p class="learning-lab__question"><span>先想一想</span>{{ lesson.question }}</p>
		<div class="learning-lab__body"><slot /></div>
		<footer class="learning-lab__footer">
			<button type="button" :aria-expanded="expanded" :aria-controls="`${id}-answer`" @click="expanded = !expanded">
				{{ expanded ? "收起解释" : "查看推理与边界" }} <span aria-hidden="true">{{ expanded ? "−" : "+" }}</span>
			</button>
			<div v-show="expanded" :id="`${id}-answer`" class="learning-lab__answer">
				<p><strong>怎样观察：</strong>{{ lesson.try }}</p>
				<p><strong>为什么：</strong>{{ lesson.answer }}</p>
				<p class="learning-lab__boundary"><strong>本实验不证明：</strong>{{ lesson.limit }}</p>
			</div>
		</footer>
	</section>
</template>
