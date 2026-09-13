<script setup lang="ts">
/**
 * 主题切换器：挂在导航栏，一键切换 data-cs-theme。
 *
 * 与 VitePress 自带的浅色 / 深色开关正交——两者可任意组合。
 * SSR 期间不访问 document，首屏取值由 config.mts 注入的内联脚本完成。
 */
import { onMounted, onBeforeUnmount, ref, useId } from 'vue'
import {
	CS_THEME_ATTRIBUTE,
	CS_THEME_STORAGE_KEY,
	csThemes,
	defaultCsTheme,
	isCsThemeId,
	type CsThemeId
} from '../theme-registry'
const current = ref<CsThemeId>(defaultCsTheme)
const open = ref(false)
const id = useId()
const root = ref<HTMLElement | null>(null)

function apply(theme: CsThemeId) {
	current.value = theme
	document.documentElement.setAttribute(CS_THEME_ATTRIBUTE, theme)

	try {
		localStorage.setItem(CS_THEME_STORAGE_KEY, theme)
	} catch {
		// 隐私模式下 localStorage 不可写；本次会话内仍生效，只是不持久化
	}
}

function select(theme: CsThemeId) {
	apply(theme)
	open.value = false
}

function onPointerDown(event: MouseEvent) {
	if (root.value && !root.value.contains(event.target as Node)) {
		open.value = false
	}
}

function onKeydown(event: KeyboardEvent) {
	if (event.key === 'Escape') {
		open.value = false
	}
}

onMounted(() => {
	// 内联脚本已写入属性，这里只读取，避免二次赋值造成闪烁
	const fromDom = document.documentElement.getAttribute(CS_THEME_ATTRIBUTE)

	if (isCsThemeId(fromDom)) {
		current.value = fromDom
	} else {
		apply(defaultCsTheme)
	}

	document.addEventListener('pointerdown', onPointerDown)
	document.addEventListener('keydown', onKeydown)
})

onBeforeUnmount(() => {
	document.removeEventListener('pointerdown', onPointerDown)
	document.removeEventListener('keydown', onKeydown)
})
</script>

<template>
	<div ref="root" class="cs-theme-switch">
		<button
			type="button"
			class="cs-theme-switch__trigger"
			:aria-expanded="open"
			:aria-controls="`${id}-menu`"
			aria-haspopup="listbox"
			title="切换配色主题"
			@click="open = !open"
		>
			<span class="cs-theme-switch__dot" aria-hidden="true" />
			<span class="cs-theme-switch__label">配色</span>
		</button>
		<ul v-show="open" :id="`${id}-menu`" class="cs-theme-switch__menu" role="listbox" aria-label="配色主题">
			<li v-for="theme in csThemes" :key="theme.id" role="none">
				<button
					type="button"
					role="option"
					:aria-selected="current === theme.id"
					class="cs-theme-switch__option"
					@click="select(theme.id)"
				>
					<span class="cs-theme-switch__swatch" :style="{ background: theme.swatch }" aria-hidden="true" />
					<span class="cs-theme-switch__text">
						<strong>{{ theme.label }}</strong>
						<small>{{ theme.description }}</small>
					</span>
					<span v-if="current === theme.id" class="cs-theme-switch__check" aria-hidden="true">✓</span>
				</button>
			</li>
		</ul>
	</div>
</template>

<style scoped>
.cs-theme-switch {
	position: relative;
	display: flex;
	align-items: center;
}

.cs-theme-switch__trigger {
	display: flex;
	align-items: center;
	gap: var(--cs-space-2);
	min-height: 36px;
	padding: 0 var(--cs-space-3);
	border: var(--cs-border);
	border-radius: var(--cs-radius-pill);
	background: var(--cs-color-bg);
	color: var(--cs-color-text-muted);
	font-size: var(--cs-text-xs);
	cursor: pointer;
	transition: var(--cs-transition-colors);
}

.cs-theme-switch__trigger:hover {
	border-color: var(--cs-color-brand);
	color: var(--cs-color-brand);
}

.cs-theme-switch__trigger:focus-visible {
	outline: var(--cs-focus-ring-width) solid color-mix(in srgb, var(--cs-color-brand) 32%, transparent);
	outline-offset: var(--cs-focus-ring-offset);
}

.cs-theme-switch__dot {
	width: 10px;
	height: 10px;
	border-radius: var(--cs-radius-circle);
	background: var(--cs-color-brand);
}

.cs-theme-switch__menu {
	position: absolute;
	top: calc(100% + var(--cs-space-2));
	right: 0;
	z-index: var(--cs-z-popover);
	min-width: 230px;
	margin: 0;
	padding: var(--cs-space-1);
	list-style: none;
	border: var(--cs-border);
	border-radius: var(--cs-radius-md);
	background: var(--cs-color-bg-elevated);
	box-shadow: var(--cs-shadow-md);
}

.cs-theme-switch__option {
	display: flex;
	align-items: center;
	gap: var(--cs-space-3);
	width: 100%;
	padding: var(--cs-space-3);
	border: 0;
	border-radius: var(--cs-radius-sm);
	background: transparent;
	color: var(--cs-color-text);
	text-align: left;
	cursor: pointer;
	transition: var(--cs-transition-colors);
}

.cs-theme-switch__option:hover {
	background: var(--cs-color-bg-soft);
}

.cs-theme-switch__option[aria-selected='true'] {
	background: var(--cs-color-brand-soft);
}

.cs-theme-switch__option:focus-visible {
	outline: var(--cs-focus-ring-width) solid color-mix(in srgb, var(--cs-color-brand) 32%, transparent);
	outline-offset: calc(var(--cs-focus-ring-offset) * -1);
}

.cs-theme-switch__swatch {
	flex: 0 0 auto;
	width: 14px;
	height: 14px;
	border-radius: var(--cs-radius-circle);
}

.cs-theme-switch__text {
	display: flex;
	flex-direction: column;
	gap: 1px;
	min-width: 0;
}

.cs-theme-switch__text strong {
	font-size: var(--cs-text-sm);
	font-weight: var(--cs-font-weight-semibold);
}

.cs-theme-switch__text small {
	color: var(--cs-color-text-muted);
	font-size: var(--cs-text-3xs);
	line-height: var(--cs-leading-tight);
}

.cs-theme-switch__check {
	margin-left: auto;
	color: var(--cs-color-brand);
	font-size: var(--cs-text-xs);
}

/* 窄屏隐藏文字，只留色点，避免挤压导航栏 */
@media (max-width: 767px) {
	.cs-theme-switch__label {
		display: none;
	}
}
</style>
