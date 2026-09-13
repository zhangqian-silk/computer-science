import DefaultTheme from 'vitepress/theme'
import type { Theme } from 'vitepress'
import { defineAsyncComponent, h, type Component } from 'vue'
import 'katex/dist/katex.min.css'
import 'markdown-it-texmath/css/texmath.css'
import './styles/index.css'
import ThemeSwitch from './system/ThemeSwitch.vue'

// components/ 下的组件按需异步注册，供 Markdown 直接书写标签使用；
// system/ 下的站点外壳组件不进入这个注册表，避免被重复打包。
const componentModules = import.meta.glob<{ default: Component }>('./components/*.vue')

export default {
	...DefaultTheme,
	// 把配色切换器放进导航栏右侧，与浅色 / 深色开关并列
	Layout: () => h(DefaultTheme.Layout, null, {
		'nav-bar-content-after': () => h(ThemeSwitch)
	}),
	enhanceApp(context) {
		DefaultTheme.enhanceApp?.(context)
		const { app } = context

		for (const [componentPath, loader] of Object.entries(componentModules)) {
			const componentName = componentPath.split('/').at(-1)?.replace(/\.vue$/, '')

			if (componentName) {
				app.component(componentName, defineAsyncComponent(async () => (await loader()).default))
			}
		}
	}
} satisfies Theme
