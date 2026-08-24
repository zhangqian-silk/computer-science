import DefaultTheme from 'vitepress/theme'
import type { Theme } from 'vitepress'
import { defineAsyncComponent, type Component } from 'vue'
import 'katex/dist/katex.min.css'
import 'markdown-it-texmath/css/texmath.css'
import './custom.css'

const componentModules = import.meta.glob<{ default: Component }>('./components/*.vue')

export default {
	...DefaultTheme,
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
