import katex from 'katex'

export function renderMath(tex: string, displayMode = false) {
	return katex.renderToString(tex, {
		displayMode,
		throwOnError: false,
		strict: 'ignore',
		output: 'html'
	})
}
