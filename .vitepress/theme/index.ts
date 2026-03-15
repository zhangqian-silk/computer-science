import DefaultTheme from 'vitepress/theme'
import type { Theme } from 'vitepress'
import 'katex/dist/katex.min.css'
import 'markdown-it-texmath/css/texmath.css'
import './custom.css'
import AttentionMathFlow from './components/AttentionMathFlow.vue'
import AttentionRoadmap from './components/AttentionRoadmap.vue'
import AttentionScoreMatrixExplorer from './components/AttentionScoreMatrixExplorer.vue'
import AttentionTradeoffExplorer from './components/AttentionTradeoffExplorer.vue'
import MaskPositionExplorer from './components/MaskPositionExplorer.vue'
import MultiHeadAttentionExplorer from './components/MultiHeadAttentionExplorer.vue'
import TransformerBlockExplorer from './components/TransformerBlockExplorer.vue'

export default {
	...DefaultTheme,
	enhanceApp({ app }) {
		DefaultTheme.enhanceApp?.({ app } as Parameters<NonNullable<Theme['enhanceApp']>>[0])
		app.component('AttentionMathFlow', AttentionMathFlow)
		app.component('AttentionRoadmap', AttentionRoadmap)
		app.component('AttentionScoreMatrixExplorer', AttentionScoreMatrixExplorer)
		app.component('AttentionTradeoffExplorer', AttentionTradeoffExplorer)
		app.component('MaskPositionExplorer', MaskPositionExplorer)
		app.component('MultiHeadAttentionExplorer', MultiHeadAttentionExplorer)
		app.component('TransformerBlockExplorer', TransformerBlockExplorer)
	}
} satisfies Theme
