import DefaultTheme from 'vitepress/theme'
import type { Theme } from 'vitepress'
import 'katex/dist/katex.min.css'
import 'markdown-it-texmath/css/texmath.css'
import './custom.css'
import AttentionMathFlow from './components/AttentionMathFlow.vue'
import AttentionRoadmap from './components/AttentionRoadmap.vue'
import AttentionScoreMatrixExplorer from './components/AttentionScoreMatrixExplorer.vue'
import AttentionTradeoffExplorer from './components/AttentionTradeoffExplorer.vue'
import LSTMandGRUComparisonExplorer from './components/LSTMandGRUComparisonExplorer.vue'
import LSTMStructureExplorer from './components/LSTMStructureExplorer.vue'
import MaskPositionExplorer from './components/MaskPositionExplorer.vue'
import MultiHeadAttentionExplorer from './components/MultiHeadAttentionExplorer.vue'
import NGramWorkflowExplorer from './components/NGramWorkflowExplorer.vue'
import NPLMStructureExplorer from './components/NPLMStructureExplorer.vue'
import RNNStructureExplorer from './components/RNNStructureExplorer.vue'
import Seq2SeqWorkflowExplorer from './components/Seq2SeqWorkflowExplorer.vue'
import TransformerBlockExplorer from './components/TransformerBlockExplorer.vue'

export default {
	...DefaultTheme,
	enhanceApp({ app }) {
		DefaultTheme.enhanceApp?.({ app } as Parameters<NonNullable<Theme['enhanceApp']>>[0])
		app.component('AttentionMathFlow', AttentionMathFlow)
		app.component('AttentionRoadmap', AttentionRoadmap)
		app.component('AttentionScoreMatrixExplorer', AttentionScoreMatrixExplorer)
		app.component('AttentionTradeoffExplorer', AttentionTradeoffExplorer)
		app.component('LSTMandGRUComparisonExplorer', LSTMandGRUComparisonExplorer)
		app.component('LSTMStructureExplorer', LSTMStructureExplorer)
		app.component('MaskPositionExplorer', MaskPositionExplorer)
		app.component('MultiHeadAttentionExplorer', MultiHeadAttentionExplorer)
		app.component('NGramWorkflowExplorer', NGramWorkflowExplorer)
		app.component('NPLMStructureExplorer', NPLMStructureExplorer)
		app.component('RNNStructureExplorer', RNNStructureExplorer)
		app.component('Seq2SeqWorkflowExplorer', Seq2SeqWorkflowExplorer)
		app.component('TransformerBlockExplorer', TransformerBlockExplorer)
	}
} satisfies Theme
