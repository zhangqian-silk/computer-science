import DefaultTheme from 'vitepress/theme'
import type { Theme } from 'vitepress'
import 'katex/dist/katex.min.css'
import 'markdown-it-texmath/css/texmath.css'
import './custom.css'
import ANNIndexStrategyExplorer from './components/ANNIndexStrategyExplorer.vue'
import AttentionMathFlow from './components/AttentionMathFlow.vue'
import AttentionRoadmap from './components/AttentionRoadmap.vue'
import AttentionScoreMatrixExplorer from './components/AttentionScoreMatrixExplorer.vue'
import AttentionTradeoffExplorer from './components/AttentionTradeoffExplorer.vue'
import ContrastiveAlignmentMatrixExplorer from './components/ContrastiveAlignmentMatrixExplorer.vue'
import EmbeddingEvolutionExplorer from './components/EmbeddingEvolutionExplorer.vue'
import EmbeddingMapExplorer from './components/EmbeddingMapExplorer.vue'
import LSTMandGRUComparisonExplorer from './components/LSTMandGRUComparisonExplorer.vue'
import LSTMStructureExplorer from './components/LSTMStructureExplorer.vue'
import MaskPositionExplorer from './components/MaskPositionExplorer.vue'
import MultiHeadAttentionExplorer from './components/MultiHeadAttentionExplorer.vue'
import MultimodalEmbeddingExplorer from './components/MultimodalEmbeddingExplorer.vue'
import NGramWorkflowExplorer from './components/NGramWorkflowExplorer.vue'
import NPLMStructureExplorer from './components/NPLMStructureExplorer.vue'
import RNNStructureExplorer from './components/RNNStructureExplorer.vue'
import Seq2SeqWorkflowExplorer from './components/Seq2SeqWorkflowExplorer.vue'
import TransformerBlockExplorer from './components/TransformerBlockExplorer.vue'
import VectorRetrievalPipelineExplorer from './components/VectorRetrievalPipelineExplorer.vue'
import Word2VecTrainingExplorer from './components/Word2VecTrainingExplorer.vue'

export default {
	...DefaultTheme,
	enhanceApp({ app }) {
		DefaultTheme.enhanceApp?.({ app } as Parameters<NonNullable<Theme['enhanceApp']>>[0])
		app.component('ANNIndexStrategyExplorer', ANNIndexStrategyExplorer)
		app.component('AttentionMathFlow', AttentionMathFlow)
		app.component('AttentionRoadmap', AttentionRoadmap)
		app.component('AttentionScoreMatrixExplorer', AttentionScoreMatrixExplorer)
		app.component('AttentionTradeoffExplorer', AttentionTradeoffExplorer)
		app.component('ContrastiveAlignmentMatrixExplorer', ContrastiveAlignmentMatrixExplorer)
		app.component('EmbeddingEvolutionExplorer', EmbeddingEvolutionExplorer)
		app.component('EmbeddingMapExplorer', EmbeddingMapExplorer)
		app.component('LSTMandGRUComparisonExplorer', LSTMandGRUComparisonExplorer)
		app.component('LSTMStructureExplorer', LSTMStructureExplorer)
		app.component('MaskPositionExplorer', MaskPositionExplorer)
		app.component('MultiHeadAttentionExplorer', MultiHeadAttentionExplorer)
		app.component('MultimodalEmbeddingExplorer', MultimodalEmbeddingExplorer)
		app.component('NGramWorkflowExplorer', NGramWorkflowExplorer)
		app.component('NPLMStructureExplorer', NPLMStructureExplorer)
		app.component('RNNStructureExplorer', RNNStructureExplorer)
		app.component('Seq2SeqWorkflowExplorer', Seq2SeqWorkflowExplorer)
		app.component('TransformerBlockExplorer', TransformerBlockExplorer)
		app.component('VectorRetrievalPipelineExplorer', VectorRetrievalPipelineExplorer)
		app.component('Word2VecTrainingExplorer', Word2VecTrainingExplorer)
	}
} satisfies Theme
