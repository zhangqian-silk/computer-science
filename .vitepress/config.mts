import { existsSync, readdirSync, readFileSync } from 'node:fs'
import path from 'node:path'
import { defineConfig, type DefaultTheme } from 'vitepress'
import texmath from 'markdown-it-texmath'
import katex from 'katex'
import { withMermaid } from 'vitepress-plugin-mermaid'
import { installAgentArchive } from './agent-archive.mjs'
import { csThemeInlineScript } from './theme/theme-registry'

const repoName = 'computer-science'

const sectionMeta = [
	{ key: 'ai', text: 'AI', link: '/ai/' },
	{ key: 'fundamentals', text: '基础', link: '/fundamentals/README' },
	{ key: 'language', text: '语言', link: '/language/golang/README' },
	{ key: 'middleware', text: '中间件', link: '/middleware/redis/redis' },
	{ key: 'distributed', text: '分布式', link: '/distributed/theory' },
	{ key: 'system_design', text: '系统设计', link: '/system_design/README' },
	{ key: 'design_patterns', text: '设计模式', link: '/design_patterns/CreationalPatterns' }
] as const

function shouldIgnoreFile(relativeDir: string, fileName: string) {
	if (fileName === 'AGENTS.md') {
		return true
	}

	return relativeDir === '' && (fileName === 'README.md' || fileName === 'CONTRIBUTING.md')
}

function sortEntries(left: string, right: string) {
	const leftIsDir = existsSync(path.join(process.cwd(), left)) && !path.extname(left)
	const rightIsDir = existsSync(path.join(process.cwd(), right)) && !path.extname(right)

	if (leftIsDir !== rightIsDir) {
		return leftIsDir ? -1 : 1
	}

	return left.localeCompare(right, 'zh-CN')
}

function formatText(name: string) {
	return name
		.replace(/\.md$/i, '')
		.replace(/[-_]/g, ' ')
}

function normalizeLink(relativePath: string) {
	const normalized = relativePath.split(path.sep).join('/')
	const withoutExtension = normalized.replace(/\.md$/i, '')

	if (withoutExtension === 'index') {
		return '/'
	}

	if (withoutExtension.endsWith('/index')) {
		return `/${withoutExtension.replace(/\/(README|index)$/i, '')}/`
	}

	return `/${withoutExtension}`
}

function buildItems(relativeDir: string, depth = 0): DefaultTheme.SidebarItem[] {
	const absoluteDir = path.join(process.cwd(), relativeDir)
	const entries = readdirSync(absoluteDir, { withFileTypes: true })
		.filter((entry) => !entry.name.startsWith('.'))
		.filter((entry) => {
			if (entry.isDirectory()) {
				return true
			}

			return entry.name.endsWith('.md') && !shouldIgnoreFile(relativeDir, entry.name)
		})
		.sort((left, right) => sortEntries(path.join(relativeDir, left.name), path.join(relativeDir, right.name)))

	const items: DefaultTheme.SidebarItem[] = []

	for (const entry of entries) {
		const nextRelativePath = path.join(relativeDir, entry.name)

		if (entry.isDirectory()) {
			const children = buildItems(nextRelativePath, depth + 1)

			if (children.length > 0) {
				items.push({
					text: formatText(entry.name),
					collapsed: depth >= 1,
					items: children
				})
			}

			continue
		}

		items.push({
			text: formatText(entry.name),
			link: normalizeLink(nextRelativePath)
		})
	}

	return items
}

function installBracketMathBlock(md: {
	core: {
		ruler: {
			before: (
				beforeName: string,
				ruleName: string,
				rule: (state: { src: string }) => void
			) => void
		}
	}
}) {
	md.core.ruler.before('normalize', 'normalize_bracket_math_block', (state) => {
		const lines = state.src.split('\n')
		const normalized: string[] = []
		let inFence = false

		for (let index = 0; index < lines.length; index += 1) {
			const line = lines[index]
			const trimmed = line.trim()

			if (/^(```|~~~)/.test(trimmed)) {
				inFence = !inFence
				normalized.push(line)
				continue
			}

			if (!inFence && trimmed === '\\[') {
				const formulaLines: string[] = []
				let cursor = index + 1

				while (cursor < lines.length && lines[cursor].trim() !== '\\]') {
					formulaLines.push(lines[cursor])
					cursor += 1
				}

				if (cursor < lines.length) {
					const formula = formulaLines.join(' ').trim()
					let previousIndex = normalized.length - 1

					while (previousIndex >= 0 && normalized[previousIndex].trim() === '') {
						previousIndex -= 1
					}

					const previousLine = previousIndex >= 0 ? normalized[previousIndex] : ''
					const isListContext = /^\s*(?:[-+*]|\d+\.)\s+/.test(previousLine)

					if (isListContext) {
						normalized[previousIndex] = `${previousLine} $${formula}$`
					} else {
						normalized.push('$$')
						normalized.push(...formulaLines)
						normalized.push('$$')
					}

					index = cursor
					continue
				}
			}

			normalized.push(line)
		}

		state.src = normalized.join('\n')
	})
}

const aiSidebar: DefaultTheme.SidebarItem[] = [
	{
		text: 'AI 总览',
		link: '/ai/'
	},
	{
		text: '基础',
		link: '/ai/foundations/',
		items: [
			{ text: '数学与符号约定', link: '/ai/foundations/math-notation' },
			{ text: '学习问题与分类', link: '/ai/foundations/learning-problem-map' },
			{ text: '神经网络训练基础', link: '/ai/foundations/neural-network-basics' },
			{ text: 'Tensor、形状与内存', link: '/ai/foundations/tensor-and-memory' },
			{ text: '数值计算', link: '/ai/foundations/numerical-computing' },
			{ text: 'NLP 方法演化', link: '/ai/foundations/nlp-history' }
		]
	},
	{
		text: '表示',
		link: '/ai/representation/',
		collapsed: false,
		items: [
			{ text: 'Tokenization', link: '/ai/representation/tokenization' },
			{ text: 'Embedding', link: '/ai/representation/embedding' },
			{ text: 'word2vec', link: '/ai/representation/word2vec' },
			{ text: '矩阵分解视角', link: '/ai/representation/embedding-matrix-factorization' },
			{ text: '文本嵌入', link: '/ai/representation/text-embedding' },
			{ text: '多模态对齐', link: '/ai/representation/multimodal-alignment' },
			{ text: '向量检索', link: '/ai/representation/vector-retrieval' }
		]
	},
	{
		text: '机制',
		link: '/ai/mechanism/',
		collapsed: false,
		items: [
			{ text: 'Attention', link: '/ai/mechanism/attention' },
			{ text: 'Self-Attention', link: '/ai/mechanism/self-attention' },
			{ text: 'MHA、MQA、GQA 与 MLA', link: '/ai/mechanism/attention-head-sharing' },
			{ text: '位置表示', link: '/ai/mechanism/positional-encoding' },
			{ text: 'RoPE', link: '/ai/mechanism/rope' },
			{ text: '长上下文位置扩展', link: '/ai/mechanism/long-context' },
			{ text: '稀疏注意力', link: '/ai/mechanism/sparse-attention' },
			{ text: '归一化、残差与 Dropout', link: '/ai/mechanism/normalization-and-residual' },
			{ text: '前馈网络与门控激活', link: '/ai/mechanism/feed-forward-and-gating' },
			{ text: 'MoE', link: '/ai/mechanism/moe' },
			{ text: 'LoRA', link: '/ai/mechanism/lora' }
		]
	},
	{
		text: '模型',
		link: '/ai/model/',
		collapsed: false,
		items: [
			{ text: 'N-gram', link: '/ai/model/n-gram' },
			{ text: 'NPLM', link: '/ai/model/nplm' },
			{ text: 'RNN', link: '/ai/model/rnn' },
			{ text: 'LSTM', link: '/ai/model/lstm' },
			{ text: 'Seq2Seq', link: '/ai/model/seq2seq' },
			{ text: 'Transformer', link: '/ai/model/transformer' },
			{ text: 'BERT', link: '/ai/model/bert' },
			{ text: 'GPT', link: '/ai/model/gpt' },
			{ text: '状态空间模型', link: '/ai/model/state-space-model' }
		]
	},
	{
		text: 'AI Infra',
		link: '/ai/infra/',
		collapsed: false,
		items: [
			{ text: '岗位能力路线', link: '/ai/infra/job-readiness' },
			{
				text: '加速器与性能',
				link: '/ai/infra/accelerator/',
				collapsed: true,
				items: [
					{ text: 'GPU 执行与存储层级', link: '/ai/infra/accelerator/gpu-architecture' },
					{ text: '性能模型', link: '/ai/infra/accelerator/performance-model' },
					{ text: '精度与 Tensor Core', link: '/ai/infra/accelerator/precision-and-tensor-cores' },
					{ text: 'CUDA、Triton 与 Kernel', link: '/ai/infra/accelerator/cuda-and-kernels' },
					{ text: '编译图与执行计划', link: '/ai/infra/accelerator/compiler-and-graphs' },
					{ text: 'Profiling', link: '/ai/infra/accelerator/profiling' }
				]
			},
			{
				text: '分布式通信',
				link: '/ai/infra/distributed/',
				collapsed: true,
				items: [
					{ text: 'Collective', link: '/ai/infra/distributed/collectives' },
					{ text: '拓扑与网络', link: '/ai/infra/distributed/topology-and-network' },
					{ text: '重叠、启动与故障', link: '/ai/infra/distributed/overlap-launch-and-failures' }
				]
			},
			{
				text: '训练 Runtime',
				link: '/ai/infra/training/',
				collapsed: true,
				items: [
					{ text: '训练循环与内存', link: '/ai/infra/training/memory-and-loop' },
					{ text: '混合精度', link: '/ai/infra/training/mixed-precision' },
					{ text: '数据管线', link: '/ai/infra/training/data-pipeline' },
					{ text: '并行策略', link: '/ai/infra/training/parallelism' },
					{ text: 'Checkpoint 与恢复', link: '/ai/infra/training/checkpoint-and-recovery' },
					{ text: '训练框架', link: '/ai/infra/training/frameworks' },
					{ text: '后训练 Runtime', link: '/ai/infra/training/post-training-runtime' },
					{ text: 'Scaling 与资源预算', link: '/ai/infra/training/scaling-and-compute' },
					{ text: '训练性能', link: '/ai/infra/training/performance' }
				]
			},
			{
				text: '推理 Runtime',
				link: '/ai/infra/inference/',
				collapsed: true,
				items: [
					{ text: '请求生命周期', link: '/ai/infra/inference/request-lifecycle' },
					{ text: 'Prefill 与 Decode', link: '/ai/infra/inference/prefill-decode' },
					{ text: 'KV Cache 管理', link: '/ai/infra/inference/kv-cache-management' },
					{ text: 'Batch 与调度', link: '/ai/infra/inference/batching-scheduling' },
					{ text: 'Attention Kernel', link: '/ai/infra/inference/attention-kernels' },
					{ text: '量化', link: '/ai/infra/inference/quantization' },
					{ text: '解码与推测执行', link: '/ai/infra/inference/decoding-and-speculation' },
					{ text: '分布式推理', link: '/ai/infra/inference/distributed-inference' },
					{ text: '前缀缓存与 PD 分离', link: '/ai/infra/inference/cache-and-disaggregation' },
					{ text: '模型服务特性', link: '/ai/infra/inference/model-serving-features' },
					{ text: 'API、可观测性与可靠性', link: '/ai/infra/inference/api-observability-reliability' },
					{ text: '基准测试与容量', link: '/ai/infra/inference/benchmarking-capacity' },
					{ text: '推理系统排障', link: '/ai/infra/inference/troubleshooting' },
					{ text: '框架案例', link: '/ai/infra/inference/frameworks' }
				]
			},
			{
				text: '服务平台',
				link: '/ai/infra/platform/',
				collapsed: true,
				items: [
					{ text: 'GPU 容器与 Kubernetes', link: '/ai/infra/platform/gpu-containers-kubernetes' },
					{ text: '部署与路由', link: '/ai/infra/platform/deployment-and-routing' },
					{ text: 'GPU Fleet 与容量', link: '/ai/infra/platform/fleet-and-capacity' },
					{ text: '安全与多租户', link: '/ai/infra/platform/security-and-multitenancy' }
				]
			},
			{
				text: '实验与项目',
				link: '/ai/infra/labs/',
				collapsed: true,
				items: [
					{ text: '模型与 Runtime 实验', link: '/ai/infra/labs/model-runtime-labs' },
					{ text: '加速器与分布式实验', link: '/ai/infra/labs/accelerator-distributed-labs' },
					{ text: '框架与服务实验', link: '/ai/infra/labs/framework-serving-labs' },
					{ text: '综合项目', link: '/ai/infra/labs/projects' }
				]
			}
		]
	},
	{
		text: '评估',
		link: '/ai/evaluation/',
		items: [
			{ text: '语言模型评估', link: '/ai/evaluation/language-model-evaluation' },
			{ text: '向量表示分析', link: '/ai/evaluation/embedding-geometry' },
			{ text: '检索评估', link: '/ai/evaluation/retrieval-evaluation' },
			{ text: '生成评估', link: '/ai/evaluation/generation-evaluation' },
			{ text: '长上下文评估', link: '/ai/evaluation/long-context-evaluation' }
		]
	},
	{
		text: 'Agent 系统',
		link: '/ai/agent/',
		collapsed: false,
		items: [
			{ text: '学习总览', link: '/ai/agent/notes/' },
			{ text: '模型交互', link: '/ai/agent/notes/model/llm-api' },
			{
				text: '上下文管理与优化',
				collapsed: true,
				items: [
					{ text: 'Prompt and Context', link: '/ai/agent/notes/context/context-engineering' },
					{ text: 'RAG 与代码检索', link: '/ai/agent/notes/context/rag-retrieval' },
					{ text: '长期记忆', link: '/ai/agent/notes/context/conversation-memory' },
					{ text: '上下文压缩', link: '/ai/agent/notes/context/compaction' }
				]
			},
			{
				text: '工具系统与能力扩展',
				collapsed: true,
				items: [
					{ text: '工具契约、执行与结果', link: '/ai/agent/notes/tools/tool-calling' },
					{ text: '文件、进程、浏览器与服务', link: '/ai/agent/notes/tools/execution-environments' },
					{ text: 'MCP 与外部连接器', link: '/ai/agent/notes/tools/mcp-connectors' },
					{ text: 'Skills 与操作知识加载', link: '/ai/agent/notes/tools/skills' },
					{ text: 'Hooks、Plugins 与扩展管理', link: '/ai/agent/notes/tools/hooks-plugins' }
				]
			},
			{
				text: '流程控制与 Agent 方法',
				collapsed: true,
				items: [
					{ text: 'Workflow 与 Agent Loop', link: '/ai/agent/notes/control/workflows' },
					{ text: '规划与计划修订', link: '/ai/agent/notes/control/reasoning-planning' },
					{ text: 'Subagent 与多 Agent 协作', link: '/ai/agent/notes/control/subagents-multi-agent' },
					{ text: 'Dynamic Workflows 与代码编排', link: '/ai/agent/notes/control/dynamic-workflows' }
				]
			},
			{
				text: 'Agent 运行时与工程化',
				collapsed: true,
				items: [
					{ text: '运行时组成与任务生命周期', link: '/ai/agent/notes/runtime/harness-architecture' },
					{ text: '人工交互与执行控制', link: '/ai/agent/notes/runtime/human-control' },
					{ text: '任务恢复与副作用', link: '/ai/agent/notes/runtime/persistence-recovery' },
					{ text: '任务调度与 Agent 接入', link: '/ai/agent/notes/runtime/scheduling-protocols' },
					{ text: '安全与隔离', link: '/ai/agent/notes/runtime/security' }
				]
			},
			{
				text: '评测、可观测性与优化',
				collapsed: true,
				items: [
					{ text: '评测与实验设计', link: '/ai/agent/notes/quality/evaluation' },
					{ text: '可观测性与故障诊断', link: '/ai/agent/notes/quality/observability' },
					{ text: '性能与成本', link: '/ai/agent/notes/quality/performance-cost' },
					{ text: '版本、回归与发布', link: '/ai/agent/notes/quality/versioning-release' }
				]
			},
			{
				text: '资料与实验',
				collapsed: true,
				items: [
					{ text: '实现案例阅读索引', link: '/ai/agent/notes/implementation-index' },
					{ text: '学习练习与实验', link: '/ai/agent/notes/labs' },
					{ text: '参考资料', link: '/ai/agent/notes/references' },
					{ text: '旧版归档', link: '/ai/agent/archive/catalog' }
				]
			}
		]
	}
]

const sidebar = Object.fromEntries(
	sectionMeta.map(({ key }) => [
		`/${key}/`,
		key === 'ai' ? aiSidebar : buildItems(key)
	])
)

sidebar['/ai/agent/archive/'] = [
	{ text: '返回新版学习总览', link: '/ai/agent/notes/' },
	{ text: '旧版归档目录', link: '/ai/agent/archive/catalog' },
	{ text: '旧版 Agent 入口', link: '/ai/agent/archive/' },
	{ text: '旧版学习总览', link: '/ai/agent/archive/notes/' },
	...['model', 'context', 'tools', 'control', 'runtime'].map((section) => ({
		text: ({ model: '模型交互', context: '上下文与记忆', tools: '工具与扩展', control: '流程与协作', runtime: '运行系统与质量' })[section]!,
		collapsed: true,
		items: buildItems(`ai/agent/archive/notes/${section}`).map((item) => ({
			...item,
			text: readFileSync(path.join(process.cwd(), `${item.link!.slice(1)}.md`), 'utf8')
				.match(/^# (.+)$/m)?.[1] ?? item.text
		}))
	})),
	{ text: '旧版实现索引', link: '/ai/agent/archive/notes/implementation-index' },
	{ text: '旧版实验说明', link: '/ai/agent/archive/notes/labs' },
	{ text: '旧版参考资料', link: '/ai/agent/archive/notes/references' }
]

export default withMermaid(
	defineConfig({
		lang: 'zh-CN',
		title: 'Computer Science',
		description: '计算机科学知识库',
		base: `/${repoName}/`,
		lastUpdated: true,
		cleanUrls: true,
		// 首屏同步写入 data-cs-theme，避免默认主题先绘制一帧再切换造成闪烁
		head: [['script', {}, csThemeInlineScript]],
		srcExclude: [
			'**/AGENTS.md',
			// 面向贡献者的工程文档，与 CONTRIBUTING.md 同类，不作为站点页面构建
			'docs/**',
			// Agent 笔记的参考代码与模板按源码方式保留（可在 GitHub 浏览），不作为站点页面构建
			'**/ai/agent/notes/examples/**',
			'**/ai/agent/notes/templates/**',
			'**/ai/agent/archive/notes/examples/**',
			'**/ai/agent/archive/notes/templates/**'
		],
		ignoreDeadLinks: [
			/^https?:\/\//
		],
		markdown: {
			languages: ['proto'],
			languageAlias: {
				thrift: 'proto'
			},
			lineNumbers: true,
			config(md) {
				installBracketMathBlock(md)
				installAgentArchive(md)

				md.use(texmath, {
					engine: katex,
					delimiters: ['brackets', 'dollars', 'beg_end'],
					katexOptions: {
						output: 'html',
						throwOnError: false,
						strict: 'ignore'
					}
				})
			}
		},
		vue: {
			template: {
				compilerOptions: {
					isCustomElement: (tag) => tag === 'eq' || tag === 'eqn'
				}
			}
		},
		themeConfig: {
			nav: sectionMeta.map(({ text, link }) => ({ text, link })),
			sidebar,
			search: {
				provider: 'local'
			},
			outline: {
				level: [2, 3],
				label: '目录'
			},
			docFooter: {
				prev: '上一页',
				next: '下一页'
			},
			returnToTopLabel: '回到顶部',
			sidebarMenuLabel: '导航',
			darkModeSwitchLabel: '主题',
			lightModeSwitchTitle: '切换到浅色模式',
			darkModeSwitchTitle: '切换到深色模式'
		},
		mermaid: {
			flowchart: {
				useMaxWidth: true,
				htmlLabels: true,
				curve: 'linear'
			},
			sequence: {
				useMaxWidth: true
			}
		}
	})
)
