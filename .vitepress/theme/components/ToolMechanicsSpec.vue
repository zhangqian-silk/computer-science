<script setup lang="ts">
// 纯展示组件：复刻原 HTML 手册中的静态规格图，不包含交互状态。
withDefaults(defineProps<{
	figure?: "pipeline" | "dimensions" | "notify" | "task-state" | "trust" | "evolution" | "approval"
}>(), {
	figure: "pipeline"
})
</script>

<template>
	<div class="atms">










		<figure v-if="figure === 'pipeline'" class="atms-figure atms-figure--plain">
			<figcaption class="atms-figcap"><b>图 1-1</b> 工具调用的九段链路 · 一次 ReAct 闭环</figcaption>
			<div class="atms-flow-scroll">
				<svg class="atms-svg atms-svg--pipeline" viewBox="0 0 760 476" role="img" aria-label="工具调用九段链路">
					<defs>
						<marker id="atms-arrow" markerWidth="9" markerHeight="9" refX="7.5" refY="4.5" orient="auto">
							<path d="M0,0 L9,4.5 L0,9 Z" fill="var(--cs-color-text-subtle)" />
						</marker>
						<marker id="atms-arrow-brand" markerWidth="9" markerHeight="9" refX="7.5" refY="4.5" orient="auto">
							<path d="M0,0 L9,4.5 L0,9 Z" fill="var(--cs-color-brand)" />
						</marker>
						<marker id="atms-arrow-warning" markerWidth="9" markerHeight="9" refX="7.5" refY="4.5" orient="auto">
							<path d="M0,0 L9,4.5 L0,9 Z" fill="var(--cs-color-warning)" />
						</marker>
						<marker id="atms-arrow-success" markerWidth="9" markerHeight="9" refX="7.5" refY="4.5" orient="auto">
							<path d="M0,0 L9,4.5 L0,9 Z" fill="var(--cs-color-success)" />
						</marker>
					</defs>

					<rect x="0" y="20" width="760" height="96" rx="8" fill="var(--cs-color-neutral-soft)" />
					<rect x="0" y="152" width="760" height="96" rx="8" fill="var(--cs-color-brand-soft)" />
					<rect x="0" y="284" width="760" height="172" rx="8" fill="var(--cs-color-warning-soft)" />

					<text x="20" y="66" class="atms-svg-title">编排层</text>
					<text x="20" y="87" class="atms-svg-kicker">ORCHESTRATOR</text>
					<text x="20" y="198" class="atms-svg-title atms-brand">模型层</text>
					<text x="20" y="219" class="atms-svg-kicker atms-brand">LLM</text>
					<text x="20" y="375" class="atms-svg-title atms-warning">执行层</text>
					<text x="20" y="396" class="atms-svg-kicker atms-warning">RUNTIME</text>

					<g font-family="inherit">
						<rect x="120" y="36" width="150" height="64" rx="6" class="atms-box atms-node" />
						<text x="138" y="60" class="atms-svg-kicker atms-brand">01 REGISTRY</text>
						<text x="138" y="84" class="atms-svg-label">注册与发现</text>

						<rect x="320" y="36" width="150" height="64" rx="6" class="atms-box atms-node" />
						<text x="338" y="60" class="atms-svg-kicker atms-brand">02 SELECT</text>
						<text x="338" y="84" class="atms-svg-label">筛选 · 注入定义</text>

						<rect x="520" y="36" width="150" height="64" rx="6" class="atms-box atms-node" />
						<text x="538" y="60" class="atms-svg-kicker atms-brand">CONTEXT</text>
						<text x="538" y="84" class="atms-svg-label">构建本轮输入</text>

						<rect x="320" y="168" width="150" height="64" rx="6" class="atms-box atms-box-brand atms-node atms-node--strong" />
						<text x="338" y="192" class="atms-svg-kicker atms-brand">03 DECIDE</text>
						<text x="338" y="216" class="atms-svg-label">选工具 · 出参数</text>

						<rect x="520" y="168" width="150" height="64" rx="6" class="atms-box atms-box-success atms-node atms-node--strong" />
						<text x="538" y="192" class="atms-svg-kicker atms-success">FINAL</text>
						<text x="538" y="216" class="atms-svg-label">最终答复</text>

						<rect x="120" y="300" width="150" height="64" rx="6" class="atms-box atms-node" />
						<text x="138" y="324" class="atms-svg-kicker atms-warning">04 VALIDATE</text>
						<text x="138" y="348" class="atms-svg-label">解析 · 校验 · 归一</text>

						<rect x="320" y="300" width="150" height="64" rx="6" class="atms-box atms-node" />
						<text x="338" y="324" class="atms-svg-kicker atms-warning">05 GUARD</text>
						<text x="338" y="348" class="atms-svg-label">鉴权 · 限流 · 确认</text>

						<rect x="520" y="300" width="150" height="64" rx="6" class="atms-box atms-box-warning atms-node atms-node--strong" />
						<text x="538" y="324" class="atms-svg-kicker atms-warning">06 EXECUTE</text>
						<text x="538" y="348" class="atms-svg-label">调用 · 超时 · 重试</text>

						<rect x="120" y="388" width="150" height="64" rx="6" class="atms-box atms-node" />
						<text x="138" y="412" class="atms-svg-kicker atms-warning">07 NOTIFY</text>
						<text x="138" y="436" class="atms-svg-label">进度 · 异步结果</text>

						<rect x="320" y="388" width="150" height="64" rx="6" class="atms-box atms-node" />
						<text x="338" y="412" class="atms-svg-kicker atms-warning">08 SHAPE</text>
						<text x="338" y="436" class="atms-svg-label">裁剪 · 落盘 · 引用</text>

						<rect x="520" y="388" width="150" height="64" rx="6" class="atms-box atms-node" />
						<text x="538" y="412" class="atms-svg-kicker atms-warning">09 FEEDBACK</text>
						<text x="538" y="436" class="atms-svg-label">结果回填</text>
					</g>

					<path d="M270 68 H316" stroke="var(--cs-color-text-subtle)" stroke-width="1.6" marker-end="url(#atms-arrow)" />
					<path d="M470 68 H516" stroke="var(--cs-color-text-subtle)" stroke-width="1.6" marker-end="url(#atms-arrow)" />
					<path d="M595 100 V132 H395 V164" stroke="var(--cs-color-brand)" stroke-width="1.7" fill="none" marker-end="url(#atms-arrow-brand)" />
					<path d="M470 212 H516" stroke="var(--cs-color-success)" stroke-width="1.8" marker-end="url(#atms-arrow-success)" />
					<text x="493" y="200" text-anchor="middle" class="atms-svg-kicker atms-success atms-svg-edge">无工具调用</text>
					<path d="M395 232 V264 H595 V296" stroke="var(--cs-color-warning)" stroke-width="1.8" fill="none" marker-end="url(#atms-arrow-warning)" />
					<text x="462" y="256" class="atms-svg-kicker atms-warning">请求调用工具</text>
					<path d="M516 332 H474" stroke="var(--cs-color-text-subtle)" stroke-width="1.6" marker-end="url(#atms-arrow)" />
					<path d="M316 332 H274" stroke="var(--cs-color-text-subtle)" stroke-width="1.6" marker-end="url(#atms-arrow)" />
					<path d="M195 364 V384" stroke="var(--cs-color-text-subtle)" stroke-width="1.6" marker-end="url(#atms-arrow)" />
					<path d="M270 420 H316" stroke="var(--cs-color-text-subtle)" stroke-width="1.6" marker-end="url(#atms-arrow)" />
					<path d="M470 420 H516" stroke="var(--cs-color-text-subtle)" stroke-width="1.6" marker-end="url(#atms-arrow)" />
					<path d="M670 420 H724 V68 H674" stroke="var(--cs-color-brand)" stroke-width="1.7" stroke-dasharray="6 4" fill="none" marker-end="url(#atms-arrow-brand)" />
				</svg>
			</div>
		</figure>


		<figure v-if="figure === 'dimensions'" class="atms-figure">
			<figcaption class="atms-figcap"><b>图 4-1</b> 工具的四个分类维度及其执行策略</figcaption>
			<div class="atms-grid atms-grid-4">
				<section class="atms-card atms-card-read">
					<span class="atms-kicker">DIM 1 · 副作用</span>
					<h4>只读、写入与不可逆操作</h4>
					<p>只读操作通常可以重试和并行执行；写入操作需要幂等标识与审计；删除、转账、外发和发布应经过人工确认。</p>
					<code>readOnlyHint · destructiveHint</code>
				</section>
				<section class="atms-card atms-card-write">
					<span class="atms-kicker">DIM 2 · 幂等性</span>
					<h4>幂等 / 非幂等</h4>
					<p>下单、发送消息和转账等操作不能因超时直接重试；幂等标识的作用域必须与业务上的重复语义一致。</p>
					<code>idempotentHint</code>
				</section>
				<section class="atms-card atms-card-brand">
					<span class="atms-kicker">DIM 3 · 时延</span>
					<h4>短时、长时与持久任务</h4>
					<p>短时操作同步返回；持续数秒至数分钟的操作报告进度；跨连接或跨重启的任务使用持久化句柄。</p>
					<code>sync · progress · task</code>
				</section>
				<section class="atms-card atms-card-neutral">
					<span class="atms-kicker">DIM 4 · 确定性</span>
					<h4>封闭环境与开放环境</h4>
					<p>网页、邮件和工单等外部内容属于不可信数据，应与高危写权限及外发通道隔离。</p>
					<code>openWorldHint</code>
				</section>
			</div>
		</figure>

		<figure v-if="figure === 'notify'" class="atms-figure">
			<figcaption class="atms-figcap"><b>图 5-1</b> 三种结果通知方式：区别在于能否跨越连接的有效期限</figcaption>
			<div class="atms-grid atms-grid-3">
				<section class="atms-mode atms-mode-read">
					<span class="atms-chip atms-chip-read">MODE A</span>
					<h4>同步阻塞</h4>
					<p><b>适用：</b>快速查询和短时操作</p>
					<p><b>有效期限：</b>请求与响应在同一次调用内完成</p>
					<p><b>断线后：</b>不能依据原连接取得结果</p>
				</section>
				<section class="atms-mode atms-mode-brand">
					<span class="atms-chip atms-chip-brand">MODE B</span>
					<h4>连接内进度</h4>
					<p><b>适用：</b>需要持续反馈的中等时长任务</p>
					<p><b>有效期限：</b>进度信息依附于当前请求</p>
					<p><b>断线后：</b>进度通道随之终止</p>
				</section>
				<section class="atms-mode atms-mode-write">
					<span class="atms-chip atms-chip-write">MODE C</span>
					<h4>任务句柄</h4>
					<p><b>适用：</b>分钟级、小时级任务及人工审批</p>
					<p><b>有效期限：</b>任务状态独立持久化</p>
					<p><b>断线后：</b>可凭任务标识继续查询</p>
				</section>
			</div>
		</figure>

		<figure v-if="figure === 'trust'" class="atms-figure">
			<figcaption class="atms-figcap"><b>图 8-1</b> MCP Host–Client–Server 信任边界</figcaption>
			<div class="atms-trust">
				<section class="atms-host">
					<span class="atms-kicker">HOST · AI 应用</span>
					<h4>必须由 Host 保留的职责</h4>
					<div class="atms-grid atms-grid-2 atms-compact">
						<p>模型访问与用户身份</p>
						<p>工具筛选与上下文预算</p>
						<p>独立鉴权与 scope</p>
						<p>人工确认与审计快照</p>
					</div>
				</section>
				<div class="atms-clients">
					<span>Client A</span>
					<span>Client B</span>
					<span>Client C</span>
				</div>
				<section class="atms-server">
					<span class="atms-kicker">SERVER · 外部依赖</span>
					<h4>Tools / Resources / Prompts</h4>
					<p>可作为本地 STDIO 进程或远程 HTTP 服务运行；其描述、注解和返回内容均不能直接构成授权依据。</p>
				</section>
			</div>
		</figure>

		<figure v-if="figure === 'evolution'" class="atms-figure">
			<figcaption class="atms-figcap"><b>图 8-2</b> MCP 的无状态化演进：从连接状态转向请求自描述</figcaption>
			<div class="atms-evolution">
				<section v-for="item in [
					{ before: '初始化握手', after: '请求元数据 + 能力发现', goal: '请求自包含' },
					{ before: '连接级会话标识', after: '显式任务句柄与普通参数', goal: '任意实例可处理' },
					{ before: '服务端反向请求', after: '返回补充输入请求，由客户端重新提交', goal: '请求方向单一' },
					{ before: '核心协议阻塞长任务', after: 'Tasks 扩展与任务状态轮询', goal: '连接可释放' },
					{ before: '流恢复与消息补发', after: '断流后以新请求重发', goal: '无连接状态' }
				]" :key="item.before" class="atms-evolution-row">
					<div class="atms-evolution-cell">{{ item.before }}</div>
					<span class="atms-evolution-arrow">→</span>
					<div class="atms-evolution-cell atms-evolution-cell--strong">{{ item.after }}</div>
					<span class="atms-chip atms-chip-brand">{{ item.goal }}</span>
				</section>
			</div>
		</figure>

		<figure v-if="figure === 'approval'" class="atms-figure">
			<figcaption class="atms-figcap"><b>图 6-1</b> 人工确认的三种恢复语义：以扣款后审批为例</figcaption>
			<div class="atms-grid atms-grid-3">
				<section class="atms-mode">
					<span class="atms-chip atms-chip-write">方式 A</span>
					<h4>异常式中断</h4>
					<p><b>恢复方式：</b>从头重新执行整个节点</p>
					<p><b>主要风险：</b>中断前的扣款可能再次执行</p>
					<p><b>使用要求：</b>中断前无副作用，或副作用具备幂等性</p>
				</section>
				<section class="atms-mode">
					<span class="atms-chip atms-chip-brand">方式 B</span>
					<h4>可序列化状态</h4>
					<p><b>恢复方式：</b>从暂停的调用处继续</p>
					<p><b>主要风险：</b>取决于状态快照的粒度</p>
					<p><b>使用要求：</b>状态完整可序列化，并可跨越进程的有效期限</p>
				</section>
				<section class="atms-mode atms-mode-read">
					<span class="atms-chip atms-chip-read">方式 C</span>
					<h4>回调式放行</h4>
					<p><b>恢复方式：</b>批准后在原执行点继续</p>
					<p><b>主要风险：</b>外部调用尚未发出，重复风险较低</p>
					<p><b>使用要求：</b>运行时能够同步等待授权结果</p>
				</section>
			</div>
		</figure>

		<figure v-if="figure === 'task-state'" class="atms-figure">
			<figcaption class="atms-figcap"><b>图 5-2</b> 持久任务状态机：两个中间态与三个终态</figcaption>
			<div class="atms-flow-scroll">
				<svg class="atms-svg atms-svg--states" viewBox="0 0 760 190" role="img" aria-label="持久任务状态机">
					<defs>
						<marker id="atms-st-arrow" markerWidth="9" markerHeight="9" refX="7.5" refY="4.5" orient="auto">
							<path d="M0,0 L9,4.5 L0,9 Z" fill="var(--cs-color-text-subtle)" />
						</marker>
						<marker id="atms-st-brand" markerWidth="9" markerHeight="9" refX="7.5" refY="4.5" orient="auto">
							<path d="M0,0 L9,4.5 L0,9 Z" fill="var(--cs-color-brand)" />
						</marker>
						<marker id="atms-st-danger" markerWidth="9" markerHeight="9" refX="7.5" refY="4.5" orient="auto">
							<path d="M0,0 L9,4.5 L0,9 Z" fill="var(--cs-color-danger)" />
						</marker>
						<marker id="atms-st-success" markerWidth="9" markerHeight="9" refX="7.5" refY="4.5" orient="auto">
							<path d="M0,0 L9,4.5 L0,9 Z" fill="var(--cs-color-success)" />
						</marker>
						<marker id="atms-st-warning" markerWidth="9" markerHeight="9" refX="7.5" refY="4.5" orient="auto">
							<path d="M0,0 L9,4.5 L0,9 Z" fill="var(--cs-color-warning)" />
						</marker>
					</defs>

					<g font-family="inherit">
						<rect x="36" y="60" width="120" height="60" rx="6" class="atms-box atms-box-brand atms-state" />
						<text x="96" y="86" text-anchor="middle" class="atms-svg-kicker atms-brand">STATE</text>
						<text x="96" y="108" text-anchor="middle" class="atms-svg-label">working</text>

						<rect x="234" y="60" width="140" height="60" rx="6" class="atms-box atms-box-brand atms-state" />
						<text x="304" y="86" text-anchor="middle" class="atms-svg-kicker atms-brand">PAUSED</text>
						<text x="304" y="108" text-anchor="middle" class="atms-svg-label">input_required</text>

						<rect x="470" y="14" width="120" height="52" rx="6" class="atms-box atms-box-success atms-state atms-state--terminal" />
						<text x="530" y="38" text-anchor="middle" class="atms-svg-kicker atms-success">FINAL</text>
						<text x="530" y="57" text-anchor="middle" class="atms-svg-label">completed</text>

						<rect x="470" y="70" width="120" height="52" rx="6" class="atms-box atms-box-danger atms-state atms-state--terminal" />
						<text x="530" y="94" text-anchor="middle" class="atms-svg-kicker atms-danger">FINAL</text>
						<text x="530" y="113" text-anchor="middle" class="atms-svg-label">failed</text>

						<rect x="470" y="126" width="120" height="52" rx="6" class="atms-box atms-box-warning atms-state atms-state--terminal" />
						<text x="530" y="150" text-anchor="middle" class="atms-svg-kicker atms-warning">FINAL</text>
						<text x="530" y="169" text-anchor="middle" class="atms-svg-label">cancelled</text>

						<rect x="628" y="60" width="104" height="60" rx="6" fill="var(--cs-color-bg-soft)" stroke="var(--cs-color-border)" stroke-dasharray="5 4" class="atms-state" />
						<text x="680" y="86" text-anchor="middle" class="atms-svg-kicker">CLIENT</text>
						<text x="680" y="108" text-anchor="middle" class="atms-svg-label" style="font-size:11.5px">补充输入 / 取消</text>
					</g>

					<path d="M156 90 H230" stroke="var(--cs-color-brand)" stroke-width="1.6" marker-end="url(#atms-st-brand)" />
					<path d="M304 60 C304 36 420 30 466 36" stroke="var(--cs-color-success)" stroke-width="1.5" fill="none" marker-end="url(#atms-st-success)" />
					<path d="M360 104 C410 112 432 102 466 98" stroke="var(--cs-color-danger)" stroke-width="1.5" fill="none" marker-end="url(#atms-st-danger)" />
					<path d="M304 120 C330 158 420 166 466 156" stroke="var(--cs-color-warning)" stroke-width="1.5" fill="none" marker-end="url(#atms-st-warning)" />
					<path d="M374 76 C430 40 560 30 624 66" stroke="var(--cs-color-brand)" stroke-width="1.5" stroke-dasharray="5 4" fill="none" marker-end="url(#atms-st-brand)" />
					<path d="M628 100 C560 130 420 128 378 104" stroke="var(--cs-color-brand)" stroke-width="1.5" stroke-dasharray="5 4" fill="none" marker-end="url(#atms-st-brand)" />
					<text x="392" y="52" class="atms-svg-kicker atms-success">完成</text>
					<text x="392" y="124" class="atms-svg-kicker atms-danger">失败</text>
					<text x="392" y="174" class="atms-svg-kicker atms-warning">取消</text>
					<text x="166" y="50" class="atms-svg-kicker atms-brand">需要补充输入</text>
					<text x="556" y="124" class="atms-svg-kicker atms-brand">补充后继续</text>
				</svg>
			</div>
		</figure>
	</div>
</template>

<style scoped>
.atms {
	margin: var(--cs-space-8) 0;
	display: grid;
	gap: var(--cs-space-9);
}

.atms-figure {
	margin: 0;
	min-width: 0;
}

.atms-figure--plain .atms-figcap {
	padding-bottom: 0;
	margin-bottom: var(--cs-space-4);
	border-bottom: 0;
}

.atms-figcap {
	padding-bottom: var(--cs-space-3);
	margin-bottom: var(--cs-space-5);
	border-bottom: 1px solid var(--cs-color-border);
	font-family: var(--cs-font-mono);
	font-size: var(--cs-text-xs);
	line-height: var(--cs-leading-normal);
	letter-spacing: 0.05em;
	color: var(--cs-color-text-subtle);
}

.atms-figcap b {
	color: var(--cs-color-brand);
	margin-right: var(--cs-space-2);
}

.atms-flow-scroll {
	overflow-x: auto;
	padding-bottom: var(--cs-space-2);
}

.atms-svg {
	display: block;
	width: 100%;
	min-width: 60rem;
	height: auto;
}

.atms-svg--pipeline {
	min-width: 44rem;
}

.atms-svg--pipeline .atms-svg-kicker {
	font-size: 10.5px;
}

.atms-svg--pipeline .atms-svg-label {
	font-size: 12.5px;
}

.atms-svg--pipeline .atms-svg-edge {
	font-size: 10px;
	letter-spacing: 0;
}

.atms-svg-title {
	font-size: 15px;
	font-weight: 700;
	fill: var(--cs-color-text);
}

.atms-node--strong {
	stroke-width: 2;
}

.atms-node--compact .atms-svg-label {
	font-size: 12.5px;
}

.atms-svg-kicker {
	font-family: var(--cs-font-mono);
	font-size: 11px;
	letter-spacing: 0.06em;
	fill: var(--cs-color-text-subtle);
}

.atms-svg-label {
	font-size: 13px;
	font-weight: 600;
	fill: var(--cs-color-text);
}

.atms-svg-muted {
	font-size: 12px;
	fill: var(--cs-color-text-subtle);
}

.atms-brand {
	color: var(--cs-color-brand);
	fill: var(--cs-color-brand);
}

.atms-warning {
	color: var(--cs-color-warning);
	fill: var(--cs-color-warning);
}

.atms-success {
	color: var(--cs-color-success);
	fill: var(--cs-color-success);
}

.atms-danger {
	color: var(--cs-color-danger);
	fill: var(--cs-color-danger);
}

.atms-box {
	fill: var(--cs-color-bg-elevated);
	stroke: var(--cs-color-border);
	stroke-width: 1.2;
}

.atms-box-brand {
	stroke: var(--cs-color-brand);
	stroke-width: 1.6;
}

.atms-box-warning {
	stroke: var(--cs-color-warning);
	stroke-width: 1.6;
}

.atms-box-success {
	stroke: var(--cs-color-success);
	stroke-width: 1.6;
}

.atms-box-danger {
	stroke: var(--cs-color-danger);
	stroke-width: 1.6;
}

.atms-svg--states {
	min-width: 46rem;
}

.atms-grid {
	display: grid;
	gap: var(--cs-space-4);
}

.atms-grid-4 {
	grid-template-columns: repeat(4, minmax(0, 1fr));
}

.atms-grid-3 {
	grid-template-columns: repeat(3, minmax(0, 1fr));
}

.atms-grid-2 {
	grid-template-columns: repeat(2, minmax(0, 1fr));
}

.atms-compact {
	gap: var(--cs-space-2);
}

.atms-card,
.atms-mode,
.atms-host,
.atms-server {
	background: var(--cs-color-bg-elevated);
	border: 1px solid var(--cs-color-border);
	border-top: 3px solid var(--cs-color-border-strong);
	border-radius: var(--cs-radius-sm);
	box-shadow: var(--cs-shadow-sm);
	padding: var(--cs-space-5) var(--cs-space-5) var(--cs-space-4);
	min-width: 0;
}

.atms-card h4,
.atms-mode h4,
.atms-host h4,
.atms-server h4 {
	margin: var(--cs-space-2) 0 var(--cs-space-3);
	font-size: var(--cs-text-md);
	line-height: var(--cs-leading-tight);
}

.atms-card p,
.atms-mode p,
.atms-host p,
.atms-server p {
	margin: 0;
	font-size: var(--cs-text-sm);
	line-height: var(--cs-leading-normal);
	color: var(--cs-color-text-muted);
}

.atms-card p + p,
.atms-mode p + p {
	margin-top: var(--cs-space-2);
}

.atms-card code {
	display: inline-block;
	margin-top: var(--cs-space-4);
	font-family: var(--cs-font-mono);
	font-size: var(--cs-text-2xs);
	color: var(--cs-color-text-subtle);
	background: var(--cs-color-bg-soft);
	border: 1px solid var(--cs-color-border);
	border-radius: var(--cs-radius-xs);
	padding: var(--cs-space-1) var(--cs-space-3);
}

.atms-kicker {
	display: block;
	font-family: var(--cs-font-mono);
	font-size: var(--cs-text-2xs);
	font-weight: 600;
	letter-spacing: 0.07em;
	color: var(--cs-color-text-subtle);
}

.atms-card-read {
	border-top-color: var(--cs-color-success);
}

.atms-card-read .atms-kicker {
	color: var(--cs-color-success);
}

.atms-card-write,
.atms-mode-write {
	border-top-color: var(--cs-color-warning);
}

.atms-card-write .atms-kicker {
	color: var(--cs-color-warning);
}

.atms-card-brand,
.atms-mode-brand {
	border-top-color: var(--cs-color-brand);
}

.atms-card-brand .atms-kicker {
	color: var(--cs-color-brand);
}

.atms-mode-read {
	border-top-color: var(--cs-color-success);
}

.atms-mode-neutral {
	border-top-color: var(--cs-color-border-strong);
}

.atms-chip {
	display: inline-flex;
	align-items: center;
	border-radius: var(--cs-radius-xs);
	padding: 0.1875rem 0.625rem;
	font-family: var(--cs-font-mono);
	font-size: var(--cs-text-2xs);
	font-weight: 600;
	letter-spacing: 0.06em;
}

.atms-chip-read {
	color: var(--cs-color-success);
	background: var(--cs-color-success-soft);
}

.atms-chip-brand {
	color: var(--cs-color-brand);
	background: var(--cs-color-brand-soft);
}

.atms-chip-write {
	color: var(--cs-color-warning);
	background: var(--cs-color-warning-soft);
}

.atms-trust {
	display: grid;
	grid-template-columns: minmax(0, 1.3fr) auto minmax(0, 1fr);
	gap: var(--cs-space-4);
	align-items: stretch;
}

.atms-host {
	border-top-color: var(--cs-color-brand);
}

.atms-server {
	border-top-color: var(--cs-color-warning);
}

.atms-clients {
	display: grid;
	gap: var(--cs-space-2);
	align-content: center;
}

.atms-clients span {
	display: grid;
	place-items: center;
	min-height: 2.5rem;
	padding: 0 var(--cs-space-4);
	border: 1px solid var(--cs-color-border);
	border-radius: var(--cs-radius-pill);
	background: var(--cs-color-bg-elevated);
	font-family: var(--cs-font-mono);
	font-size: var(--cs-text-xs);
	color: var(--cs-color-text-muted);
}

.atms-trust {
	gap: var(--cs-space-3);
}

@media (max-width: 720px) {
	.atms-clients {
		grid-template-columns: repeat(3, minmax(0, 1fr));
	}
}

.atms-host .atms-grid p {
	padding: var(--cs-space-3);
	border: 1px solid var(--cs-color-border);
	border-radius: var(--cs-radius-sm);
	background: var(--cs-color-bg-soft);
	color: var(--cs-color-text);
	font-size: var(--cs-text-sm);
}

.atms-evolution {
	display: grid;
	gap: var(--cs-space-2);
}

.atms-evolution-row {
	display: grid;
	grid-template-columns: minmax(0, 1fr) 2rem minmax(0, 1.25fr) auto;
	gap: var(--cs-space-3);
	align-items: center;
	padding: var(--cs-space-4) var(--cs-space-5);
	border: 1px solid var(--cs-color-border);
	border-radius: var(--cs-radius-sm);
	background: var(--cs-color-bg-elevated);
	box-shadow: var(--cs-shadow-sm);
}

.atms-evolution-cell {
	font-size: var(--cs-text-sm);
	line-height: var(--cs-leading-normal);
	color: var(--cs-color-text-muted);
}

.atms-evolution-cell--strong {
	color: var(--cs-color-text);
	font-weight: 600;
}

.atms-evolution-arrow {
	text-align: center;
	font-family: var(--cs-font-mono);
	color: var(--cs-color-brand);
}

.atms-evolution-row .atms-chip {
	justify-content: center;
	justify-self: end;
	white-space: nowrap;
}

@media (max-width: 720px) {
	.atms-grid-4 {
		grid-template-columns: repeat(2, minmax(0, 1fr));
	}

	.atms-trust {
		grid-template-columns: 1fr;
	}

	.atms-evolution-row {
		grid-template-columns: 1fr 1.5rem auto;
		grid-template-areas: "before arrow chip" "before after after";
		row-gap: var(--cs-space-2);
	}

	.atms-evolution-row > :nth-child(1) { grid-area: before; }
	.atms-evolution-row > :nth-child(2) { grid-area: arrow; }
	.atms-evolution-row > :nth-child(3) { grid-area: after; }
	.atms-evolution-row > :nth-child(4) { grid-area: chip; }
}

/* 桌面端利用正文两侧留白扩幅，与 context-boundary 图保持一致；
   四列卡片在 832–920px 幅宽下每张约 190–210px，避免被压成窄条。 */
@media (min-width: 1280px) {
	.atms-figure {
		width: min(920px, calc(100vw - 28rem));
		max-width: none;
		margin-right: auto;
		margin-left: auto;
	}
}

@media (max-width: 600px) {
	.atms-grid-4,
	.atms-grid-3,
	.atms-grid-2 {
		grid-template-columns: 1fr;
	}

	.atms-clients {
		grid-template-columns: 1fr;
	}

	.atms-evolution-row {
		grid-template-columns: 1fr;
		grid-template-areas: "before" "arrow" "after" "chip";
	}

	.atms-evolution-arrow {
		transform: rotate(90deg);
		justify-self: start;
	}

	.atms-evolution-row .atms-chip { justify-self: start; }
}
</style>

<style>
/* 仅作用于工具机制页的规格书式正文块；类名带页面前缀，避免进入其他文档。 */
.vp-doc .atms-callout {
	margin: var(--cs-space-6) 0;
	padding: var(--cs-space-5) var(--cs-space-6);
	border-left: 3px solid var(--cs-color-border-strong);
	border-radius: 0 var(--cs-radius-md) var(--cs-radius-md) 0;
	background: var(--cs-color-bg-soft);
	font-size: var(--cs-text-md);
	line-height: var(--cs-leading-normal);
}

.vp-doc .atms-callout__k {
	display: block;
	margin-bottom: var(--cs-space-2);
	font-family: var(--cs-font-mono);
	font-size: var(--cs-text-2xs);
	font-weight: 600;
	letter-spacing: 0.1em;
	color: var(--cs-color-text-subtle);
}

.vp-doc .atms-callout--key {
	border-left-color: var(--cs-color-brand);
	background: var(--cs-color-brand-soft);
}

.vp-doc .atms-callout--key .atms-callout__k {
	color: var(--cs-color-brand);
}

.vp-doc .atms-callout--trap {
	border-left-color: var(--cs-color-danger);
	background: var(--cs-color-danger-soft);
}

.vp-doc .atms-callout--trap .atms-callout__k {
	color: var(--cs-color-danger);
}

@media (max-width: 600px) {
	.vp-doc .atms-callout {
		font-size: var(--cs-text-base);
	}
}
</style>
