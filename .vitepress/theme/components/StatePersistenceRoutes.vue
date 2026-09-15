<script setup lang="ts">
import { computed, ref } from "vue"

/**
 * 状态持久化与恢复路径：四个实现各自把「什么」写在「哪里」，以及恢复能回到哪一点。
 *
 * 关键对照维度不是「有没有持久化」，而是三个更细的问题：
 *   1. 写入粒度——每一步都写，还是只在退出时写？决定崩溃后能回到多近的位置。
 *   2. 记录形态——追加式日志还是可覆盖的状态？决定半截写入能否被识别。
 *   3. 恢复语义——恢复是接着往下走，还是把某个单元从头再跑一遍？
 * 第三条最容易被忽略，而它直接决定了「副作用会不会被重复执行」。
 *
 * 各实现的路径、文件形态与配置项取自官方文档与官方仓库源码（核验于 2026-09-14）。
 * 这些细节随版本变化，落地前应按所用版本复核。
 */

type Impl = {
	key: string
	name: string
	store: string
	shape: string
	granularity: string
	resume: string
	gap: string
}

const impls: Impl[] = [
	{
		key: "codex",
		name: "实现 A",
		store: "会话目录下按日期组织的记录文件，文件名里编码了时间戳与线程标识；另有并行的索引与状态库",
		shape: "追加式 JSONL，每行一条带序号与时间戳的记录。半截写入可以被识别并丢弃，这是追加式相对就地覆盖的主要好处",
		granularity: "逐条追加：模型消息、推理条目、工具调用与其输出、压缩标记、令牌用量都各自成行",
		resume: "按线程标识定位记录文件后重建历史；被回退过的线程会在文件名里额外附一个区别于线程标识的记录标识，使同一线程的多条历史可以并存",
		gap: "同一份记录由写者锁裁决归属：另一个进程尝试写同一线程时会被拒绝，而不是交错写入。启动时还会清理上一次留下的陈旧锁"
	},
	{
		key: "cc",
		name: "实现 B",
		store: "会话记录与文件快照分开存放：前者是对话，后者是编辑前的文件内容",
		shape: "会话记录加上按检查点组织的文件快照。两者生命周期不同——快照有数量上限与保留期，会话记录没有",
		granularity: "每条用户提示创建一个检查点；文件快照保留最近的一批，超出后丢弃不再被任何检查点引用的快照",
		resume: "恢复会话后仍可回退；回退分三种——只回代码、只回对话、两者都回。另有把某一段对话压成摘要的选项，它改的是上下文而不是磁盘",
		gap: "只跟踪文件编辑工具做的改动。通过命令行完成的移动、删除、复制不在其中，子代理的编辑通常也不在；符号链接与硬链接目标会被跳过并给出计数提示。快照过期后回退可能直接失败"
	},
	{
		key: "pi",
		name: "实现 C",
		store: "会话目录下按工作目录组织的文件",
		shape: "JSONL，但结构是一棵「树」：每条记录都有自己的标识与父标识，当前位置是活动叶子",
		granularity: "逐条追加，包含消息、模型切换、思考档位变化、标签、压缩条目与分支摘要",
		resume: "可以跳到树上任一历史位置继续，而不必新建文件；也可以派生出新文件。离开一条分支时可以先把它总结成一条摘要挂到新位置，避免丢掉那条路径上的结论",
		gap: "树形结构让「探索多条路径」成为一等公民，代价是恢复语义更复杂——「当前上下文」取决于从根到活动叶子的那条路径，而不是文件里的全部内容"
	},
	{
		key: "lg",
		name: "实现 D",
		store: "由所选的检查点后端决定：进程内存、本地文件、或外部数据库",
		shape: "按线程组织的状态快照，配合一个与线程正交的长期存储",
		granularity: "这一档是可配的，也是最值得注意的一处权衡。三种取值分别是：每步之前同步落盘、下一步执行时异步落盘、以及只在整个运行结束时落盘。越靠后越快，崩溃后能回到的位置越远",
		resume: "用同一个线程标识再次调用即可续跑。人工介入通过一个专门的中断机制表达：运行暂停，外部给出答复后继续",
		gap: "恢复时节点从头重跑，而不是从中断那一行继续。因此中断之前的副作用会再执行一次——这是该模型里最需要提前设计的一点，不是可选的优化"
	}
]

const active = ref("lg")
const cur = computed(() => impls.find((i) => i.key === active.value)!)

/** 落盘粒度与「崩溃后丢失多少」的关系，用于呈现权衡而非精确基准 */
const durability = [
	{ key: "sync", name: "每步之前同步落盘", lose: 1, speed: 2 },
	{ key: "async", name: "下一步执行时异步落盘", lose: 2, speed: 4 },
	{ key: "exit", name: "只在运行结束时落盘", lose: 5, speed: 5 }
]
</script>

<template>
	<div class="spr">
		<div class="spr__tabs">
			<button
				v-for="i in impls"
				:key="i.key"
				type="button"
				class="spr__tab"
				:class="{ 'is-on': active === i.key }"
				:aria-pressed="active === i.key"
				@click="active = i.key"
			>{{ i.name }}</button>
		</div>

		<dl class="spr__rows">
			<div class="spr__row"><dt>存在哪里</dt><dd>{{ cur.store }}</dd></div>
			<div class="spr__row"><dt>记录形态</dt><dd>{{ cur.shape }}</dd></div>
			<div class="spr__row"><dt>写入粒度</dt><dd>{{ cur.granularity }}</dd></div>
			<div class="spr__row"><dt>恢复到哪</dt><dd>{{ cur.resume }}</dd></div>
			<div class="spr__row spr__row--gap"><dt>边界与缺口</dt><dd>{{ cur.gap }}</dd></div>
		</dl>

		<div class="spr__durability">
			<p class="spr__dtitle">落盘粒度的权衡：写得越少越快，崩溃后回退得越远</p>
			<div v-for="d in durability" :key="d.key" class="spr__drow">
				<span class="spr__dname">{{ d.name }}</span>
				<span class="spr__dpair">
					<span class="spr__dlabel">丢失窗口</span>
					<span class="spr__pips">
						<em v-for="n in 5" :key="`l${n}`" :class="{ 'is-lose': n <= d.lose }"></em>
					</span>
				</span>
				<span class="spr__dpair">
					<span class="spr__dlabel">吞吐</span>
					<span class="spr__pips">
						<em v-for="n in 5" :key="`s${n}`" :class="{ 'is-speed': n <= d.speed }"></em>
					</span>
				</span>
			</div>
		</div>

		<p class="pc-note">
			四种做法的分歧集中在一个问题上：<b>恢复之后，是接着往下走，还是把某个单元从头再跑一遍？</b>
			追加式记录与树形会话属于前者——历史是既成事实，恢复就是重建视图；
			而基于图的执行模型属于后者，中断点所在的那个节点会完整重跑，
			于是「中断之前写过的那次数据库更新」会再发生一次。
			这不是实现缺陷，而是该模型的语义，写节点时就得按它设计：
			把有副作用的操作挪到中断之后，或者让它幂等。
			另一条共同的提醒是<b>恢复覆盖不了它没记录的东西</b>——
			文件快照不跟踪命令行造成的改动，会话记录也不跟踪外部进程对同一批文件的并发修改。
			把它当成版本控制的替代品，会在最需要的那一次落空。
		</p>
	</div>
</template>

<style scoped>
.spr { margin: 1rem 0; }
.spr__tabs { display: flex; flex-wrap: wrap; gap: var(--cs-space-1); }
.spr__tab { min-height: var(--cs-tap-target); padding: var(--cs-space-1) var(--cs-space-5); cursor: pointer; font-size: var(--cs-text-xs); color: var(--cs-color-text-muted); background: var(--cs-color-bg); border: 1px solid var(--cs-color-border); border-radius: var(--cs-radius-pill); transition: var(--cs-transition-colors); }
.spr__tab:hover { background: var(--cs-color-bg-soft); }
.spr__tab.is-on { background: var(--cs-color-brand-soft); border-color: var(--cs-color-brand); color: var(--cs-color-brand); font-weight: 600; }
.spr__rows { margin: var(--cs-space-5) 0 0; display: grid; gap: var(--cs-space-2); }
.spr__row { display: grid; gap: var(--cs-space-1); padding: var(--cs-space-3) var(--cs-space-4); border: 1px solid var(--cs-color-border); border-radius: var(--cs-radius-md); background: var(--cs-color-bg); }
.spr__row--gap { background: var(--cs-color-warning-soft); border-color: var(--cs-color-warning); }
.spr__row dt { font-family: var(--cs-font-mono); font-size: var(--cs-text-3xs); letter-spacing: .08em; color: var(--cs-color-text-subtle); }
.spr__row--gap dt { color: var(--cs-color-warning); }
.spr__row dd { margin: 0; font-size: var(--cs-text-base); line-height: var(--cs-leading-relaxed); }
.spr__durability { margin-top: var(--cs-space-6); display: grid; gap: var(--cs-space-2); }
.spr__dtitle { margin: 0 0 var(--cs-space-1); font-family: var(--cs-font-mono); font-size: var(--cs-text-3xs); letter-spacing: .08em; color: var(--cs-color-text-subtle); }
.spr__drow { display: flex; flex-wrap: wrap; align-items: center; gap: var(--cs-space-3) var(--cs-space-6); padding: var(--cs-space-3) var(--cs-space-4); border: 1px solid var(--cs-color-border); border-radius: var(--cs-radius-sm); background: var(--cs-color-bg-soft); }
.spr__dname { flex: 1 1 12rem; font-size: var(--cs-text-sm); }
.spr__dpair { display: flex; align-items: center; gap: var(--cs-space-2); }
.spr__dlabel { font-family: var(--cs-font-mono); font-size: var(--cs-text-3xs); color: var(--cs-color-text-subtle); }
.spr__pips { display: flex; gap: 3px; }
.spr__pips em { width: 8px; height: 8px; border-radius: var(--cs-radius-circle); background: var(--cs-color-neutral-soft); border: 1px solid var(--cs-color-border); }
.spr__pips em.is-lose { background: var(--cs-color-danger); border-color: var(--cs-color-danger); }
.spr__pips em.is-speed { background: var(--cs-color-success); border-color: var(--cs-color-success); }
@media (min-width: 861px) {
	.spr__row { grid-template-columns: 6.5rem minmax(0, 1fr); align-items: baseline; gap: var(--cs-space-5); }
}
</style>
