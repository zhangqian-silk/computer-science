# 模型交互：Prompt、生成与多轮对话

模型调用把任务、资料和可用工具组织成输入，返回回答或工具调用。理解交互首先要看清**模型看到了什么、生成了什么、结果表达了什么**；多轮对话则是在这个基础上继续传入指令、历史消息和工具结果。

本章按「单次输入输出 → 多轮交互 → 生成机制」展开。单次调用与基础多轮构成完整学习单元；对话资料的存取、选择与压缩可在掌握消息关系后继续学习[上下文管理](../context/context-engineering.md)。

---

## 一、单次模型调用：输入、工具与输出 {#single-call}

一次调用的边界是「准备请求 → 调用模型 → 解析响应」：

```mermaid
flowchart LR
	I["任务与资料"] --> Q["请求参数"]
	Q --> M["模型调用"]
	M --> R["响应解析"]
```

图中输入不只是问题文本，还可以包含输出要求和工具说明；输出也不只有自然语言回答。先把这两个方向对应起来，再考虑后续是否继续调用。

### 最小文本请求与响应 {#request}

用户已经给出配置时，直接把配置和问题放入 `messages`：

```json
{
	"messages": [{
		"role": "user",
		"content": "配置如下：\nserver:\n  port: 8080\n  timeout_seconds: 30\n\n服务端口是多少？"
	}]
}
```

`messages` 提供消息；其中 `role: "user"` 标记用户输入，`content` 放资料与问题。本例所需配置已经给出，不需要工具。

<a id="response"></a>

对应的文本响应中，回答位于 `choices` 内：

```json
{
	"choices": [{
		"index": 0,
		"message": {
			"role": "assistant",
			"content": "服务端口是 8080，服务通过该端口接收请求。"
		},
		"finish_reason": "stop"
	}]
}
```

程序先确认请求成功、`choices` 非空、结果未被拒绝或截断，再读取 `choices[0].message.content`。`finish_reason: "stop"` 表示本轮正常结束，响应中的 `usage` 则提供用量统计。

至此已经完成一次「提供配置 → 接收回答 → 读取文本」的交互。后面的输出格式、工具和流式，分别改变这个例子的某一个环节。

### 输入应该怎样设计 {#input-design}

一个可理解的输入应让任务、资料和回答要求相互区分：

| 组成 | 应说明什么 | 配置问答中的例子 |
| --- | --- | --- |
| 任务 | 要模型完成什么 | 找到服务端口并解释用途 |
| 资料 | 回答依据是什么 | 配置正文，而不只是一个本地文件路径 |
| 限定条件 | 哪些信息不能自行假设 | 未提供字段时说明缺失，不猜测默认值 |
| 输出要求 | 结果交给谁、怎样使用 | 给人看时先回答值；给程序读时约束字段 |

例如，比「看看这个配置」更明确的输入是：

```text
任务：指出服务端口，并用一句话解释它的作用。
资料：
server:
  port: 8080
  timeout_seconds: 30

要求：仅依据已提供字段；未提供的信息明确说明。
```

若输出形式不容易用文字说清，可以补充少量「输入 → 期望输出」示例（few-shot）。示例应服务于格式或判断规则，而不是用大量相似内容掩盖不明确的任务。[[19]](../references.md#source-openai-prompt)

### Zero-shot 与 few-shot：规则与示例 {#few-shot}

Zero-shot 提示直接提供任务要求和待处理内容；few-shot 在此基础上增加少量输入输出对，使目标形式和判断规则具体可见。

| 组成 | Zero-shot | Few-shot |
| --- | --- | --- |
| 任务要求 | 只提取显式配置的服务端口，缺失时返回 null | 相同要求 |
| 输入输出示例 | 不提供 | 提供有端口、无端口两种情况 |
| 当前输入 | `server.port = 9090` | 同一份当前输入 |

例如可以把以下内容放在提示中：

```text
任务：提取配置中显式给出的服务端口。
输出：JSON 对象，只包含 port；没有服务端口时返回 null。

示例一
输入：server.port = 8080
输出：{"port":8080}

示例二
输入：server.timeout_seconds = 30
输出：{"port":null}

当前输入：server.port = 9090
```

示例一说明取值和输出格式，示例二说明缺失字段的处理。当前输入对应的目标结果是 `{"port":9090}`，不是复用示例中的 8080。**示例提供的是规则的具体表现，当前输入提供本次要处理的事实。**

这些示例是提示设计者给定的内容，不需要为每个示例分别调用模型，也不是通过本次交互更新模型权重。Zero-shot 同样可能完成任务；few-shot 的价值在于帮助表达难以仅靠文字说明的格式或分类边界，而不是保证示例越多效果越好。[[19]](../references.md#source-openai-prompt)

### 多模态输入：文字与图片共同构成消息 {#multimodal}

输入可以不只有文字。对于具备视觉理解能力的模型，一条用户消息可以同时包含文字任务和图片内容：文字说明关注什么，图片提供待观察的信息。角色仍然是 `user`，图片是一种内容类型，不是新的消息角色。[[75]](../references.md#source-openai-image-input)

设一张配置界面截图清晰显示「服务端口：8080」「超时：30 秒」，一次图片问答可以这样组织：

| 消息中的内容块 | 内容 | 作用 |
| --- | --- | --- |
| 文字任务 | 「读取截图中的服务端口和超时，说明单位。」 | 指定观察目标 |
| 图片 | 实际的配置界面截图 | 提供可见字段、数值和布局 |
| 文字要求 | 「只依据可见内容；模糊或被遮挡的字段说明无法确认。」 | 约束结论依据 |

这里的图片内容必须实际进入模型输入。只写「请看 config.png」，或在文本中描述一个本地路径，并不等于模型已经看到图片。消息中的 `content` 因而可以理解为有序的内容块集合，而不必始终是一段字符串。

读图时应区分三类信息：

- **可见事实**：截图中端口字段的显示值为 8080，超时字段的显示值为 30 秒。
- **相关解释**：说明字段含义和单位，但不把解释说成截图里额外存在的文字。
- **未被证明的状态**：截图没有证明配置已保存、服务已启动或该端口正在监听。

例如用户随后问「这个设置能否说明服务已经运行」，需要依据问题重新判断证据范围，而不是把界面上的配置值当成执行结果。

同一请求也可以携带多张图片。比较修改前后的配置时，应明确标出图片 A、图片 B，以及比较的字段；不要让图片顺序、文字描述和指代对象彼此含混。

在后续追问中，原图片或与问题相关的视觉信息仍需处于当前可用输入中；上一轮的文本回答不是整张图片的替代品。小字、模糊、旋转和遮挡会影响判断，图片也会占用输入预算，不能只按文字长度估算。[[75]](../references.md#source-openai-image-input)

### 从文本回答到结构化结果 {#structured-output}

如果端口需要交给程序使用，直接返回 `{"port":8080}` 比从一句话里截取数字更容易处理。JSON Schema 用规则描述字段、类型和必填项；例如把 `port` 限定为整数或 `null`，不允许额外字段。

在原请求中增加 `response_format`：

```json
{
	"response_format": {
		"type": "json_schema",
		"json_schema": {
			"name": "service_port",
			"strict": true,
			"schema": {
				"type": "object",
				"properties": {
					"port": {"type": ["integer", "null"]}
				},
				"required": ["port"],
				"additionalProperties": false
			}
		}
	}
}
```

输入同时说明「只提取已提供的端口；没有该字段时返回 null」。正常完成且未拒绝时，`message.content` 承载 JSON 文本；程序解析外层响应后，再解析这段文本，才能取得数值 `port`。

提示中要求 JSON、JSON mode、严格 JSON Schema 分别提供自然语言要求、可解析形式和字段约束。**Schema 约束结构，不核实事实**；字段类型正确，也仍需核对数值是否来自输入。[[10]](../references.md#source-openai-structured)

### 从工具声明到调用参数 {#tools}

现在只改变资料的提供方式：用户给出的是 `config.yaml` 路径，而不是配置正文。模型尚未看到文件，可以通过请求中的 `tools` 得知程序提供了一个读取能力：

```json
{
	"messages": [{
		"role": "user",
		"content": "请读取 config.yaml，告诉我服务端口。"
	}],
	"tools": [{
		"type": "function",
		"function": {
			"name": "read_file",
			"description": "读取指定路径的 UTF-8 配置文件，返回文件文本。",
			"parameters": {
				"type": "object",
				"properties": {
					"path": {
						"type": "string",
						"description": "要读取的配置文件路径。"
					}
				},
				"required": ["path"],
				"additionalProperties": false
			},
			"strict": true
		}
	}],
	"tool_choice": "auto"
}
```

工具定义中的 `name` 标识操作，`description` 解释什么时候使用，`parameters` 描述参数名称、类型和必填项。`strict: true` 在支持的 schema 范围内约束生成参数；本例要求 `path` 必填且不接受额外字段。[[2]](../references.md#source-openai-functions)

`tool_choice: "auto"` 允许模型选择回答或提出工具调用；`none` 禁止调用，`required` 要求产生一个或多个调用。指定某个函数时可使用 `{"type":"function","function":{"name":"read_file"}}`。

**请求携带的是工具说明，不是函数代码，也不是执行结果。** 模型据此生成名称和参数，实际文件读取仍由程序完成。[[72]](../references.md#source-openai-chat-functions)

如果模型选择工具，`message` 则可能是：

```json
{
	"role": "assistant",
	"content": null,
	"tool_calls": [{
		"id": "call_config",
		"type": "function",
		"function": {
			"name": "read_file",
			"arguments": "{\"path\":\"config.yaml\"}"
		}
	}]
}
```

这里 `arguments` 是**包含 JSON 的字符串**。读取响应对象中的 `message` 后，还要解析 `function.arguments` 中的 JSON 文本，才能取出 `path`。外层消息结构和内部参数数据是两个读取层次。[[72]](../references.md#source-openai-chat-functions)

读取顺序可以保持简单：

1. 确认请求成功；接口错误不是正常的生成结果。
2. 检查拒绝与结束原因。拒绝、截断或过滤应单独处理，不进入正常结果解析；`length` 表示达到生成上限，不能因为文本像一句完整的话就忽略截断。
3. 检查全部 `tool_calls`，保留各自 ID，解析并校验工具名称与参数。
4. 再消费 `content`，最后记录用量。

一条响应可以同时包含文本与调用。`finish_reason: "tool_calls"` 表示模型提出了调用，不表示函数已经执行；`stop` 表示本轮正常结束，也不证明回答中的事实正确。

由此可以区分两种结构化内容：`response_format` 约束的是要消费的回答结果，工具 `parameters` 约束的是要提交给函数的调用参数。

### 调用结果、生成结束与工具结果 {#response-states}

「成功」需要明确指向哪一个层次：

| 层次 | 判断的对象 | 可以确认什么 |
| --- | --- | --- |
| 模型调用结果 | 是否取得有效响应 | 本次调用是否交付了可解释的响应，而不是错误或中断 |
| 生成结束与内容 | `finish_reason`、拒绝信息、文本和调用条目 | 模型为何结束，本次产生的是回答、调用提议还是不完整内容 |
| 工具执行结果 | 工具实际返回的数据或状态 | 对应动作是否完成，以及得到什么结果 |

例如模型正常返回 `read_file` 的调用参数，但文件实际不存在：模型调用可以正常完成，工具调用提议也可以结构正确，而工具结果仍是「未找到文件」。反过来，一句「已读取配置」只是模型文本，不能代替文件读取结果。

这三个层次分别记录不同事实。`tool_calls` 不表示工具已经成功执行，生成正常结束也不等于答案已通过事实核对。

### 流式输出怎样解析 {#streaming}

当 `stream: true` 时，客户端通过 SSE 接收增量事件，读取的是 `choices[].delta`，而不是把每个片段当成完整的 `message`：

| 增量内容 | 处理方式 |
| --- | --- |
| `delta.role` | 标记消息角色，不一定带有正文 |
| `delta.content` | 按顺序拼接文本，空值不当成正文 |
| `delta.tool_calls` | 按候选索引和工具条目索引关联 ID、名称，分别累积参数片段 |
| 非空 `finish_reason` | 判断对应候选是正常结束、工具调用还是截断 |

工具参数片段不一定是完整 JSON，应拼装完再解析；网络读取边界也不等于 SSE 事件边界。连接中断、收到了第一段文字，都不能替代完整的结束判断。[[73]](../references.md#source-openai-chat-stream)

### 参数与响应字段速查 {#parameter-reference}

按当前场景选择参数，不必给每次请求填满所有可选项。

<details>
<summary>请求参数：输入、输出形式与调用设置</summary>

| 参数 | 作用 | 使用时关注什么 |
| --- | --- | --- |
| `messages` | 提供按顺序排列的输入消息 | 每条消息的角色、内容和相互关系 |
| `response_format` | 约束回答的格式 | 普通文本、JSON 对象或 JSON Schema |
| `tools` | 声明可供模型选择的工具 | 名称、用途和参数 schema，不是函数实现 |
| `tool_choice` | 控制是否允许、要求或指定工具调用 | `auto`、`none`、`required` 或具体函数 |
| `stream` | 选择完整返回还是增量返回 | 两种模式的响应解析方式不同 |
| `max_completion_tokens` | 限制本轮生成预算 | 是上限，不是要求输出达到的长度 |
| `temperature`、`top_p` | 调整生成时的采样选择 | 原理见底层知识部分 |

</details>

<details>
<summary>响应字段：回答、调用、结束原因与用量</summary>

| 字段 | 表示什么 | 解析方式 |
| --- | --- | --- |
| `choices` | 本次生成的候选结果列表 | 通常只请求一个候选，读取前仍需确认列表非空 |
| `message.content` | 模型返回的文本 | 普通回答直接读取；JSON 回答还需解析文本中的 JSON |
| `message.tool_calls` | 模型提出的工具调用列表 | 逐项读取调用 ID、名称与参数，不能只看文本 |
| `message.refusal` | 拒绝信息 | 有拒绝时单独处理，不当作业务结果 |
| `finish_reason` | 本轮为什么结束 | 区分 `stop`、`length`、`tool_calls`、`content_filter` 等 |
| `usage` | 用量统计 | 读取 `prompt_tokens`、`completion_tokens`、`total_tokens` 等 |

</details>

---

## 二、多轮模型交互：指令、消息与工具结果 {#multi-turn}

**用户对话轮次与模型调用次数不是同一个计数。** 用户继续追问可以触发下一次调用；用户只提问一次，程序也可能先请求工具参数，再把工具结果交给模型生成回答。

| 后续调用由什么触发 | 新增的输入 | 需要保留什么 |
| --- | --- | --- |
| 用户继续提问 | 新的 `user` 消息 | 原有资料、相关回答和仍然生效的指令 |
| 工具返回结果 | 对应的 `tool` 消息 | 原始问题、assistant 调用消息和调用 ID |

两种情况都是把已有交互与新增信息组成下一次请求，区别在于新增信息来自用户还是工具。

### 用 system prompt 保持回答要求 {#system-prompt}

若希望多轮配置问答一直遵循相同规则，可以在 `messages` 前面放入 system 消息：

```json
{
	"role": "system",
	"content": "你是配置解释助手。依据提供的配置回答，先给出字段值，再用一句话解释；区分当前配置与修改建议，缺少字段时明确说明。"
}
```

System prompt 放稳定的任务范围、回答依据、表达方式和信息不足时的处理要求；user 消息放当前问题和资料。它同样可以用于单次调用，只是在多轮场景中更容易看出「固定要求」与「逐轮变化的问题」的区别。

<a id="adaptation"></a>

| 消息角色 | 在多轮交互中的作用 |
| --- | --- |
| `system` | 需要持续生效的行为要求 |
| `developer` | 在支持该角色的接口中承载应用开发者指令 |
| `user` | 提供资料、提问、追问或修正条件 |
| `assistant` | 保留模型之前的回答或工具调用消息 |
| `tool` | 把某次工具调用的结果交回模型 |

角色是消息结构的一部分，把 `System:` 写进用户正文并不会改变角色。不同模型对 `system`／`developer` 的支持应按其接口选择。[[1]](../references.md#source-openai-text)

使用自行管理的 `messages` 时，每轮都应继续包含需要生效的 system prompt，不能只在第一轮发送一次就假定下一次独立请求会自动继承。System prompt 规定如何回答，但不会补回未提供的配置事实。[[8]](../references.md#source-openai-state)

### 用户的多轮消息怎样组织 {#conversation}

继续使用端口和超时配置，用户可以连续追问：

| 轮次 | 当前问题 | 本轮 `messages` 的组成 |
| --- | --- | --- |
| 第一轮 | 提供配置，问「服务端口是多少？」 | system + user₁ |
| 第二轮 | 「超时呢？」 | system + user₁ + assistant₁ + user₂ |
| 第三轮 | 「如果把它改成 60 呢？」 | system + user₁ + assistant₁ + user₂ + assistant₂ + user₃ |

此前的 `user` 消息提供资料与条件，`assistant` 消息说明已经回答到哪里，也为「展开第二点」等追问提供指代对象。新消息可以追问、纠正事实或提出假设，不能简单理解为「最后一个数字覆盖之前的数字」。

例如「如果改成 60」是在询问修改方案，当前配置仍是 30；「我刚才贴错了，实际是 60」则是在纠正事实。历史回答也可能有误，不能仅因它带有 `assistant` 角色就当作新的事实证据。

### 工具调用结果怎样传回模型 {#tool-calling}

继续使用第一部分的 `read_file` 调用：程序解析 `call_config` 的参数、读取文件后，要把**完整的 assistant 调用消息**追加到原有 `messages`，再追加工具结果：

```json
{
	"role": "tool",
	"tool_call_id": "call_config",
	"content": "server:\n  port: 8080\n  timeout_seconds: 30\n"
}
```

`tool_call_id` 对应的是上一轮 `tool_calls[].id`，不是函数名称。名称相同的多次调用可以具有不同参数，必须通过 ID 区分各自结果。

| 阶段 | 消息或返回值 |
| --- | --- |
| 第一次请求 | 用户问题，以及 `tools` 中的 `read_file` 定义 |
| 第一次响应 | assistant 消息，包含 `id: "call_config"` 的工具调用 |
| 追加输入 | 原有消息 + 完整 assistant 调用消息 + 对应的 tool 结果消息 |
| 第二次请求 | 发送更新后的 `messages`；如仍允许继续使用工具，同时携带 `tools` |
| 第二次响应 | 模型依据已看到的文件内容生成端口说明 |

第二次请求不一定需要新的用户问题，工具结果本身就是新增输入。它属于 `role: "tool"`，不应伪装成用户发言；`tool_calls` 是模型输出，也不应误放到声明能力的 `tools` 参数中。[[72]](../references.md#source-openai-chat-functions)

<details>
<summary>查看第二次调用的完整 messages：原始问题、调用消息与工具结果</summary>

```json
[
	{
		"role": "user",
		"content": "请读取 config.yaml，告诉我服务端口。"
	},
	{
		"role": "assistant",
		"content": null,
		"tool_calls": [{
			"id": "call_config",
			"type": "function",
			"function": {
				"name": "read_file",
				"arguments": "{\"path\":\"config.yaml\"}"
			}
		}]
	},
	{
		"role": "tool",
		"tool_call_id": "call_config",
		"content": "server:\n  port: 8080\n  timeout_seconds: 30\n"
	}
]
```

将这个数组作为下一次请求的 `messages`。与第一次请求相比，新增的是 assistant 调用消息和 tool 结果；没有新增用户提问。

</details>

如果同一响应提出多个调用，应分别返回对应 ID 的结果；失败也应如实表达，不能把读取失败说成配置字段不存在。需要原样续接的原生条目也应保留，不能仅留下可见文本。

<ModelConversationExplorer />

在图中切换「用户追问／工具结果」，对照两种后续输入。用户追问可以比较是否携带历史；工具结果可以检查调用 ID、结果 ID 以及每次调用真正看到的消息。

---

## 三、底层知识：生成参数、Token 与缓存 {#internals}

### 模型怎样处理消息并生成 Token {#tokens}

聊天接口接收带角色的消息，自回归语言模型处理 Token 序列。消息角色、正文和边界经聊天模板与分词形成输入；模型根据已有序列计算下一个 Token 的概率，选出 Token 后继续生成，直到满足结束条件。

| 阶段 | 发生什么 | 对调用的影响 |
| --- | --- | --- |
| 模板与分词 | 把消息边界、角色和文本转换为 Token ID | Token 数不等于字符数或单词数 |
| Prefill | 处理输入前缀，建立中间状态，并产生首个输出 Token 的分布 | 影响开始回答前的输入处理时间 |
| Decode | 将已生成 Token 接回序列，逐步预测后续 Token | 输出越长，需要推进的生成步骤越多 |

因此，输入长度影响「读多少」，输出长度影响「生成多少」。System prompt、历史消息和 few-shot 示例改变的是本轮生成条件，不会因为参与了对话就修改模型权重。

### 生成参数控制什么 {#generation}

自回归生成先为候选 Token 计算分数，再把分数转换为概率，最后按选定策略取出下一个 Token。Temperature 调整概率分布，top_p 决定参与采样的候选集合。

符号与约定：$z_i$ 表示候选 Token $i$ 的分数（logit），$\tau>0$ 表示温度，$\mathcal{V}$ 表示词表。温度调整后的概率为：

$$
q_i(\tau)=\frac{\exp(z_i/\tau)}{\sum_{j\in\mathcal{V}}\exp(z_j/\tau)}
$$

Softmax 将候选分数变成总和为 1 的概率。温度较低时，分数差异被放大，分布更集中；温度较高时，分布更平缓。

只看三个候选 A、B、C，假设温度为 1 时的概率为 60%、30%、10%。保持原始分数不变，可以得到：

| 温度 | A | B | C | 分布特点 |
| --- | ---: | ---: | ---: | --- |
| 0.5 | 78.26% | 19.57% | 2.17% | 更集中于高分候选 |
| 1 | 60.00% | 30.00% | 10.00% | 基准分布 |
| 2 | 47.27% | 33.43% | 19.30% | 较低分候选也有更大机会 |

Top_p 则按概率从高到低保留累计概率达到阈值的最小候选集合。在基准分布上取 `top_p = 0.8`，A 与 B 的累计概率为 0.9，因此保留 A、B，排除 C；重新归一化后，从约 66.67%、33.33% 的分布中采样。

这说明 **temperature 改变分布的集中程度，top_p 截取候选集合**。它们不是同一个操作；候选集合大小也不是固定值。下面再对照其他生成控制项：

| 控制项 | 作用 | 不能保证什么 |
| --- | --- | --- |
| `temperature` | 调整候选概率分布的集中程度；较低时更偏向高概率候选 | 不补充事实，也不保证答案正确 |
| `top_p` | 选择累计概率达到阈值的候选集合，再从中采样 | 不是固定保留若干个 Token |
| 输出 Token 上限 | 限制最多生成多少 | 不保证在该预算内完成答案 |
| `stop` | 在接口支持时遇到指定序列就停止 | 停止不等于语义完整 |
| 推理预算／effort | 在支持的模型上控制推理投入 | 不等于可见回答长度或正确率保证 |

上式只用于正温度；贪心选择直接取最高分候选，应与正温度下的概率采样区分。低温度可以减少采样随机性，但 `temperature = 0` 不能作为跨请求、跨版本完全复现的承诺。输入与执行条件也会影响结果。[[71]](../references.md#source-openai-reproducibility)

### 上下文窗口与输出预算 {#context-window}

上下文窗口是一次生成可使用的 Token 容量。输入包括 system prompt、历史、当前问题、工具说明和结果等内容；生成部分包括回答，在部分模型中还包括推理 Token。**可见文本短，不代表生成 Token 少。**[[8]](../references.md#source-openai-state)

设窗口为 $W$，输入为 $T_{\mathrm{in}}$，生成部分为 $T_{\mathrm{gen}}$，容量约束为：

$$
T_{\mathrm{in}} + T_{\mathrm{gen}} \le W
$$

生成还受模型输出上限和本轮预算限制。窗口为 8,000 Token、输入占 6,000 时，从容量上最多还剩 2,000 给生成，不能靠把输出参数设成 4,000 来扩大窗口。

同理，第二轮虽然只新增「超时呢？」几个字，只要携带了较长前文，本轮仍然是一个较长输入。

### 缓存复用了什么 {#cache}

| 缓存 | 复用内容 | 是否还要生成回答 |
| --- | --- | --- |
| 单次生成 KV Cache | 已处理位置的 Key／Value 中间状态 | 要，后续 Token 仍需生成 |
| 跨请求前缀缓存／Prompt Cache | 已计算过的相同输入前缀 | 要，还需处理新输入并生成回答 |
| 结果缓存 | 之前得到的完整答案 | 直接返回命中结果时，不再调用模型 |

多轮问答中，system prompt 和早期消息保持不变，新问题追加在末尾，就可能复用共同前缀。是否命中还取决于实际 Token、模型配置、缓存可用性及产品规则；语义相似不等于前缀相同。[[18]](../references.md#source-openai-cache)

**缓存复用计算，不会把前文移出上下文窗口，也不等于模型获得长期记忆。** KV Cache 省去历史 K／V 的重复计算，后续生成仍需读取相关状态。

### 用生成过程解释延迟与成本 {#latency}

| 指标 | 观察什么 |
| --- | --- |
| TTFT：首 Token 延迟 | 从发出请求到收到首个输出 Token 的时间 |
| TPOT／ITL | 后续 Token 的平均生成用时／相邻输出间隔 |
| 总耗时 | 从发出请求到响应结束的时间 |

这些指标可以用前面的机制理解：

- 输入较长时，要关注前缀处理；有效的前缀缓存可以减少重复计算。
- 输出较长时，要关注后续生成；即使命中前缀缓存，模型仍需生成新回答。
- 开启流式后，可以更早展示已有输出，但不能跳过输入处理，也不保证总耗时更短。

角色事件或空增量先到达时，用户还没有看到回答，因此「首个事件时间」和「首个可见文本时间」也应区分。

核算调用成本时，看实际输入、生成与缓存用量，再按对应价格类别计算。不要只按最新用户消息计输入，也不要把缓存 Token 当成窗口之外额外的一份输入。

### 读懂用量：总量与细项的包含关系 {#usage}

`usage` 描述本次实际发生的用量，输出预算则是事先设定的上限。假设一次调用按输入与生成分别记录如下：

```json
{
	"usage": {
		"prompt_tokens": 1000,
		"completion_tokens": 200,
		"total_tokens": 1200
	}
}
```

若输入中有 600 Token 命中缓存，生成中有 80 Token 属于推理，可以这样理解：

| 数量 | 与总量的关系 | 解释 |
| --- | --- | --- |
| 输入 1,000 | 包含全部本轮输入 | 不仅是最新用户问题，也包括指令、历史等实际参与生成的内容 |
| 缓存命中 600 | 是输入 1,000 中的一部分 | 其余 400 未命中，不应再把 600 加到输入总量上 |
| 生成 200 | 包含本轮生成用量 | 不一定全部是用户可见的回答文字 |
| 推理 80 | 是生成 200 中的一部分 | 其余 120 是其他生成用量，不应把推理再计入一次 |
| 总计 1,200 | 输入 1,000 + 生成 200 | 不是把总量和所有细项全部相加 |

其他生成用量还可能包含非可见格式等内容，因此不能直接用「生成量减推理量」断言可见回答恰好有多少 Token。图片也有相应输入计量，不能把图片输入视为零成本或只统计文字。[[11]](../references.md#source-openai-reasoning) [[75]](../references.md#source-openai-image-input)

---

## 学习与面试自检

<details>
<summary>单次调用：端口 JSON 与 read_file 调用参数，分别应该从哪里取出？</summary>

先检查请求是否成功、响应是否完整及是否拒绝。业务结果从 `message.content` 取出 JSON 文本后解析；工具调用则逐项读取 `message.tool_calls`，再解析 `function.arguments`，同时保留调用 ID。前者是回答结果，后者是交给函数的参数。

能解释这两条解析路径，也应能指出 `response_format` 和 `tools` 分别约束哪一种内容。

进一步区分输入中的三类信息：few-shot 示例说明期望的规律，当前资料提供本次事实，图片提供可见证据。不要把示例里的 8080 或截图之外的运行状态当成当前答案。

</details>

<details>
<summary>多轮交互：用户只问了一次，为什么读取配置的例子会调用模型两次？</summary>

第一次模型调用提出读取请求，程序执行后，第二次调用才看到文件内容。下一次输入应包含原始用户问题、完整 assistant 调用消息和匹配 `tool_call_id` 的 tool 结果；如果原请求还有需要生效的 system prompt，也应继续包含。

换成用户追问「超时呢？」时，新增的是 user 消息，但同样需要前文提供配置依据。用户轮次和模型调用次数应分别计数。

</details>

<details>
<summary>底层机制：窗口为 8,000 Token，输入为 6,000，其中 4,000 命中缓存，还能生成 4,000 吗？</summary>

不能。输入仍占 6,000 Token，窗口最多还剩 2,000 给生成，且还需满足模型和请求的输出上限。缓存节省重复的前缀计算，不会增加窗口容量；是否降低总耗时，还要看输入处理与后续生成各占多少。

</details>

---

## 参考资料

- 工具声明、响应解析与结果回传：[Chat Completions Function Calling](../references.md#source-openai-chat-functions)、[Function calling](../references.md#source-openai-functions)。
- 消息角色与提示：[Text generation](../references.md#source-openai-text)、[Prompt engineering](../references.md#source-openai-prompt)。
- 图片输入与视觉理解：[Images and vision](../references.md#source-openai-image-input)。
- 多轮输入与窗口：[Conversation state](../references.md#source-openai-state)。
- 推理与生成用量：[Reasoning models](../references.md#source-openai-reasoning)。
- 输出格式：[Structured model outputs](../references.md#source-openai-structured)。
- 流式与缓存：[Streaming Chat Completions](../references.md#source-openai-chat-stream)、[Hugging Face Caching](../references.md#source-hf-cache)、[Prompt caching](../references.md#source-openai-cache)。
- 生成复现边界：[OpenAI：Reproducible outputs](../references.md#source-openai-reproducibility)。
