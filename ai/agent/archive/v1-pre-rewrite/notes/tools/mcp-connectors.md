# MCP 与外部连接器

MCP 将模型宿主与外部工具、资源等能力连接起来。宿主负责发现、选择、调用和把结果编入模型输入，模型本身通常不直接管理连接或 OAuth。MCP 统一接口，不统一业务授权、结果真伪、执行幂等和任务完成条件。

本页以公开的 **2026-07-28 规范变更**和实际需要兼容的旧协议形态为边界；不能把新版规范发布等同于所有已安装客户端与 server 已升级。协议版本必须进入连接配置和验收记录。[[35]](../references.md#source-mcp-release) [[36]](../references.md#source-mcp-ts-versions)

## 工具、资源与提示

工具提供可执行动作，资源提供可读取内容，提示提供可复用的输入模板或交互入口。宿主把发现结果转换成模型可见描述或材料。一个 MCP server 可以提供多个类型，宿主也不一定全部暴露给模型。

这有助于解释为何“把 MCP 接上”不是一次模型调用的新角色：连接器在程序侧运行；模型看到的是选择后的工具 schema、资源文本或提示内容。凭据、连接地址的敏感部分和权限判定不应进入模型消息。

## 协议版本与连接生命周期

旧形态的初始化、能力协商和会话管理，与 2026-07-28 核心无状态请求设计存在差异。新版移除核心 `initialize/initialized` 握手及 `Mcp-Session-Id` 语义，使用逐请求版本/能力信息，并提供可选发现与显式状态句柄等机制；Tasks 等能力又有独立扩展边界。具体字段应按目标 SDK 实现，不应同时拼上所有年代的 header。[[35]](../references.md#source-mcp-release)

参考适配的逻辑是先选择已验证协议 profile，而不是让模型猜版本：

```python
# 教学伪代码。两个 client 都是按对应 SDK 规范实现的连接适配器。
def connect_server(config, clients):
    profile = config["protocol_profile"]
    if profile not in clients:
        raise ValueError("未验证的协议组合")
    client = clients[profile]
    connection = client.connect(config["endpoint"], config["credential_ref"])
    observed = connection.describe_capabilities()
    if not config["required_capabilities"].issubset(observed):
        raise ValueError("服务能力不满足，不能静默降级")
    return connection
```

`describe_capabilities()` 是内部统一入口，不意味着新版线上必须执行某个同名 RPC。保留协商结果、SDK 版本和实际请求 profile，后续问题才可复现。兼容层的 fallback 应明确并可观测，不能无限轮询不同协议造成副作用或鉴权噪声。

## 发现、暴露与执行授权

发现目录是 server 声称自己提供的能力，不是宿主自动授予的权限。推荐流程是 server allowlist → 使用当前身份连接 → 校验目录和 schema → 按任务选择能力 → 公布模型 → 执行时再校验授权。

如果只在工具列表里隐藏某项，但执行入口仍接受其名称，模型生成未公开调用时可能绕过选择边界。工具名也应命名空间化，避免两个 server 同名 `delete` 被错误路由。目录快照需要记录来源、版本和 schema hash。

目录缓存可以减少开销，但 schema 更新会影响在途调用。可借鉴 [版本快照](hooks-plugins.md) 的原则：本轮绑定已公布定义，下一轮刷新；安全撤销即时检查。新版缓存元数据可辅助新鲜度处理，但不替代当前用户资源权限。[[35]](../references.md#source-mcp-release)

## 远程结果与工具网关

```python
# 教学伪代码：网关已做最终参数和身份校验。
def invoke_remote(binding, call, identity, gateway):
    request = {
        "server_id": binding["server_id"],
        "tool_name": binding["remote_name"],
        "arguments": call["arguments"],
        "catalog_revision": binding["catalog_revision"],
    }
    receipt = gateway.execute_remote(identity, request)
    return {
        "call_id": call["id"],
        "status": receipt["status"],
        "content": receipt["model_content"],
        "receipt_ref": receipt["id"],
    }
```

模型的 call ID 负责本次配对，远端操作可能有自己的 request/operation ID。两者都应保留。网络超时不证明操作未执行；server 返回 `isError` 也要结合协议错误、工具业务错误和执行结果未知分类。自动重试策略仍由 [业务对账](../runtime/persistence-recovery.md) 决定。

schema 校验不应触发不受控的外部 `$ref` 下载。限制目录大小、参数体积、资源类型和响应字节数。server 提供的描述、提示和资源正文都需要按外部数据处理，不能因为它使用标准协议就升级信任。

## 凭据和委托

令牌属于特定主体、受众和授权范围。模型不应选择“换一个更大权限的 token”绕过拒绝，宿主也不应把一个下游服务的 token 原样转发给任意其他服务。远程 server 可以看到发送给它的参数，因此最小披露与出口策略同样重要。

OAuth 处理、重定向、资源指示、受众绑定等安全要求应按当前选用规范核验。本笔记不把某一版本授权流程写成通用法律或安全保证。旧版安全最佳实践仍可作为风险类别入口，但其具体传输规则要结合当前版本。[[37]](../references.md#source-mcp-security-202511)

## 与其他协议的职责区别

MCP 主要连接能力；ACP 主要连接宿主/编辑器与 Agent 交互；A2A 关注不同 Agent 服务间任务与产物等交互；Codex App Server、Pi RPC 等还有各自原生语义。协议之间不存在“名字相似就自动替换”的关系。[[38]](../references.md#source-acp) [[39]](../references.md#source-a2a)

例如调用一个 MCP tool 可以启动 Agent 任务，但它仍需要任务身份、查询状态、取消和结果交付的明确契约。把整个编码 Agent 暴露成一个阻塞工具，不自动解决十分钟等待、断线或人工审批。

## 联调矩阵与实际故障

测试维度包括客户端 SDK/协议版本、server 版本、传输、身份、工具目录、是否有人可响应审批以及响应大小。至少验证：发现成功但调用无权限；目录更新后旧调用；令牌过期；重复 tool 名；连接断开；调用返回大产物；server 文本包含伪指令；取消时远端仍完成。

测量发现次数、目录 token、工具调用延迟、认证刷新、缓存命中和出错位置。一个 server 连接稳定而工具选择错误，可能是描述问题；发现和认证失败则不应让模型反复改写参数。针对不同责任层返回可操作错误，会比“工具不可用”更容易排障。

## 自建的合理起点

先接少量可信只读能力，固定 SDK 与协议 profile，保存真实往返 fixture，并通过统一工具网关。确认模型选择、参数和结果回填后，再增加写操作、目录动态刷新和多个 server。不要把标准协议当作免去工具设计、权限与恢复设计的捷径。

---

## 一次目录更新的联调案例

宿主发现server的`lookup`接受`{"query":...}`，并向模型公布。模型生成期间server升级，改为`{"text":...}`。如果调用时只用工具名去找最新实现，就会出现参数不兼容，甚至行为改变。

合理方案包括协商不可变工具版本、连接到固定发布、或在版本改变时拒绝旧调用并刷新目录。没有版本能力的server也能接入，但应承认无法无缝热更新，并避免在关键任务中动态替换。错误不应被交给模型无限猜字段。

记录工具目录hash与请求中的schema版本，联调日志才能证明模型基于什么描述作了选择。只保存最新`tools/list`不足以复现。

## 能力发现、调用与资源读取预算

一个包含大量工具的server可能让目录占据显著上下文。可在程序侧缓存目录，在模型侧按任务暴露子集；必要时提供工具搜索入口。但每次动态发现也有模型与网络开销，不应把所有调用都先绕一次发现流程。

资源读取同样需要上限。返回PDF、图片或大日志时，约定类型、大小、分页与引用，不能把任意二进制转成超长base64文本塞回模型。缓存范围也不能跨越租户和当前授权。

## 协议兼容与拒绝行为测试

旧客户端、新server、不同transport、凭据过期、server只支持某扩展、审批无人应答，这些组合比单个正常hello更重要。为每种部署记录已验证矩阵，未覆盖组合标为未验证，而不是因为SDK自动fallback就默认安全。

fallback仅解决连接方式，不能保证语义相同。比如新协议某状态句柄在旧模式无对应能力，应明确拒绝需要该功能的任务；不应静默换成一次性调用然后显示“已恢复”。
