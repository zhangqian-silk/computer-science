# 文件、进程与浏览器环境

工具调用把模型连接到资源，执行环境决定它实际能接触哪些文件、进程、网络和凭据。本页区分结构化文件 API、通用命令和浏览器操作的安全与并发语义。

## 文件路径与版本校验

`read_file(path)` 需要确定工作区根、路径解析、符号链接策略、文件大小和返回范围。禁止 `../` 字符串不等于路径安全：符号链接、大小写规则、挂载点和运行时替换都可能改变实际目标。真正安全的实现需要利用操作系统提供的目录句柄/无跟随访问能力或隔离文件系统，并测试平台差异。

编辑还需要“读的版本就是写的基准”。模型看到旧文件后，用户或另一个 Agent 可能已修改。仅使用临时文件加原子 rename 可以防止半写文件，但不会自动避免覆盖别人的新内容。

```python
# 教学伪代码：workspace.apply_if_version 必须原子比较并应用。
def propose_edit(read_result, replacement):
    return {
        "path": read_result["path"],
        "expected_version": read_result["version"],
        "replacement": replacement,
    }

def apply_edit(workspace, proposal):
    return workspace.apply_if_version(
        proposal["path"],
        expected_version=proposal["expected_version"],
        new_content=proposal["replacement"],
    )
```

这里不能在普通 Python 中先 `read` 比较 hash、再 `write` 就声称无竞争：比较与写之间仍可变化。单进程可以串行化工作区写操作；多进程需要更强的资源控制、文件锁协议或每任务独立工作区加合并。

## Patch、全文件替换与结构化编辑

全文件替换简单，但长文件容易无意改动。Patch 表达差异，仍可能匹配错误位置；AST 编辑更精确地处理语言结构，但受解析器和语言版本限制。选择应取决于任务和可验证性，而不是默认一种方法永远可靠。

编辑工具的回执应包括旧版本、新版本、改变范围和实际差异。拒绝应用时返回冲突信息，让模型重新读取，而不是让它连续尝试不同 offset 强行覆盖。对用户未提交修改，应保留原始基线和工作区状态，不能用 checkout/reset 简单“恢复干净”。

## Shell 的能力与风险

使用 argv 而非 `shell=True` 可减少 shell 元字符解析问题，但不能让被执行程序本身无害。`python -c ...`、测试脚本、包管理器 lifecycle scripts 都可能读写网络和文件。命令名白名单也不够，例如允许解释器通常意味着允许任意代码。

```python
# 可运行形态示例：这里只执行受信任的固定测试命令。
import subprocess

def run_fixed_check(workspace):
    completed = subprocess.run(
        ["python", "-m", "unittest", "discover", "-s", "tests"],
        cwd=workspace,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
        env={"PATH": "/usr/bin:/bin", "PYTHONNOUSERSITE": "1"},
    )
    return {
        "exit_code": completed.returncode,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
    }
```

这不是沙箱。真实测试代码和路径必须可信或在隔离环境中运行；固定 PATH 是否包含目标解释器也需按部署环境调整。`capture_output` 在无上限输出时会占用大量内存，因此生产 executor 应增量读取、限量保留、转存日志，并避免 stdout/stderr 互相阻塞。[[32]](../references.md#source-python-subprocess)

## 进程生命周期与取消

启动进程后，任务拥有它及其后代的清理责任。仅取消一个 await 或杀掉父 PID，可能留下子进程、后台服务和仍持有端口的测试程序。应使用进程组或平台等价机制，先协作终止，超时后强制停止，并回收退出状态。

即便进程停止，远端提交过的事务、邮件或云资源仍可能存在。因此“取消成功”至少区分停止本地等待、终止进程树、确认外部操作结果。若 API 不支持撤销，只能停止新工作并对已发出的操作对账，不能承诺回滚。

长命令可以返回 job ID，后续 poll/wait/cancel；但 job ID 不应脱离会话、租户和所有权。丢弃 UI 页面不代表转让进程所有权。与 [取消](../runtime/human-control.md)、[任务调度](../runtime/scheduling-protocols.md) 的协议要一致。

## 浏览器操作依赖观察版本

浏览器工具通常先读取页面或截图，再执行点击、输入等动作。观察到按钮后到执行之间，页面可能重排、登录态改变或目标窗口切换。坐标或元素编号应绑定观测代次；不能让旧截图上的第3个按钮在新页面上被盲点。

关键写操作使用确认前再观察、核对目标对象和动作摘要；执行后读取结果页或后端状态验证。视觉上出现“成功”也可能只是未提交表单提示，具体业务应有可核验回执。

网页文本和截图均属于外部数据。浏览器下载文件的类型、大小、去向需要控制；登录凭据由浏览器环境持有，不发给模型。允许访问某个域名也不等于允许它接收任何秘密数据。

## 沙箱、权限与工作区的边界

| 机制 | 主要限制 | 不能自动保证 |
| --- | --- | --- |
| 工具审批 | 某项具体动作是否被允许 | 被允许程序内部没有其他效果 |
| 文件 API 范围 | 通过该 API 访问的路径 | Shell 或插件也受同样约束 |
| 工作区目录 | 输出组织与并行隔离 | 进程不能越出目录 |
| OS/container/VM 沙箱 | 进程能访问的资源 | 已允许资源绝对安全、远端动作可回滚 |
| 网络策略 | 可连接目标与出口 | 业务参数和目标账户正确 |

Claude Code 的 sandbox 文档可作为 Bash 执行约束的公开行为参考，但不能把 Bash 的隔离结论推广到所有 MCP server、插件或外部服务。其 checkpointing 也需要按实际覆盖范围理解，不能等同于任何 Shell 或数据库动作的通用回滚。[[33]](../references.md#source-cc-sandbox) [[34]](../references.md#source-cc-checkpoints)

DSH 的文件、子进程等能力可以由 provider 替换，这为接入远程执行环境提供设计思路；是否真正隔离仍取决于 provider 和部署配置，不由“插件化”三个字保证。[[30]](../references.md#source-dsh-tools)

## 并行工作区与产物集成

多个 Agent 同时写同一目录会造成测试互相影响、文件覆盖和不一致结果。每任务或每子任务独立 worktree/container 可以降低冲突，但必须定义基线 commit、未提交内容处理、共享缓存目录和合并顺序。

结果不是一段“已完成”消息，而是补丁、版本、测试命令与退出结果。合并后仍要运行集成验证，因为各分支单独通过不代表组合通过。日志需要归属执行者，避免把 A 的测试结果误绑定到 B 的代码。

## 实现与调优的检查点

压力测试长输出、卡住的 stderr、子进程逃逸、链接替换、路径冲突、磁盘满、权限撤销、浏览器导航变化和取消竞争。许多错误与模型质量无关，应该先在不调用模型的 executor 测试中覆盖。

性能上区分环境启动、依赖安装、工具实际运行和产物上传。缓存依赖或复用浏览器可降低启动成本，但可能泄漏上个任务状态。每一种复用都应有清理与隔离测试；只有“平均更快”而没有跨任务污染检查，不足以上线。

---

## 读—改—测的具体版本轨迹

工作区文件最初版本h1，模型读取后准备补丁。用户在此期间写入h2。执行器收到补丁时发现基准h1已过期，返回conflict与当前版本，要求重新读取；不能直接覆盖h2。

重新生成的补丁基于h2应用得到h3。测试必须记录测试的是h3；如果另一个任务又改到h4，h3的测试通过不能作为h4交付证据。版本并非只为Git提交服务，也约束未提交工作区。

```text
read h1 → 用户写h2 → apply expected h1：冲突
read h2 → apply expected h2：得到h3
test h3 → 产生receipt(test,h3)
交付前检查candidate版本仍为h3
```

这条轨迹可用内存VersionedFiles先测，再换真实文件系统和进程测试。两者不等价：内存锁可以原子比较写入，真实跨进程文件仍需相应策略。

## 不同超时阶段的回执

命令没启动就超过排队deadline，可明确标为not_started；进程启动后被终止，需要保存已产生输出与退出状态；远端API在进程被杀之前已提交，则仍需unknown或远端回执。不要把三者都叫timeout后直接重试。

执行器可返回`started`、`exit_observed`、`owned_children_reaped`和`external_state`等字段。上层UI不一定全显示，但恢复逻辑需要这些差异。

## 环境复用的泄漏测试

连续运行两个任务，第一任务写入临时文件、环境变量、浏览器localStorage和后台进程；第二任务不应看到未授权的前一任务状态。缓存包目录可允许共享，只读凭据和工作文件不应无区别共享。

验证清理不仅看目录空不空，还看进程、端口、浏览器会话、挂载和日志。快照恢复若省略某个对象，复用就可能把上一任务的权限带给下一任务。性能收益必须与隔离成本一起记录。
