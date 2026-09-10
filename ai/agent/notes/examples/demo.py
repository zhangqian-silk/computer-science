"""Run three offline paths and print observable results."""
import json
from core import Registry, Tool, ToolGateway, ScriptedModel, one_shot_tool, run_loop

registry = Registry()
registry.install([Tool("get_port", "v1", (), lambda args: 8080)])
proposal = {
    "complete": True, "text": "",
    "tool_calls": [{"id": "read-1", "name": "get_port", "arguments": {}}],
}
answer = {"complete": True, "text": "配置端口是 8080。", "tool_calls": []}

direct = ScriptedModel([answer])
direct_result = direct.complete([{"role": "user", "content": "解释 port: 8080"}], {})
print("1. 单次生成:", direct_result["text"], "模型次数:", len(direct.requests))

routing = ScriptedModel([proposal])
routing_result = one_shot_tool(routing, ToolGateway(registry.snapshot()), "查询端口")
print("2. 工具直接返回:", routing_result, "模型次数:", len(routing.requests))

agent = ScriptedModel([proposal, answer])
result = run_loop(agent, ToolGateway(registry.snapshot()), "查询并解释端口")
print("3. 观察驱动闭环:", result["status"], "模型次数:", len(agent.requests))
print(json.dumps(result["messages"], ensure_ascii=False, indent=2))
