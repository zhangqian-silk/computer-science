# AI Infra 领域补充

通用规范以仓库根 `AGENTS.md` 为准，数学符号沿用 `ai/AGENTS.md`。

- 可运行示例按执行环境选择语言：PyTorch 使用 Python，算子使用 CUDA C++ 或 Triton，服务控制面与压测工具优先使用 Go，部署配置使用 YAML。
- 实验选择能验证当前假设的硬件即可；CPU 上的语义验证不代表 GPU 算子、显存带宽或吞吐的性能结论。
