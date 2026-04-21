# 项目记忆

## 2026-04-21 - Foundry 边界下的 2s-AGCN 迁移
- 背景：本仓迁移官方 2s-AGCN/AAGCN 到现代 PyTorch，并接入私有 Foundry。
- 决策：本仓只放模型、旧 `.npy + .pkl` 数据适配、checkpoint best-effort remap 和显式注册桥；不维护 skeleton 高层编译器语义。
- 原因：Foundry 已拥有 dataset/protocol、stream/fusion、graph 布局/策略、`SkeletonContext` 和高层配置到 `RunConfig` 的编译逻辑，项目仓重复实现会漂移。
- 操作：新增 `src/two_stream_agcn` 包、`conf/two_stream_agcn` preset、`pyproject.toml`，并使用 `uv run pytest` 在 Python 3.12 环境验证。
- 验证：`uv run pytest` 通过 15 个测试；`uv run python -m compileall src tests` 通过。
- 后续：新增实验配置时只写 Foundry/HydraLoom preset 或显式注册入口，不要在本仓新增 protocol/stream/fusion/graph 合法性规则。
