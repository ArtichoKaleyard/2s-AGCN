"""官方 2s-AGCN/AAGCN 权重的 best-effort checkpoint remap。

迁移后的模型尽量保留官方模块命名，因此大多数单流 checkpoint 只需要清理
包装器前缀。双流包装器仍需要显式传入 stream 前缀，因为旧 checkpoint
本身不编码它属于 joint 还是 bone 流。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn


@dataclass(frozen=True, slots=True)
class CheckpointLoadReport:
    """一次 best-effort checkpoint 加载操作的摘要。

    参数：
        loaded_keys: 已复制进目标模型的 key。
        missing_keys: 仍未解析的模型 key。
        unexpected_keys: checkpoint 中无法使用的 key。
        shape_mismatched_keys: 两侧都存在但因 tensor 形状不同而跳过的 key。
    """

    loaded_keys: tuple[str, ...]
    missing_keys: tuple[str, ...]
    unexpected_keys: tuple[str, ...]
    shape_mismatched_keys: tuple[str, ...]


def _extract_state_dict(checkpoint: dict[str, Any]) -> dict[str, torch.Tensor]:
    """从常见 PyTorch checkpoint 包装结构中提取 state dict。"""

    for key in ("state_dict", "model_state_dict", "model", "weights"):
        value = checkpoint.get(key)
        if isinstance(value, dict):
            return value
    return checkpoint


def remap_official_state_dict(
    state_dict: dict[str, torch.Tensor],
    *,
    stream_prefix: str | None = None,
) -> dict[str, torch.Tensor]:
    """规范化官方 checkpoint key，但不承诺完整兼容。

    参数：
        state_dict: 原始 checkpoint state dict。
        stream_prefix: 可选目标前缀，例如将单流 checkpoint 加载到双流包装器
            时使用的 ``"models.joint"``。

    返回：
        remap 后的 state dict。函数会去除常见 ``DataParallel`` 前缀，并可选添加
        stream 前缀，但刻意不发明无法从 checkpoint 推断出来的语义映射。
    """

    normalized: dict[str, torch.Tensor] = {}
    prefix = f"{stream_prefix}." if stream_prefix else ""
    for key, tensor in state_dict.items():
        new_key = key
        while new_key.startswith("module."):
            new_key = new_key.removeprefix("module.")
        if stream_prefix and not new_key.startswith(prefix):
            new_key = f"{prefix}{new_key}"
        normalized[new_key] = tensor
    return normalized


def load_checkpoint_best_effort(
    model: nn.Module,
    checkpoint: str | Path | dict[str, Any],
    *,
    stream_prefix: str | None = None,
    map_location: str | torch.device = "cpu",
) -> CheckpointLoadReport:
    """加载兼容的 checkpoint tensor，并报告所有跳过项。

    参数：
        model: 目标模型。
        checkpoint: checkpoint 路径，或已经加载好的 checkpoint 对象。
        stream_prefix: 将官方单流 checkpoint 加载到包装器某个流时使用的可选前缀。
        map_location: 读取 checkpoint 路径时使用的设备映射。

    返回：
        描述 loaded、missing、unexpected 与 shape-mismatched key 的报告。该辅助
        函数刻意采用 best-effort 策略：不兼容 tensor 会被跳过，而不是强行转换。
    """

    if isinstance(checkpoint, str | Path):
        raw_checkpoint = torch.load(Path(checkpoint), map_location=map_location)
    else:
        raw_checkpoint = checkpoint

    candidate = remap_official_state_dict(_extract_state_dict(raw_checkpoint), stream_prefix=stream_prefix)
    target_state = model.state_dict()
    compatible: dict[str, torch.Tensor] = {}
    shape_mismatched: list[str] = []
    unexpected: list[str] = []

    for key, tensor in candidate.items():
        if key not in target_state:
            unexpected.append(key)
            continue
        if tuple(tensor.shape) != tuple(target_state[key].shape):
            shape_mismatched.append(key)
            continue
        compatible[key] = tensor

    model.load_state_dict(compatible, strict=False)
    loaded = tuple(sorted(compatible))
    missing = tuple(sorted(key for key in target_state if key not in compatible))
    return CheckpointLoadReport(
        loaded_keys=loaded,
        missing_keys=missing,
        unexpected_keys=tuple(sorted(unexpected)),
        shape_mismatched_keys=tuple(sorted(shape_mismatched)),
    )
