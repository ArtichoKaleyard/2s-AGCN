"""AGCN 系列分类器的单流/双流包装器。"""

from __future__ import annotations

from collections.abc import Callable

import torch
from torch import nn


class TwoStreamSkeletonModel(nn.Module):
    """把单流骨架分类器包装为 joint/bone/双流模型。

    参数：
        stream_builders: stream 名到构建单流分类器 callable 的映射。
        stream_mode: ``joint``、``bone`` 或 ``two_stream``。
        fusion: 双流输入的融合方式。目前支持 ``sum`` 与 ``concat_linear``。
        num_classes: 输出类别数，供 ``concat_linear`` 使用。
        feature_dim: 每个单流 backbone 输出的特征维度。
    """

    def __init__(
        self,
        stream_builders: dict[str, Callable[[], nn.Module]],
        stream_mode: str,
        fusion: str | None,
        num_classes: int,
        feature_dim: int = 256,
    ) -> None:
        super().__init__()
        self.stream_mode = stream_mode
        self.streams = tuple(stream_builders)
        self.fusion = fusion
        self.models = nn.ModuleDict({name: builder() for name, builder in stream_builders.items()})
        self.concat_head: nn.Linear | None = None
        if stream_mode == "two_stream" and fusion == "concat_linear":
            self.concat_head = nn.Linear(feature_dim * len(self.streams), num_classes)

    def _forward_single(self, stream_name: str, inputs: torch.Tensor) -> torch.Tensor:
        """将单个 stream 输入送入对应分类器。"""

        return self.models[stream_name](inputs)

    def _features_single(self, stream_name: str, inputs: torch.Tensor) -> torch.Tensor:
        """将单个 stream 输入送入对应特征提取器。"""

        model = self.models[stream_name]
        if not hasattr(model, "forward_features"):
            raise TypeError(f"Stream model `{stream_name}` does not expose `forward_features()`.")
        return model.forward_features(inputs)

    def forward(self, inputs: torch.Tensor | dict[str, torch.Tensor]) -> torch.Tensor:
        """返回融合后的分类 logits。"""

        if self.stream_mode in {"joint", "bone"}:
            stream_name = self.stream_mode
            if stream_name not in self.models:
                raise KeyError(f"Missing `{stream_name}` stream model.")
            if isinstance(inputs, dict):
                if stream_name not in inputs:
                    raise KeyError(f"Missing `{stream_name}` stream input.")
                inputs = inputs[stream_name]
            return self._forward_single(stream_name, inputs)

        if not isinstance(inputs, dict):
            raise TypeError("Two-stream AGCN/AAGCN expects `dict[str, Tensor]` inputs.")
        missing = [stream for stream in self.streams if stream not in inputs]
        if missing:
            raise KeyError(f"Missing two-stream inputs: {missing}")

        if self.fusion == "sum":
            logits = [self._forward_single(stream, inputs[stream]) for stream in self.streams]
            return torch.stack(logits, dim=0).sum(dim=0)
        if self.fusion == "concat_linear":
            if self.concat_head is None:
                raise RuntimeError("concat_linear fusion head is not initialized.")
            features = [self._features_single(stream, inputs[stream]) for stream in self.streams]
            return self.concat_head(torch.cat(features, dim=1))
        raise ValueError(f"Unsupported two-stream fusion mode: {self.fusion}")
