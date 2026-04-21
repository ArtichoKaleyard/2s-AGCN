"""AGCN 系列模型共享的小型工具。"""

from __future__ import annotations

import math

import torch
from torch import nn


def conv_branch_init(conv: nn.Conv2d, branches: int) -> None:
    """按官方仓库方式初始化图子集分支卷积。"""

    weight = conv.weight
    out_channels = weight.size(0)
    in_channels = weight.size(1)
    temporal_kernel = weight.size(2)
    nn.init.normal_(weight, 0, math.sqrt(2.0 / (out_channels * in_channels * temporal_kernel * branches)))
    if conv.bias is not None:
        nn.init.constant_(conv.bias, 0)


def conv_init(conv: nn.Conv2d) -> None:
    """使用 Kaiming 初始化二维卷积，并将 bias 置零。"""

    nn.init.kaiming_normal_(conv.weight, mode="fan_out")
    if conv.bias is not None:
        nn.init.constant_(conv.bias, 0)


def bn_init(batch_norm: nn.modules.batchnorm._BatchNorm, scale: float) -> None:
    """初始化 BatchNorm 的缩放权重和偏置。"""

    nn.init.constant_(batch_norm.weight, scale)
    nn.init.constant_(batch_norm.bias, 0)


class UnitTCN(nn.Module):
    """与官方 AGCN 实现一致的时间卷积单元。"""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 9, stride: int = 1) -> None:
        super().__init__()
        padding = int((kernel_size - 1) / 2)
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=(kernel_size, 1),
            padding=(padding, 0),
            stride=(stride, 1),
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        conv_init(self.conv)
        bn_init(self.bn, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """执行时间卷积并接 BatchNorm。"""

        return self.bn(self.conv(x))


class ZeroResidual(nn.Module):
    """返回标量零的残差分支，对齐原始 lambda 写法。"""

    def forward(self, x: torch.Tensor) -> int:
        """为禁用的残差连接返回零。"""

        del x
        return 0


def normalize_skeleton_input(
    x: torch.Tensor,
    data_bn: nn.BatchNorm1d,
    *,
    in_channels: int,
    num_point: int,
    num_person: int,
) -> tuple[torch.Tensor, int]:
    """将 ``(N, C, T, V, M)`` 骨架输入整理为 GCN stage 需要的形状。

    参数：
        x: 骨架张量。
        data_bn: 作用在 ``M * V * C`` 维度上的 BatchNorm 层。
        in_channels: 期望的通道数。
        num_point: 期望的关节点数量。
        num_person: 期望的人数维度。

    返回：
        形状为 ``(N * M, C, T, V)`` 的 stage 输入，以及原始 batch size。

    异常：
        ValueError: 当输入形状与模型配置不一致时抛出。
    """

    if x.ndim != 5:
        raise ValueError(f"Expected skeleton input `(N, C, T, V, M)`, got {tuple(x.shape)}.")
    batch_size, channels, timesteps, points, persons = x.size()
    if channels != in_channels or points != num_point or persons != num_person:
        raise ValueError(
            "Skeleton input shape does not match model configuration: "
            f"expected C={in_channels}, V={num_point}, M={num_person}; "
            f"got C={channels}, V={points}, M={persons}."
        )
    x = x.permute(0, 4, 3, 1, 2).contiguous().view(batch_size, persons * points * channels, timesteps)
    x = data_bn(x)
    x = x.view(batch_size, persons, points, channels, timesteps)
    x = x.permute(0, 1, 3, 4, 2).contiguous().view(batch_size * persons, channels, timesteps, points)
    return x, batch_size


def global_pool_person_mean(x: torch.Tensor, *, batch_size: int, num_person: int) -> torch.Tensor:
    """在时间、关节和人数维度上池化 stage 输出。"""

    channels = x.size(1)
    x = x.view(batch_size, num_person, channels, -1)
    return x.mean(3).mean(1)
