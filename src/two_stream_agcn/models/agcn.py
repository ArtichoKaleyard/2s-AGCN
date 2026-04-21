"""官方 2s-AGCN 模型的现代 PyTorch 实现。"""

from __future__ import annotations

import math

import torch
from torch import nn

from .common import (
    UnitTCN,
    ZeroResidual,
    bn_init,
    conv_branch_init,
    conv_init,
    global_pool_person_mean,
    normalize_skeleton_input,
)
from .graph import build_spatial_adjacency


class UnitGCN(nn.Module):
    """官方 AGCN 模型中的自适应图卷积单元。"""

    def __init__(self, in_channels: int, out_channels: int, adjacency: torch.Tensor, coff_embedding: int = 4) -> None:
        super().__init__()
        num_subset = int(adjacency.size(0))
        inter_channels = out_channels // coff_embedding
        self.inter_c = inter_channels
        self.num_subset = num_subset
        self.register_buffer("A", adjacency.clone(), persistent=False)
        self.PA = nn.Parameter(adjacency.clone())
        nn.init.constant_(self.PA, 1e-6)

        self.conv_a = nn.ModuleList([nn.Conv2d(in_channels, inter_channels, 1) for _ in range(num_subset)])
        self.conv_b = nn.ModuleList([nn.Conv2d(in_channels, inter_channels, 1) for _ in range(num_subset)])
        self.conv_d = nn.ModuleList([nn.Conv2d(in_channels, out_channels, 1) for _ in range(num_subset)])

        if in_channels != out_channels:
            self.down: nn.Module = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.down = nn.Identity()

        self.bn = nn.BatchNorm2d(out_channels)
        self.soft = nn.Softmax(-2)
        self.relu = nn.ReLU()

        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                conv_init(module)
            elif isinstance(module, nn.BatchNorm2d):
                bn_init(module, 1)
        bn_init(self.bn, 1e-6)
        for branch in self.conv_d:
            conv_branch_init(branch, self.num_subset)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """对 ``(N, C, T, V)`` 输入执行自适应图卷积。"""

        batch_size, channels, timesteps, num_nodes = x.size()
        adjacency = self.A.to(dtype=x.dtype, device=x.device) + self.PA
        output: torch.Tensor | None = None
        for subset_index in range(self.num_subset):
            attention_a = self.conv_a[subset_index](x)
            attention_a = attention_a.permute(0, 3, 1, 2).contiguous().view(
                batch_size, num_nodes, self.inter_c * timesteps
            )
            attention_b = self.conv_b[subset_index](x).view(batch_size, self.inter_c * timesteps, num_nodes)
            adaptive_adjacency = self.soft(torch.matmul(attention_a, attention_b) / attention_a.size(-1))
            adaptive_adjacency = adaptive_adjacency + adjacency[subset_index]
            feature = x.view(batch_size, channels * timesteps, num_nodes)
            z = self.conv_d[subset_index](
                torch.matmul(feature, adaptive_adjacency).view(batch_size, channels, timesteps, num_nodes)
            )
            output = z if output is None else output + z

        if output is None:
            raise RuntimeError("AGCN graph convolution produced no subset output.")
        output = self.bn(output)
        output = output + self.down(x)
        return self.relu(output)


class TcnGcnUnit(nn.Module):
    """官方 AGCN 模型中的一个 TCN-GCN block。"""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        adjacency: torch.Tensor,
        stride: int = 1,
        residual: bool = True,
    ) -> None:
        super().__init__()
        self.gcn1 = UnitGCN(in_channels, out_channels, adjacency)
        self.tcn1 = UnitTCN(out_channels, out_channels, stride=stride)
        self.relu = nn.ReLU()
        if not residual:
            self.residual: nn.Module = ZeroResidual()
        elif in_channels == out_channels and stride == 1:
            self.residual = nn.Identity()
        else:
            self.residual = UnitTCN(in_channels, out_channels, kernel_size=1, stride=stride)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """执行图卷积、时间卷积和残差相加。"""

        x = self.tcn1(self.gcn1(x)) + self.residual(x)
        return self.relu(x)


class AGCNModel(nn.Module):
    """现代 PyTorch 版官方 2s-AGCN 单流分类器。"""

    def __init__(
        self,
        num_class: int = 60,
        num_point: int = 25,
        num_person: int = 2,
        graph_layout: str = "ntu-rgb+d",
        graph_strategy: str = "spatial",
        in_channels: int = 3,
    ) -> None:
        super().__init__()
        if graph_strategy != "spatial":
            raise ValueError("Official AGCN migration currently preserves only `spatial` graph strategy.")
        adjacency = torch.tensor(build_spatial_adjacency(graph_layout), dtype=torch.float32)
        self.num_class = num_class
        self.num_point = num_point
        self.num_person = num_person
        self.in_channels = in_channels
        self.data_bn = nn.BatchNorm1d(num_person * in_channels * num_point)

        self.l1 = TcnGcnUnit(in_channels, 64, adjacency, residual=False)
        self.l2 = TcnGcnUnit(64, 64, adjacency)
        self.l3 = TcnGcnUnit(64, 64, adjacency)
        self.l4 = TcnGcnUnit(64, 64, adjacency)
        self.l5 = TcnGcnUnit(64, 128, adjacency, stride=2)
        self.l6 = TcnGcnUnit(128, 128, adjacency)
        self.l7 = TcnGcnUnit(128, 128, adjacency)
        self.l8 = TcnGcnUnit(128, 256, adjacency, stride=2)
        self.l9 = TcnGcnUnit(256, 256, adjacency)
        self.l10 = TcnGcnUnit(256, 256, adjacency)

        self.fc = nn.Linear(256, num_class)
        nn.init.normal_(self.fc.weight, 0, math.sqrt(2.0 / num_class))
        bn_init(self.data_bn, 1)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """返回分类头之前的 AGCN 池化特征。"""

        x, batch_size = normalize_skeleton_input(
            x,
            self.data_bn,
            in_channels=self.in_channels,
            num_point=self.num_point,
            num_person=self.num_person,
        )
        x = self.l1(x)
        x = self.l2(x)
        x = self.l3(x)
        x = self.l4(x)
        x = self.l5(x)
        x = self.l6(x)
        x = self.l7(x)
        x = self.l8(x)
        x = self.l9(x)
        x = self.l10(x)
        return global_pool_person_mean(x, batch_size=batch_size, num_person=self.num_person)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """返回分类 logits。"""

        return self.fc(self.forward_features(x))


# 兼容别名保留原仓命名风格，但不要求新代码采用这种写法。
# 这样 state-dict remap 和外部导入更容易审计。
unit_gcn = UnitGCN
TCN_GCN_unit = TcnGcnUnit
Model = AGCNModel
