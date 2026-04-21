"""官方 AAGCN 变体的现代 PyTorch 实现。"""

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
    """带可选自适应图与 attention 的 AAGCN 图卷积。"""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        adjacency: torch.Tensor,
        coff_embedding: int = 4,
        adaptive: bool = True,
        attention: bool = True,
    ) -> None:
        super().__init__()
        inter_channels = out_channels // coff_embedding
        num_subset = int(adjacency.size(0))
        num_joints = int(adjacency.size(-1))
        self.inter_c = inter_channels
        self.out_c = out_channels
        self.in_c = in_channels
        self.num_subset = num_subset
        self.adaptive = adaptive
        self.attention = attention

        self.conv_d = nn.ModuleList([nn.Conv2d(in_channels, out_channels, 1) for _ in range(num_subset)])
        if adaptive:
            self.PA = nn.Parameter(adjacency.clone())
            self.alpha = nn.Parameter(torch.zeros(1))
            self.conv_a = nn.ModuleList([nn.Conv2d(in_channels, inter_channels, 1) for _ in range(num_subset)])
            self.conv_b = nn.ModuleList([nn.Conv2d(in_channels, inter_channels, 1) for _ in range(num_subset)])
        else:
            raise ValueError("官方 AAGCN 的 adaptive=False 分支依赖未定义的 self.mask，迁移版不提供修复语义。")

        if attention:
            self.conv_ta = nn.Conv1d(out_channels, 1, 9, padding=4)
            nn.init.constant_(self.conv_ta.weight, 0)
            nn.init.constant_(self.conv_ta.bias, 0)

            kernel_joint = num_joints - 1 if not num_joints % 2 else num_joints
            padding = (kernel_joint - 1) // 2
            self.conv_sa = nn.Conv1d(out_channels, 1, kernel_joint, padding=padding)
            nn.init.xavier_normal_(self.conv_sa.weight)
            nn.init.constant_(self.conv_sa.bias, 0)

            reduction_ratio = 2
            self.fc1c = nn.Linear(out_channels, out_channels // reduction_ratio)
            self.fc2c = nn.Linear(out_channels // reduction_ratio, out_channels)
            nn.init.kaiming_normal_(self.fc1c.weight)
            nn.init.constant_(self.fc1c.bias, 0)
            nn.init.constant_(self.fc2c.weight, 0)
            nn.init.constant_(self.fc2c.bias, 0)

        if in_channels != out_channels:
            self.down: nn.Module = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.down = nn.Identity()

        self.bn = nn.BatchNorm2d(out_channels)
        self.soft = nn.Softmax(-2)
        self.tan = nn.Tanh()
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU(inplace=True)

        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                conv_init(module)
            elif isinstance(module, nn.BatchNorm2d):
                bn_init(module, 1)
        bn_init(self.bn, 1e-6)
        for branch in self.conv_d:
            conv_branch_init(branch, self.num_subset)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """执行 AAGCN 图卷积和 attention。"""

        batch_size, channels, timesteps, num_nodes = x.size()
        output: torch.Tensor | None = None
        if self.adaptive:
            adjacency = self.PA
            for subset_index in range(self.num_subset):
                attention_a = self.conv_a[subset_index](x)
                attention_a = attention_a.permute(0, 3, 1, 2).contiguous().view(
                    batch_size, num_nodes, self.inter_c * timesteps
                )
                attention_b = self.conv_b[subset_index](x).view(batch_size, self.inter_c * timesteps, num_nodes)
                adaptive_adjacency = self.tan(torch.matmul(attention_a, attention_b) / attention_a.size(-1))
                adaptive_adjacency = adjacency[subset_index] + adaptive_adjacency * self.alpha
                feature = x.view(batch_size, channels * timesteps, num_nodes)
                z = self.conv_d[subset_index](
                    torch.matmul(feature, adaptive_adjacency).view(batch_size, channels, timesteps, num_nodes)
                )
                output = z if output is None else output + z
        if output is None:
            raise RuntimeError("AAGCN graph convolution produced no subset output.")
        output = self.bn(output)
        output = output + self.down(x)
        output = self.relu(output)

        if self.attention:
            spatial = output.mean(-2)
            spatial = self.sigmoid(self.conv_sa(spatial))
            output = output * spatial.unsqueeze(-2) + output

            temporal = output.mean(-1)
            temporal = self.sigmoid(self.conv_ta(temporal))
            output = output * temporal.unsqueeze(-1) + output

            channel = output.mean(-1).mean(-1)
            channel = self.relu(self.fc1c(channel))
            channel = self.sigmoid(self.fc2c(channel))
            output = output * channel.unsqueeze(-1).unsqueeze(-1) + output
        return output


class TcnGcnUnit(nn.Module):
    """一个 AAGCN TCN-GCN block。"""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        adjacency: torch.Tensor,
        stride: int = 1,
        residual: bool = True,
        adaptive: bool = True,
        attention: bool = True,
    ) -> None:
        super().__init__()
        self.gcn1 = UnitGCN(in_channels, out_channels, adjacency, adaptive=adaptive, attention=attention)
        self.tcn1 = UnitTCN(out_channels, out_channels, stride=stride)
        self.relu = nn.ReLU(inplace=True)
        self.attention = attention
        if not residual:
            self.residual: nn.Module = ZeroResidual()
        elif in_channels == out_channels and stride == 1:
            self.residual = nn.Identity()
        else:
            self.residual = UnitTCN(in_channels, out_channels, kernel_size=1, stride=stride)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """执行 AAGCN graph block、temporal block 和残差分支。"""

        return self.relu(self.tcn1(self.gcn1(x)) + self.residual(x))


class AAGCNModel(nn.Module):
    """现代 PyTorch 版官方 AAGCN 单流分类器。"""

    def __init__(
        self,
        num_class: int = 60,
        num_point: int = 25,
        num_person: int = 2,
        graph_layout: str = "ntu-rgb+d",
        graph_strategy: str = "spatial",
        in_channels: int = 3,
        drop_out: float = 0,
        adaptive: bool = True,
        attention: bool = True,
    ) -> None:
        super().__init__()
        if graph_strategy != "spatial":
            raise ValueError("Official AAGCN migration currently preserves only `spatial` graph strategy.")
        adjacency = torch.tensor(build_spatial_adjacency(graph_layout), dtype=torch.float32)
        self.num_class = num_class
        self.num_point = num_point
        self.num_person = num_person
        self.in_channels = in_channels
        self.data_bn = nn.BatchNorm1d(num_person * in_channels * num_point)

        self.l1 = TcnGcnUnit(in_channels, 64, adjacency, residual=False, adaptive=adaptive, attention=attention)
        self.l2 = TcnGcnUnit(64, 64, adjacency, adaptive=adaptive, attention=attention)
        self.l3 = TcnGcnUnit(64, 64, adjacency, adaptive=adaptive, attention=attention)
        self.l4 = TcnGcnUnit(64, 64, adjacency, adaptive=adaptive, attention=attention)
        self.l5 = TcnGcnUnit(64, 128, adjacency, stride=2, adaptive=adaptive, attention=attention)
        self.l6 = TcnGcnUnit(128, 128, adjacency, adaptive=adaptive, attention=attention)
        self.l7 = TcnGcnUnit(128, 128, adjacency, adaptive=adaptive, attention=attention)
        self.l8 = TcnGcnUnit(128, 256, adjacency, stride=2, adaptive=adaptive, attention=attention)
        self.l9 = TcnGcnUnit(256, 256, adjacency, adaptive=adaptive, attention=attention)
        self.l10 = TcnGcnUnit(256, 256, adjacency, adaptive=adaptive, attention=attention)

        self.fc = nn.Linear(256, num_class)
        nn.init.normal_(self.fc.weight, 0, math.sqrt(2.0 / num_class))
        bn_init(self.data_bn, 1)
        self.drop_out: nn.Module = nn.Dropout(drop_out) if drop_out else nn.Identity()

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """返回分类头之前的 AAGCN 池化特征。"""

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
        x = global_pool_person_mean(x, batch_size=batch_size, num_person=self.num_person)
        return self.drop_out(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """返回分类 logits。"""

        return self.fc(self.forward_features(x))


unit_gcn = UnitGCN
TCN_GCN_unit = TcnGcnUnit
Model = AAGCNModel
