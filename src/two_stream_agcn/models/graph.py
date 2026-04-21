"""与官方 2s-AGCN 布局对齐的图构造工具。"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, slots=True)
class GraphSpec:
    """AGCN 系列模型使用的骨架图元数据。

    参数：
        layout: 规范化布局名称。
        num_node: 骨架关节点数量。
        inward: 子节点到父节点的有向边，使用零基索引。
        outward: 反向有向边，使用零基索引。
        self_link: 每个关节点的自环边。
    """

    layout: str
    num_node: int
    inward: tuple[tuple[int, int], ...]
    outward: tuple[tuple[int, int], ...]
    self_link: tuple[tuple[int, int], ...]


def edge_to_matrix(edges: tuple[tuple[int, int], ...], num_node: int) -> np.ndarray:
    """构造官方代码使用的有向邻接矩阵。"""

    adjacency = np.zeros((num_node, num_node), dtype=np.float32)
    for source, target in edges:
        adjacency[target, source] = 1.0
    return adjacency


def normalize_digraph(adjacency: np.ndarray) -> np.ndarray:
    """对有向图邻接矩阵做按列归一化。"""

    column_sum = np.sum(adjacency, axis=0)
    num_node = adjacency.shape[0]
    degree = np.zeros((num_node, num_node), dtype=np.float32)
    for index in range(num_node):
        if column_sum[index] > 0:
            degree[index, index] = column_sum[index] ** (-1)
    return np.dot(adjacency, degree).astype(np.float32)


def spatial_adjacency(spec: GraphSpec) -> np.ndarray:
    """返回官方三子集 spatial 邻接张量。"""

    identity = edge_to_matrix(spec.self_link, spec.num_node)
    inward = normalize_digraph(edge_to_matrix(spec.inward, spec.num_node))
    outward = normalize_digraph(edge_to_matrix(spec.outward, spec.num_node))
    return np.stack((identity, inward, outward)).astype(np.float32)


def get_graph_spec(layout: str) -> GraphSpec:
    """将 Foundry 或原仓布局名称解析为图元数据。

    参数：
        layout: 布局名称。支持 ``ntu-rgb+d``、``ntu_rgb_d``、``openpose18``
            和 ``kinetics``。

    返回：
        与官方 2s-AGCN 仓库一致的图元数据。

    异常：
        ValueError: 当布局未知时抛出。
    """

    normalized = layout.strip().lower().replace("_", "-")
    if normalized in {"ntu-rgb+d", "ntu-rgb-d"}:
        num_node = 25
        inward_ori = (
            (1, 2),
            (2, 21),
            (3, 21),
            (4, 3),
            (5, 21),
            (6, 5),
            (7, 6),
            (8, 7),
            (9, 21),
            (10, 9),
            (11, 10),
            (12, 11),
            (13, 1),
            (14, 13),
            (15, 14),
            (16, 15),
            (17, 1),
            (18, 17),
            (19, 18),
            (20, 19),
            (22, 23),
            (23, 8),
            (24, 25),
            (25, 12),
        )
        inward = tuple((source - 1, target - 1) for source, target in inward_ori)
        return GraphSpec(
            layout="ntu-rgb+d",
            num_node=num_node,
            inward=inward,
            outward=tuple((target, source) for source, target in inward),
            self_link=tuple((index, index) for index in range(num_node)),
        )
    if normalized in {"openpose18", "kinetics"}:
        num_node = 18
        inward = (
            (4, 3),
            (3, 2),
            (7, 6),
            (6, 5),
            (13, 12),
            (12, 11),
            (10, 9),
            (9, 8),
            (11, 5),
            (8, 2),
            (5, 1),
            (2, 1),
            (0, 1),
            (15, 0),
            (14, 0),
            (17, 15),
            (16, 14),
        )
        return GraphSpec(
            layout="openpose18",
            num_node=num_node,
            inward=inward,
            outward=tuple((target, source) for source, target in inward),
            self_link=tuple((index, index) for index in range(num_node)),
        )
    raise ValueError(f"Unsupported skeleton graph layout: {layout}")


def build_spatial_adjacency(layout: str) -> np.ndarray:
    """按布局名称构造官方 spatial 邻接张量。"""

    return spatial_adjacency(get_graph_spec(layout))
