"""迁移后 AGCN 系列模型接口的冒烟测试。"""

from __future__ import annotations

import torch

from two_stream_agcn.integration import build_aagcn_model, build_agcn_model
from two_stream_agcn.models import AAGCNModel, AGCNModel, TwoStreamSkeletonModel
from two_stream_agcn.models.graph import build_spatial_adjacency


def test_spatial_graph_shapes_match_official_layouts() -> None:
    """NTU 与 Kinetics 布局暴露官方三子集邻接矩阵。"""

    assert build_spatial_adjacency("ntu-rgb+d").shape == (3, 25, 25)
    assert build_spatial_adjacency("kinetics").shape == (3, 18, 18)


def test_agcn_forward_shape() -> None:
    """AGCN 为每个输入样本返回一行 logits。"""

    model = AGCNModel(num_class=7)
    model.eval()
    x = torch.randn(2, 3, 8, 25, 2)

    with torch.no_grad():
        logits = model(x)

    assert logits.shape == (2, 7)


def test_aagcn_forward_shape_without_attention() -> None:
    """关闭 attention 时 AAGCN 仍保持相同分类器接口。"""

    model = AAGCNModel(num_class=7, attention=False)
    model.eval()
    x = torch.randn(2, 3, 8, 25, 2)

    with torch.no_grad():
        logits = model(x)

    assert logits.shape == (2, 7)


def test_two_stream_sum_wrapper() -> None:
    """双流包装器在指定 sum 时对各流 logits 求和。"""

    wrapper = TwoStreamSkeletonModel(
        {
            "joint": lambda: AGCNModel(num_class=5),
            "bone": lambda: AGCNModel(num_class=5),
        },
        stream_mode="two_stream",
        fusion="sum",
        num_classes=5,
    )
    wrapper.eval()
    inputs = {
        "joint": torch.randn(2, 3, 8, 25, 2),
        "bone": torch.randn(2, 3, 8, 25, 2),
    }

    with torch.no_grad():
        logits = wrapper(inputs)

    assert logits.shape == (2, 5)


def test_single_stream_wrapper_uses_requested_stream_key() -> None:
    """单流包装器按 stream_mode 选择 stream，而不是依赖插入顺序。"""

    wrapper = TwoStreamSkeletonModel(
        {
            "joint": lambda: AGCNModel(num_class=5),
            "bone": lambda: AGCNModel(num_class=5),
        },
        stream_mode="bone",
        fusion=None,
        num_classes=5,
    )
    wrapper.eval()
    inputs = {
        "joint": torch.randn(2, 3, 8, 25, 2),
        "bone": torch.randn(2, 3, 8, 25, 2),
    }

    with torch.no_grad():
        expected = wrapper.models["bone"](inputs["bone"])
        actual = wrapper(inputs)

    assert torch.allclose(actual, expected)


def test_foundry_builders_return_project_models() -> None:
    """构建器函数消费具体参数，不调用编译器。"""

    agcn = build_agcn_model({"num_class": 4, "stream_mode": "joint"})
    aagcn = build_aagcn_model({"num_class": 4, "stream_mode": "joint", "attention": False})

    assert isinstance(agcn, AGCNModel)
    assert isinstance(aagcn, AAGCNModel)


def test_foundry_builders_move_model_to_requested_device() -> None:
    """构建器遵守 Foundry runtime 传入的 device。"""

    model = build_agcn_model({"num_class": 4, "stream_mode": "joint"}, device=torch.device("cpu"))

    assert next(model.parameters()).device.type == "cpu"


def test_aagcn_builder_ignores_foundry_generic_dropout_default() -> None:
    """AAGCN 构建器不把 Foundry 通用 dropout 默认值误映射为官方 drop_out。"""

    model = build_aagcn_model({"num_class": 4, "stream_mode": "joint", "dropout": 0.5})

    assert isinstance(model.drop_out, torch.nn.Identity)


def test_aagcn_rejects_broken_official_nonadaptive_branch() -> None:
    """AAGCN 不静默修复官方不可用的 adaptive=False 分支。"""

    try:
        AAGCNModel(adaptive=False)
    except ValueError as exc:
        assert "adaptive=False" in str(exc)
    else:
        raise AssertionError("AAGCNModel(adaptive=False) should be rejected.")
