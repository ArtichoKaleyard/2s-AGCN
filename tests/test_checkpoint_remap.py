"""官方 checkpoint best-effort remap 测试。"""

from __future__ import annotations

import torch

from two_stream_agcn.checkpoints import load_checkpoint_best_effort, remap_official_state_dict
from two_stream_agcn.models import AGCNModel


def test_remap_strips_data_parallel_prefix_and_adds_stream_prefix() -> None:
    """加载前会规范化官方 ``module.`` 前缀。"""

    state = {"module.fc.weight": torch.ones(2, 3)}

    remapped = remap_official_state_dict(state, stream_prefix="models.joint")

    assert set(remapped) == {"models.joint.fc.weight"}


def test_best_effort_loader_skips_shape_mismatches() -> None:
    """不兼容 tensor 会被报告，而不是被强行转换。"""

    model = AGCNModel(num_class=5, num_point=25, num_person=2)
    checkpoint = {
        "state_dict": {
            "module.fc.weight": torch.zeros(4, 256),
            "module.fc.bias": torch.zeros(5),
            "module.not_a_key": torch.zeros(1),
        }
    }

    report = load_checkpoint_best_effort(model, checkpoint)

    assert "fc.bias" in report.loaded_keys
    assert "fc.weight" in report.shape_mismatched_keys
    assert "not_a_key" in report.unexpected_keys


def test_best_effort_loader_accepts_checkpoint_path(tmp_path) -> None:
    """best-effort loader 支持从 checkpoint 路径读取。"""

    model = AGCNModel(num_class=5, num_point=25, num_person=2)
    checkpoint_path = tmp_path / "checkpoint.pt"
    torch.save({"state_dict": {"module.fc.bias": torch.zeros(5)}}, checkpoint_path)

    report = load_checkpoint_best_effort(model, checkpoint_path)

    assert "fc.bias" in report.loaded_keys
