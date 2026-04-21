"""官方 2s-AGCN split 数据适配器测试。"""

from __future__ import annotations

import pickle
from types import SimpleNamespace

import numpy as np
import torch

from two_stream_agcn.data import LegacySkeletonSplitDataset, build_legacy_split_datasets


def _write_split(root, split: str, *, count: int) -> None:
    """按官方文件命名创建一个极小旧 split。"""

    joint = np.arange(count * 3 * 4 * 5 * 2, dtype=np.float32).reshape(count, 3, 4, 5, 2)
    bone = joint + 100
    np.save(root / f"{split}_data_joint.npy", joint)
    np.save(root / f"{split}_data_bone.npy", bone)
    with (root / f"{split}_label.pkl").open("wb") as handle:
        pickle.dump(([f"{split}-{index}" for index in range(count)], list(range(count))), handle)


def test_legacy_dataset_returns_foundry_classification_item(tmp_path) -> None:
    """样本按 Foundry classification 形态暴露 ``inputs`` 和 ``target``。"""

    _write_split(tmp_path, "train", count=2)
    dataset = LegacySkeletonSplitDataset(
        {
            "joint": tmp_path / "train_data_joint.npy",
            "bone": tmp_path / "train_data_bone.npy",
        },
        tmp_path / "train_label.pkl",
        mmap_mode=None,
    )

    item = dataset[1]

    assert set(item["inputs"]) == {"joint", "bone"}
    assert item["inputs"]["joint"].shape == (3, 4, 5, 2)
    assert item["target"].dtype == torch.long
    assert item["sample_name"] == "train-1"


def test_legacy_builder_uses_explicit_params_without_compiling_semantics(tmp_path) -> None:
    """构建器只根据显式参数解析旧文件，不编译高层语义。"""

    _write_split(tmp_path, "train", count=2)
    _write_split(tmp_path, "val", count=1)
    dataset_config = SimpleNamespace(
        params={
            "data_root": str(tmp_path),
            "streams": ["joint", "bone"],
            "mmap_mode": None,
        }
    )

    train_dataset, val_dataset = build_legacy_split_datasets(dataset_config)

    assert len(train_dataset) == 2
    assert len(val_dataset) == 1
    assert set(train_dataset[0]["inputs"]) == {"joint", "bone"}


def test_legacy_builder_reads_foundry_dataset_config_root_and_memmap(tmp_path) -> None:
    """构建器兼容 Foundry 顶层 data_root 和 memmap 参数。"""

    _write_split(tmp_path, "train", count=1)
    _write_split(tmp_path, "val", count=1)
    dataset_config = SimpleNamespace(
        data_root=str(tmp_path),
        params={
            "streams": ["joint"],
            "memmap": False,
        },
    )

    train_dataset, val_dataset = build_legacy_split_datasets(dataset_config)

    assert len(train_dataset) == 1
    assert len(val_dataset) == 1
    assert set(train_dataset[0]["inputs"]) == {"joint"}
