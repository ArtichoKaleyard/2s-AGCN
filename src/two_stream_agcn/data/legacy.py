"""官方 2s-AGCN ``.npy + .pkl`` 数据布局适配器。"""

from __future__ import annotations

import pickle
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset


def _read_label_file(path: Path) -> tuple[list[str], list[int]]:
    """读取官方 label pickle 文件，并兼容 Python 2 pickle 编码。"""

    try:
        with path.open("rb") as handle:
            sample_names, labels = pickle.load(handle)
    except UnicodeDecodeError:
        with path.open("rb") as handle:
            sample_names, labels = pickle.load(handle, encoding="latin1")
    return list(sample_names), [int(label) for label in labels]


def _path_from_params(params: Mapping[str, Any], split: str, stream: str, data_root: Path) -> Path:
    """从显式参数和旧默认命名中解析某个 stream 的数据文件路径。"""

    candidates = (
        f"{split}_data_{stream}",
        f"{split}_{stream}_data",
        f"{stream}_{split}_data",
    )
    for key in candidates:
        value = params.get(key)
        if value:
            path = Path(value)
            return path if path.is_absolute() else data_root / path
    default = data_root / f"{split}_data_{stream}.npy"
    if default.exists():
        return default
    return data_root / f"{split}_{stream}.npy"


def _label_path_from_params(params: Mapping[str, Any], split: str, data_root: Path) -> Path:
    """从显式参数和旧默认命名中解析 label pickle 路径。"""

    for key in (f"{split}_label", f"{split}_label_path", f"{split}_labels"):
        value = params.get(key)
        if value:
            path = Path(value)
            return path if path.is_absolute() else data_root / path
    return data_root / f"{split}_label.pkl"


class LegacySkeletonSplitDataset(Dataset[dict[str, Any]]):
    """官方 2s-AGCN split 数组的数据集。

    参数：
        stream_files: stream 名到 ``.npy`` 文件路径的映射。数组应使用官方
            ``(N, C, T, V, M)`` 格式。
        label_file: 官方 pickle label 文件，包含 sample name 和整数标签。
        mmap_mode: 读取大规模旧数组时使用的可选 NumPy mmap 模式。
        include_sample_name: 是否在每个样本中包含 sample 元数据。
    """

    def __init__(
        self,
        stream_files: Mapping[str, str | Path],
        label_file: str | Path,
        *,
        mmap_mode: str | None = "r",
        include_sample_name: bool = True,
    ) -> None:
        super().__init__()
        if not stream_files:
            raise ValueError("At least one legacy skeleton stream file is required.")

        self.stream_names = tuple(stream_files)
        self.stream_files = {
            name: Path(path)
            for name, path in stream_files.items()
        }
        self.mmap_mode = mmap_mode
        self.sample_names, self.labels = _read_label_file(Path(label_file))
        self.include_sample_name = include_sample_name
        self.arrays: dict[str, np.ndarray] | None = None

        sizes = {
            name: int(np.load(path, mmap_mode=mmap_mode).shape[0])
            for name, path in self.stream_files.items()
        }
        sizes["labels"] = len(self.labels)
        if len(set(sizes.values())) != 1:
            raise ValueError(f"Legacy split has inconsistent sample counts: {sizes}")

    def _ensure_arrays_loaded(self) -> dict[str, np.ndarray]:
        """Lazily open legacy stream arrays.

        Spawn-based DataLoader workers need to pickle the dataset object before
        starting child processes. Holding open NumPy memmap arrays directly on
        the dataset makes that pickle extremely large and defeats the purpose of
        using worker processes. By reopening arrays lazily inside each process,
        the pickled dataset only carries paths and metadata.
        """

        if self.arrays is None:
            self.arrays = {
                name: np.load(path, mmap_mode=self.mmap_mode)
                for name, path in self.stream_files.items()
            }
        return self.arrays

    def __getstate__(self) -> dict[str, Any]:
        """Drop opened arrays before pickling for spawn/forkserver workers."""

        state = self.__dict__.copy()
        state["arrays"] = None
        return state

    def __len__(self) -> int:
        """返回当前 split 的样本数量。"""

        return len(self.labels)

    def __getitem__(self, index: int) -> dict[str, Any]:
        """返回符合 Foundry classification 约定的样本。"""

        arrays = self._ensure_arrays_loaded()
        inputs = {
            name: torch.as_tensor(np.array(array[index], copy=True), dtype=torch.float32)
            for name, array in arrays.items()
        }
        item: dict[str, Any] = {
            "inputs": inputs,
            "target": torch.tensor(self.labels[index], dtype=torch.long),
        }
        if self.include_sample_name:
            item["sample_name"] = self.sample_names[index]
            item["index"] = index
        return item


def build_legacy_split_datasets(
    dataset_config: Any,
    loader_config: Any | None = None,
) -> tuple[LegacySkeletonSplitDataset, LegacySkeletonSplitDataset]:
    """根据 Foundry dataset params 构建 train/validation 数据集。

    该适配器只翻译旧文件布局，不推断 dataset alias、protocol、stream 合法性或
    graph 合法性；这些高层语义仍由 Foundry skeleton 编译器负责。
    """

    del loader_config
    params = getattr(dataset_config, "params", dataset_config)
    if params is None:
        params = {}
    if not isinstance(params, Mapping):
        raise TypeError("Legacy 2s-AGCN dataset params must be a mapping.")

    configured_root = params.get("data_root", getattr(dataset_config, "data_root", "."))
    data_root = Path(configured_root)
    streams_value = params.get("streams", ("joint",))
    if isinstance(streams_value, str):
        streams: Sequence[str] = (streams_value,)
    else:
        streams = tuple(str(stream) for stream in streams_value)
    if "mmap_mode" in params:
        mmap_mode = params["mmap_mode"]
    else:
        mmap_mode = "r" if bool(params.get("memmap", True)) else None
    include_sample_name = bool(params.get("include_sample_name", True))

    def make_split(split: str) -> LegacySkeletonSplitDataset:
        stream_files = {
            stream: _path_from_params(params, split, stream, data_root)
            for stream in streams
        }
        return LegacySkeletonSplitDataset(
            stream_files,
            _label_path_from_params(params, split, data_root),
            mmap_mode=mmap_mode,
            include_sample_name=include_sample_name,
        )

    return make_split("train"), make_split("val")
