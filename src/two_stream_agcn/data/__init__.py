"""官方 2s-AGCN 旧 split 文件的数据适配入口。"""

from .legacy import LegacySkeletonSplitDataset, build_legacy_split_datasets

__all__ = [
    "LegacySkeletonSplitDataset",
    "build_legacy_split_datasets",
]
