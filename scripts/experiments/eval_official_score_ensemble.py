"""Evaluate official-style joint/bone score ensemble for NTU xview AGCN.

This script intentionally mirrors the original repository's workflow:
single-stream checkpoints are evaluated separately, their logits are saved, and
the final prediction is produced by score-level summation.
"""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from two_stream_agcn.data.legacy import LegacySkeletonSplitDataset
from two_stream_agcn.models.agcn import AGCNModel


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default="./data/ntu/xview")
    parser.add_argument("--joint-checkpoint", default="./artifacts/skeleton/ntu60_xview_agcn_joint_foundry/best.pt")
    parser.add_argument("--bone-checkpoint", default="./artifacts/skeleton/ntu60_xview_agcn_bone_foundry/best.pt")
    parser.add_argument("--output-dir", default="./artifacts/ensemble/ntu60_xview_agcn_official_scores")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--alpha", type=float, default=1.0)
    return parser.parse_args()


def load_model(checkpoint_path: Path, device: torch.device) -> AGCNModel:
    """Load a single-stream AGCN model from a Foundry checkpoint."""

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state_dict = checkpoint["model_state_dict"]
    model = AGCNModel().to(device)
    model.load_state_dict(state_dict)
    model.eval()
    return model


def build_val_loader(data_root: Path, stream: str, batch_size: int, num_workers: int) -> DataLoader:
    """Build validation loader for one official stream."""

    dataset = LegacySkeletonSplitDataset(
        {stream: data_root / f"val_data_{stream}.npy"},
        data_root / "val_label.pkl",
        mmap_mode="r",
        include_sample_name=True,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False,
    )


@torch.inference_mode()
def evaluate_stream(
    *,
    stream: str,
    checkpoint_path: Path,
    data_root: Path,
    output_dir: Path,
    batch_size: int,
    num_workers: int,
    device: torch.device,
) -> tuple[list[str], np.ndarray, np.ndarray]:
    """Evaluate one stream and save official-style score pickle."""

    model = load_model(checkpoint_path, device)
    loader = build_val_loader(data_root, stream, batch_size, num_workers)
    all_names: list[str] = []
    all_labels: list[np.ndarray] = []
    all_scores: list[np.ndarray] = []

    for batch in tqdm(loader, desc=f"eval {stream}", leave=True):
        inputs = batch["inputs"][stream].to(device, non_blocking=False)
        logits = model(inputs)
        all_names.extend(str(name) for name in batch["sample_name"])
        all_labels.append(batch["target"].cpu().numpy())
        all_scores.append(logits.float().cpu().numpy())

    labels = np.concatenate(all_labels, axis=0)
    scores = np.concatenate(all_scores, axis=0)
    score_dict = dict(zip(all_names, scores, strict=True))
    with (output_dir / f"{stream}_score.pkl").open("wb") as handle:
        pickle.dump(score_dict, handle)
    np.save(output_dir / f"{stream}_scores.npy", scores)
    np.save(output_dir / "labels.npy", labels)
    return all_names, labels, scores


def accuracy(scores: np.ndarray, labels: np.ndarray, top_k: int = 1) -> float:
    """Compute top-k accuracy."""

    if top_k == 1:
        return float((scores.argmax(axis=1) == labels).mean())
    ranks = np.argsort(scores, axis=1)[:, -top_k:]
    return float(np.mean([label in row for label, row in zip(labels, ranks, strict=True)]))


def save_summary(output_dir: Path, payload: dict[str, Any]) -> None:
    """Save a small text summary without adding another JSON dependency."""

    lines = [f"{key}: {value}" for key, value in payload.items()]
    (output_dir / "summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    """Run joint/bone score evaluation and fusion."""

    args = parse_args()
    data_root = Path(args.data_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    joint_names, labels, joint_scores = evaluate_stream(
        stream="joint",
        checkpoint_path=Path(args.joint_checkpoint),
        data_root=data_root,
        output_dir=output_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        device=device,
    )
    bone_names, bone_labels, bone_scores = evaluate_stream(
        stream="bone",
        checkpoint_path=Path(args.bone_checkpoint),
        data_root=data_root,
        output_dir=output_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        device=device,
    )
    if joint_names != bone_names:
        raise RuntimeError("Joint and bone score order mismatch.")
    if not np.array_equal(labels, bone_labels):
        raise RuntimeError("Joint and bone labels mismatch.")

    fused_scores = joint_scores + args.alpha * bone_scores
    np.save(output_dir / "fused_scores.npy", fused_scores)
    summary = {
        "joint_top1": accuracy(joint_scores, labels, top_k=1),
        "joint_top5": accuracy(joint_scores, labels, top_k=5),
        "bone_top1": accuracy(bone_scores, labels, top_k=1),
        "bone_top5": accuracy(bone_scores, labels, top_k=5),
        "fusion_alpha": args.alpha,
        "fusion_top1": accuracy(fused_scores, labels, top_k=1),
        "fusion_top5": accuracy(fused_scores, labels, top_k=5),
        "num_samples": int(labels.shape[0]),
    }
    save_summary(output_dir, summary)
    for key, value in summary.items():
        print(f"{key}: {value}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
