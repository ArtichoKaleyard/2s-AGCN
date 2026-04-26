"""Sweep official score-fusion alpha for NTU60 xview AGCN.

The official 2s-AGCN ensemble uses ``joint_score + alpha * bone_score``. This
script sweeps alpha over frozen train/validation logits exported by
``train_offline_fusion_head.py``. It reports both a train-selected alpha and a
validation-oracle alpha so the strict result and the diagnostic upper reference
are not conflated.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np


def parse_args() -> argparse.Namespace:
    """Parse alpha sweep arguments.

    Returns:
        Parsed command-line arguments.
    """

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--score-dir", default="./artifacts/ensemble/ntu60_xview_agcn_fusion_head")
    parser.add_argument("--output-prefix", default="alpha_sweep")
    parser.add_argument("--start", type=float, default=0.0)
    parser.add_argument("--stop", type=float, default=2.0)
    parser.add_argument("--step", type=float, default=0.001)
    return parser.parse_args()


def load_split(score_dir: Path, split: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load cached logits for one split.

    Args:
        score_dir: Directory containing cached ``*_scores.npy`` files.
        split: Split name, usually ``train`` or ``val``.

    Returns:
        ``(labels, joint_scores, bone_scores)``.

    Raises:
        FileNotFoundError: If any required cache file is missing.
    """

    labels = np.load(score_dir / f"{split}_labels.npy")
    joint_scores = np.load(score_dir / f"{split}_joint_scores.npy")
    bone_scores = np.load(score_dir / f"{split}_bone_scores.npy")
    return labels, joint_scores, bone_scores


def accuracy(scores: np.ndarray, labels: np.ndarray, top_k: int = 1) -> float:
    """Compute top-k accuracy.

    Args:
        scores: Prediction scores with shape ``(N, C)``.
        labels: Integer labels with shape ``(N,)``.
        top_k: Accuracy rank.

    Returns:
        Accuracy as a Python float.
    """

    if top_k == 1:
        return float(np.mean(np.argmax(scores, axis=1) == labels))
    top_indices = np.argpartition(scores, kth=-top_k, axis=1)[:, -top_k:]
    return float(np.mean(np.any(top_indices == labels[:, None], axis=1)))


def alpha_grid(start: float, stop: float, step: float) -> np.ndarray:
    """Build an inclusive floating-point alpha grid.

    Args:
        start: First alpha value.
        stop: Last alpha value, included within rounding tolerance.
        step: Positive alpha step.

    Returns:
        One-dimensional alpha array.

    Raises:
        ValueError: If the range is invalid.
    """

    if step <= 0:
        raise ValueError("--step must be positive.")
    if stop < start:
        raise ValueError("--stop must be greater than or equal to --start.")
    count = int(np.floor((stop - start) / step + 0.5)) + 1
    return start + np.arange(count, dtype=np.float64) * step


def evaluate_alpha(
    alpha: float,
    labels: np.ndarray,
    joint_scores: np.ndarray,
    bone_scores: np.ndarray,
) -> tuple[float, float]:
    """Evaluate one alpha value.

    Args:
        alpha: Bone stream multiplier.
        labels: Ground-truth labels.
        joint_scores: Joint-stream logits.
        bone_scores: Bone-stream logits.

    Returns:
        ``(top1, top5)``.
    """

    fused_scores = joint_scores + alpha * bone_scores
    return accuracy(fused_scores, labels, top_k=1), accuracy(fused_scores, labels, top_k=5)


def pick_best(rows: list[dict[str, float]], key_prefix: str) -> dict[str, float]:
    """Pick the best row by top-1 and then top-5.

    Args:
        rows: Sweep rows.
        key_prefix: Metric prefix, either ``train`` or ``val``.

    Returns:
        Best row dictionary.
    """

    top1_key = f"{key_prefix}_top1"
    top5_key = f"{key_prefix}_top5"
    return max(rows, key=lambda row: (row[top1_key], row[top5_key], -abs(row["alpha"] - 1.0)))


def save_csv(path: Path, rows: list[dict[str, float]]) -> None:
    """Write all sweep rows to CSV.

    Args:
        path: Destination CSV file.
        rows: Sweep rows.
    """

    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("alpha", "train_top1", "train_top5", "val_top1", "val_top5"),
        )
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    """Run alpha sweep and save summary files."""

    args = parse_args()
    score_dir = Path(args.score_dir)
    train_labels, train_joint, train_bone = load_split(score_dir, "train")
    val_labels, val_joint, val_bone = load_split(score_dir, "val")
    rows: list[dict[str, float]] = []

    for alpha in alpha_grid(float(args.start), float(args.stop), float(args.step)):
        train_top1, train_top5 = evaluate_alpha(alpha, train_labels, train_joint, train_bone)
        val_top1, val_top5 = evaluate_alpha(alpha, val_labels, val_joint, val_bone)
        rows.append(
            {
                "alpha": float(alpha),
                "train_top1": train_top1,
                "train_top5": train_top5,
                "val_top1": val_top1,
                "val_top5": val_top5,
            }
        )

    fixed_one = min(rows, key=lambda row: abs(row["alpha"] - 1.0))
    train_best = pick_best(rows, "train")
    val_best = pick_best(rows, "val")
    summary: dict[str, Any] = {
        "score_dir": str(score_dir),
        "start": float(args.start),
        "stop": float(args.stop),
        "step": float(args.step),
        "num_alphas": len(rows),
        "fixed_alpha_1": fixed_one,
        "train_selected": train_best,
        "val_oracle": val_best,
    }

    csv_path = score_dir / f"{args.output_prefix}.csv"
    json_path = score_dir / f"{args.output_prefix}_summary.json"
    save_csv(csv_path, rows)
    json_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    for key, value in summary.items():
        print(f"{key}: {value}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
