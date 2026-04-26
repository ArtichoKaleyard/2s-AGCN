"""Train a frozen-logit fusion head for NTU60 xview AGCN.

The official 2s-AGCN ensemble sums independently trained joint and bone scores.
This script keeps that single-stream training boundary intact: it loads the
existing joint/bone checkpoints, exports train/validation logits once, freezes
those logits, and trains only a lightweight fusion head on the training split.

The validation split is never used for fitting the head. This avoids the common
pitfall of learning an ensemble weight directly on the reported validation set.
"""

from __future__ import annotations

import argparse
import json
import pickle
import random
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from two_stream_agcn.data.legacy import LegacySkeletonSplitDataset
from two_stream_agcn.models.agcn import AGCNModel

SplitName = Literal["train", "val"]


class LinearFusionHead(nn.Module):
    """Linear classifier over concatenated joint and bone logits.

    Args:
        num_classes: Number of action classes in the target dataset.
    """

    def __init__(self, num_classes: int) -> None:
        super().__init__()
        self.classifier = nn.Linear(num_classes * 2, num_classes)

    def forward(self, joint_scores: torch.Tensor, bone_scores: torch.Tensor) -> torch.Tensor:
        """Return fused logits from fixed single-stream logits.

        Args:
            joint_scores: Joint-stream logits with shape ``(N, C)``.
            bone_scores: Bone-stream logits with shape ``(N, C)``.

        Returns:
            Fused logits with shape ``(N, C)``.
        """

        return self.classifier(torch.cat((joint_scores, bone_scores), dim=1))


class ScalarFusionHead(nn.Module):
    """Learn two global stream weights and per-class bias.

    This is a constrained alternative to ``LinearFusionHead``. It is close to
    official score summation, but learns the relative stream scale on the train
    split instead of fixing ``alpha=1.0`` by hand.

    Args:
        num_classes: Number of action classes in the target dataset.
    """

    def __init__(self, num_classes: int) -> None:
        super().__init__()
        self.logits = nn.Parameter(torch.zeros(2))
        self.bias = nn.Parameter(torch.zeros(num_classes))

    def forward(self, joint_scores: torch.Tensor, bone_scores: torch.Tensor) -> torch.Tensor:
        """Return weighted score fusion logits.

        Args:
            joint_scores: Joint-stream logits with shape ``(N, C)``.
            bone_scores: Bone-stream logits with shape ``(N, C)``.

        Returns:
            Fused logits with shape ``(N, C)``.
        """

        weights = F.softplus(self.logits)
        return weights[0] * joint_scores + weights[1] * bone_scores + self.bias


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for export and head training.

    Returns:
        Parsed argument namespace.
    """

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default="./data/ntu/xview")
    parser.add_argument("--joint-checkpoint", default="./artifacts/skeleton/ntu60_xview_agcn_joint_foundry/best.pt")
    parser.add_argument("--bone-checkpoint", default="./artifacts/skeleton/ntu60_xview_agcn_bone_foundry/best.pt")
    parser.add_argument("--output-dir", default="./artifacts/ensemble/ntu60_xview_agcn_fusion_head")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--head-device", default="cuda")
    parser.add_argument("--head", choices=("linear", "scalar"), default="linear")
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=49)
    parser.add_argument("--eval-every", type=int, default=25)
    parser.add_argument("--force-export", action="store_true")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch for repeatable head training.

    Args:
        seed: Integer random seed.
    """

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_model(checkpoint_path: Path, device: torch.device) -> AGCNModel:
    """Load a frozen AGCN single-stream classifier.

    Args:
        checkpoint_path: Foundry checkpoint containing ``model_state_dict``.
        device: Device used for backbone inference.

    Returns:
        Evaluation-mode AGCN model.

    Raises:
        KeyError: If the checkpoint does not contain ``model_state_dict``.
    """

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state_dict = checkpoint["model_state_dict"]
    model = AGCNModel().to(device)
    model.load_state_dict(state_dict)
    model.eval()
    return model


def build_loader(data_root: Path, split: SplitName, stream: str, batch_size: int, num_workers: int) -> DataLoader:
    """Build a deterministic loader for one split and stream.

    Args:
        data_root: Directory containing official ``.npy + .pkl`` files.
        split: Dataset split name.
        stream: Skeleton stream name, usually ``joint`` or ``bone``.
        batch_size: Evaluation batch size.
        num_workers: Number of DataLoader workers.

    Returns:
        A non-shuffled DataLoader preserving sample order.
    """

    dataset = LegacySkeletonSplitDataset(
        {stream: data_root / f"{split}_data_{stream}.npy"},
        data_root / f"{split}_label.pkl",
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
def export_stream_logits(
    *,
    split: SplitName,
    stream: str,
    checkpoint_path: Path,
    data_root: Path,
    output_dir: Path,
    batch_size: int,
    num_workers: int,
    device: torch.device,
    force: bool,
) -> tuple[list[str], np.ndarray, np.ndarray]:
    """Export or load cached logits for one split and stream.

    Args:
        split: Dataset split name.
        stream: Skeleton stream name.
        checkpoint_path: Checkpoint for the corresponding single-stream model.
        data_root: Directory containing split arrays and labels.
        output_dir: Directory used for cached logits.
        batch_size: Inference batch size.
        num_workers: DataLoader worker count.
        device: Device used for backbone inference.
        force: Recompute logits even if cache files already exist.

    Returns:
        ``(sample_names, labels, scores)`` in dataset order.

    Raises:
        RuntimeError: If cached files are incomplete.
    """

    names_path = output_dir / f"{split}_names.pkl"
    labels_path = output_dir / f"{split}_labels.npy"
    scores_path = output_dir / f"{split}_{stream}_scores.npy"
    if not force and names_path.exists() and labels_path.exists() and scores_path.exists():
        with names_path.open("rb") as handle:
            names = pickle.load(handle)
        return list(names), np.load(labels_path), np.load(scores_path)

    model = load_model(checkpoint_path, device)
    loader = build_loader(data_root, split, stream, batch_size, num_workers)
    all_names: list[str] = []
    all_labels: list[np.ndarray] = []
    all_scores: list[np.ndarray] = []

    for batch in tqdm(loader, desc=f"export {split} {stream}", leave=True):
        inputs = batch["inputs"][stream].to(device, non_blocking=False)
        logits = model(inputs)
        all_names.extend(str(name) for name in batch["sample_name"])
        all_labels.append(batch["target"].cpu().numpy())
        all_scores.append(logits.float().cpu().numpy())

    names = all_names
    labels = np.concatenate(all_labels, axis=0)
    scores = np.concatenate(all_scores, axis=0)
    with names_path.open("wb") as handle:
        pickle.dump(names, handle)
    np.save(labels_path, labels)
    np.save(scores_path, scores)
    return names, labels, scores


def export_split_logits(
    *,
    split: SplitName,
    args: argparse.Namespace,
    data_root: Path,
    output_dir: Path,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Export or load joint and bone logits for one split.

    Args:
        split: Dataset split name.
        args: Parsed runtime arguments.
        data_root: Directory containing official split files.
        output_dir: Cache/output directory.
        device: Device used for backbone inference.

    Returns:
        ``(labels, joint_scores, bone_scores)``.

    Raises:
        RuntimeError: If joint and bone sample orders or labels differ.
    """

    joint_names, labels, joint_scores = export_stream_logits(
        split=split,
        stream="joint",
        checkpoint_path=Path(args.joint_checkpoint),
        data_root=data_root,
        output_dir=output_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        device=device,
        force=bool(args.force_export),
    )
    bone_names, bone_labels, bone_scores = export_stream_logits(
        split=split,
        stream="bone",
        checkpoint_path=Path(args.bone_checkpoint),
        data_root=data_root,
        output_dir=output_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        device=device,
        force=bool(args.force_export),
    )
    if joint_names != bone_names:
        raise RuntimeError(f"{split} joint/bone sample order mismatch.")
    if not np.array_equal(labels, bone_labels):
        raise RuntimeError(f"{split} joint/bone label mismatch.")
    return labels, joint_scores, bone_scores


def accuracy(scores: torch.Tensor, labels: torch.Tensor, top_k: int = 1) -> float:
    """Compute top-k accuracy for tensors.

    Args:
        scores: Prediction logits with shape ``(N, C)``.
        labels: Integer labels with shape ``(N,)``.
        top_k: Accuracy rank.

    Returns:
        Accuracy as a Python float in ``[0, 1]``.
    """

    if top_k == 1:
        return float((scores.argmax(dim=1) == labels).float().mean().item())
    predictions = scores.topk(top_k, dim=1).indices
    return float((predictions == labels[:, None]).any(dim=1).float().mean().item())


def make_head(kind: str, num_classes: int) -> nn.Module:
    """Construct the requested fusion head.

    Args:
        kind: Fusion head kind.
        num_classes: Number of classes.

    Returns:
        Fusion head module.

    Raises:
        ValueError: If ``kind`` is unsupported.
    """

    if kind == "linear":
        return LinearFusionHead(num_classes)
    if kind == "scalar":
        return ScalarFusionHead(num_classes)
    raise ValueError(f"Unsupported fusion head: {kind}")


def train_head(
    *,
    head: nn.Module,
    train_labels: torch.Tensor,
    train_joint: torch.Tensor,
    train_bone: torch.Tensor,
    val_labels: torch.Tensor,
    val_joint: torch.Tensor,
    val_bone: torch.Tensor,
    args: argparse.Namespace,
) -> dict[str, Any]:
    """Train the fusion head on fixed train logits and evaluate on validation logits.

    Args:
        head: Trainable fusion head.
        train_labels: Train split labels.
        train_joint: Frozen joint train logits.
        train_bone: Frozen bone train logits.
        val_labels: Validation split labels.
        val_joint: Frozen joint validation logits.
        val_bone: Frozen bone validation logits.
        args: Parsed runtime arguments.

    Returns:
        Summary containing best/final metrics.
    """

    optimizer = torch.optim.AdamW(head.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay))
    best: dict[str, Any] = {"epoch": 0, "accuracy": 0.0, "top5": 0.0, "state_dict": None}

    for epoch in range(1, int(args.epochs) + 1):
        head.train()
        optimizer.zero_grad(set_to_none=True)
        logits = head(train_joint, train_bone)
        loss = F.cross_entropy(logits, train_labels)
        loss.backward()
        optimizer.step()

        should_eval = epoch == 1 or epoch == int(args.epochs) or epoch % int(args.eval_every) == 0
        if should_eval:
            head.eval()
            with torch.inference_mode():
                val_logits = head(val_joint, val_bone)
                val_loss = F.cross_entropy(val_logits, val_labels)
                val_acc = accuracy(val_logits, val_labels, top_k=1)
                val_top5 = accuracy(val_logits, val_labels, top_k=5)
            if val_acc >= float(best["accuracy"]):
                best = {
                    "epoch": epoch,
                    "accuracy": val_acc,
                    "top5": val_top5,
                    "state_dict": {key: value.detach().cpu() for key, value in head.state_dict().items()},
                }
            print(
                f"epoch={epoch} train_loss={loss.item():.6f} "
                f"val_loss={val_loss.item():.6f} val_acc={val_acc:.6f} val_top5={val_top5:.6f}"
            )

    head.eval()
    with torch.inference_mode():
        final_logits = head(val_joint, val_bone)
        final = {
            "accuracy": accuracy(final_logits, val_labels, top_k=1),
            "top5": accuracy(final_logits, val_labels, top_k=5),
        }
    return {
        "best_epoch": int(best["epoch"]),
        "best_accuracy": float(best["accuracy"]),
        "best_top5": float(best["top5"]),
        "best_state_dict": best["state_dict"],
        "final_accuracy": float(final["accuracy"]),
        "final_top5": float(final["top5"]),
    }


def to_tensor(array: np.ndarray, device: torch.device, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """Convert a NumPy array into a tensor on the requested device.

    Args:
        array: Source array.
        device: Target torch device.
        dtype: Target torch dtype.

    Returns:
        Tensor copy on ``device``.
    """

    return torch.as_tensor(array, dtype=dtype, device=device)


def main() -> int:
    """Run frozen-logit fusion head training."""

    args = parse_args()
    set_seed(int(args.seed))
    data_root = Path(args.data_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    export_device = torch.device(args.device)
    train_labels_np, train_joint_np, train_bone_np = export_split_logits(
        split="train",
        args=args,
        data_root=data_root,
        output_dir=output_dir,
        device=export_device,
    )
    val_labels_np, val_joint_np, val_bone_np = export_split_logits(
        split="val",
        args=args,
        data_root=data_root,
        output_dir=output_dir,
        device=export_device,
    )

    head_device = torch.device(args.head_device)
    train_labels = to_tensor(train_labels_np, head_device, dtype=torch.long)
    val_labels = to_tensor(val_labels_np, head_device, dtype=torch.long)
    train_joint = to_tensor(train_joint_np, head_device)
    train_bone = to_tensor(train_bone_np, head_device)
    val_joint = to_tensor(val_joint_np, head_device)
    val_bone = to_tensor(val_bone_np, head_device)

    num_classes = int(train_joint.shape[1])
    head = make_head(str(args.head), num_classes).to(head_device)
    baseline_sum = val_joint + val_bone
    baseline = {
        "sum_top1": accuracy(baseline_sum, val_labels, top_k=1),
        "sum_top5": accuracy(baseline_sum, val_labels, top_k=5),
    }
    result = train_head(
        head=head,
        train_labels=train_labels,
        train_joint=train_joint,
        train_bone=train_bone,
        val_labels=val_labels,
        val_joint=val_joint,
        val_bone=val_bone,
        args=args,
    )

    checkpoint = {
        "head": str(args.head),
        "seed": int(args.seed),
        "args": vars(args),
        "baseline": baseline,
        "result": {key: value for key, value in result.items() if key != "best_state_dict"},
        "head_state_dict": result["best_state_dict"],
    }
    torch.save(checkpoint, output_dir / f"{args.head}_fusion_head.pt")
    summary = {
        "head": str(args.head),
        "seed": int(args.seed),
        "num_train": int(train_labels.shape[0]),
        "num_val": int(val_labels.shape[0]),
        **baseline,
        **{key: value for key, value in result.items() if key != "best_state_dict"},
    }
    (output_dir / f"{args.head}_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    for key, value in summary.items():
        print(f"{key}: {value}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
