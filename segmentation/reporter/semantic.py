from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, ClassVar

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from segmentation.reporter.base import BaseTrainingReporter, safe_metrics, to_float


def _convert_class_metrics(value: Any) -> dict[int, float] | None:
    if not isinstance(value, dict):
        return None
    converted: dict[int, float] = {}
    for key, val in value.items():
        try:
            cls_id = int(key)
        except (TypeError, ValueError):
            continue
        numeric = to_float(val)
        if numeric is not None:
            converted[cls_id] = numeric
    return converted or None


def format_metrics_report(
    epoch: int,
    train_stats: dict[str, Any] | None,
    val_stats: dict[str, Any] | None,
    lr: float | None,
    metric_order: tuple[str, ...] | None = None,
) -> str:
    cleaned_train = safe_metrics(train_stats)
    cleaned_val = safe_metrics(val_stats)
    raw_val = val_stats or {}

    lines: list[str] = []
    lines.append(f"RESULTS FOR EPOCH {epoch}")
    lines.append("=" * 60)
    lines.append("")

    train_loss = cleaned_train.pop("loss_total", None)
    if train_loss is not None:
        lines.append(f"Train Loss     : {train_loss:.6f}")

    val_loss = cleaned_val.pop("loss_total", None)
    if val_loss is not None:
        lines.append(f"Val Loss       : {val_loss:.6f}")

    if lr is not None:
        lines.append(f"Learning Rate  : {lr:.6g}")

    train_time = cleaned_train.pop("elapsed_time", None)
    if train_time is not None:
        lines.append(f"Train Time (s) : {train_time:.2f}")

    val_time = cleaned_val.pop("elapsed_time", None)
    if val_time is not None:
        lines.append(f"Val Time (s)   : {val_time:.2f}")

    if cleaned_val:
        lines.append("")
        shown: set[str] = set()
        for key in metric_order or ():
            if key in cleaned_val:
                lines.append(f"{key:14s}: {cleaned_val[key]:.6f}")
                shown.add(key)
        for key, value in cleaned_val.items():
            if key not in shown:
                lines.append(f"{key:14s}: {value:.6f}")

    class_iou = _convert_class_metrics(raw_val.get("class_iou") or raw_val.get("Class IoU"))
    class_dice = _convert_class_metrics(raw_val.get("class_dice") or raw_val.get("Class Dice"))
    if class_iou:
        lines.append("")
        lines.append("Per-class IoU:")
        for cls, val in class_iou.items():
            lines.append(f"  Class {cls}: {float(val):.6f}")
    if class_dice:
        lines.append("")
        lines.append("Per-class Dice:")
        for cls, val in class_dice.items():
            lines.append(f"  Class {cls}: {float(val):.6f}")

    return "\n".join(lines)


@dataclass
class SemanticTrainingReporter(BaseTrainingReporter):
    output_dir: Path
    monitor_metric: str = "mean_iou"
    plot_every: int = 1
    append: bool = True
    save_per_class: bool = True

    _HEADER: ClassVar[tuple[str, ...]] = (
        "epoch",
        "train/loss",
        "val/loss",
        "metrics/mean_iou",
        "metrics/mean_dice",
        "metrics/overall_acc",
        "metrics/mean_acc",
        "metrics/freqw_acc",
        "lr",
        "time/train",
        "time/val",
        "time/epoch",
    )

    _METRIC_COLUMN_MAP: ClassVar[dict[str, str]] = {
        "mean_iou": "metrics/mean_iou",
        "mean_dice": "metrics/mean_dice",
        "overall_acc": "metrics/overall_acc",
        "mean_acc": "metrics/mean_acc",
        "fwavcc": "metrics/freqw_acc",
    }

    def __post_init__(self) -> None:
        self.per_class_dir = Path(self.output_dir) / "per_class"
        super().__post_init__()

    @property
    def header(self) -> list[str]:
        return list(self._HEADER)

    @property
    def metric_column_map(self) -> dict[str, str]:
        return dict(self._METRIC_COLUMN_MAP)

    def build_row(
        self,
        epoch: int,
        train_stats: dict[str, float],
        val_stats: dict[str, float],
        lr: float | None,
    ) -> dict[str, Any]:
        train_time = train_stats.get("elapsed_time")
        val_time = val_stats.get("elapsed_time")
        epoch_time = (
            (train_time + val_time)
            if train_time is not None and val_time is not None
            else None
        )
        return {
            "epoch": epoch,
            "train/loss": train_stats.get("loss_total"),
            "val/loss": val_stats.get("loss_total"),
            "metrics/mean_iou": val_stats.get("mean_iou"),
            "metrics/mean_dice": val_stats.get("mean_dice"),
            "metrics/overall_acc": val_stats.get("overall_acc"),
            "metrics/mean_acc": val_stats.get("mean_acc"),
            "metrics/freqw_acc": val_stats.get("fwavcc"),
            "lr": lr,
            "time/train": train_time,
            "time/val": val_time,
            "time/epoch": epoch_time,
        }

    def format_report(
        self,
        epoch: int,
        train_stats: dict[str, Any] | None,
        val_stats: dict[str, Any] | None,
        lr: float | None,
    ) -> str:
        return format_metrics_report(
            epoch=epoch,
            train_stats=train_stats,
            val_stats=val_stats,
            lr=lr,
            metric_order=tuple(self._METRIC_COLUMN_MAP),
        )

    def train_stats_from_row(self, row: dict[str, Any]) -> dict[str, float]:
        out: dict[str, float] = {}
        _MAP = {"train/loss": "loss_total", "time/train": "elapsed_time"}
        for col, key in _MAP.items():
            v = row.get(col)
            if v is not None:
                out[key] = float(v)
        return out

    def val_stats_from_row(self, row: dict[str, Any]) -> dict[str, float]:
        out: dict[str, float] = {}
        _MAP = {
            "val/loss": "loss_total",
            "metrics/mean_iou": "mean_iou",
            "metrics/mean_dice": "mean_dice",
            "metrics/overall_acc": "overall_acc",
            "metrics/mean_acc": "mean_acc",
            "metrics/freqw_acc": "fwavcc",
            "time/val": "elapsed_time",
        }
        for col, key in _MAP.items():
            v = row.get(col)
            if v is not None:
                out[key] = float(v)
        return out

    def on_after_log_epoch(
        self,
        epoch: int,
        train_stats: dict[str, Any] | None,
        val_stats: dict[str, Any] | None,
    ) -> None:
        if not self.save_per_class or not isinstance(val_stats, dict):
            return
        class_iou = _convert_class_metrics(val_stats.get("class_iou"))
        class_dice = _convert_class_metrics(val_stats.get("class_dice"))
        class_payload = {
            "epoch": epoch,
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "class_iou": class_iou,
            "class_dice": class_dice,
        }
        if class_iou or class_dice:
            self.per_class_dir.mkdir(parents=True, exist_ok=True)
            with (self.per_class_dir / f"epoch_{epoch:03d}.json").open(
                "w", encoding="utf-8"
            ) as handle:
                json.dump(class_payload, handle, indent=2)

    def write_plots(self) -> None:
        if not self.history:
            return
        epochs = [row["epoch"] for row in self.history]

        def series(key: str) -> list[float | None]:
            return [row.get(key) for row in self.history]

        fig, axes = plt.subplots(2, 3, figsize=(13, 7))

        ax = axes[0, 0]
        ax.plot(epochs, series("train/loss"), label="train")
        ax.plot(epochs, series("val/loss"), label="val")
        ax.set_title("Loss")
        ax.set_xlabel("epoch")
        ax.set_ylabel("loss")
        ax.legend()

        ax = axes[0, 1]
        ax.plot(epochs, series("metrics/mean_iou"), label="mean IoU")
        ax.plot(epochs, series("metrics/mean_dice"), label="mean Dice")
        ax.set_title("Segmentation Metrics")
        ax.set_xlabel("epoch")
        ax.set_ylabel("score")
        ax.legend()

        ax = axes[0, 2]
        ax.plot(epochs, series("metrics/overall_acc"), label="overall acc")
        ax.plot(epochs, series("metrics/mean_acc"), label="mean acc")
        ax.set_title("Accuracy")
        ax.set_xlabel("epoch")
        ax.set_ylabel("score")
        ax.legend()

        ax = axes[1, 0]
        ax.plot(epochs, series("metrics/freqw_acc"), label="freqw acc")
        ax.set_title("FreqW Acc")
        ax.set_xlabel("epoch")
        ax.set_ylabel("score")
        ax.legend()

        ax = axes[1, 1]
        ax.plot(epochs, series("lr"), label="lr")
        ax.set_title("Learning Rate")
        ax.set_xlabel("epoch")
        ax.set_ylabel("lr")
        ax.legend()

        ax = axes[1, 2]
        ax.plot(epochs, series("time/epoch"), label="epoch time")
        ax.set_title("Epoch Time (s)")
        ax.set_xlabel("epoch")
        ax.set_ylabel("seconds")
        ax.legend()

        fig.tight_layout()
        fig.savefig(self.plot_path, dpi=160)
        plt.close(fig)
