from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from leaf_seg.reporter.base import BaseTrainingReporter, safe_metrics


def format_instance_metrics_report(
    epoch: int,
    train_stats: dict[str, Any] | None,
    val_stats: dict[str, Any] | None,
    lr: float | None,
) -> str:
    cleaned_train = safe_metrics(train_stats)
    cleaned_val = safe_metrics(val_stats)
    lines: list[str] = []
    lines.append(f"INSTANCE RESULTS FOR EPOCH {epoch}")
    lines.append("=" * 60)
    lines.append("")

    train_loss = cleaned_train.get("loss_total")
    if train_loss is not None:
        lines.append(f"Train Loss     : {train_loss:.6f}")
    if cleaned_train.get("loss_mask") is not None:
        lines.append(f"Train Mask Loss: {cleaned_train['loss_mask']:.6f}")
    if cleaned_train.get("loss_classifier") is not None:
        lines.append(f"Train Cls Loss : {cleaned_train['loss_classifier']:.6f}")
    if cleaned_train.get("loss_box_reg") is not None:
        lines.append(f"Train Box Loss : {cleaned_train['loss_box_reg']:.6f}")

    if lr is not None:
        lines.append(f"Learning Rate  : {lr:.6g}")

    train_time = cleaned_train.get("elapsed_time")
    if train_time is not None:
        lines.append(f"Train Time (s) : {train_time:.2f}")

    val_time = cleaned_val.get("elapsed_time")
    if val_time is not None:
        lines.append(f"Val Time (s)   : {val_time:.2f}")

    metric_order = [
        "segm_AP",
        "segm_AP50",
        "segm_AP75",
        "segm_APs",
        "segm_APm",
        "segm_APl",
        "bbox_AP",
        "bbox_AP50",
        "bbox_AP75",
        "bbox_APs",
        "bbox_APm",
        "bbox_APl",
    ]
    shown_metric = False
    for key in metric_order:
        value = cleaned_val.get(key)
        if value is None:
            continue
        if not shown_metric:
            lines.append("")
            shown_metric = True
        lines.append(f"{key:14s}: {value:.6f}")
    return "\n".join(lines)


@dataclass
class InstanceTrainingReporter(BaseTrainingReporter):
    output_dir: Path
    monitor_metric: str = "segm_AP"
    plot_every: int = 1
    append: bool = True

    _HEADER: ClassVar[tuple[str, ...]] = (
        "epoch",
        "train/loss_total",
        "train/loss_classifier",
        "train/loss_box_reg",
        "train/loss_mask",
        "val/segm_AP",
        "val/segm_AP50",
        "val/segm_AP75",
        "val/segm_APs",
        "val/segm_APm",
        "val/segm_APl",
        "val/bbox_AP",
        "val/bbox_AP50",
        "val/bbox_AP75",
        "val/bbox_APs",
        "val/bbox_APm",
        "val/bbox_APl",
        "lr",
        "time/train",
        "time/val",
        "time/epoch",
    )

    _METRIC_COLUMN_MAP: ClassVar[dict[str, str]] = {
        "segm_AP": "val/segm_AP",
        "segm_AP50": "val/segm_AP50",
        "segm_AP75": "val/segm_AP75",
        "bbox_AP": "val/bbox_AP",
        "bbox_AP50": "val/bbox_AP50",
        "bbox_AP75": "val/bbox_AP75",
    }

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
            "train/loss_total": train_stats.get("loss_total"),
            "train/loss_classifier": train_stats.get("loss_classifier"),
            "train/loss_box_reg": train_stats.get("loss_box_reg"),
            "train/loss_mask": train_stats.get("loss_mask"),
            "val/segm_AP": val_stats.get("segm_AP"),
            "val/segm_AP50": val_stats.get("segm_AP50"),
            "val/segm_AP75": val_stats.get("segm_AP75"),
            "val/segm_APs": val_stats.get("segm_APs"),
            "val/segm_APm": val_stats.get("segm_APm"),
            "val/segm_APl": val_stats.get("segm_APl"),
            "val/bbox_AP": val_stats.get("bbox_AP"),
            "val/bbox_AP50": val_stats.get("bbox_AP50"),
            "val/bbox_AP75": val_stats.get("bbox_AP75"),
            "val/bbox_APs": val_stats.get("bbox_APs"),
            "val/bbox_APm": val_stats.get("bbox_APm"),
            "val/bbox_APl": val_stats.get("bbox_APl"),
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
        return format_instance_metrics_report(
            epoch=epoch,
            train_stats=train_stats,
            val_stats=val_stats,
            lr=lr,
        )

    def train_stats_from_row(self, row: dict[str, Any]) -> dict[str, float]:
        out: dict[str, float] = {}
        _MAP = {
            "train/loss_total": "loss_total",
            "train/loss_classifier": "loss_classifier",
            "train/loss_box_reg": "loss_box_reg",
            "train/loss_mask": "loss_mask",
            "time/train": "elapsed_time",
        }
        for col, key in _MAP.items():
            v = row.get(col)
            if v is not None:
                out[key] = float(v)
        return out

    def val_stats_from_row(self, row: dict[str, Any]) -> dict[str, float]:
        out: dict[str, float] = {}
        _MAP = {
            "val/segm_AP": "segm_AP",
            "val/segm_AP50": "segm_AP50",
            "val/segm_AP75": "segm_AP75",
            "val/segm_APs": "segm_APs",
            "val/segm_APm": "segm_APm",
            "val/segm_APl": "segm_APl",
            "val/bbox_AP": "bbox_AP",
            "val/bbox_AP50": "bbox_AP50",
            "val/bbox_AP75": "bbox_AP75",
            "val/bbox_APs": "bbox_APs",
            "val/bbox_APm": "bbox_APm",
            "val/bbox_APl": "bbox_APl",
            "time/val": "elapsed_time",
        }
        for col, key in _MAP.items():
            v = row.get(col)
            if v is not None:
                out[key] = float(v)
        return out

    def write_plots(self) -> None:
        if not self.history:
            return
        epochs = [row["epoch"] for row in self.history]

        def series(key: str) -> list[float | None]:
            return [row.get(key) for row in self.history]

        fig, axes = plt.subplots(2, 3, figsize=(13, 7))

        ax = axes[0, 0]
        ax.plot(epochs, series("train/loss_total"), label="total")
        ax.set_title("Train Loss")
        ax.set_xlabel("epoch")
        ax.set_ylabel("loss")
        ax.legend()

        ax = axes[0, 1]
        ax.plot(epochs, series("train/loss_mask"), label="mask")
        ax.plot(epochs, series("train/loss_classifier"), label="cls")
        ax.plot(epochs, series("train/loss_box_reg"), label="box")
        ax.set_title("Train Loss Terms")
        ax.set_xlabel("epoch")
        ax.set_ylabel("loss")
        ax.legend()

        ax = axes[0, 2]
        ax.plot(epochs, series("val/segm_AP"), label="AP")
        ax.plot(epochs, series("val/segm_AP50"), label="AP50")
        ax.plot(epochs, series("val/segm_AP75"), label="AP75")
        ax.set_title("Segm AP")
        ax.set_xlabel("epoch")
        ax.set_ylabel("score")
        ax.legend()

        ax = axes[1, 0]
        ax.plot(epochs, series("val/bbox_AP"), label="AP")
        ax.plot(epochs, series("val/bbox_AP50"), label="AP50")
        ax.plot(epochs, series("val/bbox_AP75"), label="AP75")
        ax.set_title("BBox AP")
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
