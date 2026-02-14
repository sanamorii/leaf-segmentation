"""
Extensible training run reporter for leaf-seg. We allow for:
- logging per-epoch metrics
- Maintaining in-memory history of all logged epochs.
- Track best epoch according to chosen montior metric
- Generation of human readable reports + plots.
"""


from __future__ import annotations

import csv
import json
import math
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any


def to_float(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float)):
        return float(value)
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def format_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (int, float)):
        if math.isnan(value):
            return ""
        return f"{value:.6f}"
    return str(value)


def safe_metrics(metrics: dict[str, Any] | None) -> dict[str, float]:
    if not metrics:
        return {}
    cleaned: dict[str, float] = {}
    for key, value in metrics.items():
        if isinstance(value, dict):
            continue
        numeric = to_float(value)
        if numeric is not None:
            cleaned[key] = numeric
    return cleaned


@dataclass
class BaseTrainingReporter(ABC):
    output_dir: Path
    monitor_metric: str
    plot_every: int = 1
    append: bool = True

    def __post_init__(self) -> None:
        self.output_dir = Path(self.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.csv_path = self.output_dir / "results.csv"
        self.plot_path = self.output_dir / "results.png"
        self.last_report_path = self.output_dir / "metrics_last.txt"
        self.best_report_path = self.output_dir / "metrics_best.txt"
        self.history: list[dict[str, Any]] = []
        self.best_row: dict[str, Any] | None = None

        if self.csv_path.exists() and self.append:
            self._load_existing()
            self._hydrate_best_from_history()
        else:
            self._write_header()

    # Abstract interface
    @property
    @abstractmethod
    def header(self) -> list[str]:
        """Column names for the CSV."""

    @property
    @abstractmethod
    def metric_column_map(self) -> dict[str, str]:
        """Map from short metric name to CSV column name."""

    @abstractmethod
    def build_row(
        self,
        epoch: int,
        train_stats: dict[str, float],
        val_stats: dict[str, float],
        lr: float | None,
    ) -> dict[str, Any]:
        """Build a CSV row dict from cleaned train/val stats."""

    @abstractmethod
    def format_report(
        self,
        epoch: int,
        train_stats: dict[str, Any] | None,
        val_stats: dict[str, Any] | None,
        lr: float | None,
    ) -> str:
        """Produce a human-readable report string."""

    @abstractmethod
    def train_stats_from_row(self, row: dict[str, Any]) -> dict[str, float]:
        """Reconstruct raw train stats dict from a CSV row."""

    @abstractmethod
    def val_stats_from_row(self, row: dict[str, Any]) -> dict[str, float]:
        """Reconstruct raw val stats dict from a CSV row."""

    @abstractmethod
    def write_plots(self) -> None: ...

    # Optional hook
    def on_after_log_epoch(
        self,
        epoch: int,
        train_stats: dict[str, Any] | None,
        val_stats: dict[str, Any] | None,
    ) -> None:
        """Post execution hook after logging epoch.

        Args:
            epoch (int): current epoch
            train_stats (dict[str, Any] | None): running stats returned by train
            val_stats (dict[str, Any] | None): running stats returned by validate
        """
        return

    # getters - convenience 
    @property
    def best_epoch(self) -> int | None:
        if self.best_row is None:
            return None
        return int(self.best_row.get("epoch", 0))

    @property
    def best_value(self) -> float | None:
        if self.best_row is None:
            return None
        col = self.metric_column_map.get(self.monitor_metric)
        if col is None:
            return None
        return to_float(self.best_row.get(col))


    # CSV helpers
    def _write_header(self) -> None:
        with self.csv_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(self.header)

    def _load_existing(self) -> None:
        with self.csv_path.open("r", newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                parsed: dict[str, Any] = {}
                for key in self.header:
                    value = row.get(key)
                    if value in (None, ""):
                        parsed[key] = None
                        continue
                    numeric = to_float(value)
                    parsed[key] = numeric if numeric is not None else value
                self.history.append(parsed)

    def _append_row(self, row: dict[str, Any]) -> None:
        if not self.csv_path.exists():
            self._write_header()
        with self.csv_path.open("a", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow([format_value(row.get(key)) for key in self.header])


    # Best-epoch logic
    def _hydrate_best_from_history(self) -> None:
        metric_col = self.metric_column_map.get(self.monitor_metric)
        if not metric_col:
            return
        for row in self.history:
            self._maybe_update_best(row, metric_col)

    def _maybe_update_best(
        self, row: dict[str, Any], metric_col: str | None = None
    ) -> None:
        if metric_col is None:
            metric_col = self.metric_column_map.get(self.monitor_metric)
        if metric_col is None:
            return
        value = to_float(row.get(metric_col))
        if value is None:
            return
        current_best = (
            to_float(self.best_row.get(metric_col))
            if self.best_row is not None
            else None
        )
        if current_best is None or value > current_best:
            self.best_row = dict(row)


    # Main entry point
    def log_epoch(
        self,
        epoch: int,
        epochs: int,
        train_stats: dict[str, Any] | None,
        val_stats: dict[str, Any] | None,
        lr: float | None,
    ) -> None:
        """Main entry point. Log results of current epoch to reports and memory.

        Args:
            epoch (int): current epochs
            epochs (int): total epochs
            train_stats (dict[str, Any] | None): running training stats of current epoch
            val_stats (dict[str, Any] | None): running validation stats of current epoch
            lr (float | None): current learning rate set by scheduler | none.
        """
        cleaned_train = safe_metrics(train_stats)
        cleaned_val = safe_metrics(val_stats)

        row = self.build_row(
            epoch=epoch,
            train_stats=cleaned_train,
            val_stats=cleaned_val,
            lr=lr,
        )
        self.history.append(row)
        self._append_row(row)

        self.on_after_log_epoch(
            epoch=epoch,
            train_stats=train_stats,
            val_stats=val_stats,
        )

        self._maybe_update_best(row)

        self._write_reports(
            epoch=epoch,
            train_stats=cleaned_train,
            val_stats=cleaned_val,
            lr=lr,
        )

        if self.plot_every > 0 and (
            epoch % self.plot_every == 0 or epoch == epochs
        ):
            self.write_plots()


    # Report writing
    def _write_reports(
        self,
        epoch: int,
        train_stats: dict[str, Any] | None,
        val_stats: dict[str, Any] | None,
        lr: float | None,
    ) -> None:
        last_report = self.format_report(
            epoch=epoch,
            train_stats=train_stats,
            val_stats=val_stats,
            lr=lr,
        )
        self.last_report_path.write_text(last_report, encoding="utf-8")

        if self.best_row is not None:
            best_train = self.train_stats_from_row(self.best_row)
            best_val = self.val_stats_from_row(self.best_row)
            best_report = self.format_report(
                epoch=int(self.best_row.get("epoch", 0)),
                train_stats=best_train,
                val_stats=best_val,
                lr=self.best_row.get("lr"),
            )
            self.best_report_path.write_text(best_report, encoding="utf-8")

    # Metadata
    def write_metadata(self, payload: dict[str, Any]) -> None:
        """Write training config to json.

        Args:
            payload (dict[str, Any]): training configuration
        """
        path = self.output_dir / "run_meta.json"
        with path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
