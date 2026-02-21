from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional

SplitKind = Literal["ratio", "files", "none"]

@dataclass(frozen=True, kw_only=True)
class DatasetSpec:
    name: str
    root: Path
    task: Literal["semantic", "instance"]

@dataclass(frozen=True, kw_only=True)
class SemanticDatasetSpec(DatasetSpec):
    image_dir: str = "gt"
    mask_dir: str = "masks"
    ext: str = "png"
    # manifest  ## e.g. root/manifest.jsonl

@dataclass(frozen=True, kw_only=True)
class InstanceDatasetSpec(DatasetSpec):
    image_dir: str = "gt"
    ann: Path
    ext: str = "png"
    remap: bool = True
    filter_empty: bool = True
    # manifest  ## e.g. root/manifest.jsonl

@dataclass(frozen=True, kw_only=True)
class SplitSpec:
    kind: SplitKind
    seed: int = 42
    train_ratio: float = 0.8
    val_ratio: float = 0.2
    train_files: Optional[Path] = None
    val_files: Optional[Path] = None
    