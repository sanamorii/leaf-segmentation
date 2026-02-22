from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional

SplitKind = Literal["ratio", "files", "none"]

@dataclass(frozen=True, kw_only=True)
class DatasetSpec:
    name: str
    root: Path
    task: Literal["semantic", "instance"]
    train_set: Path
    val_set: Path
    ext: str = "png"

@dataclass(frozen=True, kw_only=True)
class SemanticDatasetSpec(DatasetSpec):
    image_dir: str = "gt"
    mask_dir: str = "masks"
    manifest : Optional[Path] = None  ## e.g. root/manifest.jsonl

@dataclass(frozen=True, kw_only=True)
class InstanceDatasetSpec(DatasetSpec):
    image_dir: str = "gt"
    ann: Optional[Path] = None
    remap: bool = True
    filter_empty: bool = True
    # manifest  ## e.g. root/coco.json

# @dataclass(frozen=True, kw_only=True)
# class SplitSpec:
#     kind: SplitKind
#     seed: int = 42
#     train_ratio: float = 0.8
#     val_ratio: float = 0.2
#     train_files: Optional[Path] = None
#     val_files: Optional[Path] = None
    