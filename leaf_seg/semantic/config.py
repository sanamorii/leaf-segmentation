from dataclasses import dataclass

from leaf_seg.common.torch_utils import get_default_device


@dataclass
class SemanticTrainConfig:
    model: str
    encoder: str
    dataset: str
    num_classes: str
    batch_size: int = 8
    num_workers: int = 4
    lr: float = 1e-3
    epochs: int = 100
    gradient_clipping: float = 0.1
    patience: int = -1

    device: str = get_default_device()
    resume: str | None = None
    use_amp: bool = False
    monitor_metric: str | None = "mean_iou"
    no_report: bool = False
    out: str = "checkpoints/semantic"

    progress: bool | None = None  #TODO: rename to verbosity
    verbosity: int = 0

@dataclass
class SemanticEvalConfig:
    model: str
    encoder: str
    checkpoint: str
    num_classes: int
    dataset: str
    output: str = "results/semantic"
    device: str = get_default_device()
    resize: tuple[int,int] = (512,512)
    verbosity: int = 0

@dataclass
class SemanticInferConfig:
    model: str
    encoder: str
    num_classes: str
    image: str | None = None
    images: str | None = None
    output: str = "results/infer_semantic"
    device: str = "cuda"
    resize: tuple[int,int] = (512,512)
    verbosity: int = 0