from dataclasses import dataclass

from leaf_seg.common.torch_utils import get_default_device


@dataclass(kw_only=True)
class BaseConfig:
    model: str
    encoder: str
    dataset: str
    batch_size: int = 8
    num_workers: int = 4
    num_classes: int
    device: str = get_default_device()
    
    output: str
    verbosity: int = 0

@dataclass(kw_only=True)
class InstanceTrainConfig(BaseConfig):
    model: str | None = None
    encoder: str | None = None

    lr: float = 1e-3
    epochs: int = 100
    gradient_clipping: float = 0.1
    patience: int = -1  # 0 <= off | 0 > on

    resume: str | None = None
    use_amp: bool = False

    output: str = "checkpoints/semantic/train"

    metric_to_track: str | None = "segm_AP"
    no_report: bool = False
    report_every: int = 1
    progress: bool | None = None  #TODO: rename to verbosity

@dataclass(kw_only=True)
class SemanticTrainConfig(BaseConfig):
    lr: float = 1e-3
    epochs: int = 100
    gradient_clipping: float = 0.1
    patience: int = -1  # 0 <= off | 0 > on

    resume: str | None = None
    use_amp: bool = False

    output: str = "checkpoints/semantic/train"

    metric_to_track: str | None = "mean_iou"
    no_report: bool = False
    report_every: int = 1
    progress: bool | None = None  #TODO: rename to verbosity

@dataclass(kw_only=True)
class SemanticFinetuneConfig(SemanticTrainConfig):
    output: str = "checkpoints/semantic/finetune"

    ckpt: str
    encoder_lr : float | None = None
    decoder_lr : float | None = None
    weight_decay: float = 1e-4
    freeze_encoder : bool = True
    freeze_epochs : int = 0
    strict_load: bool = True

@dataclass(kw_only=True)
class SemanticEvalConfig(BaseConfig):
    checkpoint: str
    num_classes: int
    output: str = "results/semantic"
    device: str = get_default_device()
    resize: tuple[int,int] = (512,512)

@dataclass(kw_only=True)
class SemanticInferConfig(BaseConfig):
    image: str | None = None
    images: str | None = None
    output: str = "results/infer_semantic"
    device: str = "cuda"
    resize: tuple[int,int] = (512,512)
