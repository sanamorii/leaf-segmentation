import torch

from leaf_seg.models.utils import load_ckpt
from leaf_seg.common.config import SemanticEvalConfig


def load_model(cfg: SemanticEvalConfig):
    checkpoint = load_ckpt(path=cfg.checkpoint)
    checkpoint

    
    ...

def evaluate(cfg):
    ...