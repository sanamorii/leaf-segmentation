from contextlib import contextmanager
import os
from typing import Literal

import sys
from tqdm import tqdm

from torch.utils.data import DataLoader

def default_progress_enabled() -> bool:
    # tqdm writes to stderr; stderr.isatty() is usually the most relevant check
    return sys.stderr.isatty()

def resolve_progress_flag(progress: bool | None) -> bool:
    """
    progress is one of:
      - None: auto
      - True: forced on
      - False: forced off
    """
    if progress is None:
        return default_progress_enabled()
    return bool(progress)

def get_tqdm_bar(loader: DataLoader, epoch: int, epochs: int, stage: Literal['Train','Val'], progress: bool, leave:bool = False):
    return (
        tqdm(loader, desc=f"Epoch {epoch+1}/{epochs} [{stage}]", leave=leave) 
        if progress 
        else loader
        )


@contextmanager
def suppress_stout(enabled: bool = True):
    """
    suppress standard output (print) for methods (pycocotools)
    """
    if not enabled:
        yield
        return
    old_stdout = sys.stdout
    try:
        with open(os.devnull, "w") as devnull:
            sys.stdout = devnull
            yield
    finally:
        sys.stdout = old_stdout