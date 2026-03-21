import random
import numpy as np
from pathlib import Path
from collections import defaultdict
from PIL import Image

def class_name(instance_name: str) -> str:
    return instance_name.split("_", 1)[0] if "_" in instance_name else instance_name

def random_split_indices(
    n: int, val_ratio: float, seed: int,
) -> tuple[list[int], list[int]]:
    """Purely random train/val split."""
    rng = random.Random(seed)
    indices = list(range(n))
    rng.shuffle(indices)
    n_val = int(n * val_ratio)
    val_idx = set(indices[:n_val])
    train = [i for i in range(n) if i not in val_idx]
    val = [i for i in range(n) if i in val_idx]
    return train, val


def compute_frame_strata(
    mask_dir: Path, rel_paths: list[Path], instance_map: dict[int, str],
    n_leaf_bins: int = 3,
) -> list[int]:
    """Assign each frame a composite stratum based on per-class visibility.

    Encodes (pot_visible, stem_count, leaf_count_bin) into a single integer
    so that minority classes (Pot, Stem) are distributed evenly across splits,
    while also balancing the leaf count distribution.
    """
    ids_by_class: dict[str, set[int]] = defaultdict(set)
    for iid, name in instance_map.items():
        ids_by_class[class_name(name)].add(iid)

    pot_vis, stem_counts, leaf_counts = [], [], []
    for rel in rel_paths:
        unique_ids = set(np.unique(np.array(Image.open(mask_dir / rel), dtype=np.uint8)))
        pot_vis.append(1 if unique_ids & ids_by_class.get("Pot", set()) else 0)
        stem_counts.append(len(unique_ids & ids_by_class.get("Stem", set())))
        leaf_counts.append(len(unique_ids & ids_by_class.get("Leaf", set())))

    # Bin leaf counts into n_leaf_bins equal-width bins
    min_lc, max_lc = min(leaf_counts), max(leaf_counts)
    if min_lc == max_lc:
        leaf_bins = [0] * len(leaf_counts)
    else:
        bw = (max_lc - min_lc) / n_leaf_bins
        leaf_bins = [min(int((c - min_lc) / bw), n_leaf_bins - 1) for c in leaf_counts]

    # Composite stratum: (pot_visible, stem_count, leaf_bin) → single int
    max_stem = max(stem_counts) + 1
    return [
        pot * (max_stem * n_leaf_bins) + stem * n_leaf_bins + lb
        for pot, stem, lb in zip(pot_vis, stem_counts, leaf_bins)
    ]


def stratified_split_indices(
    strata: list[int], val_ratio: float, seed: int,
) -> tuple[list[int], list[int]]:
    """Split indices so each stratum is represented proportionally in train/val."""
    rng = random.Random(seed)

    groups = defaultdict(list)
    for i, s in enumerate(strata):
        groups[s].append(i)

    train, val = [], []
    for s in sorted(groups):
        indices = groups[s]
        rng.shuffle(indices)
        n_val = int(len(indices) * val_ratio)
        if n_val == 0 and len(indices) > 1 and val_ratio > 0:
            n_val = 1
        val.extend(indices[:n_val])
        train.extend(indices[n_val:])

    train.sort()
    val.sort()
    return train, val