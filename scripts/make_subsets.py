"""
Create shuffled percentage subsets of a paired segmentation dataset:

<dataset>/
  gt/
  mask/

Matching rule - Pair by filename *stem* (e.g., gt/img_001.jpg pairs with mask/img_001.png)

Output structure (example for 0.8 and 0.75):
<out_root>/
  80/
    gt/
    mask/
    manifest.json
  75/
    gt/
    mask/
    manifest.json
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from random import Random
from typing import Dict, List, Tuple


IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}


@dataclass(frozen=True)
class Pair:
    stem: str
    gt_path: str
    mask_path: str


def _is_image_file(p: Path) -> bool:
    return p.is_file() and p.suffix.lower() in IMG_EXTS


def _index_by_stem(folder: Path) -> Dict[str, Path]:
    out: Dict[str, Path] = {}
    for p in folder.iterdir():
        if _is_image_file(p):
            out[p.stem] = p
    return out


def _collect_pairs(gt_dir: Path, mask_dir: Path) -> List[Pair]:
    if not gt_dir.exists() or not gt_dir.is_dir():
        raise FileNotFoundError(f"Missing gt dir: {gt_dir}")
    if not mask_dir.exists() or not mask_dir.is_dir():
        raise FileNotFoundError(f"Missing mask dir: {mask_dir}")

    gt = _index_by_stem(gt_dir)
    masks = _index_by_stem(mask_dir)

    if not gt:
        raise RuntimeError(f"No image files found in gt dir: {gt_dir}")
    if not masks:
        raise RuntimeError(f"No image files found in mask dir: {mask_dir}")

    missing_masks = sorted(set(gt.keys()) - set(masks.keys()))
    missing_gts = sorted(set(masks.keys()) - set(gt.keys()))

    if missing_masks or missing_gts:
        msg = []
        if missing_masks:
            msg.append(f"{len(missing_masks)} gt stems missing masks (first 10): {missing_masks[:10]}")
        if missing_gts:
            msg.append(f"{len(missing_gts)} mask stems missing gts (first 10): {missing_gts[:10]}")
        raise RuntimeError("Pairing mismatch:\n  - " + "\n  - ".join(msg))

    pairs = [Pair(stem=s, gt_path=str(gt[s]), mask_path=str(masks[s])) for s in sorted(gt.keys())]
    return pairs


def _ensure_empty_dir(d: Path) -> None:
    d.mkdir(parents=True, exist_ok=True)


def _copy_or_link(src: Path, dst: Path, mode: str) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return
    if mode == "copy":
        shutil.copy2(src, dst)
    elif mode == "symlink":
        # Use relative symlinks for portability inside the output tree
        rel = os.path.relpath(src, start=dst.parent)
        os.symlink(rel, dst)
    else:
        raise ValueError(f"Unknown mode: {mode}")


def _write_manifest(out_dir: Path, fraction: float, seed: int, subset: List[Pair]) -> None:
    manifest = {
        "fraction": fraction,
        "percent_folder": out_dir.name,
        "seed": seed,
        "num_samples": len(subset),
        "samples": [asdict(p) for p in subset],
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", type=str, required=True, help="Path to dataset root containing gt/ and mask/")
    ap.add_argument(
        "--fractions",
        type=str,
        default="0.8",
        help="Comma-separated fractions, e.g. '0.8,0.75,0.5'",
    )
    ap.add_argument("--out_root", type=str, default=None, help="Where to write subsets (default: <dataset>/subsets)")
    ap.add_argument("--seed", type=int, default=42, help="Shuffle seed")
    ap.add_argument(
        "--rounding",
        choices=["floor", "round", "ceil"],
        default="floor",
        help="How to convert fraction*N into an integer count",
    )
    ap.add_argument(
        "--mode",
        choices=["copy", "symlink"],
        default="copy",
        help="How to place files in subsets",
    )
    ap.add_argument("--dry_run", action="store_true", help="Print what would be done without writing files")
    return ap.parse_args()


def main() -> int:
    args = parse_args()

    dataset = Path(args.dataset).resolve()
    gt_dir = dataset / "gt"
    mask_dir = dataset / "mask"

    fractions = []
    for x in args.fractions.split(","):
        x = x.strip()
        if not x:
            continue
        f = float(x)
        if not (0.0 < f <= 1.0):
            raise ValueError(f"Fraction must be in (0,1]: got {f}")
        fractions.append(f)

    if not fractions:
        raise ValueError("No valid fractions provided.")

    out_root = Path(args.out_root).resolve() if args.out_root else (dataset / "subsets")
    pairs = _collect_pairs(gt_dir, mask_dir)

    rng = Random(args.seed)
    rng.shuffle(pairs)

    n = len(pairs)

    def frac_to_k(f: float) -> int:
        raw = f * n
        if args.rounding == "floor":
            return int(math.floor(raw))
        if args.rounding == "ceil":
            return int(math.ceil(raw))
        return int(round(raw))

    print(f"Found {n} paired samples.")
    print(f"Shuffle seed: {args.seed}")
    print(f"Output root: {out_root}")
    print(f"Mode: {args.mode}")
    print(f"Rounding: {args.rounding}")
    print("Fractions:", fractions)

    for f in fractions:
        k = frac_to_k(f)
        # Ensure at least 1 if fraction > 0 and dataset not empty
        k = max(1, min(k, n))

        folder_name = str(int(round(f * 100)))
        out_dir = out_root / folder_name
        out_gt = out_dir / "gt"
        out_mask = out_dir / "mask"

        subset = pairs[:k]
        print(f"\n==> {folder_name}% subset: {k}/{n} samples -> {out_dir}")

        if args.dry_run:
            continue

        _ensure_empty_dir(out_gt)
        _ensure_empty_dir(out_mask)

        for p in subset:
            gt_src = Path(p.gt_path)
            mask_src = Path(p.mask_path)

            gt_dst = out_gt / gt_src.name
            mask_dst = out_mask / mask_src.name

            _copy_or_link(gt_src, gt_dst, args.mode)
            _copy_or_link(mask_src, mask_dst, args.mode)

        _write_manifest(out_dir, f, args.seed, subset)

    print("\nDone.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        raise
