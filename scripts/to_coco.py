"""
Convert PlantDreamer's instance segementation dataset format into a valid COCO format.
"""
import argparse
import json
import os
from pathlib import Path

import numpy as np
from PIL import Image

# COCO RLE encoding
from pycocotools import mask as mask_utils


def build_categories(instance_id_to_name, split_instance_categories: bool):
    """
    Returns:
      categories: list[dict] with COCO categories
      instid_to_catid: dict[int -> int]
    """
    if split_instance_categories:
        # each instance name becomes its own COCO category, e.g. Leaf_0, Leaf_1, Pot_0...
        unique = sorted(set(instance_id_to_name.values()))
        name_to_catid = {name: i + 1 for i, name in enumerate(unique)}
        categories = [{"id": cid, "name": name, "supercategory": name.split("_")[0]} for name, cid in name_to_catid.items()]
        instid_to_catid = {iid: name_to_catid[nm] for iid, nm in instance_id_to_name.items()}
        return categories, instid_to_catid

    # default: group instances by their prefix (Leaf/Pot/Stem)
    super_names = sorted(set(nm.split("_")[0] for nm in instance_id_to_name.values()))
    super_to_catid = {s: i + 1 for i, s in enumerate(super_names)}
    categories = [{"id": cid, "name": s, "supercategory": s} for s, cid in super_to_catid.items()]
    instid_to_catid = {iid: super_to_catid[nm.split("_")[0]] for iid, nm in instance_id_to_name.items()}
    return categories, instid_to_catid


def rle_from_binary_mask(binary_mask: np.ndarray):
    """
    COCO RLE expects:
      - uint8
      - Fortran order (column-major)
    """
    if binary_mask.dtype != np.uint8:
        binary_mask = binary_mask.astype(np.uint8)
    rle = mask_utils.encode(np.asfortranarray(binary_mask))
    # pycocotools returns bytes for counts; COCO json needs str
    rle["counts"] = rle["counts"].decode("ascii")
    return rle


def bbox_from_binary_mask(binary_mask: np.ndarray):
    ys, xs = np.where(binary_mask > 0)
    if len(xs) == 0 or len(ys) == 0:
        return None
    x_min = int(xs.min())
    y_min = int(ys.min())
    x_max = int(xs.max())
    y_max = int(ys.max())
    return [x_min, y_min, x_max - x_min + 1, y_max - y_min + 1]  # [x,y,w,h]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=str, required=True, help="Path to dataset/ (contains gt/, mask/, annotations.json)")
    ap.add_argument("--out", type=str, required=True, help="Output COCO JSON path (e.g., coco_instances.json)")
    ap.add_argument("--image-exts", nargs="*", default=None, help="Optional list of image extensions to consider")
    ap.add_argument(
        "--split-instance-categories",
        action="store_true",
        help="If set, each instance label (Leaf_0, Leaf_1, ...) becomes its own COCO category.",
    )
    args = ap.parse_args()

    root = Path(args.root)
    gt_dir = root / "gt"
    mask_dir = root / "mask"
    ann_path = root / "annotations.json"

    if not gt_dir.exists():
        raise FileNotFoundError(f"Missing: {gt_dir}")
    if not mask_dir.exists():
        raise FileNotFoundError(f"Missing: {mask_dir}")
    if not ann_path.exists():
        raise FileNotFoundError(f"Missing: {ann_path}")

    # Load mapping: "1" -> "Leaf_0"
    with open(ann_path, "r") as f:
        raw = json.load(f)
    instance_id_to_name = {int(k): str(v) for k, v in raw.items()}

    categories, instid_to_catid = build_categories(
        instance_id_to_name,
        split_instance_categories=args.split_instance_categories,
    )

    # collect ground truth images
    if args.image_exts is None:
        # common image extensions
        exts = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
    else:
        exts = {e.lower() if e.startswith(".") else f".{e.lower()}" for e in args.image_exts}

    gt_files = sorted([p for p in gt_dir.iterdir() if p.suffix.lower() in exts and p.is_file()])
    if not gt_files:
        raise RuntimeError(f"No images found in {gt_dir} with extensions {sorted(exts)}")

    images = []
    annotations = []
    image_id = 1
    ann_id = 1

    for img_path in gt_files:
        mask_path = mask_dir / img_path.name
        if not mask_path.exists():
            # try same stem different extension: mask might be png while gt is jpg
            candidates = list(mask_dir.glob(img_path.stem + ".*"))
            if len(candidates) == 1:
                mask_path = candidates[0]
            else:
                print(f"[WARN] No matching mask for {img_path.name}, skipping.")
                continue

        # read image size
        with Image.open(img_path) as im:
            w, h = im.size

        images.append(
            {
                "id": image_id,
                "file_name": img_path.name,
                "width": w,
                "height": h,
            }
        )

        # read mask (8-bit)
        with Image.open(mask_path) as m:
            m = m.convert("L")
            mask = np.array(m, dtype=np.uint8)

        # for each non-zero instance id present in this mask, create a COCO annotation
        instance_vals = sorted([v for v in np.unique(mask).tolist() if v != 0])

        for inst_val in instance_vals:
            if inst_val not in instance_id_to_name:
                print(f"[WARN] Mask value {inst_val} not found in annotations.json; skipping that instance.")
                continue

            binary = (mask == inst_val).astype(np.uint8)
            area = int(binary.sum())
            if area == 0:
                continue

            bbox = bbox_from_binary_mask(binary)
            if bbox is None:
                continue

            rle = rle_from_binary_mask(binary)
            cat_id = instid_to_catid[inst_val]

            annotations.append(
                {
                    "id": ann_id,
                    "image_id": image_id,
                    "category_id": cat_id,
                    "segmentation": rle,   # RLE for instance segmentation
                    "area": area,
                    "bbox": bbox,
                    "iscrowd": 0,
                }
            )
            ann_id += 1

        image_id += 1

    coco = {
        "info": {"description": "Converted instance segmentation dataset", "version": "1.0"},
        "licenses": [],
        "images": images,
        "annotations": annotations,
        "categories": categories,
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(coco, f)

    print(f"Done.\nImages: {len(images)}\nAnnotations: {len(annotations)}\nCategories: {len(categories)}")
    print(f"Wrote: {out_path}")


if __name__ == "__main__":
    main()
