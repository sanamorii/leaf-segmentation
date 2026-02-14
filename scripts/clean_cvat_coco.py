"""
Remove COCO images that have no annotations, and drop orphan annotations.
- Keep only images with >=1 annotation.
- Keep only annotations whose image_id is kept.
- Preserve all other top-level COCO fields (info, licenses, etc.) as-is.

- --prune-unused-categories: remove categories not referenced by remaining annotations
- --reindex-ids: reassign image and annotation IDs to be contiguous (OFF by default)
- --max-images N: after cleaning, keep at most N images (and their annotations)
"""

import argparse
import json
from pathlib import Path
from collections import Counter


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(obj: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def clean_coco(
    coco: dict,
    prune_unused_categories: bool = False,
    reindex_ids: bool = False,
    max_images: int | None = None,
) -> tuple[dict, dict]:
    images = coco.get("images", [])
    annotations = coco.get("annotations", [])
    categories = coco.get("categories", [])

    # Basic structure checks
    if not isinstance(images, list) or not isinstance(annotations, list):
        raise ValueError("COCO JSON must contain list fields 'images' and 'annotations'.")

    # Build image_id -> image mapping
    img_by_id = {}
    dup_image_ids = []
    for img in images:
        if not isinstance(img, dict) or "id" not in img:
            continue
        iid = img["id"]
        if iid in img_by_id:
            dup_image_ids.append(iid)
        img_by_id[iid] = img

    # Count annotations per image_id (and track orphans)
    ann_image_ids = []
    orphan_anns_missing_image = 0
    for ann in annotations:
        if not isinstance(ann, dict) or "image_id" not in ann:
            continue
        iid = ann["image_id"]
        ann_image_ids.append(iid)
        if iid not in img_by_id:
            orphan_anns_missing_image += 1

    ann_counts = Counter(ann_image_ids)

    # Keep images with >= 1 annotation
    keep_image_ids = {iid for iid, c in ann_counts.items() if c > 0 and iid in img_by_id}

    # Filter images and annotations accordingly
    filtered_images = [img for img in images if isinstance(img, dict) and img.get("id") in keep_image_ids]
    filtered_annotations = [
        ann for ann in annotations
        if isinstance(ann, dict) and ann.get("image_id") in keep_image_ids
    ]
    removed_images_no_annotations = len(images) - len(filtered_images)

    # Optionally cap retained images (preserve current image order)
    removed_images_due_to_cap = 0
    if max_images is not None and len(filtered_images) > max_images:
        capped_images = filtered_images[:max_images]
        capped_image_ids = {img["id"] for img in capped_images}
        removed_images_due_to_cap = len(filtered_images) - len(capped_images)
        filtered_images = capped_images
        filtered_annotations = [
            ann for ann in filtered_annotations
            if isinstance(ann, dict) and ann.get("image_id") in capped_image_ids
        ]

    # Optionally prune categories not used by remaining annotations
    filtered_categories = categories
    pruned_category_ids = set()
    if prune_unused_categories and isinstance(categories, list):
        used_cat_ids = {ann.get("category_id") for ann in filtered_annotations if "category_id" in ann}
        used_cat_ids.discard(None)
        filtered_categories = [cat for cat in categories if isinstance(cat, dict) and cat.get("id") in used_cat_ids]
        pruned_category_ids = {cat.get("id") for cat in categories if isinstance(cat, dict)} - {
            cat.get("id") for cat in filtered_categories if isinstance(cat, dict)
        }

    # Optionally reindex IDs (off by default; can break external references)
    image_id_map = {}
    ann_id_map = {}
    if reindex_ids:
        # Reindex images
        new_images = []
        for new_i, img in enumerate(filtered_images, start=1):
            old_id = img["id"]
            image_id_map[old_id] = new_i
            img2 = dict(img)
            img2["id"] = new_i
            new_images.append(img2)

        # Reindex annotations (and update image_id accordingly)
        new_annotations = []
        for new_a, ann in enumerate(filtered_annotations, start=1):
            old_ann_id = ann.get("id", None)
            ann2 = dict(ann)
            ann2["image_id"] = image_id_map[ann2["image_id"]]
            ann2["id"] = new_a
            if old_ann_id is not None:
                ann_id_map[old_ann_id] = new_a
            new_annotations.append(ann2)

        filtered_images = new_images
        filtered_annotations = new_annotations

        # If categories were pruned, keep their IDs as-is (safer). If you need category reindexing, add it explicitly.

    # Construct output while preserving any other COCO top-level fields
    out = dict(coco)
    out["images"] = filtered_images
    out["annotations"] = filtered_annotations
    if "categories" in coco:
        out["categories"] = filtered_categories

    stats = {
        "images_before": len(images),
        "images_after": len(filtered_images),
        "annotations_before": len(annotations),
        "annotations_after": len(filtered_annotations),
        "removed_images_no_annotations": removed_images_no_annotations,
        "removed_images_due_to_cap": removed_images_due_to_cap,
        "orphan_annotations_missing_image_in_input": orphan_anns_missing_image,
        "pruned_categories": len(pruned_category_ids) if prune_unused_categories else 0,
        "reindexed_ids": bool(reindex_ids),
    }
    return out, stats


def main():
    p = argparse.ArgumentParser(description="Remove COCO images with no annotations and output cleaned JSON.")
    p.add_argument("-i", "--input", required=True, type=Path, help="Path to input COCO JSON.")
    p.add_argument("-o", "--output", required=True, type=Path, help="Path to output cleaned COCO JSON.")
    p.add_argument("--prune-unused-categories", action="store_true", help="Drop categories not used by remaining annotations.")
    p.add_argument("--reindex-ids", action="store_true", help="Reindex image/annotation IDs to be contiguous (off by default).")
    p.add_argument("--max-images", type=int, default=None, help="Keep at most N images after cleaning.")
    args = p.parse_args()

    if args.max_images is not None and args.max_images < 0:
        p.error("--max-images must be >= 0")

    coco = load_json(args.input)
    cleaned, stats = clean_coco(
        coco,
        prune_unused_categories=args.prune_unused_categories,
        reindex_ids=args.reindex_ids,
        max_images=args.max_images,
    )
    save_json(cleaned, args.output)

    print("Wrote cleaned COCO JSON:", str(args.output))
    print("---- Summary ----")
    for k, v in stats.items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    main()
