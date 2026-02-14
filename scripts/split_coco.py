"""
Split coco into separate train/test files. 
"""
import argparse
import json
import random
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--coco", required=True, help="Path to COCO instances.json")
    ap.add_argument("--out-dir", required=True, help="Output directory")
    ap.add_argument("--val-ratio", type=float, default=0.2, help="Fraction of images for validation")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    random.seed(args.seed)

    with open(args.coco, "r") as f:
        coco = json.load(f)

    images = coco["images"]
    annotations = coco["annotations"]
    categories = coco["categories"]

    image_ids = [img["id"] for img in images]
    random.shuffle(image_ids)

    n_val = int(len(image_ids) * args.val_ratio)
    val_ids = set(image_ids[:n_val])
    train_ids = set(image_ids[n_val:])

    def split_images(imgs, keep_ids):
        return [img for img in imgs if img["id"] in keep_ids]

    def split_annotations(anns, keep_ids):
        return [ann for ann in anns if ann["image_id"] in keep_ids]

    coco_train = {
        "info": coco.get("info", {}),
        "licenses": coco.get("licenses", []),
        "images": split_images(images, train_ids),
        "annotations": split_annotations(annotations, train_ids),
        "categories": categories,
    }

    coco_val = {
        "info": coco.get("info", {}),
        "licenses": coco.get("licenses", []),
        "images": split_images(images, val_ids),
        "annotations": split_annotations(annotations, val_ids),
        "categories": categories,
    }

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_path = out_dir / "instances_train.json"
    val_path = out_dir / "instances_val.json"

    with open(train_path, "w") as f:
        json.dump(coco_train, f)
    with open(val_path, "w") as f:
        json.dump(coco_val, f)

    print(f"Train images: {len(coco_train['images'])}")
    print(f"Val images:   {len(coco_val['images'])}")
    print(f"Wrote:\n  {train_path}\n  {val_path}")


if __name__ == "__main__":
    main()
