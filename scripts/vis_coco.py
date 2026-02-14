"""
Visualise coco dataset
"""
import argparse
import json
import os
from pathlib import Path
from collections import defaultdict

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from pycocotools.coco import COCO
from pycocotools import mask as mask_utils


def _load_image(img_root: Path, file_name: str) -> Image.Image:
    path = img_root / file_name
    if not path.exists():
        raise FileNotFoundError(f"Image not found: {path}")
    return Image.open(path).convert("RGB")


def _ann_to_binary_mask(ann, height, width) -> np.ndarray:
    """
    Returns HxW uint8 mask {0,1} for one annotation.
    Handles RLE (compressed/uncompressed) and polygons.
    """
    seg = ann.get("segmentation", None)
    if seg is None:
        return np.zeros((height, width), dtype=np.uint8)

    # Polygon format: list[list[xy...]] or list[xy...]
    if isinstance(seg, list):
        # COCO polygons are either [ [x1,y1,...], [x1,y1,...], ... ] or [x1,y1,...]
        rles = mask_utils.frPyObjects(seg, height, width)
        rle = mask_utils.merge(rles)
        m = mask_utils.decode(rle)
        return (m > 0).astype(np.uint8)

    # RLE dict format
    if isinstance(seg, dict) and "counts" in seg and "size" in seg:
        rle = seg
        # pycocotools accepts either bytes or str for counts, but some installs are picky:
        if isinstance(rle["counts"], str):
            rle = dict(rle)
            rle["counts"] = rle["counts"].encode("ascii")
        m = mask_utils.decode(rle)
        return (m > 0).astype(np.uint8)

    return np.zeros((height, width), dtype=np.uint8)


def _alpha_blend(base_rgb: np.ndarray, mask: np.ndarray, color: np.ndarray, alpha: float) -> np.ndarray:
    """
    base_rgb: HxWx3 uint8
    mask: HxW {0,1}
    color: (3,) uint8
    alpha: float in [0,1]
    """
    out = base_rgb.astype(np.float32).copy()
    m = mask.astype(bool)
    out[m] = (1 - alpha) * out[m] + alpha * color.astype(np.float32)
    return np.clip(out, 0, 255).astype(np.uint8)


def _get_font(size=16):
    # Pillow default font fallback; tries a common system font if present
    try:
        return ImageFont.truetype("DejaVuSans.ttf", size=size)
    except Exception:
        return ImageFont.load_default()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--coco", required=True, help="Path to COCO instances JSON")
    ap.add_argument("--img-root", required=True, help="Directory containing the images referenced by file_name")
    ap.add_argument("--out-dir", default=None, help="If set, saves overlays here instead of showing interactively")
    ap.add_argument("--limit", type=int, default=0, help="Max number of images to process (0 = all)")
    ap.add_argument("--alpha", type=float, default=0.45, help="Mask overlay alpha")
    ap.add_argument("--draw-box", action="store_true", help="Draw bounding boxes")
    ap.add_argument("--seed", type=int, default=0, help="Random seed for instance colors")
    args = ap.parse_args()

    coco_path = Path(args.coco)
    img_root = Path(args.img_root)
    out_dir = Path(args.out_dir) if args.out_dir else None
    if out_dir:
        out_dir.mkdir(parents=True, exist_ok=True)

    coco = COCO(str(coco_path))

    # category_id -> name
    cats = coco.loadCats(coco.getCatIds())
    cat_id_to_name = {c["id"]: c["name"] for c in cats}

    img_ids = coco.getImgIds()
    if args.limit and args.limit > 0:
        img_ids = img_ids[: args.limit]

    rng = np.random.default_rng(args.seed)
    font = _get_font(16)

    for idx, img_id in enumerate(img_ids, start=1):
        img_info = coco.loadImgs([img_id])[0]
        pil_img = _load_image(img_root, img_info["file_name"])
        w, h = pil_img.size

        anns = coco.loadAnns(coco.getAnnIds(imgIds=[img_id]))

        base = np.array(pil_img, dtype=np.uint8)
        draw = ImageDraw.Draw(pil_img)

        # Draw each instance
        for ann in anns:
            mask = _ann_to_binary_mask(ann, h, w)
            if mask.sum() == 0:
                continue

            # deterministic random color per annotation id
            rng2 = np.random.default_rng(ann["id"] + args.seed * 1000003)
            color = rng2.integers(low=0, high=256, size=(3,), dtype=np.uint8)

            base = _alpha_blend(base, mask, color, args.alpha)

            # label at bbox top-left if available
            cat_name = cat_id_to_name.get(ann["category_id"], str(ann["category_id"]))
            bbox = ann.get("bbox", None)
            if bbox:
                x, y, bw, bh = bbox
                x0, y0 = int(x), int(y)

                if args.draw_box:
                    # draw rectangle on the PIL image (not on base array)
                    draw.rectangle([x, y, x + bw, y + bh], outline=(255, 255, 255), width=2)

                # text background + text
                text = f"{cat_name} #{ann['id']}"
                tw, th = draw.textbbox((0, 0), text, font=font)[2:]
                pad = 2
                draw.rectangle([x0, y0 - th - 2 * pad, x0 + tw + 2 * pad, y0], fill=(0, 0, 0))
                draw.text((x0 + pad, y0 - th - pad), text, fill=(255, 255, 255), font=font)

        # combine: put blended base back into a PIL image, then re-draw boxes/labels already on pil_img
        blended = Image.fromarray(base)
        # paste labels/boxes drawn on pil_img by compositing: easiest is just re-draw on blended
        final = blended
        final_draw = ImageDraw.Draw(final)

        # Re-draw boxes/labels (so they sit above the mask overlay)
        for ann in anns:
            bbox = ann.get("bbox", None)
            if not bbox:
                continue
            x, y, bw, bh = bbox
            if args.draw_box:
                final_draw.rectangle([x, y, x + bw, y + bh], outline=(255, 255, 255), width=2)
            cat_name = cat_id_to_name.get(ann["category_id"], str(ann["category_id"]))
            text = f"{cat_name} #{ann['id']}"
            x0, y0 = int(x), int(y)
            tw, th = final_draw.textbbox((0, 0), text, font=font)[2:]
            pad = 2
            final_draw.rectangle([x0, y0 - th - 2 * pad, x0 + tw + 2 * pad, y0], fill=(0, 0, 0))
            final_draw.text((x0 + pad, y0 - th - pad), text, fill=(255, 255, 255), font=font)

        if out_dir:
            out_path = out_dir / f"{Path(img_info['file_name']).stem}_overlay.png"
            final.save(out_path)
            print(f"[{idx}/{len(img_ids)}] saved {out_path}")
        else:
            print(f"[{idx}/{len(img_ids)}] showing image_id={img_id} file={img_info['file_name']}")
            final.show()


if __name__ == "__main__":
    main()
