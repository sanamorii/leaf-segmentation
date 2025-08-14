import os
import cv2
import json
import numpy as np
from pycocotools import mask as maskUtils
from tqdm import tqdm

def create_coco_dataset(cvppp_dir, output_json):
    images = []
    annotations = []
    categories = [{"id": 1, "name": "leaf"}]
    ann_id = 1

    # All label masks in folder
    mask_files = sorted([f for f in os.listdir(cvppp_dir) if f.endswith("_label.png")])

    for img_id, mask_file in enumerate(tqdm(mask_files, desc="Processing CVPPP images"), start=1):
        plant_id = mask_file.replace("_label.png", "")
        img_file = plant_id + "_rgb.png"

        img_path = os.path.join(cvppp_dir, img_file)
        mask_path = os.path.join(cvppp_dir, mask_file)

        # Load RGB image
        img = cv2.imread(img_path)
        if img is None:
            print(f"Warning: RGB image not found for {mask_file}")
            continue

        height, width = img.shape[:2]
        images.append({
            "file_name": img_file,
            "height": height,
            "width": width,
            "id": img_id
        })

        # Load label mask as grayscale (leaf IDs as integers)
        mask_img = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask_img is None:
            print(f"Warning: Mask not found for {mask_file}")
            continue

        # Unique leaf IDs (exclude background 0)
        instance_ids = np.unique(mask_img)
        instance_ids = instance_ids[instance_ids != 0]

        for inst_id in instance_ids:
            binary_mask = (mask_img == inst_id).astype(np.uint8)

            # Bounding box
            y_indices, x_indices = np.where(binary_mask)
            if len(x_indices) == 0 or len(y_indices) == 0:
                continue
            x_min, x_max = x_indices.min(), x_indices.max()
            y_min, y_max = y_indices.min(), y_indices.max()
            bbox = [int(x_min), int(y_min), int(x_max - x_min), int(y_max - y_min)]

            # Encode mask as RLE
            rle = maskUtils.encode(np.asfortranarray(binary_mask))
            rle["counts"] = rle["counts"].decode("ascii")

            annotations.append({
                "id": ann_id,
                "image_id": img_id,
                "category_id": 1,
                "segmentation": rle,
                "area": int(np.sum(binary_mask)),
                "bbox": bbox,
                "iscrowd": 0
            })
            ann_id += 1

    coco_data = {
        "images": images,
        "annotations": annotations,
        "categories": categories
    }

    with open(output_json, "w") as f:
        json.dump(coco_data, f)

    print(f"COCO JSON saved to {output_json}")

if __name__ == "__main__":
    cvppp_dir = "data\\CVPPP\\CVPPP2017_LSC_training\\CVPPP2017_LSC_training\\training\\A1"  # <-- change this
    output_json = "cvppp_a1_train.json"
    create_coco_dataset(cvppp_dir, output_json)
