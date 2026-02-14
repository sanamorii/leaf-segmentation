"""
CVAT.ai helper - convert cityscapes format into plantdreamer format:
<dataset>/
    gt/
    mask/
"""
import os
import shutil
import argparse

parser = argparse.ArgumentParser("leaf-segmentation")
parser.add_argument("--src", default="default", type=str, help="model name")
parser.add_argument("--dst", default="mask", type=str, help="encoder name")
args = parser.parse_args()

def copy_and_rename_files(src_dir, dest_dir, suffix="_gtFine_labelIds"):
    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)

    #Copies files ending with the given suffix from src_dir into 
    # dst_dir and removes the suffix from the filename.
    for root, _, files in os.walk(src_dir):
        for file in files:
            if suffix in file:
                old_path = os.path.join(root, file)
                # remove suffix before extension
                name, ext = os.path.splitext(file)
                if name.endswith(suffix):
                    new_name = name[:-len(suffix)] + ext
                else:
                    new_name = file  # fallback, shouldn't happen
                
                new_path = os.path.join(dest_dir, new_name)

                shutil.copy2(old_path, new_path)
                print(f"Copied: {old_path} -> {new_path}")


if __name__ == "__main__":
    copy_and_rename_files(args.src, args.dst)
