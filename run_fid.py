import argparse
import os
from evaluation.fid.fid import compute_fid

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--real", type=str, required=True)
    parser.add_argument("--synth", type=str, required=True)
    parser.add_argument("--exclude_bg", action="store_true")
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--device", type=str, default="cuda")

    args = parser.parse_args()
    fid_value = compute_fid(
        args.real,
        args.synth,
        batch_size=args.batch,
        device=args.device,
        exclude_bg=args.exclude_bg
    )

    print(f"FID: {fid_value:.3f}")
