import os
import numpy as np
import csv
import torch
from torchvision import transforms
import matplotlib.pyplot as plt
from pathlib import Path
from glob import glob
import json
import logging


def save_path_pairs_to_csv(pairs, filepath):
    with open(filepath, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["image_path", "mask_path"])
        for img, mask in pairs:
            writer.writerow([img, mask])

def save_results():
    return