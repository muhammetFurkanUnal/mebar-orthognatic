import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
from dataclasses import dataclass
from pathlib import Path
from skimage import io
import os
from config import cfg
from typing import Tuple

@dataclass
class Sample:
    name: str
    image: torch.tensor
    points: torch.tensor
    heatmaps: torch.tensor


def generate_heatmap(
    keypoint: Tuple[int, int],
    img_size: Tuple[int, int] = (256, 256),
    heatmap_size: Tuple[int, int] = (64, 64),
    sigma: float = 1.5
) -> torch.Tensor:
    in_w, in_h = img_size
    hm_w, hm_h = heatmap_size

    # Initialize empty heatmap
    heatmap = torch.zeros((hm_h, hm_w), dtype=torch.float32)

    # 1. Check for invalid or empty coordinates
    if keypoint is None or len(keypoint) < 2:
        return heatmap

    raw_x, raw_y = float(keypoint[0]), float(keypoint[1])

    # 2. Scale coordinate from image space to heatmap space
    x_hm = raw_x * (hm_w / in_w)
    y_hm = raw_y * (hm_h / in_h)

    # 3. 3-Sigma bounding box check
    radius = sigma * 3.0
    ul_x = x_hm - radius
    ul_y = y_hm - radius
    br_x = x_hm + radius
    br_y = y_hm + radius

    # If the 3-sigma Gaussian area is completely outside the heatmap boundaries, return zeros
    if ul_x >= hm_w or ul_y >= hm_h or br_x < 0 or br_y < 0:
        return heatmap

    # 4. Generate coordinate grid and compute Gaussian
    y_grid, x_grid = torch.meshgrid(
        torch.arange(hm_h, dtype=torch.float32),
        torch.arange(hm_w, dtype=torch.float32),
        indexing='ij'
    )

    dist_sq = (x_grid - x_hm) ** 2 + (y_grid - y_hm) ** 2
    heatmap = torch.exp(-dist_sq / (2.0 * (sigma ** 2)))

    return heatmap


class HRNetDataset(Dataset):
    def __init__(self, manifest_path, transform=None):
        self.df = pd.read_csv(manifest_path)
        self.transform = transform
        self.points = ["subnasale", "upper_lip_anterior", "lower_lip_anterior", "pogonion"]
        self.points = [[f"{point}_x", f"{point}_y"] for point in self.points]
        self.points = np.array(self.points).flatten().tolist()


    def __getitem__(self, idx):
        name = self.df.iloc[idx]["image_name"]
        img_path = self.df.iloc[idx]["image_path"]
        image = io.imread(img_path)

        keypoints_coords = self.df.iloc[idx][self.points]
        keypoints_coords = np.asarray(keypoints_coords, dtype=float).reshape(-1, 2)

        # apply transformations to image
        if self.transform:
            augmented = self.transform(image=image, keypoints=keypoints_coords)
            image_tensor = augmented["image"]
            transformed_kps = augmented["keypoints"]

        # generate heatmaps
        heatmaps = torch.zeros((len(keypoints_coords), cfg.heatmap_size[1], cfg.heatmap_size[0]), dtype=torch.float32)
        for i in range(len(keypoints_coords)):
            heatmap = generate_heatmap(
                keypoint=(transformed_kps[i]),
                img_size=cfg.img_size,
                heatmap_size=cfg.heatmap_size
            )
            heatmaps[i] = heatmap

        return Sample(
            name=name,
            image=image_tensor,
            points=torch.tensor(transformed_kps, dtype=torch.float32),
            heatmaps=heatmaps
        )

    def __len__(self):
        return len(self.df)


if __name__ == "__main__":
    from transforms import transform
    import matplotlib.pyplot as plt

    dataset = HRNetDataset("/home/furkan/projects/mebar-orthognatic/data/processed/manifest.csv", transform=transform)

    n = 0

    sample = dataset[n]
    print(sample.name)
    print(sample.points)
    plt.imshow(sample.image)

