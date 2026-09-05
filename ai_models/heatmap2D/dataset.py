import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
from dataclasses import dataclass
from pathlib import Path
from PIL import Image, ImageOps
from sklearn.model_selection import train_test_split
import os
from config import cfg
from typing import Tuple

@dataclass
class Sample:
    name: str
    img_path: str
    image: torch.Tensor
    points: torch.Tensor
    heatmaps: torch.Tensor

def load_image_with_exif(file_path: str) -> np.ndarray:
    with Image.open(file_path) as img:
        img = ImageOps.exif_transpose(img)
        img = img.convert("RGB")
        return np.array(img)

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

    # 1b. Guard against NaN / inf coordinates (missing keypoints)
    if not (np.isfinite(raw_x) and np.isfinite(raw_y)):
        return heatmap

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

def handle_RGB(image: np.ndarray) -> np.ndarray:
    """
    Ensures the input image is strictly a 3-channel RGB image.
    Handles Grayscale (2D or 3D with 1 channel) and RGBA (4 channels).
    """
    # 1. Grayscale (H, W) -> (H, W, 3)
    if image.ndim == 2:
        return np.stack([image] * 3, axis=-1)

    # 2. 3D arrays: Check channel count
    if image.ndim == 3:
        channels = image.shape[2]
        
        # RGBA -> RGB (Drop alpha channel)
        if channels == 4:
            return image[:, :, :3]
        
        # Single-channel 3D (H, W, 1) -> (H, W, 3)
        if channels == 1:
            return np.repeat(image, 3, axis=-1)
        
        # Already 3-channel RGB
        if channels == 3:
            return image

    raise ValueError(f"Unsupported image shape for RGB conversion: {image.shape}")

class Heatmap2D_Dataset(Dataset):
    def __init__(self, df, root_path, transform=None):
        self.df = df
        self.transform = transform
        self.root_path = root_path
        self.points = ["subnasale", "upper_lip_anterior", "lower_lip_anterior", "pogonion"]
        self.points = [[f"{point}_x", f"{point}_y"] for point in self.points]
        self.points = np.array(self.points).flatten().tolist()


    def __getitem__(self, idx) -> Sample:
        name = self.df.iloc[idx]["image_name"]
        img_path = self.df.iloc[idx]["image_path"]

        full_path = os.path.join(self.root_path, img_path)
        image = load_image_with_exif(full_path)

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
            img_path=img_path,
            image=image_tensor,
            points=torch.as_tensor(transformed_kps, dtype=torch.float32),
            heatmaps=heatmaps
        )

    def __len__(self):
        return len(self.df)


def create_dataset(manifest_path, root_path, train_size, train_transform, test_transform):
    df = pd.read_csv(manifest_path)

    # Drop rows with missing keypoint coordinates (they produce NaN heatmaps -> NaN loss)
    point_names = ["subnasale", "upper_lip_anterior", "lower_lip_anterior", "pogonion"]
    coord_cols = []
    for point in point_names:
        coord_cols.extend([f"{point}_x", f"{point}_y"])
    n_before = len(df)
    df = df.dropna(subset=coord_cols).reset_index(drop=True)
    n_dropped = n_before - len(df)
    if n_dropped:
        print(f"[create_dataset] Dropped {n_dropped} row(s) with missing keypoints.")

    train_df, test_df = train_test_split(df, train_size=train_size)
    train_dataset = Heatmap2D_Dataset(train_df, root_path, transform=train_transform)
    test_dataset = Heatmap2D_Dataset(test_df, root_path, transform=test_transform)

    return train_dataset, test_dataset


if __name__ == "__main__":
    from transforms import test_transform
    import matplotlib.pyplot as plt

    df = pd.read_csv("/home/furkan/projects/mebar-orthognatic/data/processed/manifest.csv")

    dataset = Heatmap2D_Dataset(
        df=df, 
        root_path="/home/furkan/projects/mebar-orthognatic",
        transform=test_transform
        )

    print(len(dataset))

