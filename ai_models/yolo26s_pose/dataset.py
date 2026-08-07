from torch.utils.data import Dataset
import pandas as pd
from torchvision.io import read_image
from typing import TypedDict

class OrthognaticDatasetSample(TypedDict):
    sample_id: str
    batch: str
    image_path: str
    image_name: str
    width: int
    height: int


class OrthognaticDataset(Dataset):

    def __init__(self, manifest_path, transform=None):
        self.transform = transform
        self.df = pd.read_csv(manifest_path)

    def __len__(self):
        return self.df.__len__()

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_path = self.df.iloc[idx, 2]
        img = read_image(img_path)


        



        