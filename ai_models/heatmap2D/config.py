import json
from dataclasses import dataclass
from typing import List

json_path = "/home/furkan/projects/mebar-orthognatic/ai_models/heatmap2D/config.json"

@dataclass
class Config:
    version: str
    img_size: List[int]
    feature_map_shape: List[int]
    heatmap_size: List[int]
    epochs: int
    checkpoint_freq: int

with open(json_path, "r") as f:
    json_text = f.read()
    json_dict = json.loads(json_text)

cfg = Config(
    version=json_dict["version"],
    img_size=[int(x) for x in json_dict["img_size"].split("x")],
    feature_map_shape=[int(x) for x in json_dict["feature_map_shape"].split("x")],
    heatmap_size=[int(x) for x in json_dict["heatmap_size"].split("x")],
    epochs=json_dict["epochs"],
    checkpoint_freq=json_dict["checkpoint_freq"]
)


if __name__ == "__main__":
    print(cfg)