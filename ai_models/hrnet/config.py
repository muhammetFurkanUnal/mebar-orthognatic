import json
from dataclasses import dataclass
from typing import List

json_path = "/home/furkan/projects/mebar-orthognatic/ai_models/hrnet/config.json"

@dataclass
class Config:
    img_size: List[int]
    heatmap_size: List[int]

with open(json_path, "r") as f:
    json_text = f.read()
    json_dict = json.loads(json_text)

cfg = Config(
    img_size=[int(x) for x in json_dict["img_size"].split("x")],
    heatmap_size=[int(x) for x in json_dict["heatmap_size"].split("x")]
)


if __name__ == "__main__":
    print(cfg)