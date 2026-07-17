import json
from dataclasses import dataclass
from typing import List, Any

@dataclass
class Config:
	base_url:str
	API_KEY_OPENROUTER:str
	test_model:str
	models:List[str]


with open("config.json", "r") as f:
	config_dict = json.loads(f.read())

# import this from other files
config = Config(
	base_url=config_dict["base_url"],
	API_KEY_OPENROUTER=config_dict["API_KEY_OPENROUTER"],
	models=config_dict["models"],
	test_model=config_dict["test_model"]
)


if __name__ == "__main__":
	print(config)
	