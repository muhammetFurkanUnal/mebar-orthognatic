from config import config
from prompting.llm_handler import llm_call
import os
import json

if __name__ == "__main__":
	# import system prompt
	system_prompt_path = "prompts/uneducated_profile.xml"
	with open(system_prompt_path, "r") as file:
		system_prompt = file.read()

	response_json = llm_call(
		base_url=config.base_url,
		model_name='anthropic/claude-opus-4.8',
		api_key=config.API_KEY_OPENROUTER,
		system_prompt=system_prompt,
		image_paths=["assets/sample.png"],
		# file_paths=["assets/Arnett part I.pdf", "assets/Arnett part II.pdf"]
	)

	print(response_json)

	# save output .json
	output_folder = "outputs"
	output_file = "uned_opus_4_8"
	os.makedirs(output_folder, exist_ok=True)
	output_path = f"{output_folder}/{output_file}.json"

	if isinstance(response_json, str):
		data = json.loads(response_json)
	else:
		data = response_json

	with open(output_path, "w", encoding="utf-8") as f:
		json.dump(data, f, indent=4, ensure_ascii=False)