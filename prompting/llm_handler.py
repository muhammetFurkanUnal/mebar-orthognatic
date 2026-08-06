import base64
import mimetypes
from pathlib import Path
from openai import OpenAI
from typing import List, Optional

def encode_file_to_base64(file_path: str) -> str:
    """Reads a local file and converts it to a base64 encoded string."""
    with open(file_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

def llm_call(
    base_url:str,
    model_name: str, 
    api_key: str,
    system_prompt:str,
    user_prompt: Optional[str] = None,
    file_paths: Optional[List[str]] = None,
    image_paths: Optional[List[str]] = None,
) -> str:

    client = OpenAI(
            base_url=base_url,
            api_key=api_key
    )

    # prepare messages
    messages = []

    # system prompt
    messages.append({
        "role":"system",
        "content":system_prompt
    })
        
    user_content = []

    # user prompt
    if user_prompt:
        user_content.append({
            "type":"text",
            "text":user_prompt
        })
            
    # Add images to the same user message as the files.
    if image_paths:
        for image_path in image_paths:
            base64_image = encode_file_to_base64(file_path=image_path)
            mime_type = mimetypes.guess_type(image_path)[0] or "image/png"
            user_content.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:{mime_type};base64,{base64_image}"
                }
            })

    # OpenRouter Chat Completions expects a `file` part.  `input_file` is the
    # Responses API schema and is not a valid OpenRouter Chat Completions part.
    if file_paths:
        for file_path in file_paths:
            base64_file = encode_file_to_base64(file_path=file_path)
            user_content.append({
                "type": "file",
                "file": {
                    "filename": Path(file_path).name,
                    "file_data": f"data:application/pdf;base64,{base64_file}",
                },
            })

    if user_content:
        messages.append({
            "role": "user",
            "content": user_content,
        })

    response = client.chat.completions.create(
        model=model_name,
        response_format={"type": "json_object"},
        messages=messages,
    )

    return response.choices[0].message.content
    

if __name__ == "__main__":
    from config import config
    # change this to 1 if dynamic prompt is what you seek. otherwise keep 0 for fixed user prompt
    input_test = 1
    user_prompt = "selam!"
    if input_test:
        print("> ", end="")
        user_prompt = input()

    system_prompt = "you are a turkish-english translator, translate user's sentence."
    
    response_text = llm_call(
        base_url=config.base_url,
        model_name=config.test_model,
        api_key=config.API_KEY_OPENROUTER,
        system_prompt=system_prompt,
        user_prompt=user_prompt
    )

    print(response_text)
