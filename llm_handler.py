import base64
from openai import OpenAI
from typing import List

def encode_image_to_base64(image_path: str) -> str:
    """Reads a local image file and converts it to a base64 encoded string."""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

def llm_call(
    base_url:str,
    model_name: str, 
    api_key: str,
    system_prompt:str,
    user_prompt:str=None,
    image_paths: List[str] = None,
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
            
    # add images to prompt
    if image_paths:
        for image_path in image_paths:
            base64_image = encode_image_to_base64(image_path=image_path)
            user_content.append({
                        "type":"image_url",
                        "image_url":{
                            "url": f"data:image/jpeg;base64,{base64_image}"
                        }
                    })
            
    if user_prompt or image_paths:
        messages.append({
            "role":"user",
            "content":user_content
        })

    response = client.chat.completions.create(
        model=model_name,
        response_format={"type": "json_object"}, # Forces the model to respond in valid JSON
        messages=messages
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

