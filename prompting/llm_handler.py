import base64
import json
import mimetypes
import re
from pathlib import Path
from openai import (
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
    AuthenticationError,
    InternalServerError,
    OpenAI,
    RateLimitError,
)
from typing import List, Optional

from prompting.exception import (
    LLMAuthenticationError,
    LLMInvalidJsonError,
    LLMProviderError,
    LLMProviderServerError,
    LLMRateLimitError,
    LLMRequestPreparationError,
    LLMResponseError,
    LLMTransportError,
)

from dataclasses import dataclass
import time

@dataclass
class LLMHandlerOutput:
    content: str | None
    api_duration_ms: int
    input_tokens: int | None
    output_tokens: int | None
    total_tokens: int | None
    cost_usd: float | None
    finish_reason: str | None


def encode_file_to_base64(file_path: str) -> str:
    """Reads a local file and converts it to a base64 encoded string."""
    with open(file_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

class LLMCallClient:

    def __init__(self):
        self.api_response : LLMHandlerOutput = None

    def llm_call(
        self,
        base_url:str,
        model_name: str,
        api_key: str,
        system_prompt:str,
        user_prompt: Optional[str] = None,
        file_paths: Optional[List[str]] = None,
        image_paths: Optional[List[str]] = None,
    ) -> None:
        
        self.api_response = None

        if not isinstance(model_name, str) or not model_name.strip():
            raise LLMRequestPreparationError("model_name must be a non-empty string.")
        if not isinstance(system_prompt, str):
            raise LLMRequestPreparationError("system_prompt must be a string.")

        client = OpenAI(
                base_url=base_url,
                api_key=api_key,
                max_retries=0
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
        try:
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
        except (OSError, TypeError, ValueError) as exc:
            raise LLMRequestPreparationError(
                f"Could not prepare the LLM request: {exc}"
            ) from exc

        if user_content:
            messages.append({
                "role": "user",
                "content": user_content,
            })

        call_start_time = time.time()
        input_tokens = None
        output_tokens = None
        total_tokens = None
        cost_usd = None
        finish_reason = None
        content = None

        try:
            response = client.chat.completions.create(
                model=model_name,
                response_format={"type": "json_object"},
                messages=messages,
            )
            usage = response.usage
            input_tokens = getattr(usage, "prompt_tokens", None)
            output_tokens = getattr(usage, "completion_tokens", None)
            total_tokens = getattr(usage, "total_tokens", None)
            cost_usd = getattr(usage, "cost", None)

            try:
                finish_reason = response.choices[0].finish_reason
            except (AttributeError, IndexError, TypeError) as exc:
                raise LLMResponseError(
                    "The LLM response did not contain a completion choice."
                ) from exc
            
        except APITimeoutError as exc:
            raise LLMTransportError("The LLM request timed out.") from exc
        except APIConnectionError as exc:
            raise LLMTransportError("Could not connect to the LLM provider.") from exc
        except AuthenticationError as exc:
            raise LLMAuthenticationError(
                "The LLM provider rejected the supplied credentials.",
                status_code=exc.status_code,
            ) from exc
        except RateLimitError as exc:
            raise LLMRateLimitError(
                "The LLM provider rate-limited the request.",
                status_code=exc.status_code,
            ) from exc
        except InternalServerError as exc:
            raise LLMProviderServerError(
                "The LLM provider returned a server error.",
                status_code=exc.status_code,
            ) from exc
        except APIStatusError as exc:
            raise LLMProviderError(
                str(exc),
                status_code=exc.status_code,
            ) from exc
        finally:
            call_end_time = time.time()
            api_duration_ms = round((call_end_time - call_start_time) * 1000)
            self.api_response = LLMHandlerOutput(
                        content=None,
                        api_duration_ms=api_duration_ms,
                        input_tokens=input_tokens,
                        output_tokens=output_tokens,
                        total_tokens=total_tokens,
                        cost_usd=cost_usd,
                        finish_reason=finish_reason
                    )

        try:
            content = response.choices[0].message.content
            self.api_response.content = content
        except (AttributeError, IndexError, TypeError) as exc:
            raise LLMResponseError("The LLM response did not contain a message.") from exc
        if not isinstance(content, str) or not content.strip():
            raise LLMResponseError("The LLM response content was empty.")

        # Attempt to parse JSON; strip Markdown code fences first if present.
        cleaned = content.strip()
        # Try to extract JSON from a Markdown code block (```json ... ``` or ``` ... ```)
        code_block_match = re.search(
            r"```(?:json)?\s*\n?(.*?)\n?```", cleaned, re.DOTALL
        )
        if code_block_match:
            cleaned = code_block_match.group(1).strip()
        try:
            parsed_content = json.loads(cleaned)
        except json.JSONDecodeError as exc:
            raise LLMInvalidJsonError("The LLM response was not valid JSON.") from exc

        if not isinstance(parsed_content, dict):
            raise LLMInvalidJsonError(
                "The LLM response was valid JSON but not a JSON object."
            )




if __name__ == "__main__":
    from config import config
    # change this to 1 if dynamic prompt is what you seek. otherwise keep 0 for fixed user prompt
    input_test = 1
    user_prompt = "senin daşşaklarını yiyerim"
    if input_test:
        print("> ", end="")
        user_prompt = input()

    system_prompt = "you are a turkish-english translator, translate user's sentence in json."
    
    response_text = llm_call(
        base_url=config.base_url,
        model_name=config.test_model,
        api_key=config.API_KEY_OPENROUTER,
        system_prompt=system_prompt,
        user_prompt=user_prompt
    )

    print(response_text)
