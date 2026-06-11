import base64
import io
import json
import logging
import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import requests
from PIL import Image

try:
    from comfy_execution.graph_utils import ExecutionBlocker
except Exception:
    ExecutionBlocker = None


log = logging.getLogger(__name__)
NODE_DIR = os.path.dirname(os.path.abspath(__file__))


PROTOCOL_OPENAI_CHAT = "openai_chat_completions"
PROTOCOL_OPENAI_RESPONSES = "openai_responses"
PROTOCOL_ANTHROPIC_MESSAGES = "anthropic_messages"
PROTOCOL_GEMINI_GENERATE_CONTENT = "gemini_generatecontent"

API_PROTOCOLS = [
    PROTOCOL_OPENAI_CHAT,
    PROTOCOL_OPENAI_RESPONSES,
    PROTOCOL_ANTHROPIC_MESSAGES,
    PROTOCOL_GEMINI_GENERATE_CONTENT,
]

PROTOCOL_TOOLTIP = (
    "Choose the JSON protocol sent to the server. "
    "openai_chat_completions -> /chat/completions; "
    "openai_responses -> /responses; "
    "anthropic_messages -> /messages; "
    "gemini_generatecontent -> /models/{model}:generateContent. "
    "Select the format expected by your API service."
)

MODEL_TOOLTIP = (
    "Model name passed through unchanged, for example gpt-4o, claude-3-5-sonnet-latest, "
    "gemini-2.5-flash, or a model alias configured by your API service."
)

BASE_URL_TOOLTIP = (
    "Base URL for the API service. The node appends the endpoint for the selected protocol. "
    "Examples: https://api.openai.com/v1, https://api.anthropic.com/v1, "
    "or https://generativelanguage.googleapis.com/v1beta. "
    "For Gemini you may also provide a full URL ending in :generateContent or include {model}."
)

KEY_FILE_TOOLTIP = (
    "File name only, read from this node directory. Do not enter an API key here. "
    "Default: api_key.txt. This keeps secrets out of workflow JSON files. "
    "Absolute paths and ../ are rejected."
)

DEBUG_OUTPUT_TOOLTIP = (
    "Second output contains request/response debug JSON. Auth headers are redacted, "
    "and large image payloads may be omitted."
)


def read_api_key(api_key_file: str = "api_key.txt") -> str:
    key_filename = os.path.basename(api_key_file.strip() or "api_key.txt")
    if key_filename != (api_key_file.strip() or "api_key.txt"):
        raise ValueError("api_key_file must be a file name in this node directory, not a path.")

    key_path = os.path.join(NODE_DIR, key_filename)
    if not os.path.exists(key_path):
        with open(key_path, "w", encoding="utf-8") as key_file:
            key_file.write("YOUR_API_KEY_HERE")
        log.warning("API key file not found. Created template at: %s", key_path)
        return ""

    with open(key_path, "r", encoding="utf-8") as key_file:
        key = key_file.read().strip()
    if not key or key == "YOUR_API_KEY_HERE":
        log.warning("API key is missing or placeholder in %s", key_path)
        return ""
    return key


def tensor_to_pil(image: Any) -> Image.Image:
    if image.ndim == 4:
        image = image[0]
    if image.ndim != 3:
        raise ValueError("IMAGE tensor must be HWC or NHWC.")

    image_np = image.detach().cpu().numpy()
    image_np = np.clip(image_np * 255.0, 0, 255).astype(np.uint8)
    if image_np.shape[-1] == 1:
        return Image.fromarray(image_np[:, :, 0], mode="L")
    if image_np.shape[-1] == 3:
        return Image.fromarray(image_np, mode="RGB")
    if image_np.shape[-1] == 4:
        return Image.fromarray(image_np, mode="RGBA")
    raise ValueError(f"Unsupported IMAGE channel count: {image_np.shape[-1]}")


def tensor_to_png_base64(image: Any) -> str:
    pil_image = tensor_to_pil(image)
    buffer = io.BytesIO()
    pil_image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def collect_image_b64(*images: Optional[Any]) -> List[str]:
    return [tensor_to_png_base64(image) for image in images if image is not None]


def normalize_base_url(api_base_url: str, default_base_url: str) -> str:
    base_url = (api_base_url or "").strip() or default_base_url
    return base_url.rstrip("/")


def redact_headers(headers: Dict[str, str]) -> Dict[str, str]:
    redacted = dict(headers)
    for key in list(redacted.keys()):
        if key.lower() in {"authorization", "x-api-key", "x-goog-api-key"}:
            redacted[key] = "***"
    return redacted


def trim_debug_payload(payload: Any) -> Any:
    text = json.dumps(payload, ensure_ascii=False)
    if len(text) <= 8000:
        return payload
    return {"notice": "Payload omitted because it is large; image data was included in the request."}


def first_text_from_openai_chat(response_json: Dict[str, Any]) -> str:
    return response_json["choices"][0]["message"].get("content") or ""


def first_text_from_openai_responses(response_json: Dict[str, Any]) -> str:
    if response_json.get("output_text"):
        return response_json["output_text"]

    chunks: List[str] = []
    for item in response_json.get("output", []):
        for content in item.get("content", []):
            if content.get("type") in {"output_text", "text"} and content.get("text"):
                chunks.append(content["text"])
    return "".join(chunks)


def first_text_from_anthropic(response_json: Dict[str, Any]) -> str:
    chunks: List[str] = []
    for item in response_json.get("content", []):
        if item.get("type") == "text" and item.get("text"):
            chunks.append(item["text"])
    return "".join(chunks)


def first_text_from_gemini(response_json: Dict[str, Any]) -> str:
    chunks: List[str] = []
    for candidate in response_json.get("candidates", []):
        for part in candidate.get("content", {}).get("parts", []):
            if part.get("text"):
                chunks.append(part["text"])
    return "".join(chunks)


class FOKMultiProtocolChatVisionAPI:
    """
    Sends text and up to four images to OpenAI, Anthropic, Gemini, or compatible API services.
    """

    CATEGORY = "FOK API Tools/API"
    DESCRIPTION = (
        "Send text and up to four images to OpenAI Chat Completions, OpenAI Responses, "
        "Anthropic Messages, Gemini GenerateContent, or another compatible API service. "
        "API keys are loaded from a local file instead of being stored in the workflow."
    )
    FUNCTION = "execute"
    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("response_text", "debug_info")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "api_protocol": (
                    API_PROTOCOLS,
                    {"default": PROTOCOL_OPENAI_CHAT, "tooltip": PROTOCOL_TOOLTIP},
                ),
                "model": ("STRING", {"default": "gpt-4o", "tooltip": MODEL_TOOLTIP}),
                "prompt": (
                    "STRING",
                    {
                        "forceInput": True,
                        "tooltip": "User prompt sent with the optional images. Connect a text node here.",
                    },
                ),
                "api_base_url": (
                    "STRING",
                    {
                        "multiline": False,
                        "default": "https://api.openai.com/v1",
                        "tooltip": BASE_URL_TOOLTIP,
                    },
                ),
                "on_error": (
                    ["skip", "stop", "raise"],
                    {
                        "default": "skip",
                        "tooltip": (
                            "Failure behavior. skip returns the original prompt; stop blocks downstream nodes "
                            "when ExecutionBlocker is available; raise throws the error and stops execution."
                        ),
                    },
                ),
                "timeout": (
                    "INT",
                    {
                        "default": 120,
                        "min": 10,
                        "max": 600,
                        "step": 10,
                        "tooltip": "HTTP request timeout in seconds. Increase for slow models or image-heavy requests.",
                    },
                ),
            },
            "optional": {
                "system_prompt": (
                    "STRING",
                    {
                        "multiline": True,
                        "default": "",
                        "tooltip": (
                            "Optional system/instruction text. Mapped to system, instructions, "
                            "or systemInstruction depending on the selected protocol."
                        ),
                    },
                ),
                "image_1": ("IMAGE", {"tooltip": "Optional first image. Sent as PNG base64."}),
                "image_2": ("IMAGE", {"tooltip": "Optional second image. Sent as PNG base64."}),
                "image_3": ("IMAGE", {"tooltip": "Optional third image. Sent as PNG base64."}),
                "image_4": ("IMAGE", {"tooltip": "Optional fourth image. Sent as PNG base64."}),
                "max_tokens": (
                    "INT",
                    {
                        "default": 2048,
                        "min": 1,
                        "max": 32768,
                        "step": 1,
                        "tooltip": (
                            "Maximum text tokens to generate. Sent as max_tokens for Chat/Anthropic, "
                            "max_output_tokens for Responses, and maxOutputTokens for Gemini."
                        ),
                    },
                ),
                "temperature": (
                    "FLOAT",
                    {
                        "default": 0.7,
                        "min": 0.0,
                        "max": 2.0,
                        "step": 0.05,
                        "tooltip": "Sampling temperature. Lower is more deterministic; higher is more varied.",
                    },
                ),
                "api_key_file": (
                    "STRING",
                    {
                        "multiline": False,
                        "default": "api_key.txt",
                        "tooltip": KEY_FILE_TOOLTIP,
                    },
                ),
                "anthropic_version": (
                    "STRING",
                    {
                        "multiline": False,
                        "default": "2023-06-01",
                        "tooltip": (
                            "Anthropic API version header used only for anthropic_messages. "
                            "Leave default unless your API service requires another value."
                        ),
                    },
                ),
            },
        }

    def execute(
        self,
        api_protocol: str,
        model: str,
        prompt: str,
        api_base_url: str,
        on_error: str,
        timeout: int,
        system_prompt: str = "",
        image_1: Optional[Any] = None,
        image_2: Optional[Any] = None,
        image_3: Optional[Any] = None,
        image_4: Optional[Any] = None,
        max_tokens: int = 2048,
        temperature: float = 0.7,
        api_key_file: str = "api_key.txt",
        anthropic_version: str = "2023-06-01",
    ) -> Tuple[Any, str]:
        debug_data: Dict[str, Any] = {"request": {}, "response": {}}

        try:
            if api_protocol not in API_PROTOCOLS:
                raise ValueError(f"Unsupported api_protocol: {api_protocol}")
            if not model.strip():
                raise ValueError("Model ID is required.")

            key = read_api_key(api_key_file)
            image_b64_list = collect_image_b64(image_1, image_2, image_3, image_4)

            url, headers, payload = self.build_request(
                api_protocol=api_protocol,
                model=model.strip(),
                prompt=prompt,
                system_prompt=system_prompt,
                api_base_url=api_base_url,
                api_key=key,
                image_b64_list=image_b64_list,
                max_tokens=max_tokens,
                temperature=temperature,
                anthropic_version=anthropic_version,
            )

            debug_data["request"] = {
                "url": url,
                "method": "POST",
                "headers": redact_headers(headers),
                "payload": trim_debug_payload(payload),
            }

            response = requests.post(url, headers=headers, json=payload, timeout=timeout)
            debug_data["response"]["status_code"] = response.status_code
            response.raise_for_status()

            response_json = response.json()
            debug_data["response"]["body"] = response_json
            response_text = self.parse_response(api_protocol, response_json)
            if not response_text:
                raise RuntimeError("API returned an empty text response.")
            return (response_text, json.dumps(debug_data, indent=2, ensure_ascii=False))

        except Exception as exc:
            log.error("%s failed: %s", self.__class__.__name__, exc)
            if isinstance(exc, requests.exceptions.RequestException) and exc.response is not None:
                debug_data["response"]["status_code"] = exc.response.status_code
                debug_data["response"]["body"] = exc.response.text
            debug_data["error"] = str(exc)
            debug_json = json.dumps(debug_data, indent=2, ensure_ascii=False)

            if on_error == "stop":
                if ExecutionBlocker is None:
                    return ("", debug_json)
                return (ExecutionBlocker(None), debug_json)
            if on_error == "skip":
                return (prompt, debug_json)
            raise

    def build_request(
        self,
        api_protocol: str,
        model: str,
        prompt: str,
        system_prompt: str,
        api_base_url: str,
        api_key: str,
        image_b64_list: List[str],
        max_tokens: int,
        temperature: float,
        anthropic_version: str,
    ) -> Tuple[str, Dict[str, str], Dict[str, Any]]:
        if api_protocol == PROTOCOL_OPENAI_CHAT:
            return self.build_openai_chat_request(
                model, prompt, system_prompt, api_base_url, api_key, image_b64_list, max_tokens, temperature
            )
        if api_protocol == PROTOCOL_OPENAI_RESPONSES:
            return self.build_openai_responses_request(
                model, prompt, system_prompt, api_base_url, api_key, image_b64_list, max_tokens, temperature
            )
        if api_protocol == PROTOCOL_ANTHROPIC_MESSAGES:
            return self.build_anthropic_messages_request(
                model,
                prompt,
                system_prompt,
                api_base_url,
                api_key,
                image_b64_list,
                max_tokens,
                temperature,
                anthropic_version,
            )
        return self.build_gemini_generate_content_request(
            model, prompt, system_prompt, api_base_url, api_key, image_b64_list, max_tokens, temperature
        )

    def build_openai_chat_request(
        self,
        model: str,
        prompt: str,
        system_prompt: str,
        api_base_url: str,
        api_key: str,
        image_b64_list: List[str],
        max_tokens: int,
        temperature: float,
    ) -> Tuple[str, Dict[str, str], Dict[str, Any]]:
        url = f"{normalize_base_url(api_base_url, 'https://api.openai.com/v1')}/chat/completions"
        headers = {"Content-Type": "application/json"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"

        messages: List[Dict[str, Any]] = []
        if system_prompt.strip():
            messages.append({"role": "system", "content": system_prompt})

        content: List[Dict[str, Any]] = [{"type": "text", "text": prompt}]
        for image_b64 in image_b64_list:
            content.append({"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_b64}"}})
        messages.append({"role": "user", "content": content})

        payload = {
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        return url, headers, payload

    def build_openai_responses_request(
        self,
        model: str,
        prompt: str,
        system_prompt: str,
        api_base_url: str,
        api_key: str,
        image_b64_list: List[str],
        max_tokens: int,
        temperature: float,
    ) -> Tuple[str, Dict[str, str], Dict[str, Any]]:
        url = f"{normalize_base_url(api_base_url, 'https://api.openai.com/v1')}/responses"
        headers = {"Content-Type": "application/json"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"

        content: List[Dict[str, Any]] = [{"type": "input_text", "text": prompt}]
        for image_b64 in image_b64_list:
            content.append({"type": "input_image", "image_url": f"data:image/png;base64,{image_b64}"})

        payload: Dict[str, Any] = {
            "model": model,
            "input": [{"role": "user", "content": content}],
            "max_output_tokens": max_tokens,
            "temperature": temperature,
        }
        if system_prompt.strip():
            payload["instructions"] = system_prompt
        return url, headers, payload

    def build_anthropic_messages_request(
        self,
        model: str,
        prompt: str,
        system_prompt: str,
        api_base_url: str,
        api_key: str,
        image_b64_list: List[str],
        max_tokens: int,
        temperature: float,
        anthropic_version: str,
    ) -> Tuple[str, Dict[str, str], Dict[str, Any]]:
        url = f"{normalize_base_url(api_base_url, 'https://api.anthropic.com/v1')}/messages"
        headers = {
            "Content-Type": "application/json",
            "anthropic-version": anthropic_version.strip() or "2023-06-01",
        }
        if api_key:
            headers["x-api-key"] = api_key

        content: List[Dict[str, Any]] = [{"type": "text", "text": prompt}]
        for image_b64 in image_b64_list:
            content.append(
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/png",
                        "data": image_b64,
                    },
                }
            )

        payload: Dict[str, Any] = {
            "model": model,
            "messages": [{"role": "user", "content": content}],
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        if system_prompt.strip():
            payload["system"] = system_prompt
        return url, headers, payload

    def build_gemini_generate_content_request(
        self,
        model: str,
        prompt: str,
        system_prompt: str,
        api_base_url: str,
        api_key: str,
        image_b64_list: List[str],
        max_tokens: int,
        temperature: float,
    ) -> Tuple[str, Dict[str, str], Dict[str, Any]]:
        base_url = normalize_base_url(api_base_url, "https://generativelanguage.googleapis.com/v1beta")
        if "{model}" in base_url:
            url = base_url.replace("{model}", model)
        elif base_url.endswith(":generateContent"):
            url = base_url
        else:
            url = f"{base_url}/models/{model}:generateContent"

        headers = {"Content-Type": "application/json"}
        if api_key:
            headers["x-goog-api-key"] = api_key

        parts: List[Dict[str, Any]] = [{"text": prompt}]
        for image_b64 in image_b64_list:
            parts.append({"inline_data": {"mime_type": "image/png", "data": image_b64}})

        payload: Dict[str, Any] = {
            "contents": [{"role": "user", "parts": parts}],
            "generationConfig": {
                "maxOutputTokens": max_tokens,
                "temperature": temperature,
            },
        }
        if system_prompt.strip():
            payload["systemInstruction"] = {"parts": [{"text": system_prompt}]}
        return url, headers, payload

    def parse_response(self, api_protocol: str, response_json: Dict[str, Any]) -> str:
        if api_protocol == PROTOCOL_OPENAI_CHAT:
            return first_text_from_openai_chat(response_json)
        if api_protocol == PROTOCOL_OPENAI_RESPONSES:
            return first_text_from_openai_responses(response_json)
        if api_protocol == PROTOCOL_ANTHROPIC_MESSAGES:
            return first_text_from_anthropic(response_json)
        return first_text_from_gemini(response_json)
