# comfyui-FOK_API_tools

ComfyUI custom nodes for calling text and vision LLM APIs through multiple request protocols.

## Nodes

### FOK Multi-Protocol Chat Vision API

Sends a prompt and up to four images to an API endpoint, then returns the response text and debug info.

Supported protocol formats:

- `openai_chat_completions`
- `openai_responses`
- `anthropic_messages`
- `gemini_generatecontent`

This is useful for native provider APIs and switching/proxy services such as `cc-switch`, where the same backend may accept different API-compatible formats.

## Installation

Place this folder under:

```text
ComfyUI/custom_nodes/comfyui-FOK_API_tools
```

Restart ComfyUI. The node should appear under:

```text
FOK API Tools/API
```

## API Key

Do not put API keys directly into ComfyUI workflows. Workflow files can be exported or shared, so direct key inputs are intentionally not provided.

Create a local key file in this node directory:

```text
api_key.txt
```

Put only the API key in that file. `api_key.txt` is ignored by Git.

An empty example file is included:

```text
api_key.txt.example
```

The node input `api_key_file` only accepts a file name in this node directory. Absolute paths and parent directory paths are rejected.

## Basic Usage

1. Add `FOK Multi-Protocol Chat Vision API`.
2. Select `api_protocol`.
3. Set `model`.
4. Set `api_base_url`.
5. Connect `prompt`.
6. Optionally connect up to four images.
7. Put your API key in `api_key.txt`.

## Protocol URLs

The node appends the protocol endpoint to `api_base_url`:

| Protocol | Endpoint |
| --- | --- |
| `openai_chat_completions` | `/chat/completions` |
| `openai_responses` | `/responses` |
| `anthropic_messages` | `/messages` |
| `gemini_generatecontent` | `/models/{model}:generateContent` |

For Gemini, `api_base_url` can also be a full URL ending with `:generateContent`, or can contain `{model}` as a placeholder.

## cc-switch Usage

Set `api_base_url` to the base URL exposed by `cc-switch`, then select the protocol format expected by that route:

- OpenAI Chat Completions-compatible route: `openai_chat_completions`
- OpenAI Responses-compatible route: `openai_responses`
- Anthropic Messages-compatible route: `anthropic_messages`
- Gemini native GenerateContent-compatible route: `gemini_generatecontent`

The node sends provider-compatible JSON bodies and image payloads as base64 PNG data.

## Error Handling

`on_error` controls failure behavior:

- `skip`: return the original prompt.
- `stop`: block downstream execution when ComfyUI supports `ExecutionBlocker`.
- `raise`: raise the error and stop execution.

## Debug Info

The second output returns debug JSON with request and response metadata. Authentication headers are redacted:

- `Authorization`
- `x-api-key`
- `x-goog-api-key`

Large payloads may be omitted from debug output to avoid dumping base64 image data.
