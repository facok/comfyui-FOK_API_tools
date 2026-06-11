# comfyui-FOK_API_tools

这是一个 ComfyUI 自定义节点包，用于通过多种供应商兼容请求协议调用文本和视觉大模型 API。

## 节点

### FOK Multi-Protocol Chat Vision API

向 API 端点发送提示词和最多四张图片，然后返回模型回复文本和调试信息。

支持的协议格式：

- `openai_chat_completions`
- `openai_responses`
- `anthropic_messages`
- `gemini_generatecontent`

这个节点适合调用需要特定供应商兼容请求格式的 API 服务。

## 安装

把本目录放到：

```text
ComfyUI/custom_nodes/comfyui-FOK_API_tools
```

重启 ComfyUI。节点会出现在：

```text
FOK API Tools/API
```

## API Key

不要把 API key 直接写进 ComfyUI 工作流。工作流文件可能会被导出或分享，所以本节点刻意不提供直接填写 key 的输入框。

请在当前节点目录创建本地密钥文件：

```text
api_key.txt
```

文件里只放 API key。

仓库里提供了一个空示例文件：

```text
api_key.txt.example
```

节点输入 `api_key_file` 只接受当前节点目录下的文件名，不允许绝对路径或上级目录路径。

## 基本用法

1. 添加 `FOK Multi-Protocol Chat Vision API` 节点。
2. 选择 `api_protocol`。
3. 填写 `model`。
4. 填写 `api_base_url`。
5. 连接 `prompt`。
6. 可选连接最多四张图片。
7. 把 API key 写入 `api_key.txt`。

## 协议 URL

节点会根据协议把端点追加到 `api_base_url` 后：

| 协议 | 端点 |
| --- | --- |
| `openai_chat_completions` | `/chat/completions` |
| `openai_responses` | `/responses` |
| `anthropic_messages` | `/messages` |
| `gemini_generatecontent` | `/models/{model}:generateContent` |

Gemini 的 `api_base_url` 也可以直接填写以 `:generateContent` 结尾的完整 URL，或者使用 `{model}` 占位符。

节点会发送对应供应商兼容的 JSON 请求体，图片会以 base64 PNG 数据发送。

## 错误处理

`on_error` 控制失败时的行为：

- `skip`：返回原始 prompt。
- `stop`：在 ComfyUI 支持 `ExecutionBlocker` 时阻断下游执行。
- `raise`：抛出错误并停止执行。

## 调试信息

第二个输出会返回调试 JSON，包含请求和响应元信息。鉴权请求头会被脱敏：

- `Authorization`
- `x-api-key`
- `x-goog-api-key`

如果请求体过大，调试信息会省略 payload，避免输出大量 base64 图片数据。
