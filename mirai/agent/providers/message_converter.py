"""Message and tool format conversion for Cloud Code Assist.

Converts OpenAI-format messages (the internal canonical format) to
Google Generative AI format used by the streamGenerateContent endpoint.
"""

import json
import time
from typing import Any

from mirai.agent.models import ProviderResponse, TextBlock, ToolUseBlock


def convert_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Convert OpenAI-format messages to Google Generative AI format.

    OpenAI format:
        - role: "user" | "assistant" | "system" | "tool"
        - content: str | None
        - tool_calls: [{id, type, function: {name, arguments}}]  (assistant)
        - tool_call_id: str  (tool)

    Google Generative AI format:
        - role: "user" | "model"
        - parts: [{text}, {functionCall}, {functionResponse}]
    """
    contents = []
    for msg in messages:
        role = msg.get("role", "")
        parts: list[dict[str, Any]] = []

        # System messages are handled separately (systemInstruction)
        if role == "system":
            continue

        if role == "tool":
            # Tool result → functionResponse
            tool_call_id = msg.get("tool_call_id", "unknown")
            result_text = msg.get("content", "")
            fr_payload: dict[str, Any] = {
                "name": tool_call_id,
                "response": {"result": str(result_text)},
            }
            fr_payload["id"] = tool_call_id
            parts.append({"functionResponse": fr_payload})
            # Tool results are always from the user's perspective
            if parts:
                contents.append({"role": "user", "parts": parts})
            continue

        if role == "assistant":
            gemini_role = "model"
        else:
            gemini_role = "user"

        # Text content
        content = msg.get("content")
        if isinstance(content, str) and content:
            parts.append({"text": content})
        elif isinstance(content, list):
            # Handle legacy list-format content (e.g. image blocks)
            for block in content:
                if isinstance(block, dict):
                    if block.get("type") == "text" and block.get("text"):
                        parts.append({"text": block["text"]})
                    elif block.get("type") == "image":
                        source = block.get("source", {})
                        if source.get("type") == "base64":
                            parts.append({
                                "inlineData": {
                                    "mimeType": source.get("media_type", "image/png"),
                                    "data": source.get("data", ""),
                                }
                            })

        # Tool calls from assistant message → functionCall parts
        for tc in msg.get("tool_calls", []):
            fn = tc.get("function", {})
            try:
                args = json.loads(fn.get("arguments", "{}"))
            except (json.JSONDecodeError, TypeError):
                args = {}
            fc_payload: dict[str, Any] = {
                "name": fn.get("name", ""),
                "args": args,
            }
            call_id = tc.get("id")
            if call_id:
                fc_payload["id"] = call_id
            fc_part: dict[str, Any] = {"functionCall": fc_payload}
            # Preserve Gemini 3 thought_signature for replay
            thought_sig = tc.get("thought_signature")
            if thought_sig:
                fc_part["thoughtSignature"] = thought_sig
            parts.append(fc_part)

        if parts:
            contents.append({"role": gemini_role, "parts": parts})
    return contents


def convert_tools(tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Convert internal tool definitions to Google Generative AI format.

    Internal: {"name": ..., "description": ..., "input_schema": {...}}
    Google:   {"functionDeclarations": [{"name": ..., "description": ..., "parameters": {...}}]}
    """
    if not tools:
        return []
    declarations = []
    for tool in tools:
        decl: dict[str, Any] = {
            "name": tool["name"],
            "description": tool.get("description", ""),
        }
        schema = tool.get("input_schema", {})
        if schema:
            decl["parameters"] = schema
        declarations.append(decl)
    return [{"functionDeclarations": declarations}]


def parse_sse_response(raw_text: str, model_id: str | None = None) -> ProviderResponse:
    """
    Parse SSE stream response and build ProviderResponse.
    Uses orjson for fast JSON parsing of each SSE data line.
    """
    import orjson

    content_blocks: list[TextBlock | ToolUseBlock] = []
    stop_reason = "end_turn"
    tool_call_counter = 0

    # Track last text block for merging consecutive text chunks
    last_text_idx = -1

    for line in raw_text.split("\n"):
        if not line.startswith("data:"):
            continue
        json_bytes = line[5:].strip().encode()
        if not json_bytes:
            continue
        try:
            chunk = orjson.loads(json_bytes)
        except orjson.JSONDecodeError:
            continue

        response_data = chunk.get("response")
        if not response_data:
            continue

        candidate = None
        candidates = response_data.get("candidates", [])
        if candidates:
            candidate = candidates[0]

        if candidate and candidate.get("content", {}).get("parts"):
            for part in candidate["content"]["parts"]:
                if "text" in part:
                    # Merge consecutive text blocks
                    if last_text_idx >= 0 and isinstance(content_blocks[last_text_idx], TextBlock):
                        # TextBlock is frozen, create new merged block
                        old = content_blocks[last_text_idx]
                        assert isinstance(old, TextBlock)
                        content_blocks[last_text_idx] = TextBlock(text=old.text + part["text"])
                    else:
                        content_blocks.append(TextBlock(text=part["text"]))
                        last_text_idx = len(content_blocks) - 1
                if "functionCall" in part:
                    fc = part["functionCall"]
                    tool_call_counter += 1
                    call_id = fc.get("id", f"{fc['name']}_{int(time.time())}_{tool_call_counter}")
                    # Extract Gemini 3 thought_signature
                    thought_sig = part.get("thoughtSignature")
                    content_blocks.append(
                        ToolUseBlock(
                            id=call_id,
                            name=fc["name"],
                            input=fc.get("args", {}),
                            thought_signature=thought_sig,
                        )
                    )
                    stop_reason = "tool_use"
                    last_text_idx = -1  # Reset text merge tracking

        if candidate and candidate.get("finishReason"):
            fr = candidate["finishReason"]
            if fr == "STOP":
                stop_reason = "end_turn"
            elif fr == "MAX_TOKENS":
                stop_reason = "max_tokens"

    if not content_blocks:
        content_blocks.append(TextBlock())

    return ProviderResponse(content=content_blocks, stop_reason=stop_reason, model_id=model_id)
