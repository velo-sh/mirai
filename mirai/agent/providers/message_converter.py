"""Message and tool format conversion for Cloud Code Assist."""

import time
from typing import Any

from mirai.agent.models import ProviderResponse, TextBlock, ToolUseBlock


def convert_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Convert Anthropic-format messages to Google Generative AI format."""
    contents = []
    for msg in messages:
        role = "user" if msg["role"] == "user" else "model"
        parts: list[dict[str, Any]] = []
        content = msg.get("content", "")

        if isinstance(content, str):
            if content:  # Skip empty strings for Claude compatibility
                parts.append({"text": content})
        elif isinstance(content, list):
            for block in content:
                if isinstance(block, dict):
                    if block.get("type") == "text":
                        if block.get("text"):  # Skip empty text for Claude compatibility
                            parts.append({"text": block["text"]})
                    elif block.get("type") == "image":
                        source = block.get("source", {})
                        if source.get("type") == "base64":
                            parts.append(
                                {
                                    "inlineData": {
                                        "mimeType": source.get("media_type", "image/png"),
                                        "data": source.get("data", ""),
                                    }
                                }
                            )
                    elif block.get("type") == "tool_use":
                        fc_payload: dict[str, Any] = {
                            "name": block["name"],
                            "args": block.get("input", {}),
                        }
                        # Preserve id for Claude model compatibility via Cloud Code Assist
                        if block.get("id"):
                            fc_payload["id"] = block["id"]
                        fc_part: dict[str, Any] = {"functionCall": fc_payload}
                        # Preserve Gemini 3 thought_signature for replay
                        if block.get("thought_signature"):
                            fc_part["thoughtSignature"] = block["thought_signature"]
                        parts.append(fc_part)  # type: ignore[arg-type]
                    elif block.get("type") == "tool_result":
                        result_text = block.get("content", "")
                        if isinstance(result_text, list):
                            result_text = " ".join(
                                b.get("text", "")
                                for b in result_text
                                if isinstance(b, dict) and b.get("type") == "text"
                            )
                        fr_payload: dict[str, Any] = {
                            "name": block.get("tool_use_id", "unknown"),
                            "response": {"result": str(result_text)},
                        }
                        # Preserve id for Claude model compatibility
                        if block.get("tool_use_id"):
                            fr_payload["id"] = block["tool_use_id"]
                        parts.append(
                            {"functionResponse": fr_payload}  # type: ignore[dict-item]
                        )

        if parts:
            contents.append({"role": role, "parts": parts})
    return contents


def convert_tools(tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Convert Anthropic-format tools to Google Generative AI format."""
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
