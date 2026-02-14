from __future__ import annotations

import json
from typing import TypeVar

from pydantic import BaseModel, ValidationError

from mirai.logging import get_logger

log = get_logger("mirai.agent.contract")

T = TypeVar("T", bound=BaseModel)


class ContractError(Exception):
    """Raised when an LLM output fails to meet the expected schema."""

    def __init__(self, message: str, raw_content: str | None = None):
        super().__init__(message)
        self.raw_content = raw_content


def validate_response(content: str, model_class: type[T]) -> T:
    """Validate a JSON string against a Pydantic model.

    Standardizes how we handle LLM outputs that should be structured data.
    Tries to extract JSON from markdown blocks if present.
    """
    cleaned = content.strip()

    # Simple markdown extraction
    if "```json" in cleaned:
        cleaned = cleaned.split("```json")[1].split("```")[0].strip()
    elif "```" in cleaned:
        cleaned = cleaned.split("```")[1].split("```")[0].strip()

    try:
        data = json.loads(cleaned)
        return model_class.model_validate(data)
    except (json.JSONDecodeError, ValidationError) as exc:
        log.error("contract_validation_failed", error=str(exc), content_snippet=content[:100])
        raise ContractError(
            f"LLM output failed contract validation {model_class.__name__}: {exc}",
            raw_content=content,
        ) from exc
