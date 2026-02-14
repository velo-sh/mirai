"""Static model capability annotations for free providers.

The OpenAI ``/models`` endpoint returns only ``id`` and ``created`` —
no metadata about reasoning, vision, or tool-use support. This module
provides a pattern-based static map to fill those gaps for well-known
open-source models.

Usage::

    from mirai.agent.free_model_capabilities import enrich_capabilities

    entry = enrich_capabilities(entry)  # mutates in-place & returns
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from mirai.agent.registry_models import RegistryModelEntry


@dataclass(frozen=True, slots=True)
class _Cap:
    """Capability flags to OR-merge onto a RegistryModelEntry."""

    reasoning: bool = False
    vision: bool = False
    tool_use: bool = False


# ------------------------------------------------------------------
# Pattern → capability map.  Order matters: first match wins.
# Patterns are matched case-insensitively against the model ID.
# ------------------------------------------------------------------
_CAPABILITY_RULES: list[tuple[re.Pattern[str], _Cap]] = [
    # ---- Reasoning models ----
    (re.compile(r"deepseek.*r1", re.I), _Cap(reasoning=True)),
    (re.compile(r"qwq", re.I), _Cap(reasoning=True)),
    (re.compile(r"o1|o3", re.I), _Cap(reasoning=True)),
    # ---- Vision models ----
    (re.compile(r"vision|vl-|pixtral|llava", re.I), _Cap(vision=True)),
    (re.compile(r"gemma.*it", re.I), _Cap(vision=True)),
    # ---- Tool-use capable models ----
    (re.compile(r"llama.*(3|4)", re.I), _Cap(tool_use=True)),
    (re.compile(r"qwen", re.I), _Cap(tool_use=True)),
    (re.compile(r"mistral|mixtral", re.I), _Cap(tool_use=True)),
    (re.compile(r"gemma", re.I), _Cap(tool_use=True)),
    (re.compile(r"deepseek.*v3|deepseek.*chat", re.I), _Cap(tool_use=True)),
    (re.compile(r"command-r", re.I), _Cap(tool_use=True)),
]


def enrich_capabilities(entry: RegistryModelEntry) -> RegistryModelEntry:
    """Apply static capability tags to *entry* based on its model ID.

    Only sets flags that are currently ``False`` — provider-native
    metadata always wins.  Returns the same object (mutated in-place).
    """
    for pattern, cap in _CAPABILITY_RULES:
        if pattern.search(entry.id):
            if cap.reasoning and not entry.reasoning:
                entry.reasoning = True
            if cap.vision and not entry.vision:
                entry.vision = True
            if cap.tool_use and not entry.supports_tool_use:
                entry.supports_tool_use = True
    return entry
