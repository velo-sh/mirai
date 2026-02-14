"""Mock providers for testing — extracted from production code."""

from typing import Any

from mirai.agent.models import ProviderResponse, TextBlock, ToolUseBlock
from mirai.agent.providers.base import ModelInfo, UsageSnapshot
from mirai.logging import get_logger

log = get_logger("mirai.providers.mock")


class MockEmbeddingProvider:
    """Provides consistent fake embeddings for testing."""

    def __init__(self, dim: int = 1536):
        self.dim = dim

    async def get_embeddings(self, text: str) -> list[float]:
        """Return a deterministic fake vector for testing."""
        return [0.1] * 1536


class MockProvider:
    """Mock provider to test the AgentLoop logic without an API key.

    Accepts messages in OpenAI format (the internal canonical format).
    """

    def __init__(self) -> None:
        self.call_count = 0
        self.model = "mock-model"

    @property
    def provider_name(self) -> str:
        return "mock"

    async def list_models(self) -> list[ModelInfo]:
        return [ModelInfo(id="mock-model", name="Mock Model")]

    async def get_usage(self) -> UsageSnapshot:
        return UsageSnapshot(provider="mock", error="not supported")

    def config_dict(self) -> dict[str, Any]:
        return {}

    async def generate_response(
        self,
        model: str,
        system: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        **kwargs: Any,
    ) -> ProviderResponse:
        self.call_count += 1

        if "Recovered Memories" in system:
            log.debug("mock_memories_present", system_tail=system[system.find("### Recovered Memories") :])

        if "# IDENTITY" in system:
            log.debug("mock_identity_anchors_present")

        last_message = messages[-1]
        last_content = last_message.get("content", "")
        if isinstance(last_content, list):
            last_text = "".join(
                [c.get("text", "") for c in last_content if isinstance(c, dict) and c.get("type") == "text"]
            )
        else:
            last_text = str(last_content) if last_content else ""

        # Extract text from all messages to find the user's original intent
        full_history_text = ""
        for m in messages:
            content = m.get("content", "")
            if isinstance(content, list):
                full_history_text += " ".join(
                    [c.get("text", "") for c in content if isinstance(c, dict) and c.get("type") == "text"]
                )
            elif content:
                full_history_text += str(content)

        # 1. Thinking Turn
        if "analyze the situation" in last_text or "perform a self-reflection" in last_text:
            text = "<thinking>I am analyzing the current request.</thinking>"
            if "maintenance_check" in last_text:
                text = "<thinking>The system heartbeat has triggered. I should scan the project and summarize progress.</thinking>"

            return ProviderResponse(
                content=[TextBlock(text=text)],
                stop_reason="end_turn",
            )

        # 2. Critique Turn
        if "Critique your response" in last_text:
            return ProviderResponse(
                content=[
                    TextBlock(
                        text="The response is aligned with my SOUL.md. Final version: I have successfully completed the proactive scan and verified alignment with SOUL.md."
                    )
                ],
                stop_reason="end_turn",
            )

        # 3. Tool Turn / Sequential Logic
        if tools:
            # 1. Executive Multi-Step Workflow (soul_summary)
            if "soul_summary" in full_history_text.lower() or "find the soul file" in full_history_text.lower():
                shell_results = self._find_tool_results(messages, "find_soul_call")
                if not shell_results:
                    return ProviderResponse(
                        content=[
                            TextBlock(text="Finding the SOUL file..."),
                            ToolUseBlock(
                                id="find_soul_call",
                                name="shell_tool",
                                input={"command": 'find . -name "*SOUL.md"'},
                            ),
                        ],
                        stop_reason="tool_use",
                    )

                editor_results = self._find_tool_results(messages, "write_summary_call")
                if not editor_results:
                    return ProviderResponse(
                        content=[
                            TextBlock(text="Writing the summary..."),
                            ToolUseBlock(
                                id="write_summary_call",
                                name="editor_tool",
                                input={
                                    "action": "write",
                                    "path": "soul_summary.txt",
                                    "content": "SOUL Summary: This is a verified identity summary.",
                                },
                            ),
                        ],
                        stop_reason="tool_use",
                    )

            # 2. E2E Proactive Maintenance Workflow
            if "maintenance_check" in full_history_text.lower():
                maint_shell_results = self._find_tool_results(messages, "call_shell_maint")

                if not maint_shell_results:
                    return ProviderResponse(
                        content=[
                            TextBlock(text="Checking for maintenance issues..."),
                            ToolUseBlock(
                                id="call_shell_maint",
                                name="shell_tool",
                                input={"command": "ls maintenance_fixed.txt"},
                            ),
                        ],
                        stop_reason="tool_use",
                    )

                maint_editor_results = self._find_tool_results(messages, "call_edit_maint")

                if not maint_editor_results:
                    return ProviderResponse(
                        content=[
                            TextBlock(text="Fixing the issue..."),
                            ToolUseBlock(
                                id="call_edit_maint",
                                name="editor_tool",
                                input={
                                    "action": "write",
                                    "path": "maintenance_fixed.txt",
                                    "content": "HEALED: Sanity check passed.",
                                },
                            ),
                        ],
                        stop_reason="tool_use",
                    )

            # 3. Workspace List / Heartbeat
            if "List the files" in last_text or "SYSTEM_HEARTBEAT" in last_text:
                if "completed the proactive scan" not in full_history_text:
                    return ProviderResponse(
                        content=[
                            TextBlock(
                                text="I have successfully completed the proactive scan and verified alignment with SOUL.md."
                            )
                        ],
                        stop_reason="end_turn",
                    )

            # 4. Memory Isolation Test
            if "wonderland" in full_history_text.lower():
                has_memorized = self._has_tool_call(messages, "memorize")

                if not has_memorized:
                    return ProviderResponse(
                        content=[
                            TextBlock(text="Memorizing Alice's secret..."),
                            ToolUseBlock(
                                id="mem_alice_secret",
                                name="memorize",
                                input={"content": "Alice's secret is: WonderLand.", "importance": 0.9},
                            ),
                        ],
                        stop_reason="tool_use",
                    )

        # Final Answer if no more tools needed
        text = "I have finished the complex task."
        if "proactive scan" in full_history_text.lower() or "system_heartbeat" in full_history_text.lower():
            text = "I have successfully completed the proactive scan and verified alignment with SOUL.md."

        return ProviderResponse(
            content=[TextBlock(text=text)],
            stop_reason="end_turn",
            model_id=self.model,
        )

    # ------------------------------------------------------------------
    # Helpers for OpenAI-format message inspection
    # ------------------------------------------------------------------

    @staticmethod
    def _find_tool_results(messages: list[dict[str, Any]], tool_call_id: str) -> list[dict[str, Any]]:
        """Find tool result messages for a specific tool_call_id.

        In OpenAI format, tool results are: {"role": "tool", "tool_call_id": "...", "content": "..."}
        """
        return [m for m in messages if m.get("role") == "tool" and m.get("tool_call_id") == tool_call_id]

    @staticmethod
    def _has_tool_call(messages: list[dict[str, Any]], tool_name: str) -> bool:
        """Check if any assistant message contains a tool_call with the given name.

        In OpenAI format, tool calls are on assistant messages:
        {"role": "assistant", "tool_calls": [{"function": {"name": "..."}}]}
        """
        for m in messages:
            if m.get("role") != "assistant":
                continue
            for tc in m.get("tool_calls", []):
                fn = tc.get("function", {})
                if fn.get("name") == tool_name:
                    return True
        return False
