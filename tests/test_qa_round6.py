"""QA tests for Round 6 — Industrial-Grade Infrastructure.

Coverage areas:
  1. ToolContext Isolation & DI
  2. Storage Model Type Safety & Round-Trip
  3. AgentLoop State Machine Transitions
  4. Dynamic Runtime Hyperparameters
  5. Automated Tracing Spans
  6. LLM Output Contract Validation
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pydantic import BaseModel

from mirai.agent.agent_loop import AgentLoop, LoopState
from mirai.agent.contract import ContractError, validate_response
from mirai.agent.tools.base import BaseTool, ToolContext
from mirai.db.duck import DuckDBStorage
from mirai.db.models import DBTrace, FeishuMessage

# ===========================================================================
# 1. ToolContext Isolation
# ===========================================================================


class TestToolContextIsolation:
    """Verify tools only use the provided context."""

    def test_context_default_init(self):
        ctx = ToolContext()
        assert ctx.start_time > 0
        assert ctx.config is None

    def test_base_tool_receives_context(self):
        class MockTool(BaseTool):
            @property
            def definition(self):
                return {"name": "mock"}

            async def execute(self, **kwargs):
                return "ok"

        ctx = ToolContext(config=MagicMock())
        tool = MockTool(context=ctx)
        assert tool.context == ctx
        assert tool.context.config is not None


# ===========================================================================
# 2. Storage Model Round-Trip
# ===========================================================================


@pytest.mark.asyncio
class TestStorageModelRoundTrip:
    """Verify Pydantic models with DuckDBStorage."""

    async def test_db_trace_roundtrip(self, tmp_path):
        db_path = str(tmp_path / "test.db")
        storage = DuckDBStorage(db_path)

        trace = DBTrace(
            id="test-ulid",
            collaborator_id="user-123",
            trace_type="thought",
            content="Thinking about life",
            metadata={"depth": 42},
        )

        await storage.append_trace(trace)

        # Verify it exists
        traces = await storage.get_recent_traces("user-123", limit=1)
        assert len(traces) == 1
        assert traces[0].id == "test-ulid"
        assert traces[0].collaborator_id == "user-123"

        storage.close()

    async def test_feishu_history_roundtrip(self, tmp_path):
        storage = DuckDBStorage(str(tmp_path / "test_im.db"))

        msg = FeishuMessage(chat_id="chat-456", role="user", content="Hello world")
        await storage.save_feishu_history(msg)

        history = await storage.get_feishu_history("chat-456", limit=1)
        assert len(history) == 1
        assert history[0].role == "user"
        assert history[0].content == "Hello world"

        storage.close()


# ===========================================================================
# 3. AgentLoop State Machine
# ===========================================================================


@pytest.mark.asyncio
class TestAgentLoopStateMachine:
    """Verify state transitions in AgentLoop."""

    async def test_state_transitions_thinking_to_done(self):
        # Mock provider to return a simple text response
        provider = MagicMock()
        provider.generate_response = AsyncMock(
            return_value=MagicMock(
                stop_reason="end_turn", content=[MagicMock(text="Final answer")], model_id="gpt-mock"
            )
        )

        loop = AgentLoop(collaborator_id="test", provider=provider, tools=[], l3_storage=AsyncMock())

        states = []
        original_transition = loop._transition

        def tracked_transition(new_state):
            states.append(new_state)
            original_transition(new_state)

        loop._transition = tracked_transition

        await loop.run("Hello")

        # Cycle: IDLE -> THINKING -> DONE -> IDLE
        assert LoopState.THINKING in states
        assert LoopState.DONE in states
        assert loop.state == LoopState.IDLE

    async def test_state_transitions_with_tools(self):
        # Mock provider to return tool call then text
        resp1 = MagicMock(stop_reason="tool_use", model_id="gpt-mock")
        # We need a real ToolUseBlock for the loop to parse it? No, loop parses it.
        # But wait, loop uses response.content.
        from mirai.agent.models import TextBlock, ToolUseBlock

        resp1.content = [ToolUseBlock(id="tc-1", name="echo", input={"message": "hi"})]

        resp2 = MagicMock(stop_reason="end_turn", content=[TextBlock(text="Done")], model_id="gpt-mock")

        provider = MagicMock()
        provider.generate_response = AsyncMock(side_effect=[resp1, resp2])

        # Mock tool
        echo_tool = MagicMock()
        echo_tool.run = AsyncMock(return_value="hi")
        echo_tool.definition = {"name": "echo"}

        loop = AgentLoop(
            collaborator_id="test",
            provider=provider,
            tools=[echo_tool],
            l3_storage=AsyncMock(),
        )

        states = []
        loop._transition = lambda s: states.append(s)

        await loop.run("Test tools")

        # Sequence should include ACTING
        assert LoopState.THINKING in states
        assert LoopState.ACTING in states
        assert states.count(LoopState.THINKING) >= 2  # Once for first call, once after tool
        assert LoopState.DONE in states


# ===========================================================================
# 4. Dynamic Runtime Config
# ===========================================================================


@pytest.mark.asyncio
class TestDynamicRuntimeConfig:
    """Verify patch_runtime action works."""

    async def test_system_tool_patch_runtime(self):
        from mirai.agent.tools.system import SystemTool

        loop = MagicMock()
        loop.runtime_overrides = {}

        ctx = ToolContext(agent_loop=loop)
        tool = SystemTool(context=ctx)

        result = await tool.execute(action="patch_runtime", patch={"temperature": 0.5, "invalid": 99})

        assert "Successfully" in result or "✅" in result
        assert loop.runtime_overrides["temperature"] == 0.5
        assert "invalid" not in loop.runtime_overrides


# ===========================================================================
# 5. Automated Tracing
# ===========================================================================


@pytest.mark.asyncio
class TestAutomatedTracing:
    """Verify spans are created during tool execution."""

    async def test_base_tool_run_creates_span(self):
        class MockTool(BaseTool):
            @property
            def definition(self):
                return {"name": "test-tool"}

            async def execute(self, **kwargs):
                return "ok"

        tool = MockTool()

        with patch("mirai.tracing.get_tracer") as mock_get_tracer:
            mock_tracer = MagicMock()
            mock_get_tracer.return_value = mock_tracer

            await tool.run(param="val")

            mock_tracer.start_as_current_span.assert_called_with("tool.test-tool")


# ===========================================================================
# 6. Contract Validation
# ===========================================================================


class TestContractValidation:
    """Verify LLM output validation."""

    def test_validate_response_success(self):
        class MyModel(BaseModel):
            score: int
            reason: str

        content = '```json\n{"score": 10, "reason": "good"}\n```'
        obj = validate_response(content, MyModel)
        assert obj.score == 10
        assert obj.reason == "good"

    def test_validate_response_failure(self):
        class MyModel(BaseModel):
            id: int

        with pytest.raises(ContractError):
            validate_response('{"id": "not-an-int"}', MyModel)
