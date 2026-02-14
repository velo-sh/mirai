"""
Antigravity Live Test — End-to-End validation with real Cloud Code Assist credentials.

Usage:
    uv run python scripts/test_antigravity_live.py
    uv run python scripts/test_antigravity_live.py --heartbeat   # also test a heartbeat pulse
    uv run python scripts/test_antigravity_live.py --model gemini-2.0-flash
"""

import argparse
import asyncio
import time
from typing import Any

from dotenv import load_dotenv

load_dotenv()


def _print_header(title: str) -> None:
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print(f"{'=' * 60}")


def _print_result(passed: bool, duration: float, detail: str) -> None:
    status = "✅ PASS" if passed else "❌ FAIL"
    print(f"  {status} ({duration:.1f}s) — {detail}")


async def test_simple_text(provider: Any, model: str) -> bool:
    """Test 1: Basic text prompt → text response."""
    _print_header("Test 1: Simple Text Response")
    print(f"  Model: {model}")
    print("  Prompt: 'What is 2+2? Reply with just the number.'")

    start = time.time()
    try:
        response = await provider.generate_response(
            model=model,
            system="You are a concise assistant.",
            messages=[{"role": "user", "content": "What is 2+2? Reply with just the number."}],
            tools=[],
        )
        duration = time.time() - start

        # Validate response structure
        assert hasattr(response, "content"), "Response missing 'content' attribute"
        assert len(response.content) > 0, "Response content is empty"

        text = ""
        for block in response.content:
            if block.type == "text":
                text += block.text

        assert "4" in text, f"Expected '4' in response, got: {text}"
        _print_result(True, duration, f"Response: '{text.strip()}'")
        print(f"  stop_reason: {response.stop_reason}")
        return True

    except Exception as e:
        duration = time.time() - start
        _print_result(False, duration, str(e))
        return False


async def test_tool_use(provider: Any, model: str) -> bool:
    """Test 2: Tool use round-trip with echo tool."""
    _print_header("Test 2: Tool Use Round-trip")

    echo_tool = {
        "name": "echo",
        "description": "Echoes the given message back to the user.",
        "input_schema": {
            "type": "object",
            "properties": {"message": {"type": "string", "description": "The message to echo."}},
            "required": ["message"],
        },
    }

    start = time.time()
    try:
        # Step A: Get tool_use response
        response = await provider.generate_response(  # type: ignore[union-attr]
            model=model,
            system="You are a helpful assistant. When asked to echo something, use the echo tool.",
            messages=[{"role": "user", "content": "Please echo the message 'hello mirai'"}],
            tools=[echo_tool],
        )
        duration_a = time.time() - start

        assert response.stop_reason == "tool_use", f"Expected stop_reason='tool_use', got '{response.stop_reason}'"

        # Find the tool_use block
        tool_use_block = None
        for block in response.content:
            if block.type == "tool_use":
                tool_use_block = block
                break

        assert tool_use_block is not None, "No tool_use block found in response"
        print(f"  Tool called: {tool_use_block.name}")
        print(f"  Tool input: {tool_use_block.input}")
        print(f"  Tool ID: {tool_use_block.id}")
        _print_result(True, duration_a, "Tool use detected correctly")

        # Step B: Send tool result back → get final response
        print("\n  Sending tool result back...")
        start_b = time.time()

        messages = [
            {"role": "user", "content": "Please echo the message 'hello mirai'"},
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "tool_use",
                        "id": tool_use_block.id,
                        "name": tool_use_block.name,
                        "input": tool_use_block.input,
                    }
                ],
            },
            {
                "role": "user",
                "content": [{"type": "tool_result", "tool_use_id": tool_use_block.id, "content": "hello mirai"}],
            },
        ]

        final_response = await provider.generate_response(  # type: ignore[union-attr]
            model=model,
            system="You are a helpful assistant.",
            messages=messages,
            tools=[echo_tool],
        )
        duration_b = time.time() - start_b

        final_text = ""
        for block in final_response.content:
            if block.type == "text":
                final_text += block.text

        _print_result(True, duration_b, f"Final response: '{final_text.strip()[:100]}'")
        return True

    except Exception as e:
        duration = time.time() - start
        _print_result(False, duration, str(e))
        import traceback

        traceback.print_exc()
        return False


async def test_multi_turn(provider: Any, model: str) -> bool:
    """Test 3: Multi-turn conversation with context."""
    _print_header("Test 3: Multi-turn Conversation")

    start = time.time()
    try:
        # Turn 1
        print("  Turn 1: 'My name is Antigravity.'")
        r1 = await provider.generate_response(  # type: ignore[union-attr]
            model=model,
            system="You are a friendly assistant. Remember what the user tells you.",
            messages=[{"role": "user", "content": "My name is Antigravity. Just acknowledge this."}],
            tools=[],
        )
        t1_text = "".join(b.text for b in r1.content if b.type == "text")
        print(f"  Response 1: '{t1_text.strip()[:80]}'")

        # Turn 2: test context carry-over
        print("  Turn 2: 'What is my name?'")
        r2 = await provider.generate_response(  # type: ignore[union-attr]
            model=model,
            system="You are a friendly assistant. Remember what the user tells you.",
            messages=[
                {"role": "user", "content": "My name is Antigravity. Just acknowledge this."},
                {"role": "assistant", "content": t1_text},
                {"role": "user", "content": "What is my name? Reply with just the name."},
            ],
            tools=[],
        )
        duration = time.time() - start
        t2_text = "".join(b.text for b in r2.content if b.type == "text")

        assert "antigravity" in t2_text.lower(), f"Name not recalled. Got: {t2_text}"
        _print_result(True, duration, f"Context preserved! Response: '{t2_text.strip()[:80]}'")
        return True

    except Exception as e:
        duration = time.time() - start
        _print_result(False, duration, str(e))
        return False


async def test_heartbeat_pulse(provider: Any, model: str) -> bool:
    """Optional Test 4: Single heartbeat pulse through AgentLoop."""
    _print_header("Test 4: Heartbeat Pulse (via AgentLoop)")

    start = time.time()
    try:
        from mirai.agent.agent_loop import AgentLoop
        from mirai.agent.tools.echo import EchoTool
        from mirai.agent.tools.workspace import WorkspaceTool
        from mirai.db.session import init_db

        await init_db()

        agent = await AgentLoop.create(
            provider=provider,
            tools=[EchoTool(), WorkspaceTool()],
            collaborator_id="01AN4Z048W7N7DF3SQ5G16CYAJ",
        )

        pulse_message = (
            "SYSTEM_HEARTBEAT: Perform a brief self-reflection. What is your current status? Keep it under 50 words."
        )

        response = await agent.run(pulse_message)
        duration = time.time() - start

        assert len(response) > 0, "Empty heartbeat response"
        _print_result(True, duration, f"Heartbeat insight: '{response[:120]}...'")
        return True

    except Exception as e:
        duration = time.time() - start
        _print_result(False, duration, str(e))
        import traceback

        traceback.print_exc()
        return False


async def main():
    parser = argparse.ArgumentParser(description="Antigravity Live E2E Test")
    parser.add_argument("--model", default="claude-sonnet-4-20250514", help="Model to test with")
    parser.add_argument("--heartbeat", action="store_true", help="Also test heartbeat pulse via AgentLoop")
    args = parser.parse_args()

    _print_header("Antigravity Live E2E Test Suite")
    print(f"  Model: {args.model}")

    # Initialize provider
    from mirai.agent.providers import create_provider

    try:
        provider = create_provider(model=args.model)
    except Exception as e:
        print(f"\n❌ Failed to create provider: {e}")
        print("   Make sure you've run: python -m mirai.auth.auth_cli")
        return

    # Run tests
    results = []
    results.append(("Simple Text", await test_simple_text(provider, args.model)))
    results.append(("Tool Use", await test_tool_use(provider, args.model)))
    results.append(("Multi-turn", await test_multi_turn(provider, args.model)))

    if args.heartbeat:
        results.append(("Heartbeat", await test_heartbeat_pulse(provider, args.model)))

    # Summary
    _print_header("Test Summary")
    passed = sum(1 for _, r in results if r)
    total = len(results)
    for name, result in results:
        status = "✅" if result else "❌"
        print(f"  {status} {name}")
    print(f"\n  Result: {passed}/{total} tests passed")

    if passed == total:
        print("  🎉 All tests passed! Antigravity integration is working.")
    else:
        print("  ⚠️  Some tests failed. Check the output above for details.")


if __name__ == "__main__":
    asyncio.run(main())
