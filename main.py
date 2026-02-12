import asyncio
import os
from contextlib import asynccontextmanager

import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

load_dotenv()

from mirai.agent.heartbeat import HeartbeatManager  # noqa: E402
from mirai.agent.loop import AgentLoop  # noqa: E402
from mirai.agent.providers import create_provider  # noqa: E402
from mirai.agent.tools.echo import EchoTool  # noqa: E402
from mirai.agent.tools.workspace import WorkspaceTool  # noqa: E402
from mirai.db.session import init_db  # noqa: E402


class ChatRequest(BaseModel):
    message: str


# Global instances
agent: AgentLoop | None = None
heartbeat: HeartbeatManager | None = None


@asynccontextmanager
async def lifespan(app_instance: FastAPI):
    global agent, heartbeat
    # Initialize SQLite tables
    await init_db()

    # Initialize Agent components
    try:
        provider = create_provider()

        from mirai.agent.tools.editor import EditorTool
        from mirai.agent.tools.shell import ShellTool

        tools = [EchoTool(), WorkspaceTool(), ShellTool(), EditorTool()]
        agent = await AgentLoop.create(
            provider=provider,
            tools=tools,
            collaborator_id="01AN4Z048W7N7DF3SQ5G16CYAJ",  # Mira's ULID
        )
        print(f"AgentLoop initialized for: {agent.name}")

        # Start Heartbeat with Optional IM Integration
        feishu_webhook = os.getenv("FEISHU_WEBHOOK_URL")
        feishu_app_id = os.getenv("FEISHU_APP_ID")
        feishu_app_secret = os.getenv("FEISHU_APP_SECRET")

        im_provider = None
        if feishu_app_id and feishu_app_secret:
            from mirai.agent.im.feishu import FeishuProvider

            im_provider = FeishuProvider(app_id=feishu_app_id, app_secret=feishu_app_secret)
            print("Feishu IM notification enabled via App API (chat auto-discovery).")
        elif feishu_webhook:
            from mirai.agent.im.feishu import FeishuProvider

            im_provider = FeishuProvider(webhook_url=feishu_webhook)
            print("Feishu IM notification enabled via Webhook.")

        heartbeat_interval = int(os.getenv("HEARTBEAT_INTERVAL", "120"))
        heartbeat = HeartbeatManager(
            agent,
            interval_seconds=heartbeat_interval,
            im_provider=im_provider,
        )
        await heartbeat.start()

        # Start Feishu WebSocket receiver for private/group chat
        if feishu_app_id and feishu_app_secret:
            from mirai.agent.im.feishu_receiver import FeishuEventReceiver

            async def handle_feishu_message(sender_id: str, text: str, chat_id: str) -> str:
                """Route incoming Feishu messages to AgentLoop."""
                if agent:
                    return await agent.run(text)
                return "Agent is not initialized."

            receiver = FeishuEventReceiver(
                app_id=feishu_app_id,
                app_secret=feishu_app_secret,
                message_handler=handle_feishu_message,
            )
            receiver.start(loop=asyncio.get_running_loop())
            print("Feishu WebSocket receiver started. You can now chat with Mira!")

    except Exception as e:
        print(f"Warning: Failed to initialize AgentLoop: {e}")
        agent = None

    yield
    if heartbeat:
        await heartbeat.stop()


app = FastAPI(lifespan=lifespan)


@app.get("/health")
async def health_check():
    """Simple health check for the watchdog."""
    return {"status": "ok", "pid": os.getpid()}


@app.post("/chat")
async def chat(request: ChatRequest):
    """Entry point for interacting with the AI Collaborator."""
    if not agent:
        raise HTTPException(status_code=500, detail="AgentLoop not initialized. Check API keys.")

    try:
        response = await agent.run(request.message)
        return {"response": response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


def main():
    print("Starting Mirai Node (FastAPI) at http://0.0.0.0:8000")
    uvicorn.run(app, host="0.0.0.0", port=8000)


if __name__ == "__main__":
    main()
