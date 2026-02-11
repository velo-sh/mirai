from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
import os
from typing import Optional

from mirai.agent.providers import AnthropicProvider
from mirai.agent.loop import AgentLoop
from mirai.agent.tools.echo import EchoTool

from contextlib import asynccontextmanager
from mirai.db.session import init_db

# Global agent instance
agent: Optional[AgentLoop] = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global agent
    # Initialize SQLite tables
    await init_db()
    
    # Initialize Agent components
    try:
        provider = AnthropicProvider()
        tools = [EchoTool()]
        agent = await AgentLoop.create(
            provider=provider,
            tools=tools,
            collaborator_id="01AN4Z048W7N7DF3SQ5G16CYAJ" # Mira's ULID
        )
        print(f"AgentLoop initialized for: {agent.name}")
    except Exception as e:
        print(f"Warning: Failed to initialize AgentLoop: {e}")
        agent = None
    
    yield

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
        raise HTTPException(status_code=500, detail=str(e))

def main():
    print("Starting Mirai Node (FastAPI) at http://0.0.0.0:8000")
    uvicorn.run(app, host="0.0.0.0", port=8000)

if __name__ == "__main__":
    main()
