from fastapi import FastAPI
import uvicorn
import os

app = FastAPI(title="Mirai Node")

@app.get("/health")
async def health_check():
    """Simple health check for the watchdog."""
    return {"status": "ok", "pid": os.getpid()}

def main():
    print("Starting Mirai Node (FastAPI)...")
    uvicorn.run(app, host="0.0.0.0", port=8000)

if __name__ == "__main__":
    main()
