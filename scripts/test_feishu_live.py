"""Live Feishu integration test using real .env credentials."""

import asyncio
import os

from dotenv import load_dotenv

load_dotenv()


async def main():
    app_id = os.getenv("FEISHU_APP_ID")
    app_secret = os.getenv("FEISHU_APP_SECRET")

    if not app_id or not app_secret:
        print("ERROR: FEISHU_APP_ID and FEISHU_APP_SECRET are not set in .env")
        return

    print(f"[1/3] Credentials loaded: APP_ID={app_id[:8]}...{app_id[-4:]}")

    from mirai.agent.im.feishu import FeishuProvider

    provider = FeishuProvider(app_id=app_id, app_secret=app_secret)

    # Step 2: Auto-discover chat
    print("[2/3] Discovering bot's group chats...")
    chat_id = await provider._discover_chat_id()

    if not chat_id:
        print("WARNING: No group chats found. Make sure the bot is added to at least one group.")
        return

    print(f"      Found chat_id: {chat_id}")

    # Step 3: Send test message
    print("[3/3] Sending test message...")
    success = await provider.send_message(
        "🤖 **Mirai Connection Test**\n\n"
        "If you see this message, Feishu integration is working!\n"
        f"Auto-discovered chat: `{chat_id}`"
    )

    if success:
        print("SUCCESS: Message sent! Check your Feishu group chat.")
    else:
        print("FAILED: Message could not be sent. Check permissions.")


if __name__ == "__main__":
    asyncio.run(main())
