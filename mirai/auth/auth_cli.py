"""
CLI tool for Antigravity (Cloud Code Assist) management.

Usage:
    python -m mirai.auth.auth_cli login    # OAuth login
    python -m mirai.auth.auth_cli usage    # Show model quotas
    python -m mirai.auth.auth_cli status   # Show account status
"""

import argparse
import asyncio
import sys
import time
from datetime import UTC, datetime

from mirai.auth.antigravity_auth import (
    ensure_valid_credentials,
    fetch_usage,
    load_credentials,
    login,
)


def _bar(pct: float, width: int = 20) -> str:
    """Render a progress bar."""
    filled = int(pct / 100 * width)
    return "█" * filled + "░" * (width - filled)


def _reset_label(iso_ts: str | None) -> str:
    """Format reset time as a human-readable relative label."""
    if not iso_ts:
        return ""
    try:
        reset = datetime.fromisoformat(iso_ts.replace("Z", "+00:00"))
        now = datetime.now(UTC)
        delta = reset - now
        total_min = int(delta.total_seconds() / 60)
        if total_min <= 0:
            return "  (resetting now)"
        if total_min < 60:
            return f"  (resets in {total_min}m)"
        hours, mins = divmod(total_min, 60)
        if hours < 24:
            return f"  (resets in {hours}h{mins}m)"
        return f"  (resets {reset.strftime('%m-%d %H:%M')} UTC)"
    except Exception:
        return ""


async def cmd_login():
    """Run the OAuth login flow, prompting for credentials if needed."""
    from mirai.auth.antigravity_auth import _get_auth_config

    existing = load_credentials()
    if existing:
        print(f"Existing credentials found for: {existing.get('email', 'unknown')}")
        answer = input("Re-authenticate? [y/N] ").strip().lower()
        if answer != "y":
            print("Keeping existing credentials.")
            return

    # Check if client credentials are available
    cfg = _get_auth_config()
    if not cfg.client_id or not cfg.client_secret:
        print("\n╔══════════════════════════════════════════════════════════╗")
        print("║         Google OAuth Client Setup Required              ║")
        print("╚══════════════════════════════════════════════════════════╝")
        print()
        print("  To authenticate, you need a Google OAuth Client ID.")
        print("  If you don't have one yet:")
        print()
        print("  1. Go to: https://console.cloud.google.com/apis/credentials")
        print("  2. Click '+ CREATE CREDENTIALS' → 'OAuth client ID'")
        print("  3. Application type: 'Desktop app'")
        print("  4. Copy the Client ID and Client Secret below")
        print()

        client_id = cfg.client_id
        client_secret = cfg.client_secret

        if not client_id:
            client_id = input("  Client ID: ").strip()
            if not client_id:
                print("\n  ❌ Client ID is required.")
                sys.exit(1)

        if not client_secret:
            client_secret = input("  Client Secret: ").strip()
            if not client_secret:
                print("\n  ❌ Client Secret is required.")
                sys.exit(1)

        # Inject into AuthConfig for this session
        from mirai.config import AuthConfig

        cfg = AuthConfig(
            client_id=client_id,
            client_secret=client_secret,
            auth_url=cfg.auth_url,
            token_url=cfg.token_url,
            redirect_uri=cfg.redirect_uri,
            code_assist_endpoints=cfg.code_assist_endpoints,
        )
        print()

    try:
        await login(auth_config=cfg)
    except KeyboardInterrupt:
        print("\nLogin cancelled.")
        sys.exit(1)
    except Exception as e:
        print(f"\nLogin failed: {e}")
        sys.exit(1)


async def cmd_usage():
    """Show per-model quota usage."""
    try:
        creds = await ensure_valid_credentials()
    except FileNotFoundError as e:
        print(str(e))
        sys.exit(1)

    print("Fetching usage data...\n")
    usage = await fetch_usage(creds["access"], creds.get("project_id", ""))

    # Header
    plan = usage.get("plan") or "Unknown"
    project = usage.get("project") or "Unknown"
    email = creds.get("email") or "Unknown"
    print(f"  Account:  {email}")
    print(f"  Plan:     {plan:<10s}")
    print(f"  Project:  {project:<10s}")
    print()

    # Model quotas
    models = usage.get("models", [])
    if not models:
        print("  No model quota data available.")
        return

    # Filter out internal models (chat_*, tab_*)
    visible = [m for m in models if not m["id"].startswith(("chat_", "tab_"))]
    if not visible:
        print("  No user-facing model quotas found.")
        return

    # Sort: most used first, then alphabetical
    visible.sort(key=lambda m: (-m["used_pct"], m["id"]))

    # Table Header
    print(f"  {'Model':<37s}  {'Quota':<28s}  Reset")
    print(f"  {'─' * 37}  {'─' * 28}  {'─' * 18}")
    for m in visible:
        pct = m["used_pct"]
        bar = _bar(pct)
        reset_label = _reset_label(m.get("reset_time")).strip()
        status = "🟢" if pct < 50 else "🟡" if pct < 80 else "🔴" if pct < 100 else "⚠️"

        # Standard: one space after emoji
        model_part = f"{status} {m['id']:<33s}"

        # Quota part: bar(20) + space(1) + pct(5.1) + space(1) = 28
        quota_part = f"{bar} {pct:5.1f}% "

        print(f"  {model_part}  {quota_part}  {reset_label}")

    print()


async def cmd_status():
    """Show brief account status."""
    creds = load_credentials()
    if not creds:
        print("Not authenticated. Run: python -m mirai.auth.auth_cli login")
        sys.exit(1)

    email = creds.get("email", "unknown")
    project = creds.get("project_id", "unknown")
    expires = creds.get("expires", 0)
    remaining = max(0, int(expires - time.time()))
    hours, mins = divmod(remaining // 60, 60)

    print(f"  Email:    {email}")
    print(f"  Project:  {project}")
    if remaining > 0:
        print(f"  Token:    valid ({hours}h{mins}m remaining)")
    else:
        print("  Token:    expired (will auto-refresh)")


def main():
    parser = argparse.ArgumentParser(
        prog="mirai.auth",
        description="Antigravity (Cloud Code Assist) CLI",
    )
    sub = parser.add_subparsers(dest="command")

    sub.add_parser("login", help="Authenticate with Google OAuth")
    sub.add_parser("usage", help="Show per-model quota usage")
    sub.add_parser("status", help="Show account status")

    args = parser.parse_args()

    if args.command == "login":
        asyncio.run(cmd_login())
    elif args.command == "usage":
        asyncio.run(cmd_usage())
    elif args.command == "status":
        asyncio.run(cmd_status())
    else:
        # Default: show usage if credentials exist, else prompt login
        creds = load_credentials()
        if creds:
            asyncio.run(cmd_usage())
        else:
            parser.print_help()
            print("\nNo credentials found. Start with: python -m mirai.auth.auth_cli login")


if __name__ == "__main__":
    main()
