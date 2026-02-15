"""
Google Antigravity (Cloud Code Assist) OAuth authentication.

Ported from openclaw's google-antigravity-auth extension.
Uses Google OAuth2 PKCE flow to obtain tokens that proxy Claude API calls
through Google Cloud's Code Assist infrastructure.
"""

import asyncio
import hashlib
import os
import secrets
import time
import webbrowser
from base64 import urlsafe_b64encode
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from threading import Thread
from typing import Any
from urllib.parse import parse_qs, urlencode, urlparse

import httpx
import orjson

# Shared HTTP client for connection pooling + HTTP/2
_http = httpx.AsyncClient(timeout=30.0, http2=True)

# OAuth constants (from openclaw's google-antigravity-auth extension)
CLIENT_ID = "***REDACTED_CLIENT_ID***"
CLIENT_SECRET = "***REDACTED_SECRET***"
REDIRECT_URI = "http://localhost:51121/oauth-callback"
AUTH_URL = "https://accounts.google.com/o/oauth2/v2/auth"
TOKEN_URL = "https://oauth2.googleapis.com/token"

DEFAULT_PROJECT_ID = "rising-fact-p41fc"
DEFAULT_MODEL = "claude-sonnet-4-5-20250514"

SCOPES = [
    "https://www.googleapis.com/auth/cloud-platform",
    "https://www.googleapis.com/auth/userinfo.email",
    "https://www.googleapis.com/auth/userinfo.profile",
    "https://www.googleapis.com/auth/cclog",
    "https://www.googleapis.com/auth/experimentsandconfigs",
]

CODE_ASSIST_ENDPOINTS = [
    "https://cloudcode-pa.googleapis.com",
    "https://daily-cloudcode-pa.sandbox.googleapis.com",
]

CREDENTIALS_PATH = Path.home() / ".mirai" / "antigravity_credentials.json"

RESPONSE_HTML = """<!DOCTYPE html>
<html lang="en">
<head><meta charset="utf-8"><title>Mirai Antigravity OAuth</title></head>
<body>
  <main>
    <h1>Authentication complete</h1>
    <p>You can return to the terminal.</p>
  </main>
</body>
</html>"""


def _generate_pkce() -> tuple[str, str]:
    """Generate PKCE code verifier and challenge."""
    verifier = secrets.token_hex(32)
    digest = hashlib.sha256(verifier.encode()).digest()
    challenge = urlsafe_b64encode(digest).rstrip(b"=").decode()
    return verifier, challenge


def _build_auth_url(challenge: str, state: str) -> str:
    """Build the Google OAuth authorization URL."""
    params = {
        "client_id": CLIENT_ID,
        "response_type": "code",
        "redirect_uri": REDIRECT_URI,
        "scope": " ".join(SCOPES),
        "code_challenge": challenge,
        "code_challenge_method": "S256",
        "state": state,
        "access_type": "offline",
        "prompt": "consent",
    }
    return f"{AUTH_URL}?{urlencode(params)}"


class _OAuthCallbackHandler(BaseHTTPRequestHandler):
    """HTTP handler that captures the OAuth callback."""

    code: str | None = None
    state: str | None = None

    def do_GET(self):
        parsed = urlparse(self.path)
        if parsed.path != "/oauth-callback":
            self.send_response(404)
            self.end_headers()
            self.wfile.write(b"Not found")
            return

        query = parse_qs(parsed.query)
        _OAuthCallbackHandler.code = query.get("code", [None])[0]
        _OAuthCallbackHandler.state = query.get("state", [None])[0]

        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.end_headers()
        self.wfile.write(RESPONSE_HTML.encode())

    def log_message(self, format, *args):
        pass  # Suppress server logs


async def exchange_code(code: str, verifier: str) -> dict:
    """Exchange authorization code for access + refresh tokens."""
    response = await _http.post(
        TOKEN_URL,
        data={
            "client_id": CLIENT_ID,
            "client_secret": CLIENT_SECRET,
            "code": code,
            "grant_type": "authorization_code",
            "redirect_uri": REDIRECT_URI,
            "code_verifier": verifier,
        },
    )
    response.raise_for_status()
    data = response.json()

    access = data.get("access_token", "").strip()
    refresh = data.get("refresh_token", "").strip()
    expires_in = data.get("expires_in", 0)

    if not access:
        raise ValueError("Token exchange returned no access_token")
    if not refresh:
        raise ValueError("Token exchange returned no refresh_token")

    expires = int(time.time()) + expires_in - 300  # 5 min buffer
    return {"access": access, "refresh": refresh, "expires": expires}


async def refresh_access_token(refresh_token: str) -> dict:
    """Refresh an expired access token using the refresh token."""
    response = await _http.post(
        TOKEN_URL,
        data={
            "client_id": CLIENT_ID,
            "client_secret": CLIENT_SECRET,
            "refresh_token": refresh_token,
            "grant_type": "refresh_token",
        },
    )
    response.raise_for_status()
    data = response.json()

    access = data.get("access_token", "").strip()
    expires_in = data.get("expires_in", 0)

    if not access:
        raise ValueError("Token refresh returned no access_token")

    expires = int(time.time()) + expires_in - 300
    return {"access": access, "expires": expires}


async def fetch_project_id(access_token: str) -> str:
    """Fetch the Cloud AI Companion project ID."""
    headers = {
        "Authorization": f"Bearer {access_token}",
        "Content-Type": "application/json",
        "User-Agent": "google-api-nodejs-client/9.15.1",
        "X-Goog-Api-Client": "google-cloud-sdk vscode_cloudshelleditor/0.1",
        "Client-Metadata": orjson.dumps(
            {
                "ideType": "IDE_UNSPECIFIED",
                "platform": "PLATFORM_UNSPECIFIED",
                "pluginType": "GEMINI",
            }
        ).decode(),
    }

    body = orjson.dumps(
        {
            "metadata": {
                "ideType": "IDE_UNSPECIFIED",
                "platform": "PLATFORM_UNSPECIFIED",
                "pluginType": "GEMINI",
            }
        }
    )

    for endpoint in CODE_ASSIST_ENDPOINTS:
        try:
            response = await _http.post(
                f"{endpoint}/v1internal:loadCodeAssist",
                headers=headers,
                content=body,
            )
            if not response.is_success:
                continue

            data = response.json()
            project = data.get("cloudaicompanionProject")
            if isinstance(project, str) and project.strip():
                return str(project)
            if isinstance(project, dict) and project.get("id"):
                return str(project["id"])
        except Exception:
            continue

    return DEFAULT_PROJECT_ID


async def fetch_user_email(access_token: str) -> str | None:
    """Fetch the authenticated user's email."""
    try:
        response = await _http.get(
            "https://www.googleapis.com/oauth2/v1/userinfo?alt=json",
            headers={"Authorization": f"Bearer {access_token}"},
        )
        if response.is_success:
            data = response.json()
            email = data.get("email")
            if isinstance(email, str):
                return email
    except Exception:
        pass
    return None


def save_credentials(credentials: dict) -> None:
    """Save credentials to ~/.mirai/antigravity_credentials.json."""
    CREDENTIALS_PATH.parent.mkdir(parents=True, exist_ok=True)
    CREDENTIALS_PATH.write_text(orjson.dumps(credentials, option=orjson.OPT_INDENT_2).decode())
    os.chmod(CREDENTIALS_PATH, 0o600)
    print(f"Credentials saved to {CREDENTIALS_PATH}")


def load_credentials() -> dict | None:
    """Load credentials from disk. Returns None if not found."""
    if not CREDENTIALS_PATH.exists():
        return None
    try:
        return dict(orjson.loads(CREDENTIALS_PATH.read_bytes()))
    except (orjson.JSONDecodeError, OSError):
        return None


async def ensure_valid_credentials() -> dict:
    """Load credentials and refresh if expired. Returns valid credentials."""
    creds = load_credentials()
    if not creds:
        raise FileNotFoundError(
            f"No Antigravity credentials found at {CREDENTIALS_PATH}. "
            "Run `python -m mirai.auth.auth_cli` to authenticate."
        )

    if time.time() >= creds.get("expires", 0):
        print("[antigravity] Access token expired, refreshing...")
        refreshed = await refresh_access_token(creds["refresh"])
        creds["access"] = refreshed["access"]
        creds["expires"] = refreshed["expires"]
        save_credentials(creds)
        print("[antigravity] Token refreshed successfully.")

    return creds


async def login() -> dict:
    """
    Run the full OAuth PKCE login flow:
    1. Start local callback server on :51121
    2. Open Google sign-in in browser
    3. Wait for callback with auth code
    4. Exchange code for tokens
    5. Fetch projectId and email
    6. Save credentials to disk
    """
    verifier, challenge = _generate_pkce()
    state = secrets.token_hex(16)
    auth_url = _build_auth_url(challenge, state)

    # Reset handler state
    _OAuthCallbackHandler.code = None
    _OAuthCallbackHandler.state = None

    # Start callback server
    server = HTTPServer(("127.0.0.1", 51121), _OAuthCallbackHandler)
    server_thread = Thread(target=server.serve_forever, daemon=True)
    server_thread.start()

    print("\n=== Google Antigravity OAuth Login ===\n")
    print("Opening Google sign-in in your browser...\n")
    print(f"If it doesn't open automatically, visit:\n{auth_url}\n")
    webbrowser.open(auth_url)

    # Wait for callback
    print("Waiting for OAuth callback...")
    while _OAuthCallbackHandler.code is None:
        await asyncio.sleep(0.5)

    server.shutdown()

    code = _OAuthCallbackHandler.code
    returned_state = _OAuthCallbackHandler.state

    if not code:
        raise ValueError("Missing OAuth code from callback")
    if returned_state != state:
        raise ValueError("OAuth state mismatch. Please try again.")

    print("Exchanging code for tokens...")
    tokens = await exchange_code(code, verifier)

    print("Fetching project ID...")
    project_id = await fetch_project_id(tokens["access"])

    print("Fetching user email...")
    email = await fetch_user_email(tokens["access"])

    credentials = {
        **tokens,
        "project_id": project_id,
        "email": email,
    }

    save_credentials(credentials)

    print("\n✅ Antigravity OAuth complete!")
    print(f"   Email: {email or 'unknown'}")
    print(f"   Project: {project_id}")

    return credentials


async def fetch_usage(access_token: str, project_id: str = "") -> dict:
    """
    Fetch Antigravity account usage and per-model quotas.

    Returns dict with keys:
        plan: str | None
        project: str | None
        credits_available: float | None
        credits_monthly: float | None
        models: list of {id, used_pct, reset_time}
    """
    base = "https://cloudcode-pa.googleapis.com"
    headers = {
        "Authorization": f"Bearer {access_token}",
        "Content-Type": "application/json",
        "User-Agent": "antigravity",
        "X-Goog-Api-Client": "google-cloud-sdk vscode_cloudshelleditor/0.1",
    }
    metadata = {
        "ideType": "ANTIGRAVITY",
        "platform": "PLATFORM_UNSPECIFIED",
        "pluginType": "GEMINI",
    }

    result: dict[str, Any] = {
        "plan": None,
        "project": None,
        "credits_available": None,
        "credits_monthly": None,
        "models": [],
    }

    async with httpx.AsyncClient(timeout=15.0, http2=True) as client:
        # Account info
        try:
            r = await client.post(
                f"{base}/v1internal:loadCodeAssist",
                headers=headers,
                content=orjson.dumps({"metadata": metadata}),
            )
            if r.status_code == 200:
                data = r.json()
                result["plan"] = data.get("currentTier", {}).get("name") or data.get("planType")
                proj = data.get("cloudaicompanionProject")
                if isinstance(proj, str):
                    result["project"] = proj
                elif isinstance(proj, dict):
                    result["project"] = proj.get("id")
                plan_info = data.get("planInfo", {})
                if plan_info.get("monthlyPromptCredits"):
                    result["credits_monthly"] = float(plan_info["monthlyPromptCredits"])
                avail = data.get("availablePromptCredits")
                if avail is not None:
                    result["credits_available"] = float(avail)
        except Exception:
            pass

        # Per-model quotas
        try:
            body = orjson.dumps({"project": project_id} if project_id else {})
            r = await client.post(
                f"{base}/v1internal:fetchAvailableModels",
                headers=headers,
                content=body,
            )
            if r.status_code == 200:
                models = r.json().get("models", {})
                for model_id, info in sorted(models.items()):
                    quota = info.get("quotaInfo", {})
                    remaining = quota.get("remainingFraction")
                    reset_time = quota.get("resetTime")

                    if remaining is not None:
                        used_pct = (1.0 - float(remaining)) * 100
                    elif reset_time:
                        # Missing remainingFraction but has a resetTime usually means exhausted/restricted
                        used_pct = 100.0
                    else:
                        used_pct = 0.0

                    result["models"].append(
                        {
                            "id": model_id,
                            "used_pct": used_pct,
                            "reset_time": reset_time,
                        }
                    )
        except Exception:
            pass

    return result
