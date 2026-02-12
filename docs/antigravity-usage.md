# Antigravity Authentication & Usage

Mirai integrates with [Google Antigravity](https://cloudcode-pa.googleapis.com) (Cloud Code Assist) to access Claude and Gemini models without a separate API key.

## Quick Start

```bash
# 1. Login (opens browser for Google OAuth)
uv run python -m mirai.auth.auth_cli login

# 2. Check model quotas
uv run python -m mirai.auth.auth_cli usage

# 3. Check account status
uv run python -m mirai.auth.auth_cli status
```

Running without a subcommand defaults to `usage` if credentials exist.

## Commands

### `login`

Opens a browser for Google OAuth2 PKCE authentication. On success, saves credentials to `~/.mirai/antigravity_credentials.json`.

```
$ uv run python -m mirai.auth.auth_cli login

=== Google Antigravity OAuth Login ===
Opening Google sign-in in your browser...

✅ Antigravity OAuth complete!
   Email: user@gmail.com
   Project: inspirational-medium-bs8qv
```

### `usage`

Displays per-model quota usage with progress bars and reset countdowns.

```
$ uv run python -m mirai.auth.auth_cli usage

  Account:  user@gmail.com
  Plan:     Antigravity
  Project:  inspirational-medium-bs8qv

  Model                                                  Quota  Reset
  ────────────────────────────────────  ──────────────────────  ──────────────────
  🟢 claude-sonnet-4-5                   ████████░░░░░░░░░░░░  40.0%  (resets in 37m)
  🟢 gemini-3-flash                      ░░░░░░░░░░░░░░░░░░░░   0.0%  (resets in 4h11m)
```

Status indicators:
- 🟢 `< 50%` used — healthy
- 🟡 `50–80%` used — approaching limit
- 🔴 `> 80%` used — near exhaustion, expect 429 errors

### `status`

Shows account info and token validity.

```
$ uv run python -m mirai.auth.auth_cli status

  Email:    user@gmail.com
  Project:  inspirational-medium-bs8qv
  Token:    valid (0h45m remaining)
```

## Available Models

| Model | Type | Notes |
|---|---|---|
| `claude-sonnet-4-5` | Claude | Default for Mirai |
| `claude-sonnet-4-5-thinking` | Claude | Extended thinking |
| `claude-opus-4-5-thinking` | Claude | Most capable |
| `claude-opus-4-6-thinking` | Claude | Latest Opus |
| `gemini-3-flash` | Gemini | Fast, generous quota |
| `gemini-3-pro-high` | Gemini | High quality |
| `gemini-2.5-flash` | Gemini | Fast, lightweight |
| `gemini-2.5-pro` | Gemini | Balanced |
| `gpt-oss-120b-medium` | GPT-OSS | Community model |

## How It Works

1. **Authentication**: OAuth2 PKCE flow via `accounts.google.com` → tokens saved locally
2. **API Routing**: All model calls go through `cloudcode-pa.googleapis.com/v1internal:streamGenerateContent` using Google Generative AI format
3. **Auto-Refresh**: Expired access tokens are refreshed automatically using the stored refresh token
4. **Provider Detection**: `main.py` auto-detects Antigravity credentials at startup; falls back to `ANTHROPIC_API_KEY` if not found

## Credential Storage

```
~/.mirai/antigravity_credentials.json
```

Contains `access` token, `refresh` token, `project_id`, `email`, and `expires` timestamp. File is created with `0600` permissions.

## Rate Limits

Antigravity uses a sliding-window quota system per model. When a model quota is exhausted (429 error), Mirai retries up to 3 times with exponential backoff. Use `usage` command to check remaining quota and reset times.
