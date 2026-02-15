"""Tests for mirai.auth.antigravity_auth — OAuth helpers and credential management."""

import time
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from mirai.auth.antigravity_auth import (
    _build_auth_url,
    _generate_pkce,
    ensure_valid_credentials,
    exchange_code,
    fetch_project_id,
    fetch_usage,
    fetch_user_email,
    load_credentials,
    refresh_access_token,
    save_credentials,
)

# ---------------------------------------------------------------------------
# PKCE helpers
# ---------------------------------------------------------------------------


class TestPKCE:
    """Tests for PKCE code generation."""

    def test_generate_pkce_returns_tuple(self):
        """Returns a (verifier, challenge) tuple of strings."""
        verifier, challenge = _generate_pkce()
        assert isinstance(verifier, str)
        assert isinstance(challenge, str)

    def test_generate_pkce_verifier_length(self):
        """Verifier is 64 hex chars (32 random bytes)."""
        verifier, _ = _generate_pkce()
        assert len(verifier) == 64

    def test_generate_pkce_challenge_is_base64url(self):
        """Challenge is base64url-encoded without padding."""
        _, challenge = _generate_pkce()
        assert "=" not in challenge
        assert "+" not in challenge
        assert "/" not in challenge

    def test_generate_pkce_unique(self):
        """Each call produces a unique verifier."""
        v1, _ = _generate_pkce()
        v2, _ = _generate_pkce()
        assert v1 != v2


class TestBuildAuthUrl:
    """Tests for OAuth URL construction."""

    def test_contains_required_params(self):
        """URL contains all required OAuth parameters."""
        url = _build_auth_url("test_challenge", "test_state")
        assert "client_id=" in url
        assert "response_type=code" in url
        assert "redirect_uri=" in url
        assert "code_challenge=test_challenge" in url
        assert "code_challenge_method=S256" in url
        assert "state=test_state" in url
        assert "access_type=offline" in url
        assert "prompt=consent" in url

    def test_starts_with_auth_url(self):
        """URL starts with Google's OAuth endpoint."""
        url = _build_auth_url("c", "s")
        assert url.startswith("https://accounts.google.com/o/oauth2/v2/auth?")

    def test_includes_scopes(self):
        """URL includes at least the cloud-platform scope."""
        url = _build_auth_url("c", "s")
        assert "cloud-platform" in url


# ---------------------------------------------------------------------------
# Credential persistence
# ---------------------------------------------------------------------------


class TestCredentialPersistence:
    """Tests for save/load credentials."""

    def test_save_and_load_roundtrip(self, tmp_path, monkeypatch):
        """Credentials survive a save → load cycle."""
        creds_path = tmp_path / "creds.json"
        monkeypatch.setattr("mirai.auth.antigravity_auth.CREDENTIALS_PATH", creds_path)

        creds = {"access": "tok123", "refresh": "ref456", "expires": 9999999999}
        save_credentials(creds)
        loaded = load_credentials()

        assert loaded is not None
        assert loaded["access"] == "tok123"
        assert loaded["refresh"] == "ref456"

    def test_load_missing_file(self, tmp_path, monkeypatch):
        """load_credentials returns None when file doesn't exist."""
        creds_path = tmp_path / "nonexistent.json"
        monkeypatch.setattr("mirai.auth.antigravity_auth.CREDENTIALS_PATH", creds_path)
        assert load_credentials() is None

    def test_load_corrupt_file(self, tmp_path, monkeypatch):
        """load_credentials returns None for invalid JSON."""
        creds_path = tmp_path / "bad.json"
        creds_path.write_text("not valid json {{{")
        monkeypatch.setattr("mirai.auth.antigravity_auth.CREDENTIALS_PATH", creds_path)
        assert load_credentials() is None

    def test_save_creates_parent_dirs(self, tmp_path, monkeypatch):
        """save_credentials creates parent directories if needed."""
        creds_path = tmp_path / "subdir" / "deep" / "creds.json"
        monkeypatch.setattr("mirai.auth.antigravity_auth.CREDENTIALS_PATH", creds_path)
        save_credentials({"access": "x", "refresh": "y", "expires": 0})
        assert creds_path.exists()


# ---------------------------------------------------------------------------
# ensure_valid_credentials
# ---------------------------------------------------------------------------


class TestEnsureValidCredentials:
    """Tests for credential validation and refresh logic."""

    @pytest.mark.asyncio
    async def test_raises_when_no_credentials(self, monkeypatch):
        """Raises FileNotFoundError if no credentials on disk."""
        monkeypatch.setattr("mirai.auth.antigravity_auth.load_credentials", lambda: None)
        with pytest.raises(FileNotFoundError, match="No Antigravity credentials"):
            await ensure_valid_credentials()

    @pytest.mark.asyncio
    async def test_returns_valid_credentials(self, monkeypatch):
        """Returns credentials directly when not expired."""
        future_time = int(time.time()) + 3600
        creds = {"access": "tok", "refresh": "ref", "expires": future_time}
        monkeypatch.setattr("mirai.auth.antigravity_auth.load_credentials", lambda: creds)
        result = await ensure_valid_credentials()
        assert result["access"] == "tok"

    @pytest.mark.asyncio
    async def test_refreshes_expired_credentials(self, monkeypatch):
        """Refreshes and saves when credentials are expired."""
        expired_creds = {"access": "old", "refresh": "ref", "expires": 0}
        monkeypatch.setattr("mirai.auth.antigravity_auth.load_credentials", lambda: expired_creds)

        future = int(time.time()) + 3600
        mock_refresh = AsyncMock(return_value={"access": "new_tok", "expires": future})
        monkeypatch.setattr("mirai.auth.antigravity_auth.refresh_access_token", mock_refresh)

        mock_save = MagicMock()
        monkeypatch.setattr("mirai.auth.antigravity_auth.save_credentials", mock_save)

        result = await ensure_valid_credentials()
        assert result["access"] == "new_tok"
        mock_refresh.assert_awaited_once_with("ref")
        mock_save.assert_called_once()


# ---------------------------------------------------------------------------
# Async API functions (mocked HTTP)
# ---------------------------------------------------------------------------


class TestExchangeCode:
    """Tests for OAuth code exchange."""

    @pytest.mark.asyncio
    async def test_exchange_code_success(self, monkeypatch):
        """Successful token exchange returns access + refresh + expires."""
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "access_token": "at_123",
            "refresh_token": "rt_456",
            "expires_in": 3600,
        }
        mock_response.raise_for_status = MagicMock()

        mock_http = AsyncMock()
        mock_http.post.return_value = mock_response
        monkeypatch.setattr("mirai.auth.antigravity_auth._http", mock_http)

        result = await exchange_code("code123", "verifier456")
        assert result["access"] == "at_123"
        assert result["refresh"] == "rt_456"
        assert result["expires"] > 0

    @pytest.mark.asyncio
    async def test_exchange_code_no_access_token(self, monkeypatch):
        """Raises ValueError if token exchange returns no access_token."""
        mock_response = MagicMock()
        mock_response.json.return_value = {"refresh_token": "rt"}
        mock_response.raise_for_status = MagicMock()

        mock_http = AsyncMock()
        mock_http.post.return_value = mock_response
        monkeypatch.setattr("mirai.auth.antigravity_auth._http", mock_http)

        with pytest.raises(ValueError, match="no access_token"):
            await exchange_code("code", "verifier")

    @pytest.mark.asyncio
    async def test_exchange_code_no_refresh_token(self, monkeypatch):
        """Raises ValueError if token exchange returns no refresh_token."""
        mock_response = MagicMock()
        mock_response.json.return_value = {"access_token": "at"}
        mock_response.raise_for_status = MagicMock()

        mock_http = AsyncMock()
        mock_http.post.return_value = mock_response
        monkeypatch.setattr("mirai.auth.antigravity_auth._http", mock_http)

        with pytest.raises(ValueError, match="no refresh_token"):
            await exchange_code("code", "verifier")


class TestRefreshAccessToken:
    """Tests for token refresh."""

    @pytest.mark.asyncio
    async def test_refresh_success(self, monkeypatch):
        """Successful refresh returns new access token + expires."""
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "access_token": "new_at",
            "expires_in": 3600,
        }
        mock_response.raise_for_status = MagicMock()

        mock_http = AsyncMock()
        mock_http.post.return_value = mock_response
        monkeypatch.setattr("mirai.auth.antigravity_auth._http", mock_http)

        result = await refresh_access_token("rt_123")
        assert result["access"] == "new_at"

    @pytest.mark.asyncio
    async def test_refresh_no_access_token(self, monkeypatch):
        """Raises ValueError if refresh returns no access_token."""
        mock_response = MagicMock()
        mock_response.json.return_value = {}
        mock_response.raise_for_status = MagicMock()

        mock_http = AsyncMock()
        mock_http.post.return_value = mock_response
        monkeypatch.setattr("mirai.auth.antigravity_auth._http", mock_http)

        with pytest.raises(ValueError, match="no access_token"):
            await refresh_access_token("rt_123")


class TestFetchUserEmail:
    """Tests for user email fetching."""

    @pytest.mark.asyncio
    async def test_fetch_email_success(self, monkeypatch):
        """Returns email on successful API call."""
        mock_response = MagicMock()
        mock_response.is_success = True
        mock_response.json.return_value = {"email": "user@example.com"}

        mock_http = AsyncMock()
        mock_http.get.return_value = mock_response
        monkeypatch.setattr("mirai.auth.antigravity_auth._http", mock_http)

        email = await fetch_user_email("token123")
        assert email == "user@example.com"

    @pytest.mark.asyncio
    async def test_fetch_email_failure(self, monkeypatch):
        """Returns None on API failure."""
        mock_response = MagicMock()
        mock_response.is_success = False

        mock_http = AsyncMock()
        mock_http.get.return_value = mock_response
        monkeypatch.setattr("mirai.auth.antigravity_auth._http", mock_http)

        email = await fetch_user_email("token123")
        assert email is None

    @pytest.mark.asyncio
    async def test_fetch_email_exception(self, monkeypatch):
        """Returns None on network exception."""
        mock_http = AsyncMock()
        mock_http.get.side_effect = httpx.ConnectError("timeout")
        monkeypatch.setattr("mirai.auth.antigravity_auth._http", mock_http)

        email = await fetch_user_email("token123")
        assert email is None


class TestFetchProjectId:
    """Tests for Cloud AI Companion project ID fetching."""

    @pytest.mark.asyncio
    async def test_fetch_project_id_string(self, monkeypatch):
        """Returns project ID when response contains a string."""
        mock_response = MagicMock()
        mock_response.is_success = True
        mock_response.json.return_value = {"cloudaicompanionProject": "my-project-123"}

        mock_http = AsyncMock()
        mock_http.post.return_value = mock_response
        monkeypatch.setattr("mirai.auth.antigravity_auth._http", mock_http)

        project_id = await fetch_project_id("token")
        assert project_id == "my-project-123"

    @pytest.mark.asyncio
    async def test_fetch_project_id_dict(self, monkeypatch):
        """Returns project ID when response contains a dict with 'id' key."""
        mock_response = MagicMock()
        mock_response.is_success = True
        mock_response.json.return_value = {"cloudaicompanionProject": {"id": "proj-456"}}

        mock_http = AsyncMock()
        mock_http.post.return_value = mock_response
        monkeypatch.setattr("mirai.auth.antigravity_auth._http", mock_http)

        project_id = await fetch_project_id("token")
        assert project_id == "proj-456"

    @pytest.mark.asyncio
    async def test_fetch_project_id_falls_back_to_default(self, monkeypatch):
        """Returns default project ID when all endpoints fail."""
        mock_http = AsyncMock()
        mock_http.post.side_effect = httpx.ConnectError("fail")
        monkeypatch.setattr("mirai.auth.antigravity_auth._http", mock_http)

        project_id = await fetch_project_id("token")
        assert project_id == "rising-fact-p41fc"


class TestFetchUsage:
    """Tests for account usage fetching."""

    @pytest.mark.asyncio
    async def test_fetch_usage_returns_structure(self):
        """Returns expected dict structure with default values when APIs fail."""
        with patch("httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client.post.side_effect = httpx.ConnectError("offline")
            mock_client_cls.return_value = mock_client

            result = await fetch_usage("token", "project")
            assert "plan" in result
            assert "project" in result
            assert "credits_available" in result
            assert "credits_monthly" in result
            assert "models" in result
            assert result["plan"] is None  # API failed

    @pytest.mark.asyncio
    async def test_fetch_usage_parses_models(self):
        """Parses per-model quota data correctly."""
        account_resp = MagicMock()
        account_resp.status_code = 200
        account_resp.json.return_value = {
            "currentTier": {"name": "Pro"},
            "cloudaicompanionProject": "proj-1",
            "availablePromptCredits": 100.0,
        }

        models_resp = MagicMock()
        models_resp.status_code = 200
        models_resp.json.return_value = {
            "models": {
                "model-a": {"quotaInfo": {"remainingFraction": 0.5, "resetTime": "2026-02-16T00:00:00Z"}},
                "model-b": {"quotaInfo": {"resetTime": "2026-02-16T00:00:00Z"}},
            }
        }

        with patch("httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client.post.side_effect = [account_resp, models_resp]
            mock_client_cls.return_value = mock_client

            result = await fetch_usage("token", "project")
            assert result["plan"] == "Pro"
            assert result["credits_available"] == 100.0
            assert len(result["models"]) == 2

            # model-a: 50% remaining → 50% used
            model_a = next(m for m in result["models"] if m["id"] == "model-a")
            assert abs(model_a["used_pct"] - 50.0) < 0.01

            # model-b: no remainingFraction but has resetTime → exhausted (100%)
            model_b = next(m for m in result["models"] if m["id"] == "model-b")
            assert model_b["used_pct"] == 100.0
