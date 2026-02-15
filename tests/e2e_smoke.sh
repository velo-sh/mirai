#!/usr/bin/env bash
# E2E smoke test: boot Mirai via Docker Compose, check /health, then shutdown.
#
# Usage:  ./tests/e2e_smoke.sh
# Exit 0 on success, 1 on failure.
#
# Requires: docker compose, curl

set -euo pipefail

COMPOSE_FILE="docker-compose.yml"
SERVICE_NAME="mirai"
HEALTH_URL="http://localhost:8000/health"
TIMEOUT=30
POLL_INTERVAL=2

echo "=== E2E Smoke Test ==="

# 1. Build and start in background
echo "[1/4] Starting services..."
docker compose -f "$COMPOSE_FILE" up -d --build 2>&1

# 2. Wait for health endpoint
echo "[2/4] Waiting for health endpoint ($TIMEOUT sec timeout)..."
elapsed=0
while [ "$elapsed" -lt "$TIMEOUT" ]; do
    if curl -sf "$HEALTH_URL" > /dev/null 2>&1; then
        echo "  ✅ Health check passed at ${elapsed}s"
        break
    fi
    sleep "$POLL_INTERVAL"
    elapsed=$((elapsed + POLL_INTERVAL))
done

if [ "$elapsed" -ge "$TIMEOUT" ]; then
    echo "  ❌ Health check timed out after ${TIMEOUT}s"
    echo "--- Container logs ---"
    docker compose -f "$COMPOSE_FILE" logs "$SERVICE_NAME" --tail=50
    docker compose -f "$COMPOSE_FILE" down
    exit 1
fi

# 3. Validate health response
echo "[3/4] Validating health response..."
RESPONSE=$(curl -sf "$HEALTH_URL")
echo "  Response: $RESPONSE"

# Check that the response contains expected fields
if echo "$RESPONSE" | python3 -c "
import json, sys
data = json.load(sys.stdin)
assert 'status' in data, 'Missing status field'
assert data['status'] == 'ok', f'Bad status: {data[\"status\"]}'
print('  ✅ Health response valid')
" 2>&1; then
    RESULT=0
else
    echo "  ❌ Health response validation failed"
    RESULT=1
fi

# 4. Teardown
echo "[4/4] Shutting down..."
docker compose -f "$COMPOSE_FILE" down 2>&1

if [ "$RESULT" -eq 0 ]; then
    echo "=== E2E Smoke Test PASSED ==="
else
    echo "=== E2E Smoke Test FAILED ==="
fi

exit "$RESULT"
