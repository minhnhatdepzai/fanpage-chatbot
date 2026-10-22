#!/usr/bin/env bash
set -Eeuo pipefail

image="${1:-fanpage-chatbot:smoke}"
container="fanpage-chatbot-smoke-$$"

cleanup() {
  docker rm -f "$container" >/dev/null 2>&1 || true
}
trap cleanup EXIT

docker run --detach --name "$container" \
  --env BOT_ENV_FILE= \
  --env APP_ENV=test \
  --env LLM_PROVIDER=fake \
  --env MESSENGER_SEND_MODE=disabled \
  --env RAG_DOCS_ENABLED=false \
  --env VISION_ENABLED=false \
  --env WEB_SEARCH_ENABLED=false \
  --env TTS_ENABLED=false \
  "$image" >/dev/null

for _ in $(seq 1 40); do
  status="$(docker inspect --format '{{if .State.Health}}{{.State.Health.Status}}{{else}}none{{end}}' "$container")"
  if [[ "$status" == "healthy" ]]; then
    break
  fi
  if [[ "$status" == "unhealthy" ]] || ! docker inspect --format '{{.State.Running}}' "$container" | grep -q true; then
    docker logs "$container"
    exit 1
  fi
  sleep 1
done

if [[ "$(docker inspect --format '{{.State.Health.Status}}' "$container")" != "healthy" ]]; then
  docker logs "$container"
  echo "Container did not become healthy" >&2
  exit 1
fi

docker exec "$container" python - <<'PY'
import json
import urllib.request

with urllib.request.urlopen("http://127.0.0.1:8000/health", timeout=5) as response:
    assert response.status == 200
    assert json.load(response) == {"status": "ok"}

with urllib.request.urlopen("http://127.0.0.1:8000/web/math", timeout=5) as response:
    assert response.status == 200
    assert b"<!doctype html>" in response.read(256).lower()

request = urllib.request.Request(
    "http://127.0.0.1:8000/web/math/solve",
    data=json.dumps({"question": "2 + 3", "grade": 2}).encode(),
    headers={"Content-Type": "application/json"},
    method="POST",
)
with urllib.request.urlopen(request, timeout=5) as response:
    result = json.load(response)
    assert response.status == 200
    assert result["status"] == "verified"
    assert result["answer"] == "5"
    assert result["verification"]["passed"] is True
PY

echo "container smoke: OK ($image)"
