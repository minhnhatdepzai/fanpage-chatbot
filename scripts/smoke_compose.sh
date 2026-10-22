#!/usr/bin/env bash
set -Eeuo pipefail

image="${1:-fanpage-chatbot:smoke}"
project="fanpage-chatbot-smoke-$$"
offset=$(( $$ % 1000 ))

export APP_ENV_FILE=/dev/null
export POSTGRES_PASSWORD=compose-smoke-password
export POSTGRES_BIND_PORT="${POSTGRES_BIND_PORT:-$((55000 + offset))}"
export API_BIND_PORT="${API_BIND_PORT:-$((18000 + offset))}"
export FANPAGE_CHATBOT_IMAGE="$image"

cleanup() {
  code=$?
  if (( code != 0 )); then
    docker compose -p "$project" --profile app ps --all || true
    docker compose -p "$project" --profile app logs --no-color --tail=100 || true
  fi
  docker compose -p "$project" --profile app down --volumes --remove-orphans >/dev/null 2>&1 || true
  exit "$code"
}
trap cleanup EXIT

docker compose -p "$project" --profile app up --detach --no-build --wait --wait-timeout 60

base_url="http://127.0.0.1:${API_BIND_PORT}"
curl --fail --silent "$base_url/health" | python3 -c \
  'import json,sys; assert json.load(sys.stdin) == {"status": "ok"}'
curl --fail --silent "$base_url/web/math" >/dev/null
curl --fail --silent \
  --header 'Content-Type: application/json' \
  --data '{"question":"2 + 3","grade":2}' \
  "$base_url/web/math/solve" | python3 -c \
  'import json,sys; x=json.load(sys.stdin); assert x["answer"] == "5" and x["verification"]["passed"] is True'

docker compose -p "$project" --profile app ps --status running --services | grep -qx api
docker compose -p "$project" --profile app ps --status running --services | grep -qx worker

echo "compose smoke: OK ($image)"
