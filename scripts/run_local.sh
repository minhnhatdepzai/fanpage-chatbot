#!/usr/bin/env bash
# Chạy toàn bộ bot trên máy này: PostgreSQL (docker) -> model server (GPU) -> API -> worker -> tunnel HTTPS.
#
#   ./scripts/run_local.sh start     # khởi động, in URL webhook công khai
#   ./scripts/run_local.sh status    # tình trạng từng thành phần
#   ./scripts/run_local.sh stop      # dừng (giữ nguyên PostgreSQL và dữ liệu)
#
# Tunnel là Cloudflare quick tunnel (miễn phí, không cần tài khoản): URL ĐỔI mỗi lần start, nên sau start cần chạy
#   uv run botctl meta set-webhook <URL>/webhook
# Log: runs/logs/*.log (không chứa token; nội dung tin nhắn chỉ ghi khi LOG_CONTENT=true).
set -euo pipefail
cd "$(dirname "$0")/.."
LOG=runs/logs
PIDS=runs/pids
mkdir -p "$LOG" "$PIDS"

OLLAMA_BIN="${OLLAMA_BIN:-$HOME/.local/ollama/bin/ollama}"

BOOT_ID=$(cat /proc/sys/kernel/random/boot_id 2>/dev/null || echo unknown)

pid_of() { cut -d' ' -f1 "$PIDS/$1.pid" 2>/dev/null; }

# Tệp pid ghi "PID BOOT_ID": sau khi máy khởi động lại, PID cũ có thể đã thuộc tiến trình khác -> bỏ qua.
alive() {
  [ -f "$PIDS/$1.pid" ] || return 1
  local pid boot
  read -r pid boot <"$PIDS/$1.pid"
  [ "$boot" = "$BOOT_ID" ] && kill -0 "$pid" 2>/dev/null
}

start_one() {
  local name=$1
  shift
  if alive "$name"; then
    echo "$name: đang chạy (pid $(pid_of "$name"))"
    return
  fi
  nohup "$@" >"$LOG/$name.log" 2>&1 &
  echo "$! $BOOT_ID" >"$PIDS/$name.pid"
  echo "$name: khởi động (pid $!)"
}

wait_http() {
  local url=$1 secs=$2
  for _ in $(seq 1 "$secs"); do
    curl -fs -o /dev/null "$url" && return 0
    sleep 1
  done
  return 1
}

tunnel_url() { grep -oE "https://[a-z0-9-]+\.trycloudflare\.com" "$LOG/tunnel.log" 2>/dev/null | head -1; }

cmd_start() {
  docker compose up -d postgres >/dev/null
  uv run botctl db migrate
  if [ -x "$OLLAMA_BIN" ]; then  # model dự phòng 4-bit (LLM_FALLBACK_*), tự nạp khi có yêu cầu
    OLLAMA_HOST=127.0.0.1:11434 OLLAMA_KEEP_ALIVE=2m start_one ollama "$OLLAMA_BIN" serve
  fi
  start_one model uv run python -m serving.server
  wait_http http://127.0.0.1:8100/health 180 || { echo "model server chưa lên - xem $LOG/model.log"; exit 1; }
  start_one api uv run botctl api
  wait_http http://127.0.0.1:8000/health 60 || { echo "API chưa lên - xem $LOG/api.log"; exit 1; }
  start_one worker uv run botctl worker
  if command -v cloudflared >/dev/null; then
    start_one tunnel cloudflared tunnel --no-autoupdate --url http://127.0.0.1:8000
    for _ in $(seq 1 30); do [ -n "$(tunnel_url)" ] && break; sleep 1; done
  else
    echo "tunnel: chưa cài cloudflared -> webhook chỉ truy cập được nội bộ"
  fi
  cmd_status
  local url
  url=$(tunnel_url)
  if [ -n "$url" ]; then
    echo
    echo "Webhook công khai: $url/webhook"
    echo "Trỏ Meta tới URL này:  uv run botctl meta set-webhook $url/webhook"
  fi
}

cmd_stop() {
  for name in tunnel worker api model ollama; do
    if alive "$name"; then
      kill "$(pid_of "$name")" && echo "$name: đã dừng"
    fi
    rm -f "$PIDS/$name.pid"
  done
}

cmd_status() {
  for name in model ollama api worker tunnel; do
    if alive "$name"; then echo "$name: chạy (pid $(pid_of "$name"))"; else echo "$name: KHÔNG chạy"; fi
  done
  echo -n "ready: "
  curl -s http://127.0.0.1:8000/ready | python3 -c "import json,sys; d=json.load(sys.stdin); print(d['ready'], d['checks'].get('worker_heartbeat'), 'send_mode=' + str(d['checks'].get('send_mode')))" 2>/dev/null || echo "API không phản hồi"
  local url
  url=$(tunnel_url)
  [ -n "$url" ] && alive tunnel && echo "tunnel: $url"
  return 0
}

case "${1:-status}" in
  start) cmd_start ;;
  stop) cmd_stop ;;
  status) cmd_status ;;
  *) echo "dùng: $0 start|stop|status"; exit 2 ;;
esac
