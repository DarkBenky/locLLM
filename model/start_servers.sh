#!/usr/bin/env bash
# Launch the locLLM service stack (RAG + inference API) — default on the RTX 3060.
#
# Usage:
#   ./model/start_servers.sh            start RAG (:8234) + inference API (:8000) on GPU 1 (RTX 3060)
#   GPU=0 ./model/start_servers.sh      start both on torch CUDA index 0 (RTX 3090)
#   ./model/start_servers.sh stop       stop both services
#   ./model/start_servers.sh status     show health of both
#
# Notes:
#   - torch.cuda index vs nvidia-smi are SWAPPED on this machine:
#       torch idx 0 = RTX 3090 | torch idx 1 = RTX 3060 (default).
#   - The RAG server needs --gpu <local-index> (its GPU picker is otherwise
#     interactive); with CUDA_VISIBLE_DEVICES=1 the local index is 0.
set -u

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RAG_DIR="$ROOT/tok/fimData"
INF_DIR="$ROOT/model"
LOG_DIR="$ROOT/model/log"
PIDS_DIR="$ROOT/model/log"
RAG_LOG="$LOG_DIR/rag_server.log"
INF_LOG="$LOG_DIR/infrance.log"
RAG_PID="$PIDS_DIR/rag_server.pid"
INF_PID="$PIDS_DIR/infrance.pid"

GPU="${GPU:-1}"           # torch CUDA index (1 = RTX 3060)
RAG_LOCAL_GPU=0           # index inside CUDA_VISIBLE_DEVICES (after filtering)

RAG_URL="http://localhost:8234/health"
INF_URL="http://localhost:8000/health"

log() { echo "[start_servers] $*"; }

healthy() {  # healthy <url>
  python3 - "$1" <<'PY'
import sys, urllib.request
try:
    urllib.request.urlopen(sys.argv[1], timeout=2)
    sys.exit(0)
except Exception:
    sys.exit(1)
PY
}

wait_health() {  # wait_health <name> <url> <logfile> <seconds>
  local name="$1" url="$2" logf="$3" secs="${4:-600}" waited=0
  while [ "$waited" -lt "$secs" ]; do
    if healthy "$url"; then
      log "$name is UP (${url})"
      return 0
    fi
    sleep 5; waited=$((waited + 5))
  done
  log "ERROR: $name did not become ready in ${secs}s — see $logf" >&2
  return 1
}

start_rag() {
  if healthy "$RAG_URL"; then
    log "RAG service already running — skip"
    return 0
  fi
  log "starting RAG service (${RAG_URL}) on GPU ${GPU} (embedding model)"
  (
    cd "$RAG_DIR" || exit 1
    nohup env CUDA_VISIBLE_DEVICES="$GPU" \
      python server.py --gpu "$RAG_LOCAL_GPU" > "$RAG_LOG" 2>&1 &
    echo $! > "$RAG_PID"
  )
  log "RAG pid: $(cat "$RAG_PID") | log: $RAG_LOG"
  wait_health "RAG" "$RAG_URL" "$RAG_LOG" 900 || return 1
}

start_inf() {
  if healthy "$INF_URL"; then
    log "inference API already running — skip"
    return 0
  fi
  log "starting inference API (${INF_URL}) on GPU ${GPU}"
  (
    cd "$INF_DIR" || exit 1
    nohup env CUDA_VISIBLE_DEVICES="$GPU" \
      python infrance.py > "$INF_LOG" 2>&1 &
    echo $! > "$INF_PID"
  )
  log "inference pid: $(cat "$INF_PID") | log: $INF_LOG"
  wait_health "inference" "$INF_URL" "$INF_LOG" 600 || return 1
}

stop_one() {  # stop_one <name> <pidfile> <pattern>
  local name="$1" pidfile="$2" pattern="$3"
  local pid=""
  [ -f "$pidfile" ] && pid="$(cat "$pidfile" 2>/dev/null)"
  if [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null; then
    kill "$pid" 2>/dev/null
    log "stopping $name (pid $pid)..."
  else
    # fallback: pattern match
    local found
    found="$(pgrep -f "$pattern" | head -1 || true)"
    if [ -n "$found" ]; then
      kill "$found" 2>/dev/null
      log "stopping $name (pid $found)..."
    else
      log "$name not running"
      rm -f "$pidfile"
      return 0
    fi
  fi
  # escalate to SIGKILL if it doesn't exit (e.g. stuck in a CUDA call)
  sleep 3
  if [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null; then
    kill -9 "$pid" 2>/dev/null
    log "force-killed $name (pid $pid)"
  fi
  rm -f "$pidfile"
}

status_all() {
  for entry in "RAG:$RAG_URL" "inference:$INF_URL"; do
    local name="${entry%%:*}" url="${entry#*:}"
    if healthy "$url"; then
      log "$name: UP ($url)"
    else
      log "$name: DOWN ($url)"
    fi
  done
  [ -f "$RAG_PID" ] && log "RAG pid file: $(cat "$RAG_PID")"
  [ -f "$INF_PID" ] && log "inference pid file: $(cat "$INF_PID")"
  log "logs: $RAG_LOG, $INF_LOG"
}

case "${1:-start}" in
  start)
    status_all
    start_rag || log "RAG failed to start (inference can still run without it)"
    start_inf
    log "done. RAG: $RAG_URL | inference: $INF_URL | GPU: $GPU"
    ;;
  stop)
    stop_one "inference" "$INF_PID" "infrance.py"
    stop_one "RAG" "$RAG_PID" "server.py --gpu"
    ;;
  status)
    status_all
    ;;
  *)
    echo "usage: $0 [start|stop|status]" >&2
    exit 1
    ;;
esac
