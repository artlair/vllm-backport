#!/usr/bin/env bash
# DeepSeek-V4.1-Flash dummy-weight boot harness (podman).
#
#   IMAGE=<sm86 image> MODEL_DIR=/path/to/trunc ./boot.sh serve   # foreground
#   IMAGE=... ./boot.sh start                                      # detached
#   ./boot.sh smoke                                                # wait + 1 chat completion
#   ./boot.sh logs | stop
#
# Knobs (env): IMAGE (required for serve/start), MODEL_DIR (default ./trunc),
# TP=1 PP=1 CTX=8192 UTIL=0.5 SEQS=2 SPEC=0 ENGRAM_OFFLOAD=1 EAGER=1 PORT=8080
# NAME=dsv41-dummy BATCHED=2048 EXTRA (extra vllm args, word-split).
# OVERLAY=1 mounts the repo (SRC_DIR, default the worktree containing this
# script) at /src and runs the image's compiled ops with the worktree's python:
# /src/vllm is copied to /work/vllm, every build-only file of the installed
# package (.so, vllm_flash_attn, _version.py) is linked in, PYTHONPATH=/work.
# Anything after `serve`/`start` is appended to the vllm command line.
#
# Conventions mirror the GLM lane scripts on x299 (glm-head*.sh): host network,
# --ipc host, pids-limit -1, big stack ulimit, /model:ro mount, expandable
# segments, HF offline.
set -euo pipefail

HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
MODEL_DIR=${MODEL_DIR:-$HERE/trunc}
NAME=${NAME:-dsv41-dummy}
PORT=${PORT:-8080}
TP=${TP:-1}
PP=${PP:-1}
CTX=${CTX:-8192}
UTIL=${UTIL:-0.5}
SEQS=${SEQS:-2}
SPEC=${SPEC:-0}
ENGRAM_OFFLOAD=${ENGRAM_OFFLOAD:-1}
EAGER=${EAGER:-1}
BATCHED=${BATCHED:-2048}
HOST=${HOST:-127.0.0.1}
OVERLAY=${OVERLAY:-0}
SRC_DIR=${SRC_DIR:-$(cd "$HERE/../.." && pwd)}

# dsv41 boot: OVERLAY=1 wrapper, run inside the container before vllm.
overlay_prelude() {
  cat <<'EOS'
set -e
inst=/usr/local/lib/python3.12/dist-packages/vllm
mkdir -p /work && cp -r /src/vllm /work/vllm
cp "$inst/_version.py" /work/vllm/_version.py
(cd "$inst" && find . -type f ! -path '*/__pycache__/*' -print0) |
  while IFS= read -r -d '' f; do
    [ -e "/work/vllm/$f" ] && continue
    mkdir -p "/work/vllm/$(dirname "$f")"
    ln -s "$inst/$f" "/work/vllm/$f"
  done
echo "overlay: $(find /work/vllm -type l | wc -l) build-only files linked from $inst" >&2
export PYTHONPATH=/work
EOS
}

usage() { sed -n '2,15p' "$0"; exit 1; }

vllm_args() {
  # Printed one-per-line so `serve` can show the exact command.
  local args=(
    vllm serve /model
    --load-format dummy
    --kv-cache-dtype fp8_ds_mla
    --disable-custom-all-reduce
    --trust-remote-code
    --max-num-batched-tokens "$BATCHED"
    --max-model-len "$CTX"
    --gpu-memory-utilization "$UTIL"
    --max-num-seqs "$SEQS"
    --tensor-parallel-size "$TP"
    --pipeline-parallel-size "$PP"
    --served-model-name dsv41 dsv41-dummy
    --host 0.0.0.0 --port "$PORT"
  )
  if [ "$ENGRAM_OFFLOAD" = "1" ]; then
    args+=(--engram-config '{"cpu_offload": true}')
  else
    args+=(--engram-config '{"cpu_offload": false}')
  fi
  [ "$EAGER" = "1" ] && args+=(--enforce-eager)
  if [ "${SPEC}" -gt 0 ]; then
    # V4.1 DSpark: n_predict := dspark_block_size (5); SPEC > 5 must be a multiple of 5.
    args+=(--speculative-config "{\"method\":\"dspark\",\"num_speculative_tokens\":${SPEC}}")
  fi
  # shellcheck disable=SC2206
  [ -n "${EXTRA:-}" ] && args+=(${EXTRA})
  printf '%q ' "${args[@]}" "$@"
}

run_podman() {
  # $1 = -d or --rm-foreground marker, rest = vllm args string
  local mode=$1; shift
  [ -n "${IMAGE:-}" ] || { echo "IMAGE is required" >&2; exit 1; }
  [ -f "$MODEL_DIR/config.json" ] || { echo "no config.json in MODEL_DIR=$MODEL_DIR (run make_trunc_config.py --out $MODEL_DIR)" >&2; exit 1; }
  local podman_args=(
    --name "$NAME"
    --network host --ipc host --pids-limit=-1 --ulimit stack=67108864
    --device nvidia.com/gpu=all
    -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
    -e HF_HUB_OFFLINE=1
    -e VLLM_LOGGING_LEVEL="${LOGLEVEL:-INFO}"
    ${VLLM_USE_V2_MODEL_RUNNER:+-e VLLM_USE_V2_MODEL_RUNNER=$VLLM_USE_V2_MODEL_RUNNER}
    ${CUDA_VISIBLE_DEVICES:+-e CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES}
    -v "$MODEL_DIR":/model:ro
  )
  local prelude=""
  if [ "$OVERLAY" = "1" ]; then
    podman_args+=(-v "$SRC_DIR":/src:ro)
    prelude=$(overlay_prelude)
  fi
  podman_args+=(--entrypoint bash "$IMAGE" -c "$prelude
exec $*")
  echo "+ vllm command: $*" >&2
  if [ "$mode" = "-d" ]; then
    podman run -d --rm "${podman_args[@]}"
    echo "started $NAME; ./boot.sh logs | ./boot.sh smoke"
  else
    exec podman run --rm "${podman_args[@]}"
  fi
}

cmd=${1:-serve}; shift || true
case "$cmd" in
  serve) run_podman fg "$(vllm_args "$@")" ;;
  start) run_podman -d "$(vllm_args "$@")" ;;
  print) vllm_args "$@"; echo ;;
  logs)  exec podman logs "${@:--f}" "$NAME" ;;
  stop)  podman stop -t 10 "$NAME" ;;
  smoke)
    base="http://$HOST:$PORT"
    timeout=${TIMEOUT:-1800}
    echo "waiting for $base/health (up to ${timeout}s)"
    for ((i = 0; i < timeout; i += 5)); do
      if curl -sf "$base/health" >/dev/null 2>&1; then break; fi
      if [ -n "$(podman ps -q -f name="^$NAME\$" 2>/dev/null)" ] || [ "$i" -lt 60 ]; then sleep 5; else
        echo "container $NAME is not running and /health never came up" >&2; exit 1; fi
    done
    curl -sf "$base/health" >/dev/null || { echo "timed out waiting for /health" >&2; exit 1; }
    prompt=${PROMPT:-"Write one sentence about the ocean."}
    max_tokens=${MAX_TOKENS:-32}
    body=$(printf '{"model":"dsv41","messages":[{"role":"user","content":%s}],"temperature":0,"max_tokens":%d}' \
      "$(printf '%s' "$prompt" | python3 -c 'import json,sys; print(json.dumps(sys.stdin.read()))')" "$max_tokens")
    t0=$(date +%s.%N)
    resp=$(curl -sf "$base/v1/chat/completions" -H 'content-type: application/json' -d "$body")
    t1=$(date +%s.%N)
    printf '%s' "$resp" | python3 -c '
import json, sys
t0, t1 = float(sys.argv[1]), float(sys.argv[2])
r = json.load(sys.stdin)
u = r.get("usage", {})
text = r["choices"][0]["message"].get("content")
out = u.get("completion_tokens", 0)
dt = t1 - t0
print("text:", repr(text))
print(f"prompt_tokens={u.get("prompt_tokens")} completion_tokens={out} wall={dt:.2f}s tokens/s={out / dt if dt else 0:.1f}")
' "$t0" "$t1"
    ;;
  *) usage ;;
esac
