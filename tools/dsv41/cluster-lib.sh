#!/usr/bin/env bash
# Shared pieces of the DeepSeek-V4.1-Flash two-node cluster lane (sourced by
# cluster-head.sh and cluster-worker.sh; not meant to be run on its own).
#
# Topology (identical to the GLM-5.3 lane): x299 = ray head + PP stages 0-1
# (8x3090, TP=4), rome = ray worker + PP stages 2-4 (12x3090, TP=4), i.e.
# TP=4 x PP=5 over 10 GbE. Conventions mirror ~/glm-head-ubatch.sh,
# ~/glm-worker.sh and my-salt formulas/vllm (vllm-cluster.service +
# cluster-entrypoint.sh): host network, --ipc host, pids-limit -1, big stack
# ulimit, /model:ro at the same in-container path on both nodes, NCCL/GLOO
# socket ifname per node, VLLM_HOST_IP per node, ray GCS on 6379.
#
# Orchestration knobs are DSV41_* on purpose (the salt lane uses GLM_*): vLLM
# scans the VLLM_ namespace, so only genuine vLLM env vars are passed as-is.

# ---- shared knobs (env) ------------------------------------------------------
DSV41_HEAD_IP=${DSV41_HEAD_IP:-192.168.1.31}
DSV41_WORKER_IP=${DSV41_WORKER_IP:-192.168.1.7}
DSV41_HEAD_GPUS=${DSV41_HEAD_GPUS:-8}
DSV41_WORKER_GPUS=${DSV41_WORKER_GPUS:-12}
DSV41_WORLD_GPUS=${DSV41_WORLD_GPUS:-$((DSV41_HEAD_GPUS + DSV41_WORKER_GPUS))}
RAY_PORT=${RAY_PORT:-6379}
OVERLAY=${OVERLAY:-1}
MEMLOCK=${MEMLOCK:-1}
# VLLM_* knobs: exported into BOTH containers (ray also copies the driver's
# VLLM_* env to its workers, so setting them on the worker is belt and braces).
PARTITION=${PARTITION:-8,8,8,9,7}
RELAY=${RELAY:-1}
# dsv41 cluster: SLOTTRACE=1 turns on the fork's existing per-step worker trace
# (gpu_worker.py WTRACE lines: sendwait / mdrv / run / tot ms per PP stage,
# TP rank 0 only); it costs a log line per step, so use it on short benches.
SLOTTRACE=${SLOTTRACE:-}
LOGLEVEL=${LOGLEVEL:-INFO}

# dsv41 boot: OVERLAY=1 wrapper, run inside the container before ray/vllm.
# /src/vllm (the synced worktree) is copied to /work/vllm, every build-only
# file of the installed package (.so, vllm_flash_attn, _version.py) is linked
# in, PYTHONPATH=/work. Ray worker processes inherit the env of `ray start`,
# so running this in the worker container is what makes rome's ranks import
# the same python as the head's driver.
# dsv41 engram-mmap: --engram-config JSON from ENGRAM_OFFLOAD (pinned or
# resident) and, when set, ENGRAM_MODE (pinned | resident | mmap).
# dsv41 engram-warm: ENGRAM_WARM (none | async | sync), when set, adds
# mmap_warm (boot-time page-cache warmup of the mmap slices); ENGRAM_DROP
# (0 | 1), when set, adds drop_weight_pages (unset = drop with mmap tables).
engram_config_json() {
  local offload=false
  [ "${ENGRAM_OFFLOAD:-1}" = "1" ] && offload=true
  local json="{\"cpu_offload\": $offload"
  [ -n "${ENGRAM_MODE:-}" ] && json="$json, \"table_mode\": \"$ENGRAM_MODE\""
  [ -n "${ENGRAM_WARM:-}" ] && json="$json, \"mmap_warm\": \"$ENGRAM_WARM\""
  case "${ENGRAM_DROP:-}" in
    1) json="$json, \"drop_weight_pages\": true" ;;
    0) json="$json, \"drop_weight_pages\": false" ;;
  esac
  printf '%s}' "$json"
}

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

# common_podman_args <role> : prints nothing, fills the PODMAN_ARGS array.
# Needs NAME, IMAGE, MODEL_DIR, SRC_DIR, NODE_IP, NCCL_IFNAME set by the caller.
common_podman_args() {
  [ -n "${IMAGE:-}" ] || { echo "IMAGE is required" >&2; exit 1; }
  [ -f "$MODEL_DIR/config.json" ] || {
    echo "no config.json in MODEL_DIR=$MODEL_DIR (make_trunc_config.py --full-dummy --out ..., then cluster-sync.sh)" >&2
    exit 1
  }
  PODMAN_ARGS=(
    --name "$NAME"
    --network host --ipc host --pids-limit=-1 --ulimit stack=67108864
    --device nvidia.com/gpu=all
    -e VLLM_HOST_IP="$NODE_IP"
    -e NCCL_SOCKET_IFNAME="$NCCL_IFNAME"
    -e GLOO_SOCKET_IFNAME="$NCCL_IFNAME"
    -e VLLM_USE_V2_MODEL_RUNNER="${VLLM_USE_V2_MODEL_RUNNER:-1}"
    -e VLLM_PP_LAYER_PARTITION="$PARTITION"
    -e VLLM_DSV41_PP_KV_RELAY="$RELAY"
    -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
    -e HF_HUB_OFFLINE=1
    -e VLLM_LOGGING_LEVEL="$LOGLEVEL"
    ${PPMETA:+-e VLLM_PP_CACHED_METADATA=$PPMETA}
    ${SLOTTRACE:+-e VLLM_SLOT_TRACE=$SLOTTRACE}
    ${NCCLALGO:+-e NCCL_ALGO=$NCCLALGO}
    ${NCCLPROTO:+-e NCCL_PROTO=$NCCLPROTO}
    ${NCCL_DEBUG:+-e NCCL_DEBUG=$NCCL_DEBUG}
    ${CUDA_VISIBLE_DEVICES:+-e CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES}
    -v "$MODEL_DIR":/model:ro
  )
  # Pinned engram tables (cpu_offload) + NCCL/ray shared buffers: lift the
  # locked-memory cap inside the container (the salt lane does not need it
  # because GLM has no host-pinned tables).
  [ "$MEMLOCK" = "1" ] && PODMAN_ARGS+=(--ulimit memlock=-1:-1)
  if [ "$OVERLAY" = "1" ]; then
    [ -d "$SRC_DIR/vllm" ] || { echo "OVERLAY=1 but no vllm/ under SRC_DIR=$SRC_DIR (run cluster-sync.sh)" >&2; exit 1; }
    PODMAN_ARGS+=(-v "$SRC_DIR":/src:ro)
  fi
}

# Refuse to touch GPUs that something else is using (another lane's
# experiment, the production GLM cluster). FORCE=1 overrides.
gpu_preflight() {
  [ "${FORCE:-0}" = "1" ] && return 0
  local busy
  busy=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits 2>/dev/null |
    awk -F', *' '$2 > 1024 {printf "GPU%s=%sMiB ", $1, $2}')
  if [ -n "$busy" ]; then
    echo "refusing to start: GPUs already in use ($busy); FORCE=1 to override" >&2
    exit 1
  fi
  local others
  others=$(podman ps --format '{{.Names}}' 2>/dev/null | grep -E 'glm|vllm|dsv41' || true)
  if [ -n "$others" ]; then
    echo "refusing to start: other lane containers running: $(echo "$others" | tr '\n' ' '); FORCE=1 to override" >&2
    exit 1
  fi
}

# dsv41 cluster: keep a host-side copy of the container log. `podman run
# --rm` discards it the moment the container exits, which is exactly when a
# failed boot needs reading. LOGDIR [~/dsv41-test], LOGTAG [timestamp].
LOGDIR=${LOGDIR:-$HOME/dsv41-test}
start_log_capture() {
  LOGFILE="$LOGDIR/$NAME-${LOGTAG:-$(date +%Y%m%d-%H%M%S)}.log"
  mkdir -p "$LOGDIR"
  nohup podman logs -f "$NAME" >"$LOGFILE" 2>&1 </dev/null &
  echo "log: $LOGFILE"
}

# dsv41 cluster: snapshot the ray per-worker logs (/tmp/ray/session_latest
# /logs/worker-*.out|err inside the container) to the host. The head log
# dedups identical worker lines, so per-rank evidence only lives here; it is
# lost with the container, hence `stop` calls this first and DSV41_KEEP=1
# keeps a failed container alive (sleep) so it can still be collected.
ray_logs_snapshot() {
  local dir="$LOGDIR/$NAME-${LOGTAG:-$(date +%Y%m%d-%H%M%S)}-ray"
  [ -n "$(podman ps -q -f name="^$NAME\$" 2>/dev/null)" ] || { echo "$NAME is not running; no ray logs to snapshot" >&2; return 0; }
  mkdir -p "$dir"
  if podman exec "$NAME" bash -c 'cd /tmp/ray/session_latest/logs 2>/dev/null && tar -cf - worker-*.out worker-*.err raylet.err raylet.out 2>/dev/null' | tar -C "$dir" -xf - 2>/dev/null; then
    echo "ray logs: $dir ($(find "$dir" -type f | wc -l) files)"
  else
    echo "ray logs: nothing collected from $NAME" >&2
  fi
}

status_common() {
  echo "== podman"
  podman ps --filter "name=^$NAME\$" --format 'table {{.Names}}\t{{.Status}}\t{{.Image}}'
  if [ -n "$(podman ps -q -f name="^$NAME\$")" ]; then
    echo "== ray status (inside $NAME)"
    podman exec "$NAME" ray status 2>&1 | sed -n '1,40p' || true
  fi
  echo "== gpus"
  nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv 2>/dev/null || true
}
