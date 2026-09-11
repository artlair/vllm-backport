#!/usr/bin/env bash
# DeepSeek-V4.1-Flash two-node HEAD (x299): ray head + `vllm serve` in podman.
#
#   IMAGE=<sm86 image> MODEL_DIR=~/dsv41-test/models/full-dummy ./cluster-head.sh start
#   ./cluster-head.sh smoke | bench | status | logs | raylogs | stop | print | serve (foreground)
#   (bench: CONC [4] concurrent requests of MAX_TOKENS [256] each, ignore_eos,
#   prints per-request completion and aggregate tokens/s)
#
# Run it from the synced copy on x299 (~/dsv41-test/src/tools/dsv41) after
# cluster-sync.sh, and start cluster-worker.sh on rome FIRST (the head waits
# up to DSV41_CLUSTER_WAIT=900s for DSV41_WORLD_GPUS=20 GPUs to join ray).
#
# Knobs (env), defaults in brackets:
#   IMAGE (required)  MODEL_DIR [~/dsv41-test/models/full-dummy]  NAME [dsv41-head]
#   OVERLAY [1] SRC_DIR [worktree containing this script, i.e. ~/dsv41-test/src]
#   TP [4] PP [5] PARTITION [8,8,8,9,7]  (VLLM_PP_LAYER_PARTITION; the x299
#     stages are the first two entries: 8,7 keeps layers 0..14, and with them
#     BOTH engram layers (1 and 14) and their pinned host tables, on x299)
#   CTX [32768] UTIL [0.9] SEQS [4] BATCHED [engine default] PORT [8080]
#   SPEC [5]   DSpark tokens; 0 disables (V4.1 n_predict is dspark_block_size=5,
#              larger values must be multiples of 5)
#   EAGER [1]  1 = --enforce-eager; 0 = cudagraphs per CGMODE/CAPSIZES
#   CGMODE [PIECEWISE]  PIECEWISE | FULL_AND_PIECEWISE | FULL_DECODE_ONLY
#   CAPSIZES [1,2,4,8,12,16,20,24,28,32]  CAPMAX [max of CAPSIZES]
#   RELAY [1]  VLLM_DSV41_PP_KV_RELAY (partition splits kv groups 14 and 20)
#   SLOTTRACE []  1 = VLLM_SLOT_TRACE per-step worker trace (WTRACE log lines)
#   PPMETA []  VLLM_PP_CACHED_METADATA (unset = off)
#   ENGRAM_OFFLOAD [1]  --engram-config cpu_offload (pinned host tables)
#   ENGRAM_MODE []  pinned | resident | mmap: --engram-config table_mode
#              (mmap = tables read from the page cache, nothing pinned;
#              docs/dsv41-engram-mmap.md); unset keeps ENGRAM_OFFLOAD
#   ENGRAM_WARM []  none | async | sync: --engram-config mmap_warm, reads
#              each rank's mmap slices into the page cache after the
#              weights load (sync blocks, async runs in the background)
#   MEMLOCK [1]  --ulimit memlock=-1 (pinned tables need it)
#   NCCLALGO / NCCLPROTO []  unset = NCCL auto
#   LOADFORMAT [dummy]  dummy | auto (real weights)
#   KVDTYPE [fp8_ds_mla]
#   LIMITMM []  --limit-mm-per-prompt JSON, e.g. '{"image":0}' to stub the ViT
#   EXTRA []   extra vllm args (word-split); anything after start/serve/print
#              is appended too
#   NCCL_DEBUG [] LOGLEVEL [INFO] CUDA_VISIBLE_DEVICES [] FORCE [0]
#   DSV41_HEAD_IP [192.168.1.31] DSV41_HEAD_GPUS [8] DSV41_WORLD_GPUS [20]
#   NCCL_IFNAME [br1] RAY_PORT [6379] DSV41_CLUSTER_WAIT [900]
#   LOGDIR [~/dsv41-test] LOGTAG [timestamp]  host copy of the container log
#   DSV41_KEEP [0]  1 = keep the container alive after a failed vllm exit so
#              `raylogs` can still snapshot /tmp/ray per-worker logs
set -euo pipefail

HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
# shellcheck source=cluster-lib.sh
. "$HERE/cluster-lib.sh"

NAME=${NAME:-dsv41-head}
MODEL_DIR=${MODEL_DIR:-$HOME/dsv41-test/models/full-dummy}
SRC_DIR=${SRC_DIR:-$(cd "$HERE/../.." && pwd)}
NODE_IP=$DSV41_HEAD_IP
NCCL_IFNAME=${NCCL_IFNAME:-br1}
PORT=${PORT:-8080}
HOST=${HOST:-127.0.0.1}
TP=${TP:-4}
PP=${PP:-5}
CTX=${CTX:-32768}
UTIL=${UTIL:-0.9}
SEQS=${SEQS:-4}
SPEC=${SPEC:-5}
EAGER=${EAGER:-1}
CGMODE=${CGMODE:-PIECEWISE}
CAPSIZES=${CAPSIZES:-1,2,4,8,12,16,20,24,28,32}
CAPMAX=${CAPMAX:-${CAPSIZES##*,}}
ENGRAM_OFFLOAD=${ENGRAM_OFFLOAD:-1}
ENGRAM_MODE=${ENGRAM_MODE:-}
ENGRAM_WARM=${ENGRAM_WARM:-}
LOADFORMAT=${LOADFORMAT:-dummy}
KVDTYPE=${KVDTYPE:-fp8_ds_mla}
DSV41_CLUSTER_WAIT=${DSV41_CLUSTER_WAIT:-900}

usage() { sed -n '2,40p' "$0"; exit 1; }

vllm_args() {
  local args=(
    vllm serve /model
    --load-format "$LOADFORMAT"
    --kv-cache-dtype "$KVDTYPE"
    --distributed-executor-backend ray
    --disable-custom-all-reduce
    --trust-remote-code
    --max-model-len "$CTX"
    --gpu-memory-utilization "$UTIL"
    --max-num-seqs "$SEQS"
    --tensor-parallel-size "$TP"
    --pipeline-parallel-size "$PP"
    --served-model-name dsv41 dsv41-dummy
    --host 0.0.0.0 --port "$PORT"
  )
  [ -n "${BATCHED:-}" ] && args+=(--max-num-batched-tokens "$BATCHED")
  args+=(--engram-config "$(engram_config_json)")
  if [ "$EAGER" = "1" ]; then
    args+=(--enforce-eager)
  else
    args+=(--compilation-config "{\"cudagraph_mode\":\"$CGMODE\",\"cudagraph_capture_sizes\":[$CAPSIZES],\"max_cudagraph_capture_size\":$CAPMAX}")
  fi
  if [ "$SPEC" -gt 0 ]; then
    args+=(--speculative-config "{\"method\":\"dspark\",\"num_speculative_tokens\":$SPEC}")
  fi
  [ -n "${LIMITMM:-}" ] && args+=(--limit-mm-per-prompt "$LIMITMM")
  # shellcheck disable=SC2206
  [ -n "${EXTRA:-}" ] && args+=(${EXTRA})
  printf '%q ' "${args[@]}" "$@"
}

# The in-container script: overlay, ray head, wait for the worker's GPUs,
# then exec vllm serve (mirrors cluster-entrypoint.sh's head branch).
container_script() {
  local vllm_cmd=$1
  [ "$OVERLAY" = "1" ] && overlay_prelude
  cat <<EOS
ray start --head --node-ip-address $NODE_IP --port $RAY_PORT --num-gpus $DSV41_HEAD_GPUS --disable-usage-stats
echo "dsv41-head: waiting for $DSV41_WORLD_GPUS GPUs to join ray (up to ${DSV41_CLUSTER_WAIT}s)..." >&2
deadline=\$(( SECONDS + $DSV41_CLUSTER_WAIT ))
while ! python3 -c "import ray, sys; ray.init(address='auto', log_to_driver=False); sys.exit(0 if ray.cluster_resources().get('GPU', 0) >= $DSV41_WORLD_GPUS else 1)" 2>/dev/null; do
  if [ "\$SECONDS" -ge "\$deadline" ]; then echo "dsv41-head: timed out waiting for $DSV41_WORLD_GPUS GPUs" >&2; exit 1; fi
  sleep 5
done
echo "dsv41-head: ray has $DSV41_WORLD_GPUS+ GPUs; starting vllm" >&2
EOS
  # dsv41 cluster: exec keeps podman stop's SIGTERM reaching vllm; DSV41_KEEP=1
  # trades that for a container that survives a failed boot (raylogs).
  if [ "${DSV41_KEEP:-0}" = "1" ]; then
    cat <<EOS
$vllm_cmd
rc=\$?
echo "dsv41-head: vllm exited \$rc; DSV41_KEEP=1 so staying up for log collection (./cluster-head.sh raylogs; stop)" >&2
sleep infinity
EOS
  else
    echo "exec $vllm_cmd"
  fi
}

run_podman() {
  local mode=$1; shift
  gpu_preflight
  common_podman_args
  PODMAN_ARGS+=(--entrypoint bash "$IMAGE" -c "$(container_script "$*")")
  echo "+ vllm command: $*" >&2
  if [ "$mode" = "-d" ]; then
    podman run -d --rm "${PODMAN_ARGS[@]}"
    start_log_capture
    echo "started $NAME; ./cluster-head.sh logs | ./cluster-head.sh smoke"
  else
    exec podman run --rm "${PODMAN_ARGS[@]}"
  fi
}

smoke() {
  local base="http://$HOST:$PORT" timeout=${TIMEOUT:-2400} i
  echo "waiting for $base/health (up to ${timeout}s)"
  for ((i = 0; i < timeout; i += 5)); do
    if curl -sf "$base/health" >/dev/null 2>&1; then break; fi
    if [ -n "$(podman ps -q -f name="^$NAME\$" 2>/dev/null)" ] || [ "$i" -lt 60 ]; then sleep 5; else
      echo "container $NAME is not running and /health never came up" >&2; exit 1; fi
  done
  curl -sf "$base/health" >/dev/null || { echo "timed out waiting for /health" >&2; exit 1; }
  local prompt=${PROMPT:-"Write one sentence about the ocean."} max_tokens=${MAX_TOKENS:-32} body t0 t1 resp
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
pt = u.get("prompt_tokens")
dt = t1 - t0
print("text:", repr(text))
print(f"prompt_tokens={pt} completion_tokens={out} wall={dt:.2f}s tokens/s={out / dt if dt else 0:.1f}")
' "$t0" "$t1"
}

# dsv41 cluster: concurrency smoke. CONC requests of MAX_TOKENS tokens each
# (ignore_eos so every request runs to max_tokens), aggregate tokens/s.
bench() {
  local base="http://$HOST:$PORT" n=${CONC:-4} max_tokens=${MAX_TOKENS:-256} i t0 t1 tmp body
  curl -sf "$base/health" >/dev/null || { echo "$base is not serving" >&2; exit 1; }
  tmp=$(mktemp -d)
  t0=$(date +%s.%N)
  for ((i = 0; i < n; i++)); do
    body=$(printf '{"model":"dsv41","messages":[{"role":"user","content":"Write a long story about request %d."}],"temperature":0,"max_tokens":%d,"ignore_eos":true}' "$i" "$max_tokens")
    curl -sf "$base/v1/chat/completions" -H 'content-type: application/json' -d "$body" -o "$tmp/$i.json" &
  done
  wait
  t1=$(date +%s.%N)
  python3 - "$tmp" "$t0" "$t1" "$n" <<'EOF'
import glob, json, os, sys
tmp, t0, t1, n = sys.argv[1], float(sys.argv[2]), float(sys.argv[3]), int(sys.argv[4])
tot = ok = 0
for f in sorted(glob.glob(os.path.join(tmp, "*.json"))):
    try:
        with open(f) as fh:
            r = json.load(fh)
        c = r["usage"]["completion_tokens"]
    except Exception as e:  # noqa: BLE001
        print(f"{os.path.basename(f)}: {e}")
        continue
    ok += 1
    tot += c
    print(f"{os.path.basename(f)}: completion_tokens={c} finish={r['choices'][0].get('finish_reason')}")
dt = t1 - t0
print(f"completed={ok}/{n} completion_tokens={tot} wall={dt:.2f}s aggregate tokens/s={tot / dt if dt else 0:.1f}")
EOF
  rm -rf "$tmp"
}

cmd=${1:-start}; shift || true
case "$cmd" in
  serve)  run_podman fg "$(vllm_args "$@")" ;;
  start)  run_podman -d "$(vllm_args "$@")" ;;
  print)  vllm_args "$@"; echo ;;
  script) container_script "$(vllm_args "$@")" ;;
  logs)   exec podman logs "${@:--f}" "$NAME" ;;
  status) status_common; echo "== health"; curl -sf "http://$HOST:$PORT/health" >/dev/null && echo "OK $HOST:$PORT" || echo "not serving" ;;
  raylogs) ray_logs_snapshot ;;
  stop)   ray_logs_snapshot; podman stop -t 30 "$NAME" ;;
  smoke)  smoke ;;
  bench)  bench ;;
  *) usage ;;
esac
