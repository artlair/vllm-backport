#!/usr/bin/env bash
# DeepSeek-V4.1-Flash two-node WORKER (rome): ray worker joining the x299 head.
#
#   IMAGE=<sm86 image> MODEL_DIR=~/dsv41-test/models/full-dummy ./cluster-worker.sh start
#   ./cluster-worker.sh status | logs | raylogs | stop | run (foreground)
#
# Start this BEFORE cluster-head.sh on x299; the same IMAGE, the same MODEL_DIR
# contents at the same in-container path (/model) and, with OVERLAY=1, the same
# synced vllm/ tree (cluster-sync.sh) are required on both nodes. The overlay
# runs inside this container too so the ray worker processes rome spawns for
# the head's engine import the synced python, not the image's.
#
# Knobs (env), defaults in brackets:
#   IMAGE (required)  MODEL_DIR [~/dsv41-test/models/full-dummy]  NAME [dsv41-worker]
#   OVERLAY [1] SRC_DIR [worktree containing this script]
#   PARTITION [8,7,9,8,8] RELAY [1] PPMETA [] NCCLALGO/NCCLPROTO [] NCCL_DEBUG []
#     (VLLM_* env is mirrored here for parity with the head; ray copies the
#     driver's VLLM_* env to workers anyway)
#   MEMLOCK [1] LOGLEVEL [INFO] CUDA_VISIBLE_DEVICES [] FORCE [0]
#   DSV41_HEAD_IP [192.168.1.31] DSV41_WORKER_IP [192.168.1.7]
#   DSV41_WORKER_GPUS [12] NCCL_IFNAME [enp66s0f0] RAY_PORT [6379]
#   LOGDIR [~/dsv41-test] LOGTAG [timestamp]  host copy of the container log
#   DSV41_KEEP [0]  1 = keep the container alive after ray exits (raylogs)
set -euo pipefail

HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
# shellcheck source=cluster-lib.sh
. "$HERE/cluster-lib.sh"

NAME=${NAME:-dsv41-worker}
MODEL_DIR=${MODEL_DIR:-$HOME/dsv41-test/models/full-dummy}
SRC_DIR=${SRC_DIR:-$(cd "$HERE/../.." && pwd)}
NODE_IP=$DSV41_WORKER_IP
NCCL_IFNAME=${NCCL_IFNAME:-enp66s0f0}

usage() { sed -n '2,23p' "$0"; exit 1; }

container_script() {
  [ "$OVERLAY" = "1" ] && overlay_prelude
  echo "ray start --address $DSV41_HEAD_IP:$RAY_PORT --node-ip-address $NODE_IP --num-gpus $DSV41_WORKER_GPUS --block --disable-usage-stats"
  # dsv41 cluster: `ray start --block` exits when the head's GCS goes away;
  # DSV41_KEEP=1 keeps the container (and /tmp/ray) around for `raylogs`.
  if [ "${DSV41_KEEP:-0}" = "1" ]; then
    echo 'echo "dsv41-worker: ray exited $?; DSV41_KEEP=1 so staying up for log collection (./cluster-worker.sh raylogs; stop)" >&2; sleep infinity'
  fi
}

run_podman() {
  local mode=$1
  gpu_preflight
  common_podman_args
  PODMAN_ARGS+=(--entrypoint bash "$IMAGE" -c "$(container_script)")
  if [ "$mode" = "-d" ]; then
    podman run -d --rm "${PODMAN_ARGS[@]}"
    start_log_capture
    echo "started $NAME (joining $DSV41_HEAD_IP:$RAY_PORT); ./cluster-worker.sh logs"
  else
    exec podman run --rm "${PODMAN_ARGS[@]}"
  fi
}

cmd=${1:-start}; shift || true
case "$cmd" in
  run)    run_podman fg ;;
  start)  run_podman -d ;;
  script) container_script ;;
  logs)   exec podman logs "${@:--f}" "$NAME" ;;
  status) status_common ;;
  raylogs) ray_logs_snapshot ;;
  stop)   ray_logs_snapshot; podman stop -t 5 "$NAME" ;;  # ray start --block ignores SIGTERM; nothing to flush
  *) usage ;;
esac
