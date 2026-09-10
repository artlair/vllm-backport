#!/usr/bin/env bash
# Sync this worktree's vllm/ + tools/dsv41/ (and optionally a model dir) to
# both cluster nodes so OVERLAY=1 boots on x299 and rome see the same python.
#
#   ./cluster-sync.sh                      # code only
#   MODEL_DIR=./full-dummy ./cluster-sync.sh   # code + model dir
#   ./cluster-sync.sh --dry-run            # rsync -n
#
# Targets: <host>:~/dsv41-test/src/{vllm,tools/dsv41} and, when MODEL_DIR is
# given, <host>:~/dsv41-test/models/<basename of MODEL_DIR> (MODEL_NAME
# overrides the basename). Hosts: HEAD_HOST [x299], WORKER_HOST [192.168.1.7]
# (ssh names/IPs; rome resolves by IP only). DEST [~/dsv41-test].
# Excludes .git, __pycache__, *.pyc, .so build outputs and the trunc/ scratch
# dirs; --delete keeps the remote trees exact mirrors.
set -euo pipefail

HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
SRC_DIR=${SRC_DIR:-$(cd "$HERE/../.." && pwd)}
HEAD_HOST=${HEAD_HOST:-x299}
WORKER_HOST=${WORKER_HOST:-192.168.1.7}
DEST=${DEST:-dsv41-test}
HOSTS=${HOSTS:-"$HEAD_HOST $WORKER_HOST"}
RSYNC_OPTS=(-az --delete --info=stats1 "$@")
EXCLUDES=(--exclude .git --exclude __pycache__ --exclude '*.pyc' --exclude '*.so' --exclude 'tools/dsv41/trunc*' --exclude 'tools/dsv41/full-dummy*')

[ -d "$SRC_DIR/vllm" ] || { echo "no vllm/ under SRC_DIR=$SRC_DIR" >&2; exit 1; }
if [ -n "${MODEL_DIR:-}" ]; then
  [ -f "$MODEL_DIR/config.json" ] || { echo "no config.json in MODEL_DIR=$MODEL_DIR" >&2; exit 1; }
  MODEL_NAME=${MODEL_NAME:-$(basename "$(cd "$MODEL_DIR" && pwd)")}
fi

for host in $HOSTS; do
  echo "== $host: code -> $DEST/src"
  # shellcheck disable=SC2029  # DEST expands client-side on purpose
  ssh "$host" "mkdir -p $DEST/src/tools $DEST/models"
  rsync "${RSYNC_OPTS[@]}" "${EXCLUDES[@]}" "$SRC_DIR/vllm/" "$host:$DEST/src/vllm/"
  rsync "${RSYNC_OPTS[@]}" "${EXCLUDES[@]}" "$SRC_DIR/tools/dsv41/" "$host:$DEST/src/tools/dsv41/"
  if [ -n "${MODEL_DIR:-}" ]; then
    echo "== $host: model -> $DEST/models/$MODEL_NAME"
    rsync "${RSYNC_OPTS[@]}" "$MODEL_DIR/" "$host:$DEST/models/$MODEL_NAME/"
  fi
done
echo "synced $(git -C "$SRC_DIR" rev-parse --short HEAD 2>/dev/null || echo '?') to: $HOSTS"
echo "on each node: SRC_DIR=~/$DEST/src${MODEL_DIR:+ MODEL_DIR=~/$DEST/models/$MODEL_NAME}"
