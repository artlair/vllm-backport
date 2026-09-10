# DeepSeek-V4.1-Flash dummy-weight boot harness

Boots the V4.1 model code with `--load-format dummy` on a truncated config so every structural feature
(SWA-only layers, ratio-2 and ratio-1 compressed KV, kv/index sources, candidate filtering, engram,
DSpark MTP, vision tower) is exercised without downloading the 300 GB checkpoint.

1. Generate the model directory (config.json + tokenizer + minimal chat template):
   `python3 make_trunc_config.py --out trunc --bf16` (first boot; drop `--bf16` for the fp8/fp4 path).
   Knobs: `--layers 6 --experts 16 --nextn 1 --engram-num-embeddings 1000000 --vision-layers 2`,
   `--keep 0,1,2,3,20,24` for an explicit subset, `--max-position-embeddings 65536` to shrink rope tables.
   It prints the layer table and refuses topologies the model code would reject (see the docstring).
2. Boot on one 3090: `IMAGE=<sm86 image> MODEL_DIR=$PWD/trunc ./boot.sh start`, then `./boot.sh smoke`
   (waits for /health, sends one chat completion at temperature 0, prints text and tokens/s),
   `./boot.sh logs`, `./boot.sh stop`. `./boot.sh serve` runs in the foreground; `./boot.sh print` shows the vllm argv.
   Env knobs: `TP PP CTX UTIL SEQS SPEC ENGRAM_OFFLOAD EAGER PORT NAME BATCHED EXTRA`.
   `SPEC=5` enables DSpark (V4.1 sets n_predict to dspark_block_size=5; values above 5 must be multiples of 5).
   Dummy weights produce garbage text; the point is that the server boots and the request round-trips.
   Note: `tokenizer.json`, `tokenizer_config.json` come from the HF repo; `chat_template.jinja` is ours
   (the repo ships only a Python encoder), so chat formatting is approximate but sufficient for smoke tests.

## Two-node cluster lane (x299 + rome, TP=4 x PP=5)

Same topology as the production GLM-5.3 lane: x299 = ray head + PP stages 0-1 (8x3090, TP=4),
rome = ray worker + PP stages 2-4 (12x3090, TP=4), 10 GbE between them. Scripts:
`cluster-sync.sh` (rsync code + model dir to both nodes), `cluster-worker.sh` (rome),
`cluster-head.sh` (x299), sharing `cluster-lib.sh`. They mirror `~/glm-head-ubatch.sh`,
`~/glm-worker.sh` and my-salt `formulas/vllm` (host network, `--ipc host`, `--pids-limit=-1`,
stack ulimit, `/model:ro` at the same path on both nodes, `VLLM_HOST_IP` + `NCCL_SOCKET_IFNAME`
per node (br1 / enp66s0f0), ray GCS 6379, `--distributed-executor-backend ray`,
`--disable-custom-all-reduce`, `VLLM_USE_V2_MODEL_RUNNER=1`). Differences from the GLM lane:
the overlay runs in BOTH containers (rome's ray worker processes inherit `PYTHONPATH=/work`
from `ray start`, so they import the synced python, not the image's), a preflight refuses to
start while any GPU has >1 GiB in use or another lane container is running (`FORCE=1`
overrides), `start` keeps a host copy of the container log (`LOGDIR` [~/dsv41-test],
`LOGTAG` [timestamp], since `podman run --rm` discards it on exit), and `stop` first snapshots
the ray per-worker logs (`/tmp/ray/session_latest/logs/worker-*` inside the container, where
per-rank evidence lives once the head log dedups it) to `$LOGDIR/<name>-<tag>-ray/`
(`raylogs` does it on demand). `MEMLOCK=1` passes `--ulimit memlock=-1`, but rootless podman
clamps it to the user's hard limit (8 MiB on x299 and rome) and the pinned engram tables are
CUDA host allocations that do not need it, so it is best-effort. Orchestration env is
`DSV41_*` (never `VLLM_*`, which vLLM scans).

### Model directory

`python3 make_trunc_config.py --full-dummy --out full-dummy` writes the REAL 40-layer topology
(all layer lists, 384 experts, 3 DSpark layers with the real targets 37..39, the real 32-layer
vision config, the real fp8/fp4 quantization_config) with only `engram_num_embeddings` shrunk
(default 50,000,000 rows per layer; the real 384M-row tables are 94 GiB each and both would have
to be pinned on x299 under the default partition, which does not fit 125 GiB of host RAM).
The prime-bucket logic re-derives `engram_vocab_size` for the smaller table. It also prints the
expected weight VRAM per PP stage for `--partition` (repeatable, default `8,7,9,8,8`), `--tp 4`,
`--util 0.9`. The per-stage estimate is prorated from the checkpoint totals (fp4 experts
275.7 GiB over 40 layers, attention 5.2, shared experts 1.4, embed + head 2.6, vision 0.9
replicated per GPU); the DSpark layers are an estimate (128 of 384 experts at the fp4 rate,
about 2.5 GiB each). The GLM lane rule of thumb applies: keep weights under ~19 GiB per GPU.

Partition notes: the first two entries are x299. `8,7` keeps layers 0..14 on x299, and with
them both engram layers (1 and 14) and their pinned host tables (2 x 12.3 GiB at 50M rows;
2 x 94 GiB with real tables). Groups are split at 14|15 and 20|23|31 (`docs/dsv41-pp-kv-relay.md`),
so `RELAY=1` is mandatory for any 5-stage split. `8,7,9,9,7` moves one layer off the last stage
(which also carries lm_head + the three DSpark layers) and has the best worst-stage headroom;
`7,8,9,8,8` additionally splits group 2 at 6|7 (intra-node relay, cheap). With real weights and
a partition that puts layer 14 on rome (e.g. `7,7,...`), each node pins one 94 GiB table.

### Procedure

1. Sync (from this worktree, on z20): `MODEL_DIR=./full-dummy ./cluster-sync.sh`
   (code to `x299:~/dsv41-test/src` and `rome:~/dsv41-test/src`, model to `~/dsv41-test/models/full-dummy`).
   The same image tag must exist on both nodes.
2. Worker first, on rome (`ssh 192.168.1.7`):
   `cd ~/dsv41-test/src/tools/dsv41 && IMAGE=<tag> ./cluster-worker.sh start` (then `logs` until
   "Ray runtime started").
3. Head, on x299: `cd ~/dsv41-test/src/tools/dsv41 && IMAGE=<tag> ./cluster-head.sh start`.
   It starts the ray head, waits up to `DSV41_CLUSTER_WAIT=900`s for 20 GPUs, then execs
   `vllm serve`. `./cluster-head.sh print` shows the vllm argv, `script` the whole in-container script.
4. Wait + smoke, on x299: `./cluster-head.sh smoke` (polls /health up to `TIMEOUT=2400`s, one chat
   completion at temperature 0, prints text and tokens/s; dummy weights produce garbage text).
   `./cluster-head.sh status` shows the container, `ray status` and GPU memory.
5. Stop: `./cluster-head.sh stop` on x299, then `./cluster-worker.sh stop` on rome. The head's
   ray dies with its container and `ray start --block` on rome exits with it, so after a failed
   boot both containers are already gone (and with them `/tmp/ray`). `DSV41_KEEP=1` on both
   sides keeps a failed container alive (`sleep infinity`) so `raylogs` can still collect the
   per-rank logs; `stop` then removes it.
6. Concurrency smoke: `CONC=4 MAX_TOKENS=256 ./cluster-head.sh bench` (ignore_eos, prints
   per-request completion counts and aggregate tokens/s).

Head knobs (defaults): `TP=4 PP=5 PARTITION=8,7,9,8,8 CTX=32768 UTIL=0.9 SEQS=4 SPEC=5 EAGER=1
CGMODE=PIECEWISE CAPSIZES=1,2,4,8,12,16,20,24,28,32 RELAY=1 PPMETA= ENGRAM_OFFLOAD=1 MEMLOCK=1
NCCLALGO= NCCLPROTO= LOADFORMAT=dummy KVDTYPE=fp8_ds_mla LIMITMM= BATCHED= EXTRA= OVERLAY=1`.
`LIMITMM='{"image":0}'` stubs the vision tower (saves 0.9 GiB per GPU). Real weights later:
`LOADFORMAT=auto MODEL_DIR=<checkpoint with the real config>` on both nodes.

### Experiments, in order

(a) dummy TP4xPP5 relay boot, full topology, small engram, eager, no DSpark:
    `IMAGE=<tag> SPEC=0 ./cluster-head.sh start && ./cluster-head.sh smoke`
    (worker: `IMAGE=<tag> ./cluster-worker.sh start`). Proves ray + NCCL over 10 GbE, the
    relay's extra intermediate tensors, engram offload with TP-sharded pinned tables, KV sizing.
(b) cudagraphs: `EAGER=0` (PIECEWISE), then `CGMODE=FULL_AND_PIECEWISE`, `FULL_DECODE_ONLY`;
    the relay buffers are static so capture should hold, and `PPMETA=1` once graphs are in.
(c) DSpark: `SPEC=5` (default), draft layers on the last stage.
(d) real weights: `LOADFORMAT=auto` with the real config (real engram tables need a partition
    whose pinned tables fit each node's RAM, see the partition notes).

### Dummy results, 2026-09-10 (image 67c4aec4f-serve, PARTITION=8,7,9,9,7, CTX=32768, UTIL=0.9, SEQS=4)

All boots reached `Application startup complete` in 7-8 minutes (the x299 stages spend ~4.5 of
those dummy-filling their 3 GiB pinned engram shards on the CPU); the 20-GPU ray cluster forms
in ~25 s. Relay plan (`pp_relay_runtime.py`): rank 1 sends `latent_14 + topk_14` (3072 B/token)
across the 10 GbE hop to rank 2 (mirror 14); rank 2 sends `latent_20 + cand_20` (9216 B/token)
to rank 3 (mirror 20 with K cache); rank 3 forwards those plus `topk_32` (11264 B/token) to rank
4 (mirror 20 with K cache). Both engram tables land on x299 (12.38M rows x 256 per rank, 3.04 GiB
pinned, layers 1 and 14). KV: `GPU KV cache size: 288,206 tokens` (8.8x 32k), from 3.17 GiB
free on stage 0 and 1.92 GiB on stage 2 (the tightest). VRAM after load (MiB per GPU): x299
stage 0 19.2k, stage 1 17.4k; rome stage 2 21.8k, stage 3 21.2k, stage 4 17.5k (20.6k with
DSpark). Host RAM on x299 while serving: ~88 GB used of 125 (24 GB of VMs + 8 x 3 GiB pinned
tables + worker RSS), so the 50M-row tables are already close to the ceiling there.

| run | cudagraphs | PPMETA | SPEC | 1 stream, 128 tok | 4 x 256 concurrent |
| --- | --- | --- | --- | --- | --- |
| a | eager | | 0 | 9.7 tok/s | (not run) |
| b1 | PIECEWISE | | 0 | 34.1 tok/s | 56 tok/s |
| b2 | FULL_AND_PIECEWISE | | 0 | 49.1 tok/s | 150 tok/s |
| b3 | FULL_AND_PIECEWISE | 1 | 0 | 49.1 tok/s | 152-162 tok/s |
| c | FULL_AND_PIECEWISE | 1 | 5 | 35.1 tok/s | 85-108 tok/s |

DSpark on dummy weights accepts nothing (mean acceptance length 1.00), so (c) only proves the
draft path boots and round-trips; its cost/benefit needs real weights. Two fork bugs surfaced
by the full topology (fixed on this branch): the dummy fill of fp8 params tripled the host
footprint of the pinned engram shards (ray's memory monitor OOM-killed an x299 rank), and
`eager_break_during_capture` checked `VLLM_USE_BREAKABLE_CUDAGRAPH` at import time, which ray
workers (unlike multiproc ones) do not have yet, so every rank captured attention inline and the
first host sync in the prefill path invalidated the PIECEWISE capture.
