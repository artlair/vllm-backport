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
`--ulimit memlock=-1` by default (pinned engram tables), the overlay runs in BOTH containers
(rome's ray worker processes inherit `PYTHONPATH=/work` from `ray start`, so they import the
synced python, not the image's), and a preflight refuses to start while any GPU has >1 GiB in
use or another lane container is running (`FORCE=1` overrides). Orchestration env is `DSV41_*`
(never `VLLM_*`, which vLLM scans).

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
5. Stop: `./cluster-head.sh stop` on x299, then `./cluster-worker.sh stop` on rome (the head's
   ray dies with its container; the worker keeps retrying to reconnect until stopped).

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
