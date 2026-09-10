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
