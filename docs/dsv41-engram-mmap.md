# DeepSeek V4.1: engram tables from the page cache (`engram_config.table_mode = "mmap"`)

Fork feature, off by default (`table_mode: "auto"` keeps the fork's
`cpu_offload` choice: pinned host memory read over UVA, or HBM). Everything
here is tagged `# dsv41 engram-mmap:` in the code. Nothing applies to GLM-5.3
or DeepSeek V4.0.

## Why

V4.1-Flash carries two n-gram hash tables (`layers.{1,14}.engram.embed`):
384M rows x 256 fp8 plus 384M x 8 ue8m0 block scales, 94.4 GiB + 2.9 GiB
per table in `model-0004{7,8}-of-00048.safetensors`. Upstream keeps each
rank's row shard in *pinned* host memory and gathers rows on the GPU over
UVA (`ParallelEngramEmbedding`, `_engram_lookup_kernel`). Our hosts cannot
pin that: x299 has ~99 GB free of 125, rome ~82, and every TP rank adds
~4 GB of RSS. The tables must stay exactly fp8 (no requantisation).

A step needs almost nothing from the tables: per token 3 n-grams x 8 heads
= 24 rows x 256 B (+ 8 B of scales), i.e. 6 KiB per token per layer. Only
the working set matters, and the page cache already handles that.

## What

`table_mode = "mmap"` (third mode next to `pinned` and `resident`):

* **Storage.** Each rank memory-maps its row range
  `[vocab_start_idx, vocab_end_idx)` of both checkpoint tensors straight
  from the shard file: read-only, `MAP_SHARED`, `madvise(MADV_RANDOM)` (a
  cold row is one 4 KiB read, no readahead). Nothing is pinned and nothing
  is copied; `EngramMmapTable` in `common/engram.py`. The row range is
  contiguous per rank because heads are sharded by row ranges (same
  `vocab_start_idx`/`vocab_end_idx` as the other modes), so the TP layout
  is unchanged and the existing `tensor_model_parallel_all_gather` stays.
* **Lookup.** `Engram.prepare_embeddings` still runs before the decoder
  layers with the hash ids from the GPU `_hash_ids_kernel` (the single
  source of truth for the indices, chunked prefill, spec-decode draft
  tokens and lookback windows included). In mmap mode
  `ParallelEngramEmbedding.stage_from_mmap`:
  1. copies this rank's `[T, local_heads]` slice of the ids to a pinned
     host buffer and syncs the stream;
  2. gathers the raw fp8 rows and their 8 scale bytes on the host with
     `np.take` into pinned staging (rows owned by other ranks and TP
     padding heads are zero, which dequantises to 0 exactly like the UVA
     kernel's masked loads);
  3. copies the staging to a persistent device buffer (sized
     `max_num_batched_tokens x local_heads`) and runs the *same*
     `_engram_lookup_kernel` over it with identity indices
     (`vocab_start = 0`, `vocab_end = rows`). Same kernel, same
     `scale << 23` math, so the bf16 output is bit-identical.
* **Cudagraphs.** The host gather can never be captured. Two cases:
  * PIECEWISE (breakable capture, `vllm/compilation/breakable_cudagraph.py`,
    prefill and mixed batches): the lookup registers itself as an eager
    segment via `capture.add_eager` with weak-referenced ids / output (like
    the attention breaks), writing into the static `staged_rows` buffer that
    the next graph segment reads.
  * FULL (the V2 runner's `torch.cuda.graph` decode graphs in
    `FULL_AND_PIECEWISE` / `FULL_DECODE_ONLY`): the runner calls
    `model.engram_prefetch(**model_inputs)` under the step's forward context
    right before `run_fullgraph` (`gpu/model_runner.py`). It runs the same
    hash kernel on the same static inputs and metadata the graph reads and
    stages the rows; the captured lookup is a no-op
    (`ParallelEngramEmbedding.full_graph_prefetch`). The hash kernel also
    runs inside the graph (redundant, cheap, idempotent).
  With `VLLM_USE_BREAKABLE_CUDAGRAPH=0` and cudagraphs on, config
  verification refuses the mode; on the V1 runner FULL modes fall back to
  PIECEWISE with a warning; `--enforce-eager` always works.
* **Loading.** `DeepseekV41ForCausalLM.lazy_mmap_weight_names` marks
  `layers.N.engram.embed.{weight,scale}`; `safetensors_weights_iterator`
  yields those as `SafetensorsMmapRef` (header offsets only, parsed from
  the 8-byte length + JSON header, no bytes read) and the engram
  `weight_loader` maps the rank's slice. The params are 0-row placeholders
  so the names stay in `named_parameters()` and the loaded-weights
  bookkeeping is unchanged; PP ranks without the layer drop the ref
  (previously every rank *read* both tables just to discard them). Only
  the default safetensors path honours the refs (not
  `enable_multithread_load`, prefetch or torchao strategies); the loader
  raises if a real tensor reaches an mmap param.
* **Dummy load.** `--load-format dummy` never calls the weight loader; the
  first lookup attaches anonymous zero-filled tables of the shard's size
  (calloc-backed, only touched pages materialise) so the whole gather path
  still runs.

## Usage

```
--engram-config '{"table_mode": "mmap"}'
```

Harness: `ENGRAM_MODE=mmap` in `tools/dsv41/boot.sh` and
`cluster-head.sh` (unset keeps `ENGRAM_OFFLOAD`; `pinned` / `resident`
are also accepted). Real-table check without the model:
`tools/dsv41/engram_mmap_check.py --shard .../model-00047-of-00048.safetensors`
(x299: `~/dsv41-test/engram_mmap_check.sh`, runs it in the image with the
overlay). Unit test: `tests/kernels/test_engram_mmap.py`.

## Boot-time warmup (`engram_config.mmap_warm`, tagged `# dsv41 engram-warm:`)

The page cache starts cold after every boot (the ~200 GB weight stream
evicts it) and a cold row is one NVMe read. With DSpark the rejected
drafts hash to never-seen n-grams, so the first 8-stream bench on x299
decayed from 124 to 12 tok/s while the pages warmed and later passes ran
at 270 tok/s. x299 now has 256 GB with the VMs stopped, so both tables
(2 x 94.6 GiB, each rank maps a contiguous quarter) fit in the page cache
with room to spare.

`mmap_warm` (default `"none"`) reads each rank's mapped slices of *both*
tables (weight and scales, sequentially) into the page cache once
**every** rank has loaded its weights: `ParallelEngramEmbedding.warm_mmap`
starts one thread per engram layer on the rank, `EngramMmapTable.warm`
walks the slice in 64 MiB steps with `madvise(MADV_WILLNEED)` one step
ahead and `preadv` of the current step into a single scratch buffer
(readahead-friendly, paced by the disk, `preadv` releases the GIL;
`mmap.madvise` holds it, hence per step rather than over the whole slice).
No copy of the table is kept: only the page cache fills. `"sync"` blocks
until the slices are cached, `"async"` returns at once and logs completion
later; each rank logs one line at start and one at the end with bytes and
seconds. Only ranks holding an engram layer do anything; a dummy load
(anonymous tables) skips with a log line. The cold-miss gather path is
untouched. Harness: `ENGRAM_WARM` in `boot.sh` / `cluster-head.sh`.

### Where the warm runs (and why not in the model's post-load hook)

The first version warmed from
`DeepseekV41LLMForCausalLM.process_weights_after_loading`
(right after this rank's `load_weights`). On x299 that reported "done" on
every rank and left ~45% of each table resident: the other ranks on the
host were still streaming ~110 GiB of weight shards through the page
cache during and after the warm, and those pages (useless once the weights
are in VRAM) evicted the warmed ones. Two changes fix it:

* **The warm moved to the worker** (`Worker._warm_engram_tables` in
  `v1/worker/gpu_worker.py`, calling `model.warm_engram_tables`). The
  executor's collective RPC returns only when every rank has answered, so
  the first RPC after `load_model`, `determine_available_memory`, is the
  first point where every rank on the host is done streaming; the warm
  starts there (before the profile run; `"sync"` blocks it, `"async"`
  overlaps the profile run, KV allocation and graph capture). A **second
  pass** runs at the end of `compile_or_warm_up_model` (`final=True`,
  logged as `Engram mmap warm (<mode>, final pass)`, own thread and stats,
  waits for the first): it reads the same bytes again, which costs a few
  seconds at memory speed when the slices stayed resident and re-reads
  from disk whatever got evicted meanwhile. The logged GB/s of the final
  pass is the residency check (memory speed = resident).
  `process_weights_after_loading` no longer warms.
* **The weight shards are dropped from the page cache**
  (`engram_config.drop_weight_pages`, `None` = on with `table_mode =
  "mmap"`, off otherwise, so other models are untouched; `ENGRAM_DROP=0|1`
  in the harness). `WeightPageCacheDropper` in
  `model_loader/weight_utils.py`, wired by `DefaultModelLoader.load_weights`
  when the model exposes `drop_weight_pages` (the V4.1 VL wrapper does):
  the safetensors iterator tells it after each shard whether an mmap
  `SafetensorsMmapRef` was yielded from it, and it `posix_fadvise(fd, 0,
  0, POSIX_FADV_DONTNEED)`s every shard that holds none, one shard behind
  the stream (`safe_open` hands out views of a private mapping of the
  whole shard and the consumer still holds the previous yield when the
  iterator moves on; DONTNEED skips mapped pages, so an early drop would
  be a no-op), then once more over every shard after the model's
  `load_weights` returned (`finish`). That last pass is the one that
  counts for the VL wrapper, which sorts the whole stream before loading
  (every shard stays mapped until then). Shards holding an mmap engram
  slice (`model-0004{7,8}`) are never dropped, whether or not this rank
  maps them: another rank on the host does. DONTNEED is advisory and only
  discards clean, unmapped pages, so it cannot affect correctness; the
  cost is that a sibling TP rank lagging on the same shard re-reads the
  tail from disk. One log line per rank:
  `Dropped the page cache of N safetensors shard(s) (X GiB) after loading
  the weights (POSIX_FADV_DONTNEED); kept K shard(s) backing mmap engram
  slices`.

Expected picture on x299 after a boot (256 GB, both tables 2 x 94.6 GiB
plus 2 x 2.9 GiB of scales): `fincore` on `model-00047` and `model-00048`
shows both fully resident (every rank's quarter, warmed after the drops,
re-read at memory speed by the final pass), the other 46 shards show ~0
resident pages, and `buff/cache` sits at ~195 GiB plus the KV/CUDA host
buffers instead of the weight shards competing for it.

Expected time: a quarter of the weight table (23.6 GiB) plus its scales
(0.74 GiB) is ~26 GB per rank, ~13 s at ~2 GB/s sequential; the four TP
ranks of a stage stream their quarters concurrently from the same NVMe,
so a stage holding one table should warm in ~1 minute at the drive's
sequential rate, both tables on one node in ~2 minutes.

## Measured (2026-09-11, image 67c4aec4f-serve)

Unit test (3090): mmap vs pinned on a random table, TP 1/2/4, every rank,
1 / 5 / 700 tokens: `torch.equal` (bit-identical), eager-break replay with
fresh ids: identical to the pinned path.

Real shard 47 on x299 GPU 0 (layer 1, TP=1, all 384,006,168 rows mapped):
2016 random rows through the host path vs `safe_open().get_slice()` + torch
dequant: **IDENTICAL**. Gather cost (host gather + H2D + kernel, cuda sync):

| rows | page cache | 8 threads | 16 threads | 32 threads (default) |
| --- | --- | --- | --- | --- |
| 98,304 (4096 tok) | dropped (`fadvise DONTNEED`) | 3020 ms | 1950 ms | 1403 ms |
| 98,304 | fresh random rows, no drop | 2886 ms | 1618 ms | 1011 ms |
| 98,304 | warm (same rows) | 24 ms | 27 ms | 22 ms |
| 192 (8 tok) | fresh random rows | 39 ms (serial) | | 6.7 ms |
| 192 | warm | 0.54 ms (serial) | | 2.0 ms (first pooled call) |
| 24 (1 tok) | fresh random rows | 5 ms (serial) | | serial |
| 24 | warm | 0.49 ms | | serial |

A cold row is one ~200 us NVMe read; 32 gather threads reach ~100k rows/s,
close to the drive's ~122k 4K reads/s at QD32. Warm rows cost ~0.25 us each
(memcpy) plus ~0.4 ms of fixed cost (D2H sync, two H2D copies, kernel).

Chunk policy for decode-sized gathers (192 rows, host gather only, median
of 15, x299): serial 0.005 ms warm / 27 ms cold; thread pool with min chunk
16: 0.52 / 4.0 ms; 32 (chosen): 0.31 / 5.9 ms; 48: 0.26 / 8.1 ms; 64:
0.23 / 10.9 ms; `madvise(MADV_WILLNEED)` per row then serial: 0.40 / 9.3
ms. So the pool costs ~0.3 ms per warm step above two tokens in exchange
for a 4-5x bound on cold-miss latency; one-token steps stay serial.

Dummy boot on the 3090 (`ENGRAM_MODE=mmap EAGER=0`, i.e. the default
`FULL_AND_PIECEWISE`, trunc fp8 config: 6 layers, engram on layer 1 with a
1M-row anonymous table, V2 runner, `--max-num-seqs 8`): the PIECEWISE
breakable graphs capture with `eager_breaks=7` (6 attention + 1 engram), the
4 FULL decode graphs capture with the prefetch hook, the smoke completion
round-trips. 600-token completions with `ignore_eos`, two passes each:

| mode | batch 1 tok/s (ms/step) | batch 8 agg tok/s (ms/step) |
| --- | --- | --- |
| pinned (default) | 179.7 / 177.9 (5.6) | 1306 / 1310 (6.12) |
| mmap | 177.9 / 178.4 (5.6) | 1257 / 1268 (6.34) |

So the break costs nothing measurable at batch 1 and ~0.2 ms per step
(3.5%) at batch 8. The split timer (`engram mmap gather` debug line,
cumulative) puts the steady-state host gather + H2D + launch at 0.15-0.25
ms per step (the first 500 calls average 1.9 ms because they include the
one-off Triton compile and first-touch page faults) and the wait for the
ids at ~3.2 ms per step, which is the previous step's GPU work draining
before the CPU can read the hash ids: that wait is GPU time the step would
spend anyway, not added latency, but the GPU idles during the gather.

## Limits and risks

* **Page-cache pressure.** The hot rows must stay cached; the kernel
  evicts them like any file page. With 128 GB the two tables (2 x 97 GiB)
  do not fit x299's page cache, one does barely; with 256 GB both fit only
  if the weight shards' pages are gone: `drop_weight_pages` evicts them
  after the load and `mmap_warm` (run once every rank has loaded, plus
  the final pass) preloads the tables. Without both, the tables start
  cold or half-evicted after every boot. Real text hashes are Zipfian,
  so the steady-state working set is far smaller than the table, but it
  is unmeasured. Watch `buff/cache`, `fincore` on the two table shards,
  and the `engram mmap gather` debug stats (`VLLM_LOGGING_LEVEL=DEBUG`,
  every 500 calls).
* **First-request / cold latency.** A fully cold 4096-token prefill chunk
  costs ~1-1.4 s per engram layer on x299's NVMe (both layers on the same
  stage: double). A cold decode step is ~200 us per missing row on the
  calling thread for one token, spread over up to 32 threads above that.
  There is no prefetch: the ids only exist after the hash kernel runs.
* **The host sync itself.** Every step pays a stream sync (PIECEWISE: at
  the engram layer, after the hash kernel; FULL: before the replay, which
  waits for the previous step's graph to drain) plus the gather, and the
  GPU idles for that long. On the dummy 3090 boot this is invisible at
  batch 1 and ~0.2 ms per step at batch 8 (see above); on the cluster the
  same sync lands on the two x299 stages only.
* **PP ranks.** Only the ranks holding layer 1 or 14 map a table and pay
  the break; with the default partition `8,8,8,9,7` that is stages 0 and 1
  on x299 (both tables on one node, 2 x 97 GiB of page cache demand). A
  partition that puts layer 14 on rome spreads the cache load.
* **rome** has no local copy of the checkpoint yet (the rsync in
  `~/dsv41-test/rsync-to-rome.log` failed with `Permission denied
  (publickey)`); mmap mode needs the shard files on every node that owns
  an engram layer.
* **Loader coupling.** Only the default `safetensors_weights_iterator`
  path produces the lazy refs; other load strategies fail loudly rather
  than silently reading 94 GiB.
* **Not a prefetch.** A cleaner design would gather during scheduling from
  host-known token ids (hash on the CPU); it would need the lookback
  windows and spec-decode draft ids on the host and a CPU re-implementation
  of the hash, so it was not attempted (the GPU indices are the reference).
