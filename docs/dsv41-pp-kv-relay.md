# DeepSeek V4.1: PP relay of shared caches (`VLLM_DSV41_PP_KV_RELAY`)

Fork feature, gated behind `VLLM_DSV41_PP_KV_RELAY=1`. With the env off the
model keeps raising `NotImplementedError` when a pipeline stage holds consumers
of a V4.1 kv-sharing group but not its source layer. Nothing in this note
applies to GLM-5.3 or DeepSeek V4.0.

## Why

V4.1 shares caches across layers:

* `kv_source_layer_ids [2, 8, 14, 20]`: each source owns a compressor, the
  compressed-KV `MLAAttentionSpec` cache (`<layers.N>.attn`) and the paged
  indexer K cache (`<layers.N>.attn.indexer.k_cache`). Layers between two
  sources read the source's compressed KV through `_static_forward_context`.
* `index_source_layer_ids [2, 8, 14, 20, 24, 28, 32, 36]`: each runs the
  sparse indexer and publishes top-k indices into the model-wide
  `topk_indices_buffer`; layers between two index sources only read that
  buffer. Non-kv index sources (24..36) borrow the K cache of kv source 20.
* `candidate_source_layer_id 20`: layer 20's indexer also publishes
  `candidate_topk_blocks` (2048) block ids per query row into
  `candidate_block_buffer`; index sources 24..36 mask their scores with them.

Layer 20's group is layers 20..39 (about 138 GiB of fp4 experts), far more
than one TP=4 stage of 4x24 GB can hold, so the deployment partition
`VLLM_PP_LAYER_PARTITION=8,7,9,8,8` (x299 stages 0-1, rome stages 2-4) splits
groups 14 (14 on stage 1, 15..19 on stage 2) and 20 (20..23 on stage 2, 24..31
on stage 3, 32..39 on stage 4).

## What is relayed

Three payload kinds, each keyed by its source layer id so a later group can
never see a stale payload:

| key | shape | dtype | bytes/token | producer | consumer |
| --- | --- | --- | --- | --- | --- |
| `dsv41_relay_latent_<kv>` | `[T, 512]` | bf16 | 1024 | kv source `kv`'s compressor (`DeepseekCompressor.forward`, written straight into the relay buffer via `latent_out`) | the mirror on every later stage holding consumers of `kv` |
| `dsv41_relay_topk_<idx>` | `[T, index_topk=512]` | int32 | 2048 | index source `idx` (snapshot of `topk_indices_buffer[:T]` right after layer `idx` runs) | stages holding non-index-source consumers of `idx` |
| `dsv41_relay_cand_<c>` | `[T, candidate_topk_blocks=2048]` | int32 | 8192 | candidate source `c` (snapshot of `candidate_block_buffer[:T]` after layer `c`) | stages holding index sources above `c` |

`T` is the padded token count of the step, the same `T` the existing
`hidden_states` (`[T, hc_mult=4, 5120]` bf16 = 40960 bytes/token) and
`pre_mix` (`[T, 4]` fp32) entries use, so shapes stay a deterministic function
of `T` and the `VLLM_PP_CACHED_METADATA` signature cache keeps working.
Buffers are allocated once at `max_num_batched_tokens` rows and sliced
`[:T]`, so the copied-into addresses are static for cudagraph capture.

The latent is relayed instead of the quantized cache rows because both shared
caches are pure functions of it: the compressed KV row is
`rope_quant_insert(latent)` and the indexer K row is
`indexer_k_norm_rope_store(k_norm(wk(latent)))`. Recomputing on the receiver
reuses the source's kernels verbatim, is layout-agnostic (fp8_ds_mla uint8 or
plain bf16 rows, ROCm tiled K layout), handles ratio 2 for free (the kernels
skip non-boundary rows by position) and is smaller than the 584 + 132 byte
quantized pair. The cost is that the mirror carries the source indexer's `wk`
(128x512 bf16) and `k_norm` weights, loaded from the checkpoint through a small
name redirect in `DeepseekV4Model.load_weights`.

Per-hop cost for `8,7,9,8,8` (on top of 40976 bytes/token of hidden state):

* stage 0 -> 1: nothing (groups 2 and 8 are stage-local)
* stage 1 -> 2 (x299 -> rome over 10 GbE): latent 14 + topk 14 = 3072 B/token
* stage 2 -> 3: latent 20 + cand 20 = 9216 B/token
* stage 3 -> 4: latent 20 + cand 20 = 9216 B/token (forwarded by stage 3)

Decode with 8 requests x 4 MTP rows = 32 rows: 96 KB and 288 KB per hop per
step. A 8192-token prefill chunk: 24 MB and 72 MB per hop.

## Slot derivation (nothing slot-related is relayed)

The mirror registers, under the source's exact layer names, the same
`KVCacheSpec`s the source registers (built by the same helper,
`compressed_kv_cache_spec`, and the same `DeepseekV4IndexerCache` class) and
the same attention backend class. `get_kv_cache_configs` unions the per-worker
spec dicts by layer name (`vllm/v1/core/kv_cache_utils.py`), asserting equal
specs for duplicate names, plans groups on the union and projects them back
per worker. The union is unchanged by mirrors (the names already exist on the
source rank), so the scheduler's block tables are identical, and on the mirror
rank `init_attn_backend` builds the same metadata builders for those names:
`attn_metadata["<layers.kv>.attn"].slot_mapping` (compressed slot mapping for
ratio 2) and `attn_metadata["<layers.kv>.attn.indexer.k_cache"].slot_mapping`
are exactly what the source's compressor and `_produce_k` consume. The mirror
therefore calls the same two kernels with the relayed latent, the runner's
`positions` and its own metadata. The compressor state ring
(`CircularBufferSpec`) is not mirrored: only the source's compressor reads it.

Top-k and candidate rows are per token and are consumed by row index
(`topk_indices_buffer[:num_decode_tokens]`, `candidate_blocks[chunk...]`), so
they are copied into the receiving rank's buffers at the same row offsets.

## Hop forwarding

`plan_pp_kv_relay` (`vllm/models/deepseek_v4_1/pp_relay.py`) computes, per
stage, from the topology and the stage bounds (`get_pp_indices` for every
rank, i.e. `VLLM_PP_LAYER_PARTITION` if set): the payloads it receives
(provider stage < r <= last consumer stage), the payloads it sends (exactly the
next stage's receive set), the payloads with local consumers, and the mirrors
it must build (kv sources with local consumers; the indexer K cache only when
a local non-kv index source borrows it). A stage that only forwards a payload
(possible for candidates: a stage holding layers 29..31 has no index source)
returns the received tensor unchanged in its outgoing `IntermediateTensors`.
`make_empty_intermediate_tensors` on rank r and the outgoing dict on rank r-1
are built from the same plan, so keys and shapes always match.

Order inside a step on a receiving rank: the mirror writes both caches and
copies top-k / candidate rows before the first local decoder layer runs; a
later local index source overwrites `topk_indices_buffer` only after the
relayed consumers above it have run, as in the unsplit model.

## TP interaction

`gpu_worker` sends intermediate tensors with the TP all-gather split (each TP
rank ships 1/TP of every tensor, the receivers all-gather). That is valid for
tensors replicated across TP: the latent (replicated compressor GEMM over the
all-reduced hidden state) and the top-k buffer (all-gathered / all-reduced
across TP by the indexer even when `VLLM_INDEXER_QUERY_SHARD=1`). The
candidate buffer is row-sharded across TP when `VLLM_INDEXER_QUERY_SHARD=1`,
so the model exposes `pp_all_gather_tensors` and the worker disables the split
for candidate keys in that mode (each TP rank then sends its full buffer to
its own PP peer, which is the rank that reads the same rows).

## Spec decode

DSpark draft layers (40..42) have `compress_ratio 0`: `is_kv_source` and
`is_index_source` are False, `kv_source_layer_id` is None, they register only a
sliding-window cache and never look a source up. They live on the last stage
and are untouched by the relay.

## Unverified (multi-GPU only)

No PP>1 run is possible outside the production cluster, so the following are
reasoned from code only:

* NCCL send/recv of the extra keys under ray + `VLLM_PP_CACHED_METADATA`,
  including the metadata signature on the first steps.
* That the mirror's per-rank metadata (`DeepseekV4SparseMLAMetadataBuilder`,
  `DeepseekV32IndexerMetadataBuilder`) produces the same slot mapping the
  source rank sees for the same scheduler output (the builders are the same
  classes fed the same block tables; equality was not observed).
* Cudagraph capture on receiving ranks with the mirror kernels in the captured
  region, and the eager-break ordering of the candidate / top-k snapshots.
* Weight loading of the mirror's `wk` / `k_norm` through the name redirect on
  a real checkpoint (the redirect is exercised by name only in the CPU tests).
* Numerical parity of consumers reading a mirrored cache versus the unsplit
  model (the kernels are identical, so the only difference is the `wk` GEMM
  running on another GPU of the same architecture).
* End-to-end throughput impact of the extra bytes on the 10 GbE hop.

## Not supported

* FlashInfer plain per-tensor fp8 rows (`float8_e4m3fn` caches): the row
  insert needs the source layer's `_flashinfer_fp8_kv_scale`, which the mirror
  does not carry. The mirror raises at construction. Ampere / ROCm
  (`fp8_ds_mla` uint8) and bf16 rows are supported.
* Sequence parallel (already PP=1 only) and DCP with candidates (already
  rejected by the indexer).
* The AMD model file (`amd/model.py`) is not wired; Ampere uses
  `nvidia/model.py`.
