# Port proposal: upstream vLLM -> artlair/vllm-backport (dsv41-1017 lane)

Lane: DeepSeek-V4.1-Flash-1017, 24x RTX 3090 (sm86, 24 GB, 250 W), TP4 x PP6 across
rome2 (head, stages 0/1/5) + rome (stages 2-4). DSpark k=4 with local-argmax reduction,
MXFP4 experts + fp8 linears (no native fp8/fp4 silicon), TRITON_MLA_SPARSE_DSV41,
kv_cache_dtype fp8_ds_mla, FULL_AND_PIECEWISE graphs (capture sizes 5..80),
prefix caching, max-num-seqs 16, max-num-batched-tokens 2048, max_model_len 524288,
Engram tables cpu_offloaded with gpu_resident_heads {1:2, 14:1}.

Proposal baseline: `dsv41-1017` @ `c41e5a8de` (then deployed as
`zot.lan/vllm-backport/dsv41:c41e5a8de-serve`).
Upstream base: `29e66dad46` (2026-09-26) + hand-ported PRs through ~#63427.
Upstream swept: 2026-09-26 .. 2026-10-30 (tip `5d35ce3934`, #63812).

**Status, 2026-10-31:** Batch A (P1-P5 + F2) and A2 (F1) are implemented and
running as `9842ce767-serve` on rome2 and rome after operator approval. See
[the rollout validation](dsv41_sm86_batch_a_validation.md) for evidence and limits.
Other tiers remain proposals. The operator confirmed no `response_format` or
grammar-constrained traffic, so grammar-specific items remain out of scope.

Every candidate was checked twice: by PR number in commit bodies, and by code
equivalence in the current tree. Several upstream fixes turn out to be in HEAD already
under fork-local commits (`#59166`, `#64458`, `#61451`, `#61161`, `#62046`, `#58231`,
`#62467`/`#62936`, `#61824`, `#56119`, `#56720`), so a number-only grep would have
proposed them again.

---

## Tier 0 - port now

Ordered by (severity x cheapness). At the proposal baseline, all five were unported
and sat on paths this lane executes.

### P1. `#63475` persistent_topk: sync the CTA around radix barrier arrivals

**Correctness, silent wrong attention indices. One line.**

- Upstream `3090266d88`, merged 2026-10-26. Diff: a single added `__syncthreads();`
  before `red_release(&state->arrival_counter, 1)`.
- Fork: `csrc/libtorch_stable/persistent_topk.cuh:814-831` still publishes the
  inter-CTA arrival with no CTA-wide barrier after the `current_hist` atomics and the
  `next_hist` clear. Thread 0 can signal while peer threads are still contributing.
  Peer CTAs then consume a partial histogram, pick a different pivot, and emit
  missing/duplicate indices.
- Why this lane: the radix path activates above `RADIX_THRESHOLD = 32768`
  (`persistent_topk.cuh:36`, dispatch `:1047`, `:1093`), the lane permits 524288 tokens
  and its own harness runs 32K/64K prompts, and `sparse_attn_indexer.py:896-925`
  routes DSA decode top-k (k=2048) straight into this kernel on sm86. FULL graph replay
  does not remove it.
- Evidence it is real: PR author measured a HIP port of the same algorithm on MI355X,
  169 runs / 882,700 rows: 146 wrong rows in 57 runs before, 0 after (plus 712 wrong
  rows in the separate end-of-row variant).
- Port: the pre-arrival barrier maps directly. Review the end-of-row hunk against the
  fork's deterministic exact refinement (`2370f0bb8c`), which already barriers inside
  its final sweep. Gate: long rows (>32K), 1-32 rows, k in {512,1024,2048}, ties,
  mostly `-inf`, repeated runs on 3090.
- Cost: one CTA barrier per radix round next to an inter-CTA spin barrier. No tok/s
  claim to make.

### P2. `#64137` do not specialize MRV2 logits-stats kernels on `num_logits`

**Runtime JIT cliff. Two decorator edits.**

- Upstream `0d6f92ae7c`, merged 2026-10-29: `@triton.jit` ->
  `@triton.jit(do_not_specialize=["num_logits"])` on
  `_compute_local_logits_stats_kernel` and `_compute_local_residual_mass_kernel`.
- Fork: both still plain `@triton.jit`
  (`vllm/v1/worker/gpu/spec_decode/rejection_sampler_utils.py:207-218`, `406-417`).
  Warmup covers one divisibility class; c1..c16 traffic at k=4 produces many.
- Upstream measurement (H100, TP8, MTP k=4, author Ayushman Singh): two runtime JITs
  per rank before, zero after. No throughput claim, and none should be made.
- Cost: negligible; keep the fork's FLy/DSpark rejection paths under test.

### P3. `#62748` skip idle sampler staged writes

**Fixed host cost on every no-admission decode step.**

- Upstream `d7f68d4c7d`, merged 2026-10-28: move `sampler.apply_staged_writes()`
  inside the `scheduled_new_reqs` branch.
- Fork: flush still runs unconditionally
  (`vllm/v1/worker/gpu/model_runner.py:1297-1301`); `Sampler.add_request` is only
  reachable from the admission loop, so the invariant holds.
- Upstream measurement (B300, OPT-125M, author): B=1 1210/1236/1205 -> 1128/1156/1145 us
  per step; B=8 1339/1355/1332 -> 1272/1286/1276 us. Do not attach the PR's 10-11%
  headline, which included persistent staging buffers never merged.
- Lane relevance: only stage 5 owns the sampler, but every PP step waits on it.

### P4. `#63819` fixed `BLOCK_SIZE` in DSpark draft input preparation

**Removes shape-driven Triton compiles mid-serving.**

- Upstream `458334f00f`, merged 2026-10-28. The DSpark hunk is one line:
  `BLOCK_SIZE = min(256, next_power_of_2(...))` -> `128` in `prepare_dflash_inputs`.
  (The grid was already 2-D; ignore reports of a grid change.)
- Fork: same dynamic expression at
  `vllm/v1/worker/gpu/spec_decode/dflash/speculator.py:779`. Because `BLOCK_SIZE` is a
  constexpr, new prompt-tail buckets compile new kernels while serving.
- Upstream measurement (B300, CUPTI, author @mgoin, k=7): decode bs1 19.7 -> 7.5 us,
  bs256 14.3 -> 5.3 us; runtime variants observed before, none after.
- Port only the speculator hunk plus its test. **Omit** the `acceptance_estimator.py`
  and `warmup.py` hunks: those belong to adaptive verification, which this lane
  deliberately leaves off.
- Companion nit: fork uses `input_batch.num_scheduled_tokens.max()` over the padded
  buffer where upstream slices `[:num_reqs]` (#58984, pre-base). Perf-only; largely
  moot once the block size is fixed.

### P5. `#63118` NaN-stable key conversion in top-k helpers (hardening)

- Upstream fixes `bits | 0x80000000u` (which NVCC may contract to `-|x|`, canonicalizing
  NaNs) by reading float bits through opaque `mov.b32` PTX.
- Fork still has the vulnerable expression at
  `csrc/libtorch_stable/persistent_topk.cuh:50-52` and
  `csrc/libtorch_stable/topk_histogram_4096.cuh:71-73`, and the indexer runs with
  `clean_logits=False` (`sparse_attn_indexer.py:837`, `:851`).
- Honest limit: upstream's `cudaErrorIllegalAddress` reproducer came from its own
  overflow-recovery code, which this fork replaced (`2370f0bb8c`). Do not claim the IMA
  reproduces here; port it as hardening with NaN/finite/inf tests.

---

## Tier 0b - fork imports (not upstream; no PR number exists)

The six remotes were diffed against HEAD commit-by-commit. Most fork work is either already
in HEAD in a better form (see "already present" below) or does not fire on TP4xPP6. Two
items are worth taking now, two after a measurement, one is a low-priority hygiene item.

### F1. PP can schedule DSpark drafts before the sampled target token exists

**Correctness. From buliao `84d7523838`.**

- HEAD has the PP cadence check (`scheduler.py:667-671`) but then proceeds straight to
  `num_new_tokens` (`:690-704`), with neither a draft-before-target guard nor a cap that
  stops speculative rows consuming the target token slot (`:838-850`). Buliao adds both
  (`84d7523838:scheduler.py:768-774`, `:943-953`) plus a CPU regression that builds the
  offending state and asserts the first schedule is empty.
- Follow-up audit corrected the initial lane claim: the guard is live for synchronous
  PP, but defensive on this deployed async lane. Normal `AsyncScheduler` accounting
  maintains `num_tokens + num_output_placeholders - num_computed_tokens >= 1`.
  A real sync PP6 regression drives scheduling, early draft delivery, target output
  delivery and the unblocked verification step; it fails when the guard is removed.
- Effect: fail-closed hardening against verifying a draft without its anchor, not a
  demonstrated async-lane race fix or performance gain. The draft-count clamp is
  redundant under the current wait guard and is retained as a defensive bound.
  Buliao validated 32K/128K requests and 8 concurrent on its own rig.
- Port only the two invariants and the regression, not the composite commit (it also
  carries CPU-hybrid scheduler changes). Medium-high risk against the fork's adaptive
  step-id and confidence-relay machinery.

### F2. Indexer prefill scratch is sized for 40 requests, but this lane allows 16

**Memory, ~1.5 GiB per rank. Two lines. From the same buliao commit.**

- `vllm/v1/attention/backends/mla/indexer.py:845` returns `max_model_len * 40`; the
  custom op reserves FP8 K plus scales from it during profiling
  (`sparse_attn_indexer.py:440-450`). With `max_model_len=524288` that is 20,971,520
  entries per ratio-1 indexer; capping at `min(40, max_num_seqs)` = 16 gives 8,388,608.
  At 132 B/entry that is ~2.58 GiB -> ~1.03 GiB of shared profiling workspace, i.e.
  ~1.55 GiB per rank, halved for ratio-2 indexers. Code-derived, not measured here.
- Why this lane: stage 0 runs two GPU-resident Engram heads with ~2 GiB headroom and has
  OOMed in profile before; this is the single largest free memory win in the sweep.
  Buliao measured 800 -> 832 cache slots at the same utilization on its own rig.
- Port: the two-line cap plus the chunk-coverage test (request counts 1/4/8/40/64),
  verifying dummy-profile and real chunking agree for ratio-1 and ratio-2 indexers and
  for PP-local request counts. Low-medium risk, isolate it from F1.

### F3. Exact DSpark draft CUDA-graph buckets (measure; precondition now verified)

Luixiao `e9ec85d14a` derives draft-specific capture sizes instead of rounding target
sizes to `decode_query_len`. With k=4 and anchor mode (`sample_from_anchor=true`,
`dspark/speculator.py:52-61`) draft captures become `[8,12,16,20,32,40,60,64/80]`, so C1
replays an 8-row graph for 4 live rows and C6/C7 pad to 32. **Precondition confirmed on the
served checkpoint:** `/home/alec/dsv41-1017/config.json` has no `sample_from_anchor`
under `text_config` (keys present: `dspark_block_size`, `dspark_markov_rank`,
`dspark_n_routed_experts`, `dspark_noise_token_id`, `dspark_num_experts_per_tok`,
`dspark_target_layer_ids`, `num_nextn_predict_layers: 3`), and the code defaults the
attribute to `True`, so `num_query_per_req == k == 4` and the mismatch fires.
Unmeasured by the fork; small graph-descriptor memory cost.

### F4. Build Ampere prefill indices on device (measure)

PP170 `2fd36330e4` replaces the shared Torch chain (`repeat_interleave`, `arange`, masks,
advanced indexing, `index_copy_`) in `deepseek_v4_1/amd/rocm.py:166-245` with one Triton
launch, behind an Ampere-only override, removing launch count and an implicit D2H sync
from TTFT. Unmeasured; exact-output tests and 64K/256K needle retrieval only.

### F5. Power-of-two reuse for prefill-logits allocations (low priority)

PP170 `0f61e15c64` rounds `[M,N]` fp32 logits capacity up to `next_power_of_2(N)` so the
allocator can reuse it as contexts grow. Prefer buliao's later fixed-workspace variant
(`56132ba33a`) over this older allocator tactic; HEAD has since gained a shared workspace
manager. Unmeasured.

**Fork work explicitly NOT to port.** Candidate-only sparse MQA (luixiao `ca8a3cde14`
et al.): gated to exact (8,0), A100-only, and it measured 128K/C1 warm throughput
**-4.89%** while helping 128K/C8 +15.2%, which is backwards for a C1-sensitive lane.
mHC overlap (#57603/#57874, luixiao `b43d9aa70b`): exact SM80 gate, A100-only numbers,
needs sm86 side-stream measurement first. Compact MXFP8 grouped projection
(`7dadeeb17f`): its own commit reports latency regressions. Engram UVA prefetch
(`b3d5e18081`): falls back to main-stream lookup under capture, so replay gets nothing.
TOMY's 60-commit `VLLM_SM86_DCP` series: lane has no DCP and no KV offload tier.
Dormant non-bug: the 2D paged-MQA context-length fix (buliao/Luixiao) does not fire while
HEAD forces flattened `next_n=5` decode without DeepGEMM
(`vllm/v1/attention/backends/mla/indexer.py:875-897`); it becomes live only if k drops to
1/2 or native SM8x multi-token decode is re-enabled.

---

## Strategic: the model tree question (do not rebase)

Upstream rebuilt DSv4.1 as `vllm/models/deepseek_v41/` (109 files, `nvidia/`, `amd/`,
`common/`) while the fork runs `deepseek_v4_1/` (29 files) with its own `pp_relay.py` and
Engram GPU-resident heads. Diff: 114 files, 31,860 insertions. All 91 upstream DSv4.1-tree
PRs since the base were classified (5 already in HEAD: `#57407`, `#58560`, `#58678`,
`#59119`, `#62208`).

**Verdict: stay on the fork tree.** Upstream's NVIDIA DSv4.1 path cannot serve this lane
at all, independent of performance:

1. its sparse backend declares compute capability 9/10 only
   (`upstream/main:vllm/models/deepseek_v41/sparse_mla.py:145-152`), and RTX 3090 is sm86;
2. it raises when an index-K or compressed-KV source sits on another PP rank
   (`upstream/main:.../attention.py:454-468`, `:598-613`), and the lane's `5,9,6,7,8,5`
   partition deliberately crosses those sharing groups;
3. it creates target embeddings only on the first PP rank and collects only locally
   produced aux states (`upstream/main:.../nvidia/model.py:789-797`, `1094-1161`), while
   the fork declares `supports_aux_hidden_states_over_pp` for DSpark-at-PP>1.

So a tree replacement fails at backend selection or PP construction. If those checks were
bypassed, the failure mode is silent attention/Engram corruption, not a clean error. The
fork-local mechanisms this lane depends on (`common/pp_relay.py`, `ampere/ampere_sparse.py`,
Engram `gpu_resident_heads`) would all have to be reimplemented upstream-side first.

Concrete consequences for this proposal:

- **Decoder replay graphs** (`#58132`/`#59532`/`#63155`): real GB200 prefill gains, but the
  implementation assumes one PP rank owns layers 20-39. The lane splits that tail over
  three stages, so adopting it needs a new cross-stage replay-row and metadata protocol.
  Future architecture project, not a cherry-pick. (It is also why `#63155` is rejected.)
- **Upstream Engram peer/cumem work** (`#62091`, `#59923`): requires GPU tables, TP1, and
  Engram DP; the lane is TP4/DP1 with CPU-offloaded tables over two 10G-connected nodes.
  None of the activation conditions holds. Keep the fork's per-layer GPU-resident heads.
- **Wholesale `common/ops/**` import: no.** Those files are now tied to a newer DeepGEMM
  ABI, PCP tails, NVFP4 and newer FlashMLA workspace contracts; copying them risks cache
  geometry mismatches (584/528/288 B) with no measured sm86 upside.
- **`#59327` sorted host lookups: no**, thresholds are SM100-gated, and the refutation
  record's rule against transferring unmeasured micro-tuning across hardware applies.

### P6 (conditional, do not ship blind). `#59986` keep DSv4.1 sparse prefill out of FULL graphs

Two children disagreed about this one; the code settles it. Upstream `8977b13a1c` flips
`DeepseekV4SparseMLAMetadataBuilder` and `CompressorMetadataBuilder` from `ALWAYS` to
`UNIFORM_BATCH` and adds `get_varlen_cudagraph_max_query_len() = max_decode_query_len()`;
the fork still has `ALWAYS` at `compressor.py:87` and
`sparse_mla.py:221` (`DeepseekV4FlashMLAMetadataBuilder`).

It is **not live today**: `get_attn_cg_support()`
(`vllm/v1/worker/gpu/attn_utils.py:168-181`) takes the *weakest* support across all groups,
and this lane's Ampere builders return `UNIFORM_BATCH` while adaptive verification is off,
so prefill cannot enter a FULL graph. It becomes live exactly when adaptive verification
is enabled, because `ampere_sparse.py:43-67` then returns `ALWAYS` too and graph-unsafe
prefill metadata would be the only thing left blocking a mixed FULL capture.

So: **port it as a prerequisite for any adaptive-verification experiment, not now, and not
as the upstream commit** (the fork lacks `max_decode_query_len` and
`get_varlen_cudagraph_max_query_len` entirely; the equivalent already lives at the worker
level, since `c41e5a8de` "withhold[s] both uniform count and max_query_len for ineligible
prefills"). Make the builders honest, keep the worker withholding, and re-run the
cached-tail FULL graph regression suite before and after.

---

## Tier 1 - measure before adopting (conditional on traffic shape)

### M1. `#59804` keep tokens top-p must keep (open, unapproved)

Real sampling-distribution bug: the monolithic Triton pivot search uses strict
`probs > pivot` and falls back to the midpoint after 18 iterations, dropping a boundary
token needed to reach the requested mass (author's fuzz: 20 -> 0 standalone and
262 -> 0 combined violations over 4000 rows). Exposure here is **only** for requests
that set top-k and top-p together: with max-num-seqs 16, pure top-p uses the split path.
Fork already carries the two split-top-p tie fixes, so port the final algorithm, not an
old patch.

### M2. `#61213` masked-logit fast path in `_topk_topp_kernel`

One branch: if `num_finite_total <= k`, skip the cutoff search. Dormant without
structured/forced decoding; author measured 6.672 -> 0.863 ms (8 finite logits, top-k 50,
H100) and +13.7% throughput on a restrictive-regex DFlash workload. Worth taking only
if grammar-constrained traffic is real; the lane's own harness never sends
`response_format`.

### M3. `#61996` small-batch split top-p pipeline

Fork caps split count at `splits * batch_size >= SM count` and always launches the
split pipeline. This lane's c8 + k=4 shape is exactly 40 rows, inside the affected
range. Author measured 2.11x at B=38 and 1.44x at B=64 for a 129,280 vocab on B300, and
1.89x/1.14x on H200; a real DFlash c8 case saved 257 us/step. **The PR explicitly
never measured below 132 SMs and a 3090 has 82**, so benchmark on the lane before
merging tuning; the fork's #58231/#62046 tie fixes must be preserved.

### M4. `#63710` skip redundant DSpark context projections on long prefills (open)

Measured on **this exact model**: DeepSeek-V4.1-Flash-1017, 8x H20, TP8+EP, 64K in/128 out

- TPOT 99.19 -> 93.92 ms, TTFT 7847.73 -> 7278.88 ms, throughput 49.34 -> 51.17 tok/s
(+3.7%). One run per arm, low acceptance (1.26), so treat as directional. Under the
lane's 2048-token batch cap the removed rows are fewer, so expect less. Medium-high port
risk: the file is fork-modified (`deepseek_v4_1/nvidia/dspark.py` PP support, local
argmax hooks, graph-captured context KV).

---

## Tier 2 - design decisions, not ports

### D1. `#63219` per-request draft retention for structured output under PP

Silent-wrongness class: `DraftTokensHandler` keeps one global `req_ids` + `draft_tokens_np`
(`vllm/v1/worker/gpu/spec_decode/utils.py:13-42`), so a PP microbatch can overwrite batch
A's drafts before the scheduler generates grammar masks for A. Requires structured
output traffic; the lane's harness only exercises tool calling, never `response_format`,
so the bench cannot see it. Upstream fixed it with per-request retention plus changed
grammar/rejection semantics, author measured +204%/+444% throughput under PP2 with
constrained decoding (Qwen3-4B, k=8, H100). Port risk high: collides with the fork's
draft/confidence relay and adaptive verification. Decision needed: is grammar-constrained
traffic served? If yes, this is the highest-severity unported item on the list.

### D2. `#61261` `draft_context_window` (open, mergeable, unreviewed)

Bounds draft attention to a trailing window. Author measured mean accepted length
1.45 -> 2.29 at 128-256K and 1.26 -> 2.22 above 256K with a 32K window (Kimi-K3, B300),
+31%/+23% throughput. Directly targets the acceptance collapse above ~128K tracked in
issue #47930, which the lane can reach (max_model_len 524288). But: 32K is a Kimi number,
and the port must preserve the fork's PP relay, cached-tail FULL graphs and
per-layer draft cache groups. Evaluate as an opt-in flag, not a default, after the
Tier 0 batch.

### D3. `#64109` back only the tail of drafter sliding-window KV (open)

Capacity, not tok/s: author measured 1,256,723 -> 1,390,712 KV tokens (+10.7%) on
Qwen3.8-27B + 5-layer DSpark, GB300. Amplified here because the generic
`SlidingWindowSpec` reserves `max_in_flight_tokens = PP6 x 2048 = 12288` transient
tokens on top of each window. Touches seven core scheduler/KV files the fork has already
patched heavily (null-block guards, PP projection, NaN hardening). High risk; take it
only if KV pressure actually shows up with prefix caching on.

### D4. Prefill-only wins (no steady-decode effect)

- `#58539` shared prefill chunk plan (ROCm-authored, but the fork's Ampere path inherits
  the loop): author saw 2.6-6.9% throughput at c32-512 on MI355X. Port first: it makes
  chunk identity/workspace explicit.
- `#59131` gather compressed KV once per KV-source group in prefill: +1.2% throughput,
  TTFT 1150 -> 956 ms at c64 on 4x B200. Must preserve "no extra workspace span"
  because stage 0 has ~2 GiB headroom.

---

## Rejected, with reasons

**Already in HEAD (do not re-port):** `#59166`, `#64458` (primary race), `#61451`,
`#61161`, `#62046`, `#58231`, `#62467`/`#62936` (superseded by the fork's deterministic
exact refinement), `#61824` (local argmax sharding), `#56119` (fp8 Marlin `is_bmm`),
`#56720` (in substance, with a wider 1024-worker Ampere grid), `#61039`, `#63619`,
`#55184`. `#64458` wholesale is additionally unsafe: the fork's adaptive-confidence
broadcast framing would desync the four-operation FIFO.

**Fork work already in HEAD, often better than the fork's own version:** cached prompt-tail
FULL graphs (#58400 via luixiao `5d823cef5e`, landed as `c41e5a8de6` with extra PP
regression coverage), Engram WKV TP sharding (#58678, `b84498a565`), length-aware
candidate scoring (`2118a9f873` + `eb1dd750e8` + live-length bound `b0d3b95ba0`),
cross-stage V4.1 sharing sources (HEAD `041b7cf924` vs PP170 `3dee16caa7`), exact-size
Engram pinning (HEAD `142c980399` vs PP170 `2b652f14dd`), Marlin workspace/SM8x GEMV/fp8
scale/local-argmax consolidation (`1194bb39f9`, `e8138ee737`), per-group KV block zeroing
(TOMY `51783a09b4`).

**Measured loss on this hardware class (refutation record section 1):** inductor,
expert parallel, PCP, sequence parallel, custom all-reduce tuning, NCCL proto tuning,
every Marlin occupancy/smem/warps flag, GEMV rewrite, cuBLAS routing, MoE CTA-cap lift,
W4A8-INT8, dequant-to-bf16 grouped GEMM, tile-config forcing, atomic-add, `act_and_mul`
fold, fused `moe_align`, indexer top-k glue fusion, mHC pair fusion, fp8-direct KV in
sparse prefill, `num_warps` sweeps on sparse decode, drafter-side optimization,
rejection/resampling micro-kernels.

**sm90/sm100-only or wrong backend:** `#63101` (CUTLASS MXFP4; lane is Marlin W4A16),
`#60927`/`#61343` (native-fp8 quant paths), `#62011`/`#62240` (DeepGEMM/DeepSelect),
`#64580`, `#64609`, `#64442` (requires probabilistic drafting; lane uses local argmax),
`#63885` (ROCm monokernel), `#61938`, `#60942`, `#60156`, `#62102` (wrong attention
backend), `#63155` (needs upstream's decoder-replay classes; see gulf section),
`#63063` (measured +0.34% end-to-end and duplicates WKV weights per TP rank, bad trade
at ~2 GiB stage-0 headroom), `#63047`/`#62860` (no lane memory win; 160 MiB worst case
under a 512 MiB reservation), `#59986` (needs adaptive verification), `#60692`
(unchanged upstream step time, and the fork's builder disables draft metadata update).

---

## Lane hygiene found during the sweep (independent of ports)

1. **Dead env knob, documentation only in the verified deployment.**
   `VLLM_DSV41_PP_KV_RELAY=1` has no consumer in the fork source; the relay in
   `vllm/models/deepseek_v4_1/common/pp_relay.py` is automatic. Contrary to the initial
   sweep's pillar claim, neither the verified baseline nor candidate service sets it.
   The formula comment claiming V4.1 needs it is stale, not runtime configuration drift.
2. **Padding nit.** `prepare_dflash_inputs` takes `.max()` over the padded
   `num_scheduled_tokens` buffer; upstream slices `[:num_reqs]` (#58984).
3. **Known open deadlock, not from any port:** the fork's own 256k campaign records an
   engine deadlock at ctx ~245k when two prefills are admitted ~0.3 s apart during
   active decode ("No available shared memory broadcast block found"). This rollout
   does not test that operating point or claim to fix it. Separate this known baseline
   issue from any new regression; 64K checks do not certify 245K behavior.
4. **Second dead knob: `table_mode`.** my-salt documents `engram_config` as accepting
   `'{"table_mode":"mmap"}'`, but HEAD has no `table_mode` field at all
   (`vllm/config/engram.py:38-60`; no `table_mode` anywhere in the tree). The live
   `engram_config` does not set it. Both verified service units use `cpu_offload` and
   `gpu_resident_heads`, which the source accepts. The formula example is stale
   documentation, not an unsupported runtime setting in this rollout.

---

## Sequencing

1. **Batch A (one build, low risk):** P1 + P2 + P3 + P4 + P5 + F2. One line of kernel,
   two decorators, one indent, one constant, one PTX helper, one two-line memory cap.
   Expect correctness hardening on >32K contexts, ~1.5 GiB/rank back on stage 0, and
   removal of two classes of runtime-JIT stalls. No throughput claim.
2. **Batch A2 (separate commit, own gate):** F1 scheduler invariants plus the regression,
   kept apart from the memory cap so a failure is attributable.
3. **Gate A:** fork test suites for sparse indexer/persistent top-k (long rows, ties,
   k=2048), rejection sampler at c1/c8/c16, PP6 DSpark end-to-end on the 3090 harness,
   plus the lane's own cold/prefix A/B (`tools/dsv41-prefix-ab.py`) against
   `c41e5a8de` with the same 5-repeat protocol. Do not change capture sizes, k, or
   prefix policy between arms.
4. **Batch B (per-traffic decisions):** M1/M2 if grammar-constrained traffic exists,
   M3 behind a measurement on 82 SMs, M4 directly (the fork-tree question is now settled
   in favour of the fork, so nothing blocks it).
5. **Batch C (design):** D1 (if structured output is served), D2 as opt-in long-context
   arm, D3/D4 only on evidence of KV or prefill pressure, P6 before any
   adaptive-verification experiment.

Nothing above touches the two things that have already cost this lane real time: the
PP confidence relay framing and the cached-tail FULL graph path. Every fork-local
mechanism those depend on is preserved, and the only item that would rewrite them
(`#64458`) is explicitly excluded.
