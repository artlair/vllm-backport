# DSv4.1 sm86 Batch A/A2 rollout validation

## Decision and deployed state

Batch A and A2 are implemented and running on rome2 and rome as
`zot.lan/vllm-backport/dsv41:9842ce767-serve`.

- Image source: `9842ce767f8f73322d650000857973c609a0b7eb`.
- Both container image IDs:
  `cd282af1ef349ee2f1c8b94723e499bbcee6d9c179e446819db63fc3fe430caa`.
- Successful build: [41056530266](https://github.com/artlair/vllm-backport/actions/runs/41056530266).
- Salt: main at `181ba3c` after the operator-approved merge of
  [my-salt PR #812](https://github.com/artlair/my-salt/pull/812).
  The initial rollout used `ai/dsv41-batch-a-rollout` at `80aac5f`.
- Both initial scoped Salt applies succeeded with zero failed states. The net pillar
  diff is exactly the two image pins. After the merge, both hosts were converged from
  main without restarting either service or changing the running image.

The operator authorized interrupting the live lane overnight and deploying follow-up
fixes. The head and worker were stopped before native tests, then the worker was
applied before the head. Ray registered 24/24 GPUs. The first real completion returned
`42` with fingerprint `vllm-0.1.dev1+g9842ce767-tp4-pp6-aac207c5`.

This is an incremental rollout against the previously running `c41e5a8de` image.
It does not retroactively certify the older original-to-correctness release gates.
Operational and bounded correctness checks passed. Blanket performance certification
remains inconclusive because of noisy controls; no throughput improvement is claimed.

## Changes

- `4f94910ef2`: P1-P5, upstream #63475, #64137, #62748, the DFlash input-preparation
  hunk of #63819, and #63118 hardening. This adds the top-k radix CTA barrier, opaque
  PTX float-bit reads, admission-only sampler flushing, fixed draft tiles and
  non-specialized logits counts.
- `e46f79e50e`: F2 concurrency-capped profiling workspace. Real indexer chunk planning
  also uses the same compression-normalized bound as profiling.
- `7abb52a9a9`: separate A2 scheduler guard and draft-count bound, adapted to preserve
  diffusion's zero-target-token semantics. No PP relay or confidence framing changed.
- `9842ce767f`: qualify the shared key-conversion helper through `hist4096` inside
  `filtered_topk`. The initial build at `7abb52a9a9` failed at this call; it was never
  deployed. The corrected image compiled successfully.
- Test follow-ups cover K4 C1/C8/C16 verification, both statistics kernels' compile
  caches, a real synchronous PP6 wait/unblock flow, and nonempty DFlash compile caches.
  `49602100b7` adds only a test assertion after the image source, not serving code.

No response-format or grammar-specific ports were taken: the operator confirmed that
traffic is absent. Adaptive verification remains off. Other proposal tiers remain
unimplemented.

## Held constant and dependency caveat

Measured by this session on 24 stock RTX 3090 GPUs, TP4 x PP6 across rome2 and rome:
250 W, partition `5,9,6,7,8,5`, DSpark K4, 16 sequences, 2048 batched tokens,
524288 model length, prefix caching, FULL_AND_PIECEWISE graphs with captures
`[5,10,15,20,30,40,60,80]`, and the existing CPU-offloaded Engram placement.

Image-normalized service-unit hashes match baseline on each host. Boot IDs, kernel,
GPU driver and all power limits also match. The entrypoint and non-image serving
configuration were unchanged.

Torch `2.13.0+cu130`, Triton `3.7.1`, Transformers `5.20.1` and FlashInfer
`0.6.18.post1` are unchanged. The rebuild advanced five supporting dependencies:
`platformdirs` 4.13.6 to 4.14.0, `python-discovery` 1.6.5 to 1.6.7,
`smart-open` 8.0.4 to 9.0.1, `virtualenv` 21.15.0 to 21.15.1, and
`wrapt` 2.5.1 to 2.5.2. Timing is therefore an image-rollout comparison, not strict
isolation of a single source hunk.

## Native and serving correctness

Exact-image native validation ran on one drained RTX 3090 on rome (sm86, 82 SMs):

| Selection | Passed |
| --- | ---: |
| Top-k, DFlash, rejection sampling, graph profiling and Ampere metadata | 232 |
| Indexer capacity and chunk planning | 16 |
| V4.1 compressor | 29 |
| Real synchronous PP6 delayed-target regression | 1 |
| Total | 278 |

There were no skips in those selected cases. Top-k coverage includes long rows,
padding, ties, positive NaNs, infinities and K2048. Statistics tests check actual
acceptance and reuse of warmed Triton variants at C1/C8/C16. The first native attempt
passed 231 cases but lacked an offline Llama config for one metadata test; a local
config-only fixture resolved that environment failure. No weights were loaded for
that fixture, and serving configuration was not changed.

Implementation-side scheduler CPU suites passed 300 cases before the test-only
replacement of the manufactured state. The replacement's focused scheduler selection
passed 103 cases; its real-flow case also passed in the exact image above. Python
pre-commit checks passed. Whole-file clang-format was skipped after it rewrote unrelated
pre-existing header formatting; the actual sm86 compilation and native gate passed.
An earlier GPU-hidden local selection had five CUDA-unavailable failures; the
matching-image GPU run covered those cases successfully.

Both primary and confirmation serving arms passed:

- cold 4K/16K/64K: five clean measured requests per length, 256 output tokens;
- cached 8193/32769-token prompts at C1/C8: five clean measured waves per cell,
  exact SSE counts and residual-one cache proofs;
- cold and cached-tail `42` and `ORCHID` canaries;
- outbound `echo(value=7)` and inbound tool-result CHAT semantics.

No measured request errors, missing waves or detected contamination were recorded.
Final metrics show zero running/waiting requests, zero preemptions and zero inference
errors/aborts. Ray still reports 24/24 GPUs with no failed nodes; both service units
remain active, the head has no automatic restarts, and the final kernel check is
clean for new Xid, bus-loss and OOM events throughout this session.
Tool-call rendering has exact token-ID proof. Tool-result `/tokenize` retains its
sealed HTTP 400 diagnostic, so its rendering proof remains absent, not passed.

The existing GSM8K evaluator, first 32 questions, five shots, greedy seed 42,
512 maximum tokens and client concurrency limit 16 scored **23/32 to 24/32**;
invalid answers were **1 to 0**. This meets the declared smoke threshold, not a
statistically meaningful accuracy improvement. The evaluator retains aggregate scores,
not raw completions; invocation details were recorded from the session afterward.

## Capacity and performance

vLLM-reported KV capacity increased from **2,001,504 to 2,292,262 tokens**, or
**14.53%**. GPU blocks increased from **22,558 to 25,835**. This is capacity metadata,
not throughput, free-VRAM measurement or a change to the 524288 context limit.

The sealed fixture and frozen R2 script/plan hashes match across all arms. The
original-image prefix repeat was designated primary before candidate measurement;
both old controls are retained. The first candidate remains primary. One bounded
candidate confirmation repeated both prefix and cold workloads without changing the
plan, scripts, image or serving settings. It is diagnostic, not a replacement chosen
for speed.

Primary cold independent medians, old to new:

| Prompt | TTFT (s) | E2E (s) | Decode rate (tok/s) |
| --- | ---: | ---: | ---: |
| 4096 | 2.189 to 2.188 | 4.306 to 4.387 | 120.58 to 115.43 |
| 16384 | 4.790 to 4.798 | 7.070 to 7.089 | 112.17 to 111.27 |
| 65536 | 16.239 to 16.267 | 18.501 to 18.475 | 113.98 to 115.55 |

Paired five-trial ratios, not ratios of these independent medians, determine the
gate. No comparison reached the predeclared regression rule: slowdown beyond
`max(5%, 3 x relative MAD)`, at least four of five worse pairs, and the applicable
5 ms TTFT or 0.5 ms observed-ITL floor. Stable cold TTFT/E2E and cached C1 metrics
have no detected regression. The 4K cold decode rate is lower, but below the trigger.

**Do not claim a blanket throughput pass.** Original C8 controls remain noisy, and
initial cold 16K/64K decode rates have relative MAD 5.17%/6.46%. Confirmation found
no regression signal but cannot clear noisy original controls; 64K cold decode rate
remains inconclusive. C8 wave medians are correlated requests, not eight independent
replicates. Client-observed ITL is not GPU kernel latency.

A separate read-only metrics audit independently checked hashes, wave membership,
paired statistics, confirmations and limitations. It caught a conservative analysis
helper discrepancy: the frozen R2 5 ms TTFT floor applies to every TTFT, not only
sub-100 ms controls. Correcting the helper changed none of 82 metric verdicts or
regression signals. Original helper/reports are retained beside the corrected ones.

## Review decisions and limits

- Normal AsyncScheduler accounting maintains `D = tokens + placeholders - computed`
  at least one. A2 is defensive on this async lane, not a demonstrated live-race fix.
  Its real sync PP6 regression reaches early drafts through scheduler APIs, waits,
  then unblocks on target delivery. The count clamp is redundant under the wait guard
  and remains a defensive bound.
- The F2 single-request overflow concern is not reachable on this V2 DSpark lane:
  target bounds are at most `max_model_len - 1`, and the drafter clamps its bounds and
  has no sparse-indexer gather. Other model families using the shared helper are not
  certified by this rollout.
- Positive-NaN ordering was tested; negative-NaN semantics were not changed or claimed.
- Pi behavior/tests reviews completed. Claude authentication failed, and the review
  merge process returned no report; the individual Pi reports were triaged directly.
- These checks are bounded canaries and kernel oracles, not full model equivalence or
  long-term stability certification. Model traffic beyond 64K was not tested. The known
  baseline deadlock near 245K with staggered prefills remains outside this rollout.

## Evidence

Local artifact root: `/home/alec/ab-results/dsv41-batch-a-20261030/`.

It retains build provenance, immutable image IDs, Salt dry runs/applies, hardware and
unit fingerprints, package freezes, native output, frozen scripts/data, primary and
confirmation raw requests/waves, comparison JSONs, GSM summaries, review/audit reports
and kernel-health snapshots. `PLAN.md` records the prospective gate and confirmation
choices. The original sealed fixture under `dsv41-ports-20261027/` was never resealed.

On 2026-10-31 the operator approved merging PR #812 as `181ba3c`. Gitfs and pillar
were refreshed, both main dry runs showed no unit change, and scoped main applies
completed with zero failures. Only the unconditional service-enable command reported
a change. Both hosts retained their service PIDs/start times and `9842ce767-serve`
image; the registry manifest was unchanged and `/health` succeeded. Artifacts are
under `main-convergence/` in the same root. Neither host remains on the trial branch.
