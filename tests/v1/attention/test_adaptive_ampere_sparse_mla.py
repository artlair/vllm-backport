# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Adaptive verification on the DeepSeek-V4.1 Ampere sparse attention path.

Adaptive verification trims verification drafts on the device, so a captured
FULL decode graph can be replayed with fewer live rows, a different per-request
split, a different request order and zero-draft requests. That is only safe if
the attention metadata comes from the device query start locations and every
persistent buffer is either rewritten for the padded tail or provably unread.

These tests pin the two claims the Ampere builders now make:

* the SM8x sparse-MLA, compressed-slot, SWA and indexer decode metadata is
  identical whether or not the CPU copy of the query lengths is stale, and the
  indexer takes the flattened per-token route for any ``next_n``;
* a captured sparse decode replayed at a padded bucket with a changed live split
  matches an independent per-token reference, and padded rows stay inert.
"""

import math
from types import SimpleNamespace

import pytest
import torch

from tests.kernels.attention.test_rocm_triton_attn_dsv4 import (
    HEAD_DIM,
    NOPE_HEAD_DIM,
    _read_fp8_ds_mla_cache_rows,
)
from tests.v1.attention.utils import create_vllm_config
from vllm.models.deepseek_v4_1.amd.rocm import (
    DeepseekV4ROCMAiterMLASparseBackend,
    DeepseekV4ROCMAiterSparseSWAMetadataBuilder,
    DeepseekV41ROCMAiterMLAAttention,
    DeepseekV41ROCMAiterSparseSWABackend,
    compute_global_topk_ragged_indices_and_indptr,
)
from vllm.models.deepseek_v4_1.ampere.ampere_sparse import (
    DeepseekV41AmpereMLAAttention,
    DeepseekV41AmpereMLASparseBackend,
    DeepseekV41AmpereMLASparseMetadataBuilder,
    DeepseekV41AmpereSparseSWABackend,
    DeepseekV41AmpereSparseSWAMetadataBuilder,
)
from vllm.models.deepseek_v4_1.common.ops.cache_utils import (
    quantize_and_insert_k_cache,
)
from vllm.models.deepseek_v4_1.sparse_mla import DeepseekV4SparseMLAMetadataBuilder
from vllm.platforms import current_platform
from vllm.platforms.interface import DeviceCapability
from vllm.v1.attention.backend import AttentionCGSupport, CommonAttentionMetadata
from vllm.v1.attention.backends.mla.indexer import (
    DeepseekV32IndexerMetadataBuilder,
    DeepseekV41IndexerBackend,
)
from vllm.v1.attention.ops.rocm_aiter_mla_sparse import rocm_sparse_attn_decode
from vllm.v1.kv_cache_interface import (
    MLAAttentionSpec,
    SlidingWindowMLASpec,
    get_kv_quant_mode,
)
from vllm.v1.worker.block_table import get_block_table_width
from vllm.v1.worker.gpu.attn_utils import (
    get_attn_cg_support,
    get_query_lens_mismatch_unsupported_backend,
)
from vllm.v1.worker.gpu.spec_decode.adaptive_verification import (
    maybe_create_adaptive_verification_manager,
)
from vllm.v1.worker.utils import AttentionGroup

NUM_SPECULATIVE_TOKENS = 4
DECODE_QUERY_LEN = 1 + NUM_SPECULATIVE_TOKENS
WINDOW = 128
SWA_BLOCK_SIZE = 32
COMPRESSED_BLOCK_SIZE = 128
ROPE_HEAD_DIM = HEAD_DIM - NOPE_HEAD_DIM
HEAD_SIZE = HEAD_DIM
NUM_HEADS = 16
TOPK = 32
# Positions of the synthetic indexer top-k entries, one per row of the buffer.
TOPK_POSITIONS = list(range(0, 8 * TOPK, 8))
# Every request owns this many slots in each cache, so a slot id encodes
# (request, position) and the reference can be written by hand.
SLOTS_PER_REQUEST = 256
MAX_SEQ_LEN = SLOTS_PER_REQUEST

GPU = pytest.mark.skipif(
    not current_platform.is_cuda(), reason="Ampere sparse path needs CUDA"
)


def _spec_config(adaptive: bool, num_speculative_tokens: int = NUM_SPECULATIVE_TOKENS):
    return SimpleNamespace(
        method="dspark",
        num_speculative_tokens=num_speculative_tokens,
        parallel_drafting=True,
        enable_adaptive_verification=adaptive,
        use_dspark=lambda: True,
    )


def _lane_config(*, adaptive: bool, max_num_seqs: int = 16):
    config = create_vllm_config(
        model_name="deepseek-ai/DeepSeek-V4.1-Flash-1017",
        max_model_len=4096,
        block_size=COMPRESSED_BLOCK_SIZE,
        max_num_seqs=max_num_seqs,
        max_num_batched_tokens=256,
    )
    config.speculative_config = _spec_config(adaptive)
    return config


def _compressed_spec(compress_ratio: int) -> MLAAttentionSpec:
    return MLAAttentionSpec(
        block_size=COMPRESSED_BLOCK_SIZE,
        num_kv_heads=1,
        head_size=HEAD_SIZE,
        dtype=torch.uint8,
        tokens_per_state=compress_ratio,
    )


def _swa_spec() -> SlidingWindowMLASpec:
    return SlidingWindowMLASpec(
        block_size=SWA_BLOCK_SIZE,
        num_kv_heads=1,
        head_size=HEAD_SIZE,
        dtype=torch.uint8,
        sliding_window=WINDOW,
        cache_dtype_str="fp8_ds_mla",
        state_content_bytes=584,
        alignment=576,
        model_version="deepseek_v4",
        kv_quant_mode=get_kv_quant_mode("fp8_ds_mla"),
    )


def _request_blocks(num_reqs: int, block_size: int) -> torch.Tensor:
    """Block table giving request ``r`` the slot range [r*SLOTS_PER_REQUEST, ...)."""
    per_req = -(-SLOTS_PER_REQUEST // block_size)
    table = torch.empty((num_reqs, per_req), dtype=torch.int32)
    for r in range(num_reqs):
        base = r * per_req
        table[r] = torch.arange(base, base + per_req, dtype=torch.int32)
    return table


def _slot(request: int, position: int) -> int:
    return request * SLOTS_PER_REQUEST + position


def _common_metadata(
    query_lens: list[int],
    seq_lens: list[int],
    cpu_query_lens: list[int],
    block_table: torch.Tensor,
    slot_mapping: torch.Tensor,
    causal: bool = True,
    max_query_len: int | None = None,
) -> CommonAttentionMetadata:
    device = block_table.device
    query_start_loc = torch.zeros(len(query_lens) + 1, dtype=torch.int32, device=device)
    query_start_loc[1:] = torch.tensor(
        query_lens, dtype=torch.int32, device=device
    ).cumsum(0)
    cpu_query_start_loc = torch.zeros(len(cpu_query_lens) + 1, dtype=torch.int32)
    cpu_query_start_loc[1:] = torch.tensor(cpu_query_lens, dtype=torch.int32).cumsum(0)
    seq_lens_t = torch.tensor(seq_lens, dtype=torch.int32, device=device)
    return CommonAttentionMetadata(
        query_start_loc=query_start_loc,
        query_start_loc_cpu=cpu_query_start_loc,
        seq_lens=seq_lens_t,
        seq_lens_cpu_upper_bound=seq_lens_t.cpu(),
        _seq_lens_cpu=seq_lens_t.cpu(),
        num_reqs=len(query_lens),
        num_actual_tokens=sum(query_lens),
        max_query_len=max_query_len or DECODE_QUERY_LEN,
        max_seq_len=int(seq_lens_t.max()),
        block_table_tensor=block_table,
        slot_mapping=slot_mapping,
        causal=causal,
    )


@pytest.mark.parametrize(
    "adaptive,expected",
    [(True, AttentionCGSupport.ALWAYS), (False, AttentionCGSupport.UNIFORM_BATCH)],
)
def test_ampere_builders_report_always_only_for_adaptive_runs(adaptive, expected):
    """Varlen decode is only proven for the device-trimmed batch.

    With adaptive verification off a trimmed batch cannot occur, so the Ampere
    builders keep the shared UNIFORM_BATCH contract instead of newly accepting
    varlen replay.
    """
    config = SimpleNamespace(speculative_config=_spec_config(adaptive))
    for builder_cls in (
        DeepseekV41AmpereMLASparseMetadataBuilder,
        DeepseekV41AmpereSparseSWAMetadataBuilder,
    ):
        assert builder_cls.get_cudagraph_support(config, None) is expected

    # Every other user of the shared builders keeps its contract.
    for builder_cls in (
        DeepseekV4SparseMLAMetadataBuilder,
        DeepseekV4ROCMAiterSparseSWAMetadataBuilder,
    ):
        assert (
            builder_cls.get_cudagraph_support(config, None)
            is AttentionCGSupport.UNIFORM_BATCH
        )


def test_ampere_attention_binds_ampere_scoped_builders():
    """The SWA cache reads ``swa_backend_cls``, so the layer must rebind it."""
    assert DeepseekV41AmpereMLAAttention.backend_cls is (
        DeepseekV41AmpereMLASparseBackend
    )
    assert DeepseekV41AmpereMLAAttention.swa_backend_cls is (
        DeepseekV41AmpereSparseSWABackend
    )
    assert DeepseekV41AmpereMLASparseBackend.get_builder_cls() is (
        DeepseekV41AmpereMLASparseMetadataBuilder
    )
    assert DeepseekV41AmpereSparseSWABackend.get_builder_cls() is (
        DeepseekV41AmpereSparseSWAMetadataBuilder
    )

    # The ROCm layer and the shared backends stay on the shared builders.
    assert DeepseekV41ROCMAiterMLAAttention.backend_cls is (
        DeepseekV4ROCMAiterMLASparseBackend
    )
    assert DeepseekV41ROCMAiterMLAAttention.swa_backend_cls is (
        DeepseekV41ROCMAiterSparseSWABackend
    )
    assert DeepseekV4ROCMAiterMLASparseBackend.get_builder_cls() is (
        DeepseekV4SparseMLAMetadataBuilder
    )
    assert DeepseekV41ROCMAiterSparseSWABackend.get_builder_cls() is (
        DeepseekV4ROCMAiterSparseSWAMetadataBuilder
    )

    assert DeepseekV41AmpereMLASparseBackend.supports_compute_capability(
        DeviceCapability(8, 6)
    )
    assert not DeepseekV41AmpereMLASparseBackend.supports_compute_capability(
        DeviceCapability(9, 0)
    )


@GPU
def test_adaptive_gate_accepts_the_ampere_group_shape():
    """The boot gate must pass for the Ampere target groups and fail on ROCm ones.

    ``maybe_create_adaptive_verification_manager`` rejects a backend whose decode
    plan reads the CPU query lengths and any target builder that is not ALWAYS.
    This is the exact set of groups a DSv4.1 Ampere target presents.
    """
    device = torch.device("cuda")
    config = _lane_config(adaptive=True)
    max_num_seqs = config.scheduler_config.max_num_seqs
    max_model_len = config.model_config.max_model_len
    swa_spec = _swa_spec()
    specs = [_compressed_spec(1), _compressed_spec(2)]

    def groups(mla_backend, swa_backend):
        result = []
        for ratio, spec in zip((1, 2), specs):
            group = AttentionGroup(mla_backend, [f"compressed.{ratio}"], spec, 0)
            group.metadata_builders = [
                mla_backend.get_builder_cls()(spec, group.layer_names, config, device)
            ]
            result.append([group])
        swa_group = AttentionGroup(swa_backend, ["swa"], swa_spec, 0)
        swa_group.metadata_builders = [
            swa_backend.get_builder_cls()(swa_spec, ["swa"], config, device)
        ]
        result.append([swa_group])
        indexer_spec = specs[0]
        width = get_block_table_width(
            indexer_spec.max_num_blocks_per_req(config, max_model_len),
            indexer_spec.block_size,
        )
        indexer_group = AttentionGroup(
            DeepseekV41IndexerBackend, ["indexer"], indexer_spec, 0
        )
        indexer_group.metadata_builders = [
            DeepseekV32IndexerMetadataBuilder(
                indexer_spec,
                ["indexer"],
                config,
                device,
                block_table_width=width,
            )
        ]
        result.append([indexer_group])
        return result

    ampere = groups(
        DeepseekV41AmpereMLASparseBackend, DeepseekV41AmpereSparseSWABackend
    )
    assert get_query_lens_mismatch_unsupported_backend(ampere) is None
    assert (
        get_attn_cg_support(ampere, config).min_cg_support == AttentionCGSupport.ALWAYS
    )

    req_states = SimpleNamespace(
        num_speculative_steps=NUM_SPECULATIVE_TOKENS,
        device=device,
        max_num_reqs=max_num_seqs,
        max_num_batched_tokens=config.scheduler_config.max_num_batched_tokens,
    )
    manager = maybe_create_adaptive_verification_manager(
        enable_adaptive_verification=True,
        attn_groups=ampere,
        attn_cg_support=get_attn_cg_support(ampere, config),
        req_states=req_states,
        query_start_loc=torch.zeros(max_num_seqs + 1, dtype=torch.int32, device=device),
        num_bonus_tokens=1,
        max_total_logits=1024,
        vllm_config=config,
    )
    assert manager is not None

    # The shared ROCm backends are untouched, so the same set still fails on the
    # builder that never declared varlen decode.
    rocm = groups(
        DeepseekV4ROCMAiterMLASparseBackend, DeepseekV41ROCMAiterSparseSWABackend
    )
    with pytest.raises(ValueError, match="UNIFORM_BATCH"):
        maybe_create_adaptive_verification_manager(
            enable_adaptive_verification=True,
            attn_groups=rocm,
            attn_cg_support=get_attn_cg_support(rocm, config),
            req_states=req_states,
            query_start_loc=torch.zeros(
                max_num_seqs + 1, dtype=torch.int32, device=device
            ),
            num_bonus_tokens=1,
            max_total_logits=1024,
            vllm_config=config,
        )


@GPU
@pytest.mark.parametrize("causal", [True, False])
@pytest.mark.parametrize("compress_ratio", [1, 2])
def test_sparse_metadata_is_independent_of_the_cpu_query_lens(compress_ratio, causal):
    """Sparse-MLA, compressed-slot and SWA metadata follow the device split.

    Under adaptive verification the CPU query lengths are only exact in total:
    the draft budget is redistributed evenly on the CPU while the device admits
    it best-first. Every per-row decode bound has to come from the device query
    start locations. Causal covers the target layers; non-causal covers the
    DSpark draft block, which shares the same SWA builder.
    """
    device = torch.device("cuda")
    # 10 live tokens in a 16-row bucket. The CPU copy keeps the exact total but
    # a different per-request split, which is all adaptive verification leaves.
    query_lens = [3, 2, 4, 1]
    cpu_query_lens = [4, 1, 3, 2]
    seq_lens = [131, 96, 200, 129]
    live = sum(query_lens)
    num_reqs = len(query_lens)
    assert sum(cpu_query_lens) == live

    config = _lane_config(adaptive=True)
    swa_block_table = _request_blocks(num_reqs, SWA_BLOCK_SIZE).to(device)
    compressed_block_table = _request_blocks(num_reqs, COMPRESSED_BLOCK_SIZE).to(device)
    slot_mapping = torch.cat(
        [
            torch.arange(live, dtype=torch.int64, device=device),
            torch.full((6,), -1, dtype=torch.int64, device=device),
        ]
    )

    def build(cpu_lens: list[int]):
        swa_builder = DeepseekV41AmpereSparseSWAMetadataBuilder(
            _swa_spec(), ["swa"], config, device
        )
        mla_builder = DeepseekV41AmpereMLASparseMetadataBuilder(
            _compressed_spec(compress_ratio), ["compressed"], config, device
        )
        mla_meta = mla_builder.build(
            0,
            _common_metadata(
                query_lens,
                seq_lens,
                cpu_lens,
                compressed_block_table,
                slot_mapping,
            ),
        )
        return (
            swa_builder.build(
                0,
                _common_metadata(
                    query_lens,
                    seq_lens,
                    cpu_lens,
                    swa_block_table,
                    slot_mapping,
                    causal,
                ),
            ),
            mla_meta,
            mla_builder,
        )

    stale_swa, stale_mla, mla_builder = build(cpu_query_lens)
    truth_swa, truth_mla, _ = build(query_lens)

    expected_req = torch.repeat_interleave(
        torch.arange(num_reqs, device=device), torch.tensor(query_lens, device=device)
    )
    assert stale_swa.num_decodes == num_reqs
    assert stale_swa.num_decode_tokens == live
    torch.testing.assert_close(
        stale_swa.token_to_req_indices[:live], expected_req.int(), rtol=0, atol=0
    )
    torch.testing.assert_close(
        stale_swa.is_valid_token[:live],
        torch.ones(live, dtype=torch.bool, device=device),
    )
    assert not stale_swa.is_valid_token[live:].any()

    for field in (
        "decode_swa_lens",
        "decode_swa_indices",
        "decode_swa_ragged_indices",
        "decode_swa_ragged_indptr",
    ):
        stale = getattr(stale_swa, field)
        truth = getattr(truth_swa, field)
        assert stale is not None and truth is not None
        torch.testing.assert_close(stale, truth, rtol=0, atol=0)

    torch.testing.assert_close(
        stale_mla.slot_mapping, truth_mla.slot_mapping, rtol=0, atol=0
    )
    torch.testing.assert_close(
        stale_mla.req_id_per_token[:live], expected_req.int(), rtol=0, atol=0
    )

    # Every row's window length follows its own device position, and the
    # graph-padding tail is cleared so a replayed bucket cannot read a previous
    # step's bounds. The DSpark draft block anchors one window per request and
    # widens it by the whole query block.
    if causal:
        expected_lens = [
            min(seq_len - qlen + j + 1, WINDOW)
            for seq_len, qlen in zip(seq_lens, query_lens)
            for j in range(qlen)
        ]
    else:
        expected_lens = [
            min(seq_len - qlen, WINDOW) + qlen
            for seq_len, qlen in zip(seq_lens, query_lens)
            for _ in range(qlen)
        ]
    assert stale_swa.decode_swa_lens[:live].tolist() == expected_lens
    assert not stale_swa.decode_swa_lens[live:].any()
    ragged_indptr = stale_swa.decode_swa_ragged_indptr
    assert ragged_indptr is not None
    ragged_lens = ragged_indptr[1 : live + 1] - ragged_indptr[:live]
    assert ragged_lens.tolist() == expected_lens

    if compress_ratio > 1:
        compressed = stale_mla.slot_mapping
        assert compressed.numel() == live
        # One stored state per completed ratio group, and each lands in the
        # request's own slot range.
        expected_valid = sum(
            seq_len // compress_ratio - (seq_len - qlen) // compress_ratio
            for seq_len, qlen in zip(seq_lens, query_lens)
        )
        stored = compressed[compressed >= 0]
        assert stored.numel() == expected_valid
        stored_rows = torch.nonzero(compressed >= 0).flatten()
        owner = expected_req[stored_rows.long()]
        assert torch.equal(stored // (SLOTS_PER_REQUEST // compress_ratio), owner)
        # The persistent graph buffer is refilled with -1, so rows outside this
        # step's token range cannot keep a stale slot.
        assert not mla_builder.compressed_slot_mapping_buffer[live:].ge(0).any()


@GPU
def test_mixed_batch_keeps_the_decode_prefill_boundary():
    """Only the CPU decode split is stale; the boundary and prefill rows are exact.

    The CPU copy redistributes the draft budget across decode requests but keeps
    the batch total, and prefill rows are scheduled exactly. So the number of
    decode tokens (a cumsum at the decode/prefill boundary) and the prefill query
    lengths still describe the device batch, which is what lets the eager CPU
    slicing around a captured decode region stay correct.
    """
    device = torch.device("cuda")
    # Two decode requests plus a chunked prefill long enough to stay clear of the
    # 6..9 band where the SWA builder (threshold 9) and the indexer (threshold 5)
    # disagree about what is a decode.
    query_lens = [3, 1, 12]
    cpu_query_lens = [1, 3, 12]
    seq_lens = [131, 96, 200]
    live = sum(query_lens)
    num_decodes = 2
    assert sum(cpu_query_lens) == live

    config = _lane_config(adaptive=True)
    swa_block_table = _request_blocks(len(query_lens), SWA_BLOCK_SIZE).to(device)
    width = get_block_table_width(
        _compressed_spec(1).max_num_blocks_per_req(
            config, config.model_config.max_model_len
        ),
        COMPRESSED_BLOCK_SIZE,
    )
    compressed_block_table = _request_blocks(len(query_lens), COMPRESSED_BLOCK_SIZE).to(
        device
    )
    compressed_block_table = torch.cat(
        [
            compressed_block_table,
            torch.zeros(
                (len(query_lens), width - compressed_block_table.shape[1]),
                dtype=torch.int32,
                device=device,
            ),
        ],
        dim=1,
    )
    slot_mapping = torch.arange(live, dtype=torch.int64, device=device)
    max_query_len = max(query_lens)

    swa_meta = DeepseekV41AmpereSparseSWAMetadataBuilder(
        _swa_spec(), ["swa"], config, device
    ).build(
        0,
        _common_metadata(
            query_lens,
            seq_lens,
            cpu_query_lens,
            swa_block_table,
            slot_mapping,
            max_query_len=max_query_len,
        ),
    )
    indexer_builder = DeepseekV32IndexerMetadataBuilder(
        _compressed_spec(1), ["indexer"], config, device, block_table_width=width
    )
    indexer_meta = indexer_builder.build(
        0,
        _common_metadata(
            query_lens,
            seq_lens,
            cpu_query_lens,
            compressed_block_table,
            slot_mapping,
            max_query_len=max_query_len,
        ),
    )

    decode_tokens = sum(query_lens[:num_decodes])
    assert swa_meta.num_decodes == num_decodes
    assert swa_meta.num_decode_tokens == decode_tokens
    assert swa_meta.num_prefills == 1
    assert swa_meta.num_prefill_tokens == live - decode_tokens
    assert swa_meta.prefill_query_lens_cpu.tolist() == [12]
    assert swa_meta.prefill_seq_lens.tolist() == [200]

    assert indexer_meta.num_decodes == num_decodes
    assert indexer_meta.num_decode_tokens == decode_tokens
    assert indexer_meta.num_prefills == 1
    assert indexer_meta.prefill is not None
    (chunk,) = indexer_meta.prefill.chunks
    # The prefill region starts at the exact decode total, not at the stale CPU
    # per-request cumsum.
    assert chunk.token_start == decode_tokens
    assert chunk.token_end == live
    assert chunk.total_seq_lens == seq_lens[-1]


@GPU
@pytest.mark.parametrize("num_speculative_tokens", [1, NUM_SPECULATIVE_TOKENS])
def test_indexer_decode_metadata_uses_device_rows_for_any_next_n(
    num_speculative_tokens,
):
    """The indexer must flatten decode from device lengths even at k=1.

    At ``next_n=2`` SM8x would otherwise take the native per-request branch,
    whose ``requires_padding`` flag is decided from the CPU query lengths when
    the graph is captured and then frozen. With adaptive verification the device
    rows are the truth, so the flattened per-token route has to be taken for
    every ``next_n``.
    """
    device = torch.device("cuda")
    if num_speculative_tokens == 1:
        # decode_query_len == 2, so no row may be longer than two tokens.
        query_lens = [2, 1, 2, 1]
        cpu_query_lens = [1, 2, 2, 1]
    else:
        query_lens = [3, 1, 4, 2]
        cpu_query_lens = [2, 4, 2, 2]
    seq_lens = [131, 96, 200, 129]
    live = sum(query_lens)
    config = _lane_config(adaptive=True)
    config.speculative_config.num_speculative_tokens = num_speculative_tokens
    spec = _compressed_spec(2)
    width = get_block_table_width(
        spec.max_num_blocks_per_req(config, config.model_config.max_model_len),
        spec.block_size,
    )
    block_table = _request_blocks(len(query_lens), COMPRESSED_BLOCK_SIZE).to(device)
    if block_table.shape[1] < width:
        # Production block tables are max-model-length wide; the flattened
        # kernel asserts the builder's row width.
        block_table = torch.cat(
            [
                block_table,
                torch.zeros(
                    (block_table.shape[0], width - block_table.shape[1]),
                    dtype=torch.int32,
                    device=device,
                ),
            ],
            dim=1,
        )
    slot_mapping = torch.arange(live, dtype=torch.int64, device=device)
    builder = DeepseekV32IndexerMetadataBuilder(
        spec, ["indexer"], config, device, block_table_width=width
    )
    assert builder.use_flattening, "adaptive SM8x decode must flatten"
    metadata = builder.build(
        0,
        _common_metadata(
            query_lens, seq_lens, cpu_query_lens, block_table, slot_mapping
        ),
    )

    decode = metadata.decode
    assert decode is not None
    assert decode.requires_padding is False
    assert int(decode.seq_lens.shape[0]) == live
    assert decode.seq_lens.shape[-1] == 1
    assert decode.decode_lens[:live].tolist() == [1] * live

    expected_req = torch.repeat_interleave(
        torch.arange(len(query_lens), device=device),
        torch.tensor(query_lens, device=device),
    )
    expected_seq_lens = [
        (seq_len - qlen + j + 1) // 2  # compressed indexer cache
        for seq_len, qlen in zip(seq_lens, query_lens)
        for j in range(qlen)
    ]
    assert decode.seq_lens[:live, 0].tolist() == expected_seq_lens
    torch.testing.assert_close(
        decode.block_table[:live], block_table[expected_req.long()], rtol=0, atol=0
    )
    # One row per query token means no batch-index indirection is emitted; the
    # per-token request map already is the row map.
    assert decode.indices is None

    # The graph-padding tail of the seq_lens buffer is cleared, so a replayed
    # bucket cannot read a previous step's context length.
    assert not builder.decode_seq_lens_buffer[live:].any()


class _DecodeRig:
    """Persistent decode buffers shared across capture and replay.

    Mirrors the runner: the metadata builders write into their own persistent
    buffers and the graph reads those buffers, so replay sees whatever the last
    eager build wrote.
    """

    def __init__(self, config, *, max_num_reqs: int, bucket: int):
        self.device = torch.device("cuda")
        self.bucket = bucket
        self.max_num_reqs = max_num_reqs
        self.scale = 1.0 / math.sqrt(HEAD_SIZE)
        self.qsl = torch.zeros(max_num_reqs + 1, dtype=torch.int32, device=self.device)
        self.qsl_cpu = torch.zeros(max_num_reqs + 1, dtype=torch.int32)
        self.seq_lens = torch.zeros(max_num_reqs, dtype=torch.int32, device=self.device)
        self.slot_mapping = torch.full(
            (bucket,), -1, dtype=torch.int64, device=self.device
        )
        self.swa_blocks = _request_blocks(max_num_reqs, SWA_BLOCK_SIZE).to(self.device)
        self.compressed_blocks = _request_blocks(
            max_num_reqs, COMPRESSED_BLOCK_SIZE
        ).to(self.device)
        self.swa_block_table = torch.zeros_like(self.swa_blocks)
        self.compressed_block_table = torch.zeros_like(self.compressed_blocks)
        # Graph padding rows keep the previous step's top-k entries and hidden
        # states; only is_valid_token protects them.
        self.topk = torch.zeros((bucket, TOPK), dtype=torch.int32, device=self.device)
        self.q = torch.zeros(
            (bucket, NUM_HEADS, HEAD_SIZE), dtype=torch.bfloat16, device=self.device
        )
        self.out = torch.zeros_like(self.q)
        self.swa_builder = DeepseekV41AmpereSparseSWAMetadataBuilder(
            _swa_spec(), ["swa"], config, self.device
        )
        self.mla_builder = DeepseekV41AmpereMLASparseMetadataBuilder(
            _compressed_spec(1), ["compressed"], config, self.device
        )
        self.swa_cache = self._pack_cache(SWA_BLOCK_SIZE, seed=1)
        self.compressed_cache = self._pack_cache(COMPRESSED_BLOCK_SIZE, seed=2)

    def _pack_cache(self, block_size: int, seed: int) -> torch.Tensor:
        torch.manual_seed(seed)
        kv = (
            torch.randn(
                self.max_num_reqs * SLOTS_PER_REQUEST,
                HEAD_SIZE,
                dtype=torch.bfloat16,
                device=self.device,
            )
            * 0.125
        )
        num_blocks = self.max_num_reqs * SLOTS_PER_REQUEST // block_size
        cache = torch.zeros(
            (num_blocks, block_size, 584), dtype=torch.uint8, device=self.device
        )
        slots = torch.arange(
            self.max_num_reqs * SLOTS_PER_REQUEST,
            dtype=torch.int64,
            device=self.device,
        )
        quantize_and_insert_k_cache(
            kv, cache, slots, block_size=block_size, use_fnuz=False
        )
        return cache

    def load(self, batch: list[tuple[int, int, int]]) -> CommonAttentionMetadata:
        """Write one step's batch into the persistent buffers and build metadata."""
        num_reqs = len(batch)
        live = sum(qlen for _, qlen, _ in batch)
        assert live <= self.bucket, batch
        self.qsl[0] = 0
        self.qsl_cpu[0] = 0
        self.slot_mapping.fill_(-1)
        for row, (request, qlen, seq_len) in enumerate(batch):
            assert qlen <= DECODE_QUERY_LEN and seq_len <= MAX_SEQ_LEN
            self.qsl[row + 1] = self.qsl[row] + qlen
            self.qsl_cpu[row + 1] = self.qsl_cpu[row] + qlen
            self.seq_lens[row] = seq_len
            self.swa_block_table[row] = self.swa_blocks[request]
            self.compressed_block_table[row] = self.compressed_blocks[request]
        self.qsl[num_reqs + 1 :] = live
        self.qsl_cpu[num_reqs + 1 :] = live
        self.slot_mapping[:live] = torch.arange(live, device=self.device)
        torch.manual_seed(1234 + live + num_reqs)
        self.q[:live] = torch.randn(
            (live, NUM_HEADS, HEAD_SIZE), dtype=torch.bfloat16, device=self.device
        )
        topk_row = torch.tensor(TOPK_POSITIONS, dtype=torch.int32, device=self.device)
        for row in range(live):
            self.topk[row] = topk_row
        seq_lens_cpu = torch.tensor(
            [seq_len for _, _, seq_len in batch], dtype=torch.int32
        )
        return CommonAttentionMetadata(
            query_start_loc=self.qsl,
            query_start_loc_cpu=self.qsl_cpu,
            seq_lens=self.seq_lens,
            seq_lens_cpu_upper_bound=seq_lens_cpu,
            _seq_lens_cpu=seq_lens_cpu,
            num_reqs=num_reqs,
            num_actual_tokens=live,
            max_query_len=DECODE_QUERY_LEN,
            max_seq_len=int(seq_lens_cpu.max()),
            block_table_tensor=self.swa_block_table,
            slot_mapping=self.slot_mapping,
            causal=True,
        )

    def build(self, batch):
        common = self.load(batch)
        swa_meta = self.swa_builder.build(0, common)
        compressed_common = CommonAttentionMetadata(
            query_start_loc=common.query_start_loc,
            query_start_loc_cpu=common.query_start_loc_cpu,
            seq_lens=common.seq_lens,
            seq_lens_cpu_upper_bound=common.seq_lens_cpu_upper_bound,
            num_reqs=common.num_reqs,
            num_actual_tokens=common.num_actual_tokens,
            max_query_len=common.max_query_len,
            max_seq_len=common.max_seq_len,
            block_table_tensor=self.compressed_block_table,
            slot_mapping=self.slot_mapping,
            causal=True,
        )
        mla_meta = self.mla_builder.build(0, compressed_common)
        return swa_meta, mla_meta

    def run_decode(self, metadata) -> None:
        swa_meta, _ = metadata
        assert swa_meta.is_valid_token is not None
        topk_ragged, topk_indptr, topk_lens = (
            compute_global_topk_ragged_indices_and_indptr(
                self.topk,
                swa_meta.token_to_req_indices[: self.bucket],
                self.compressed_block_table[: self.bucket],
                COMPRESSED_BLOCK_SIZE,
                swa_meta.is_valid_token[: self.bucket],
            )
        )
        rocm_sparse_attn_decode(
            q=self.q,
            kv_cache=self.compressed_cache,
            swa_k_cache=self.swa_cache,
            swa_only=False,
            topk_indices=None,
            topk_lens=topk_lens,
            swa_indices=swa_meta.decode_swa_indices,
            swa_lens=swa_meta.decode_swa_lens,
            swa_ragged_indices=swa_meta.decode_swa_ragged_indices,
            swa_ragged_indptr=swa_meta.decode_swa_ragged_indptr,
            topk_ragged_indices=topk_ragged,
            topk_ragged_indptr=topk_indptr,
            attn_sink=None,
            scale=self.scale,
            head_dim=HEAD_SIZE,
            nope_head_dim=NOPE_HEAD_DIM,
            rope_head_dim=ROPE_HEAD_DIM,
            output=self.out,
        )

    def reference(self, batch: list[tuple[int, int, int]]) -> torch.Tensor:
        """Per-token attention over that token's own window plus top-k slots."""
        live = sum(qlen for _, qlen, _ in batch)
        q = self.q[:live]
        rows = []
        for request, qlen, seq_len in batch:
            prefix = seq_len - qlen
            for j in range(qlen):
                position = prefix + j
                window = range(max(0, position + 1 - WINDOW), position + 1)
                rows.append(
                    (
                        [_slot(request, p) for p in window],
                        [_slot(request, p) for p in TOPK_POSITIONS],
                    )
                )
        out = torch.empty_like(q)
        for i, (window_slots, topk_slots) in enumerate(rows):
            kv = torch.cat(
                [
                    _read_fp8_ds_mla_cache_rows(
                        self.swa_cache,
                        torch.tensor(window_slots, device=self.device),
                        SWA_BLOCK_SIZE,
                        False,
                    ),
                    _read_fp8_ds_mla_cache_rows(
                        self.compressed_cache,
                        torch.tensor(topk_slots, device=self.device),
                        COMPRESSED_BLOCK_SIZE,
                        False,
                    ),
                ]
            )
            scores = (kv.float() @ q[i].float().T) * self.scale
            probs = torch.softmax(scores, dim=0)
            out[i] = (probs.T @ kv).to(torch.bfloat16)
        return out


# Capture with a full bucket (every row live), then replay a different order, a
# single request at the widest verify step, zero-draft rows and batches that
# leave part of the bucket padded.
_REPLAY_BATCHES = {
    5: [
        [(0, 1, 150), (1, 1, 96), (2, 1, 200), (3, 1, 129), (4, 1, 170)],
        [(4, 1, 152), (3, 1, 98), (2, 1, 202), (1, 1, 131), (0, 1, 172)],
        [(2, 5, 204)],
        [(0, 1, 154), (2, 2, 206)],
        [(1, 1, 98)],
    ],
    10: [
        [
            (0, 1, 150),
            (1, 1, 96),
            (2, 1, 200),
            (3, 1, 129),
            (4, 1, 170),
            (5, 1, 110),
            (6, 1, 190),
            (7, 1, 140),
            (8, 1, 230),
            (9, 1, 100),
        ],
        [
            (9, 1, 102),
            (2, 3, 202),
            (5, 1, 112),
            (0, 2, 152),
            (8, 1, 232),
        ],
        [(6, 5, 194), (3, 4, 133)],
        [
            (1, 1, 98),
            (7, 1, 142),
            (4, 1, 172),
            (2, 1, 204),
            (0, 1, 154),
            (9, 1, 104),
            (5, 1, 114),
            (8, 1, 234),
            (6, 1, 192),
            (3, 1, 131),
        ],
        [(2, 1, 206)],
    ],
}


@GPU
@pytest.mark.parametrize("bucket", sorted(_REPLAY_BATCHES))
def test_captured_decode_replay_matches_per_request_reference(bucket):
    """A FULL decode graph must follow the live split it is replayed with.

    Capture happens with a full bucket, where every row is live. Replay then
    presents fewer live rows, a different request order and zero-draft requests,
    while the persistent metadata buffers are rewritten eagerly. Every live row
    must match an independent per-token reference and the padded rows must stay
    finite garbage instead of leaking into the live rows.
    """
    config = _lane_config(adaptive=True)
    rig = _DecodeRig(config, max_num_reqs=16, bucket=bucket)
    capture_batch, *replay_batches = _REPLAY_BATCHES[bucket]
    assert sum(qlen for _, qlen, _ in capture_batch) == bucket
    # Guard the premise: at least one replay must leave part of the bucket padded.
    assert any(sum(qlen for _, qlen, _ in b) < bucket for b in replay_batches)

    capture_metadata = rig.build(capture_batch)
    # Warm up outside the graph so the Triton kernels are compiled.
    rig.run_decode(capture_metadata)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        rig.run_decode(capture_metadata)

    for batch in [capture_batch, *replay_batches]:
        live = sum(qlen for _, qlen, _ in batch)
        rig.build(batch)
        expected = rig.reference(batch)
        graph.replay()
        torch.testing.assert_close(
            rig.out[:live].float(), expected.float(), rtol=2e-2, atol=2e-2
        )
        # Padding rows keep the previous step's hidden states and top-k entries,
        # and the ragged indptr tail is not cleared, so a dead row can still walk
        # its capture-time (in-bounds) index range. It must stay finite and the
        # live rows above must be untouched by it.
        assert torch.isfinite(rig.out).all()
        indptr = rig.swa_builder.decode_swa_ragged_indptr_buffer[: bucket + 1]
        assert int(indptr[0]) == 0
        assert bool((indptr[1:] >= indptr[:-1]).all())
        # Bounded read: at most a full window per captured row, which is what
        # the persistent ragged buffer was sized for.
        assert int(indptr[-1]) <= bucket * WINDOW
        assert (
            bucket * WINDOW <= rig.swa_builder.decode_swa_ragged_indices_buffer.numel()
        )
