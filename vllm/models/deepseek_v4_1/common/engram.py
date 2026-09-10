# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Engram: n-gram hash lookups gated into the hyper-connection stream.

Port of the reference ``inference/engram.py`` + ``Engram`` /
``ParallelEngramEmbedding`` from ``inference/model.py`` (DeepSeek V4.1
checkpoint layout). Engram modules live on the backbone layers listed in
``engram_layer_ids`` only.

Two pieces of cross-forward state are needed because vLLM streams tokens
chunk-by-chunk while an n-gram at position ``p`` needs the token ids at
``p-1..p-3``:

- ``token_map``: token id -> compressed vocab id, built once from the
  model's tokenizer at init (deterministic; asserted against
  ``engram_compressed_vocab_size``).
- ``hash_cache``: one int32 slot per KV slot of the first local layer's
  sliding-window cache, holding the compressed id (or DEAD) of the token
  last written to that slot. Slots are stable per (request, position) —
  the block table pins a position to a physical slot, prefix-cache hits
  reuse both the physical blocks and the identical token ids, and
  spec-decode rollbacks rewrite the same slots — so lookbacks read back
  exactly what the owning request wrote. Lookback depth (3) is far inside
  the sliding window (128), so window eviction never frees a block a
  live lookback still needs.

  Slots are not part of the KV cache, so KV loaded from another instance
  (P/D, offload connectors) leaves them unwritten. The runner therefore
  passes ``lookback_token_ids``, the ids just before each request's chunk
  start, which take precedence over the slots. The V2 runner reads them
  from its device-resident token history and needs no slot cache; the V1
  runner's CPU token table holds placeholders for generated tokens under
  async scheduling, so it passes prompt positions only and keeps the slot
  cache for the rest.
"""

import mmap
import os
import re
import time
import weakref
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch
from torch import nn

from vllm.compilation.breakable_cudagraph import BreakableCUDAGraphCapture
from vllm.config import VllmConfig, get_current_vllm_config
from vllm.distributed import (
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
    tensor_model_parallel_all_gather,
)
from vllm.logger import init_logger
from vllm.model_executor.layers.linear import ReplicatedLinear
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.model_loader.weight_utils import SafetensorsMmapRef
from vllm.model_executor.utils import set_weight_attrs
from vllm.models.common.ops.sequence_parallel import sp_shard
from vllm.triton_utils import tl, triton
from vllm.utils.platform_utils import is_uva_available
from vllm.utils.torch_utils import (
    get_accelerator_view_from_cpu_tensor,
    weak_ref_tensor,
)
from vllm.v1.attention.ops.fp8_sm80 import _decode_fp8_f32

logger = init_logger(__name__)

# Cache value for tokens that take no part in an n-gram (image spans).
DEAD_ID = -1

# dsv41 engram-mmap: checkpoint names of the n-gram tables. In mmap mode the
# safetensors iterator yields these as `SafetensorsMmapRef` instead of
# reading 94 GiB per table into memory (`lazy_mmap_weight_names`).
ENGRAM_TABLE_WEIGHT_RE = re.compile(
    r"(?:^|\.)layers\.\d+\.engram\.embed\.(?:weight|scale)$"
)

# dsv41 engram-mmap: host gathers of more than two tokens' rows are split
# over a thread pool so page faults on cold rows overlap (numpy's take
# releases the GIL; a cold row is one ~200 us NVMe read, and x299's NVMe
# sustains ~120k random 4K reads/s at queue depth 32). Chunks shrink to
# _MMAP_GATHER_MIN_CHUNK rows so a decode batch spreads too. Measured on
# x299 (shard 47): 98k cold rows 3.0 s / 1.9 s / 1.0 s with 8 / 16 / 32
# threads; 192 rows (8 decode tokens) warm 5 us serial vs 0.3 ms pooled at
# chunk 32, cold 27 ms serial vs 6 ms pooled (4 ms at chunk 16, but 0.5 ms
# warm). The pool overhead is the price of bounded cold-miss latency.
_MMAP_GATHER_THREADS = 32
_MMAP_GATHER_MIN_CHUNK = 32
_MMAP_GATHER_MAX_CHUNK = 1024
_mmap_gather_pool: ThreadPoolExecutor | None = None


def _mmap_gather_executor() -> ThreadPoolExecutor:
    global _mmap_gather_pool
    if _mmap_gather_pool is None:
        _mmap_gather_pool = ThreadPoolExecutor(
            max_workers=_MMAP_GATHER_THREADS, thread_name_prefix="engram-mmap"
        )
    return _mmap_gather_pool


def is_engram_table_weight(name: str) -> bool:
    """dsv41 engram-mmap: True for `layers.N.engram.embed.{weight,scale}`."""
    return ENGRAM_TABLE_WEIGHT_RE.search(name) is not None


class EngramMmapTable:
    """dsv41 engram-mmap: one rank's row range of a checkpoint table, mapped
    read-only from the safetensors shard (MAP_SHARED, page-cache backed,
    MADV_RANDOM so a cold row costs one 4 KiB read and no readahead).

    `rows` is a [row_count, row_bytes] uint8 view over the mapping; nothing
    is read until a row is gathered. An anonymous zero-filled table (dummy
    load) uses the same interface.
    """

    def __init__(
        self,
        rows: np.ndarray,
        mapping: mmap.mmap | None,
        source: str,
    ) -> None:
        self.rows = rows
        self._mapping = mapping
        self.source = source

    @classmethod
    def from_ref(
        cls, ref: SafetensorsMmapRef, row_start: int, row_count: int
    ) -> "EngramMmapTable":
        row_bytes = ref.nbytes // ref.shape[0]
        byte_start = ref.data_begin + row_start * row_bytes
        byte_count = row_count * row_bytes
        assert byte_start + byte_count <= ref.data_end
        # mmap offsets must be page aligned; map from the page below and
        # skip the slack in the numpy view.
        aligned = byte_start - byte_start % mmap.ALLOCATIONGRANULARITY
        slack = byte_start - aligned
        fd = os.open(ref.path, os.O_RDONLY)
        try:
            mapping = mmap.mmap(
                fd,
                slack + byte_count,
                flags=mmap.MAP_SHARED,
                prot=mmap.PROT_READ,
                offset=aligned,
            )
        finally:
            os.close(fd)  # the mapping keeps its own reference
        mapping.madvise(mmap.MADV_RANDOM)
        rows = np.frombuffer(mapping, dtype=np.uint8, count=byte_count, offset=slack)
        return cls(
            rows.reshape(row_count, row_bytes), mapping, f"{ref.path}:{ref.name}"
        )

    @classmethod
    def anonymous(cls, row_count: int, row_bytes: int) -> "EngramMmapTable":
        # calloc-backed: pages stay unmapped until a gather touches them.
        return cls(np.zeros((row_count, row_bytes), dtype=np.uint8), None, "anonymous")

    def gather(self, local_rows: np.ndarray, out: np.ndarray) -> None:
        """out[i] = rows[local_rows[i]] (indices already validated)."""
        n = local_rows.shape[0]
        chunk = min(
            _MMAP_GATHER_MAX_CHUNK,
            max(_MMAP_GATHER_MIN_CHUNK, -(-n // _MMAP_GATHER_THREADS)),
        )
        if n <= 2 * _MMAP_GATHER_MIN_CHUNK:
            np.take(self.rows, local_rows, axis=0, out=out, mode="clip")
            return
        futures = [
            _mmap_gather_executor().submit(
                np.take,
                self.rows,
                local_rows[c : c + chunk],
                axis=0,
                out=out[c : c + chunk],
                mode="clip",
            )
            for c in range(0, n, chunk)
        ]
        for f in futures:
            f.result()

    def close(self) -> None:
        self.rows = None  # type: ignore[assignment]
        if self._mapping is not None:
            self._mapping.close()
            self._mapping = None


def _is_prime(n: int) -> bool:
    """Deterministic Miller-Rabin for n < 2**32 (avoids a sympy import)."""
    if n < 2:
        return False
    for p in (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37):
        if n % p == 0:
            return n == p
    d = n - 1
    r = 0
    while d % 2 == 0:
        d //= 2
        r += 1
    for a in (2, 7, 61):
        x = pow(a, d, n)
        if x in (1, n - 1):
            continue
        for _ in range(r - 1):
            x = x * x % n
            if x == n - 1:
                break
        else:
            return False
    return True


def find_next_prime(start: int, seen_primes: set[int]) -> int:
    """The smallest prime above `start` that has not been handed out yet."""
    candidate = start + 1
    while not _is_prime(candidate) or candidate in seen_primes:
        candidate += 1
    return candidate


def build_compressed_token_map(tokenizer) -> tuple[list[int], int]:
    """Map every token id onto a smaller id space where tokens that normalize
    alike collapse together.

    N-grams are hashed over these compressed ids, so " The", "the" and "THE"
    all hash the same way. The compressed size matters beyond bounds checking:
    every hash multiplier is derived from it.
    """
    from tokenizers import Regex, normalizers

    # A private-use char, so a token that is exactly one space survives
    # Strip() instead of collapsing to the empty string and merging with
    # unrelated tokens.
    sentinel = "\ue000"
    normalizer = normalizers.Sequence(
        [
            normalizers.NFKC(),
            normalizers.NFD(),
            normalizers.StripAccents(),
            normalizers.Lowercase(),
            normalizers.Replace(Regex(r"[ \t\r\n]+"), " "),
            normalizers.Replace(Regex(r"^ $"), sentinel),
            normalizers.Strip(),
            normalizers.Replace(sentinel, " "),
        ]
    )

    # The raw Rust tokenizer, matching what training decodes with
    # (no clean_up_tokenization_spaces).
    backend = tokenizer.backend_tokenizer
    key_to_new: dict[str, int] = {}
    lookup = [0] * len(tokenizer)
    for token_id in range(len(tokenizer)):
        text = backend.decode([token_id], skip_special_tokens=False)
        if "\ufffd" in text:
            # A partial UTF-8 byte token: nothing to normalize, so key it
            # by its raw form.
            key = backend.id_to_token(token_id)
        else:
            normalized = normalizer.normalize_str(text)
            key = normalized if normalized else text

        new_id = key_to_new.get(key)
        if new_id is None:
            new_id = len(key_to_new)
            key_to_new[key] = new_id
        lookup[token_id] = new_id

    return lookup, len(key_to_new)


def compute_hash_multipliers(
    layer_ids: tuple[int, ...], max_ngram_size: int, compressed_vocab_size: int
) -> torch.Tensor:
    """One multiplier per (layer, lookback), from a per-layer RNG so layers
    hash differently. Kept odd and bounded so `token_id * multiplier` cannot
    overflow int64.
    """
    max_long = np.iinfo(np.int64).max
    multiplier_bound = max(1, (max_long // compressed_vocab_size) // 2)
    rows = []
    for layer_id in layer_ids:
        generator = np.random.default_rng(10007 * layer_id)
        values = generator.integers(
            low=0,
            high=multiplier_bound,
            size=(max_ngram_size,),
            dtype=np.int64,
        )
        rows.append(torch.tensor(values * 2 + 1))
    return torch.stack(rows)


class EngramLayout:
    """Bucket layout of the n-gram hash tables.

    A position is hashed as `max_ngram_size - 1` n-grams (2-gram .. max), each
    split over `n_heads` heads. Every (n-gram size, head) pair owns its own
    prime-sized bucket range in the layer's table; the primes are drawn in
    order and never reused, which keeps the ranges disjoint.
    """

    def __init__(self, config) -> None:
        self.layer_ids: tuple[int, ...] = tuple(config.engram_layer_ids)
        self.num_embeddings: tuple[int, ...] = tuple(config.engram_num_embeddings)
        self.max_ngram_size: int = config.engram_max_ngram_size
        self.n_heads: int = config.engram_n_heads
        self.head_dim: int = config.engram_head_dim
        self.compressed_vocab_size: int = config.engram_compressed_vocab_size
        self.pad_token_id: int = config.engram_pad_token_id
        assert len(self.layer_ids) == len(self.num_embeddings)

        primes = []
        seen: set[int] = set()
        for _ in self.layer_ids:
            per_ngram = []
            for _ in range(self.max_ngram_size - 1):
                sizes, current = [], config.engram_vocab_size - 1
                for _ in range(self.n_heads):
                    current = find_next_prime(current, seen)
                    seen.add(current)
                    sizes.append(current)
                per_ngram.append(tuple(sizes))
            primes.append(tuple(per_ngram))
        self.primes: tuple[tuple[tuple[int, ...], ...], ...] = tuple(primes)
        self.n_hash_cols = (self.max_ngram_size - 1) * self.n_heads
        flat = [[p for per_ngram in layer for p in per_ngram] for layer in primes]
        offsets = [np.cumsum([0, *sizes[:-1]]) for sizes in flat]
        self.offsets = torch.tensor(np.array(offsets))  # [n_layers, n_hash_cols]

    @classmethod
    def from_config(cls, config) -> "EngramLayout | None":
        if not getattr(config, "engram_layer_ids", None):
            return None
        return cls(config)


@triton.jit(do_not_specialize=["num_tokens"])
def _write_hash_cache_kernel(
    input_ids,
    token_map,
    dead_mask,
    slot_mapping,
    cache,
    num_tokens,
    input_stride,
    mask_stride,
    slot_stride,
    BLOCK_SIZE: tl.constexpr,
    dead_id,
):
    token_idx = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    slot = tl.load(
        slot_mapping + token_idx * slot_stride, token_idx < num_tokens, other=-1
    ).to(tl.int64)
    valid = (token_idx < num_tokens) & (slot >= 0)
    token = tl.load(input_ids + token_idx * input_stride, valid, other=0)
    value = tl.load(token_map + token, valid, other=0)
    dead = tl.load(dead_mask + token_idx * mask_stride, valid, other=False)
    value = tl.where(dead, dead_id, value)
    tl.store(cache + slot, value, valid)


@triton.jit(
    do_not_specialize=[
        "num_tokens",
        "num_slots",
        "num_query_rows",
        "num_table_rows",
        "max_blocks",
    ]
)
def _hash_ids_kernel(
    input_ids,
    token_map,
    dead_mask,
    positions,
    block_table,
    query_start_loc,
    multipliers,
    primes,
    offsets,
    cache,
    lookback_token_ids,
    lookback_dead_mask,
    output,
    num_tokens,
    num_slots,
    pad_id,
    input_stride,
    mask_stride,
    position_stride,
    table_stride,
    table_col_stride,
    query_stride,
    num_query_rows,
    num_table_rows,
    max_blocks,
    cache_block_size,
    MAX_NGRAM: tl.constexpr,
    num_heads,
    BLOCK_T: tl.constexpr,
    BLOCK_H: tl.constexpr,
    dead_id,
    lookback_depth,
    lookback_row_stride,
    lookback_col_stride,
    lookback_mask_row_stride,
    lookback_mask_col_stride,
):
    token = tl.program_id(0) * BLOCK_T + tl.arange(0, BLOCK_T)
    layer = tl.program_id(1)
    num_layers = tl.num_programs(1)
    valid = token < num_tokens
    # Upper bound in query_start_loc[1:], including repeated padding boundaries.
    lo = tl.full((BLOCK_T,), 0, tl.int32)
    hi = tl.full((BLOCK_T,), num_query_rows, tl.int32)
    while tl.sum((lo < hi).to(tl.int32), 0) > 0:
        mid = (lo + hi) // 2
        end = tl.load(
            query_start_loc + (mid + 1) * query_stride,
            lo < hi,
            other=0,
        )
        right = token >= end
        active = lo < hi
        lo = tl.where(active & right, mid + 1, lo)
        hi = tl.where(active & ~right, mid, hi)
    req = tl.minimum(lo, num_query_rows - 1).to(tl.int64)
    chunk_idx = tl.load(query_start_loc + req * query_stride)
    chunk_idx = tl.minimum(chunk_idx, num_tokens - 1).to(tl.int64)
    chunk_start = tl.load(positions + chunk_idx * position_stride)
    position = tl.load(positions + token * position_stride, valid, other=0).to(tl.int64)
    head = tl.arange(0, BLOCK_H)
    blocked = tl.full((BLOCK_T,), False, tl.int1)
    rolling = tl.full((BLOCK_T,), 0, tl.int64)
    for shift in tl.static_range(MAX_NGRAM):
        lookback = position - shift
        in_batch = lookback >= chunk_start
        batch_idx = tl.maximum(token - shift, 0)
        batch_token = tl.load(
            input_ids + batch_idx * input_stride, valid & in_batch, other=0
        )
        batch_source = tl.load(token_map + batch_token, valid & in_batch, other=0)
        batch_dead = tl.load(
            dead_mask + batch_idx * mask_stride, valid & in_batch, other=False
        )
        batch_source = tl.where(batch_dead, dead_id, batch_source)

        col = chunk_start - 1 - lookback
        in_window = valid & ~in_batch & (col >= 0) & (col < lookback_depth)
        col = tl.minimum(tl.maximum(col, 0), lookback_depth - 1)
        window_token = tl.load(
            lookback_token_ids + req * lookback_row_stride + col * lookback_col_stride,
            in_window,
            other=-1,
        )
        known = in_window & (window_token >= 0)
        window_source = tl.load(token_map + window_token, known, other=0)
        window_dead = tl.load(
            lookback_dead_mask
            + req * lookback_mask_row_stride
            + col * lookback_mask_col_stride,
            known,
            other=False,
        )
        window_source = tl.where(window_dead, dead_id, window_source)

        if cache is not None:
            clamped = tl.minimum(
                tl.maximum(lookback, 0), max_blocks * cache_block_size - 1
            )
            block_row = tl.minimum(req, num_table_rows - 1)
            needs_cache = valid & ~in_batch & ~known
            block = tl.load(
                block_table
                + block_row * table_stride
                + (clamped // cache_block_size) * table_col_stride,
                needs_cache,
                other=0,
            ).to(tl.int64)
            slot = tl.minimum(
                tl.maximum(block * cache_block_size + clamped % cache_block_size, 0),
                num_slots - 1,
            )
            fallback = tl.load(cache + slot, needs_cache, other=0)
        else:
            fallback = tl.full((BLOCK_T,), pad_id, tl.int32)
        source = tl.where(
            in_batch, batch_source, tl.where(known, window_source, fallback)
        ).to(tl.int64)
        blocked |= (lookback < 0) | (source == dead_id)
        value = tl.where(blocked, pad_id, source)
        multiplier = tl.load(multipliers + layer * MAX_NGRAM + shift)
        rolling ^= value * multiplier
        if shift > 0:
            col = (shift - 1) * num_heads + head
            param_offset = layer * (MAX_NGRAM - 1) * num_heads + col
            prime = tl.load(primes + param_offset, head < num_heads, other=1)
            offset = tl.load(offsets + param_offset, head < num_heads, other=0)
            hashed = rolling[:, None] % prime[None, :] + offset[None, :]
            out_offset = (token.to(tl.int64) * num_layers + layer)[:, None] * (
                (MAX_NGRAM - 1) * num_heads
            ) + col[None, :]
            tl.store(output + out_offset, hashed, valid[:, None] & (head < num_heads))


class NgramHashState(nn.Module):
    """Maps each position to the hash ids of the n-grams ending there.

    Stateless on the V2 runner, which supplies every lookback token id. On
    the V1 runner it also keeps `hash_cache`, the slot-keyed rolling store
    of compressed ids (see module docstring), for generated tokens.
    """

    def __init__(
        self,
        vllm_config: VllmConfig,
        layout: EngramLayout,
        swa_cache_module: nn.Module,
    ) -> None:
        super().__init__()
        self.layout = layout
        self.swa_cache_module = swa_cache_module
        self.block_size: int = swa_cache_module.block_size
        self.lookback_depth: int = layout.max_ngram_size - 1
        self.use_slot_cache: bool = not vllm_config.use_v2_model_runner
        self._cache: torch.Tensor | None = None
        self._kv_cache_ref: weakref.ReferenceType[torch.Tensor] | None = None

        model_config = vllm_config.model_config
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(
            model_config.tokenizer,
            trust_remote_code=model_config.trust_remote_code,
            revision=model_config.revision,
        )
        token_map, vocab_size = build_compressed_token_map(tokenizer)
        if vocab_size != layout.compressed_vocab_size:
            raise ValueError(
                f"Compressed vocab size mismatch: built {vocab_size} from the "
                f"tokenizer, config expects {layout.compressed_vocab_size}; "
                "every hash multiplier derives from it, so the engram tables "
                "would be silently rehashed."
            )
        self.pad_id = token_map[layout.pad_token_id]
        multipliers = compute_hash_multipliers(
            layout.layer_ids, layout.max_ngram_size, vocab_size
        )
        self.register_buffer(
            "token_map", torch.tensor(token_map, dtype=torch.int32), persistent=False
        )
        self.register_buffer("primes", torch.tensor(layout.primes), persistent=False)
        self.register_buffer("offsets", layout.offsets, persistent=False)
        self.register_buffer("multipliers", multipliers, persistent=False)
        logger.info(
            "Built engram token map (%d -> %d ids) for layers %s",
            len(token_map),
            vocab_size,
            layout.layer_ids,
        )

    def ensure_cache(self) -> bool:
        """Lazily size the slot-keyed cache from the bound SWA KV cache.

        Returns False while the KV cache is unbound (profile run); the caller
        skips engram hashing then. Without the slot cache only that check
        remains.
        """
        kv_cache = self.swa_cache_module.kv_cache
        if kv_cache.numel() == 0:
            self._cache = None
            self._kv_cache_ref = None
            return False
        if not self.use_slot_cache:
            return True
        if self._kv_cache_ref is not None and self._kv_cache_ref() is kv_cache:
            return True
        # Graph memory profiling binds a temporary, smaller KV cache first.
        # Rebinding must discard its hash history without retaining KV storage.
        self._cache = torch.zeros(
            kv_cache.shape[0] * self.block_size,
            dtype=torch.int32,
            device=kv_cache.device,
        )
        self._kv_cache_ref = weakref.ref(kv_cache)
        return True

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        query_start_loc: torch.Tensor,
        dead_mask: torch.Tensor,
        lookback_token_ids: torch.Tensor,
        lookback_dead_mask: torch.Tensor,
        slot_mapping: torch.Tensor | None,
        block_table: torch.Tensor | None,
    ) -> torch.Tensor:
        """Compute [tokens, layers, hash columns] int32 n-gram hashes.

        History comes from the current chunk, then the runner's lookback
        window, then the optional V1 slot cache. V2 needs only one launch.
        """
        cache = self._cache if self.use_slot_cache else None
        num_tokens = input_ids.shape[0]
        num_layers, max_ngram = self.multipliers.shape
        num_heads = self.primes.shape[-1]
        output = input_ids.new_empty(
            (num_tokens, num_layers, (max_ngram - 1) * num_heads), dtype=torch.int32
        )
        if num_tokens == 0:
            return output
        if self.use_slot_cache:
            assert cache is not None and slot_mapping is not None
            assert block_table is not None
            # Finish writes before other thread blocks read fallback history.
            _write_hash_cache_kernel[(triton.cdiv(num_tokens, 256),)](
                input_ids,
                self.token_map,
                dead_mask,
                slot_mapping,
                cache,
                num_tokens,
                input_ids.stride(0),
                dead_mask.stride(0),
                slot_mapping.stride(0),
                256,
                DEAD_ID,
            )
        _hash_ids_kernel[(triton.cdiv(num_tokens, 32), num_layers)](
            input_ids,
            self.token_map,
            dead_mask,
            positions,
            block_table,
            query_start_loc,
            self.multipliers,
            self.primes,
            self.offsets,
            cache,
            lookback_token_ids,
            lookback_dead_mask,
            output,
            num_tokens,
            cache.shape[0] if cache is not None else 0,
            self.pad_id,
            input_stride=input_ids.stride(0),
            mask_stride=dead_mask.stride(0),
            position_stride=positions.stride(0),
            table_stride=block_table.stride(0) if block_table is not None else 0,
            table_col_stride=block_table.stride(1) if block_table is not None else 0,
            query_stride=query_start_loc.stride(0),
            num_query_rows=query_start_loc.numel() - 1,
            num_table_rows=block_table.shape[0] if block_table is not None else 0,
            max_blocks=block_table.shape[1] if block_table is not None else 0,
            cache_block_size=self.block_size,
            MAX_NGRAM=max_ngram,
            num_heads=num_heads,
            BLOCK_T=32,
            BLOCK_H=triton.next_power_of_2(num_heads),
            dead_id=DEAD_ID,
            lookback_depth=lookback_token_ids.shape[1],
            lookback_row_stride=lookback_token_ids.stride(0),
            lookback_col_stride=lookback_token_ids.stride(1),
            lookback_mask_row_stride=lookback_dead_mask.stride(0),
            lookback_mask_col_stride=lookback_dead_mask.stride(1),
            num_warps=4,
        )
        return output


def _engram_head_shard_weight_loader(
    param: torch.nn.Parameter, loaded_weight: torch.Tensor
) -> None:
    """Load this rank's complete head buckets. ue8m0 scales arrive as
    float8_e8m0fnu; keep the raw bytes (the param stores uint8)."""
    attach = getattr(param, "engram_mmap_attach", None)
    if attach is not None:
        # dsv41 engram-mmap: the param is a 0-row placeholder; map the rank's
        # row slice of the checkpoint tensor instead of copying it.
        if not isinstance(loaded_weight, SafetensorsMmapRef):
            raise RuntimeError(
                "engram table_mode='mmap' needs the default safetensors loader "
                "(no enable_multithread_load / prefetch / torchao strategy), "
                f"got a {type(loaded_weight).__name__} for the table"
            )
        attach(loaded_weight)
        return
    part_rows = param.shape[0]
    if loaded_weight.dtype == torch.float8_e8m0fnu:
        loaded_weight = loaded_weight.view(torch.uint8)
    shard = loaded_weight.narrow(0, param.engram_vocab_start, part_rows)
    assert shard.shape == param.shape, (
        f"engram shard {tuple(shard.shape)} does not fit param {tuple(param.shape)}"
    )
    param.data.copy_(shard)


@triton.jit
def _engram_lookup_kernel(
    weight,
    scales,
    ids,
    out,
    vocab_start,
    vocab_end,
    num_rows,
    ids_stride_t,
    ids_stride_h,
    HEAD_START: tl.constexpr,
    LOCAL_HEADS: tl.constexpr,
    TOTAL_HEADS: tl.constexpr,
    DIM: tl.constexpr,
    QUANT_BLOCK: tl.constexpr,
    BLOCK_R: tl.constexpr,
    GRID: tl.constexpr,
):
    """Gather fp8 rows, apply their ue8m0 block scales, write bf16.

    Only this rank's heads are read; padded heads write zeros for all-gather.
    `weight`/`scales` may address pinned host memory through UVA. `weight`
    is the e4m3fn table viewed as uint8: the bytes are decoded with the
    fp8_sm80 helper, which is the hardware convert on SM89+ and integer
    math below it (Triton refuses native fp8 converts there).
    """
    cols = tl.arange(0, DIM)
    scale_cols = cols // QUANT_BLOCK
    for base in tl.range(tl.program_id(0) * BLOCK_R, num_rows, GRID * BLOCK_R):
        rows = base + tl.arange(0, BLOCK_R)
        valid = rows < num_rows
        head = HEAD_START + rows % LOCAL_HEADS
        token = (rows // LOCAL_HEADS).to(tl.int64)
        index = tl.load(
            ids + token * ids_stride_t + head * ids_stride_h,
            mask=valid & (head < TOTAL_HEADS),
            other=-1,
        ).to(tl.int64)
        owned = valid & (head < TOTAL_HEADS)
        owned &= (index >= vocab_start) & (index < vocab_end)
        local = tl.where(owned, index - vocab_start, 0)
        values_u8 = tl.load(
            weight + local[:, None] * DIM + cols[None, :],
            mask=owned[:, None],
            other=0,
        )
        values = _decode_fp8_f32(values_u8, False)
        scale = tl.load(
            scales + local[:, None] * (DIM // QUANT_BLOCK) + scale_cols[None, :],
            mask=owned[:, None],
            other=0,
        )
        # ue8m0 is a power of two, so its byte *is* the fp32 exponent field.
        scale = (scale.to(tl.int32) << 23).to(tl.float32, bitcast=True)
        tl.store(
            out + rows[:, None] * DIM + cols[None, :],
            (values * scale).to(tl.bfloat16),
            mask=valid[:, None],
        )


class ParallelEngramEmbedding(nn.Module):
    """The n-gram hash table, sharded by complete hash heads over TP ranks.
    Rows stay fp8 and are dequantized with ue8m0 per-32 scales on lookup.

    With `cpu_offload` the shard lives in pinned host memory and is read over
    UVA instead of HBM; the TP sharding is unchanged either way.

    dsv41 engram-mmap: `table_mode="mmap"` keeps no copy of the shard at all.
    The rank's row range is memory-mapped from the checkpoint file, the rows
    a step needs are gathered on the host (raw fp8 bytes + ue8m0 scale bytes)
    into pinned staging, copied to a device staging buffer and dequantised
    there by `_engram_lookup_kernel` itself (identity indices over the
    staging rows), so the maths is the same kernel as the other modes.
    """

    # dsv41 engram-mmap: class default so instances built without __init__
    # (tests) take the non-mmap paths.
    table_mode: str | None = None
    # dsv41 engram-mmap: set by the V2 runner when it stages the rows itself
    # before every FULL cudagraph replay (`DeepseekV4Model.engram_prefetch`);
    # a lookup met inside a plain (non-breakable) capture is then a no-op
    # instead of an error, and the graph reads the prefilled static buffer.
    full_graph_prefetch: bool = False

    def __init__(
        self,
        num_embeddings: int,
        dim: int,
        head_sizes: tuple[int, ...],
        block_size: int = 32,
        cpu_offload: bool = False,
        table_mode: str | None = None,
    ):
        super().__init__()
        tp_size = get_tensor_model_parallel_world_size()
        tp_rank = get_tensor_model_parallel_rank()
        assert head_sizes and all(size > 0 for size in head_sizes)
        assert sum(head_sizes) <= num_embeddings
        # dsv41 engram-mmap: `table_mode` wins over the legacy flag.
        if table_mode is None:
            table_mode = "pinned" if cpu_offload else "resident"
        assert table_mode in ("pinned", "resident", "mmap"), table_mode
        cpu_offload = table_mode == "pinned"
        self.table_mode = table_mode
        if cpu_offload and not is_uva_available():
            raise RuntimeError("Engram CPU offload requires UVA support")
        self.num_embeddings = num_embeddings
        self.dim = dim
        self.block_size = block_size
        self.n_hash_cols = len(head_sizes)
        self.part_n_hash_cols = triton.cdiv(self.n_hash_cols, tp_size)
        self.head_start = tp_rank * self.part_n_hash_cols
        head_end = self.head_start + self.part_n_hash_cols
        self.vocab_start_idx = sum(head_sizes[: self.head_start])
        self.vocab_end_idx = sum(head_sizes[:head_end])
        self.part_num_embeddings = self.vocab_end_idx - self.vocab_start_idx
        self.tp_size = tp_size
        self.cpu_offload = cpu_offload
        self._views: tuple[torch.Tensor, torch.Tensor] | None = None
        self._view_src: tuple[int, int] | None = None
        self._num_sms = torch.cuda.get_device_properties(
            torch.accelerator.current_device_index()
        ).multi_processor_count

        # Explicit device: model init runs under a `torch.device("cuda")`
        # context, which would otherwise put the shard in HBM.
        kwargs = {"device": "cpu", "pin_memory": True} if cpu_offload else {}
        # dsv41 engram-mmap: 0-row placeholders keep the checkpoint names in
        # `named_parameters()` (the loader routes the lazy refs through them)
        # without allocating any host or device storage.
        param_rows = 0 if table_mode == "mmap" else self.part_num_embeddings
        if table_mode == "mmap":
            kwargs = {"device": "cpu"}
        self.weight = nn.Parameter(
            torch.empty(param_rows, dim, dtype=torch.float8_e4m3fn, **kwargs),
            requires_grad=False,
        )
        self.weight_scale_inv = nn.Parameter(
            torch.empty(
                param_rows,
                dim // block_size,
                dtype=torch.uint8,
                **kwargs,
            ),
            requires_grad=False,
        )
        for param in (self.weight, self.weight_scale_inv):
            set_weight_attrs(
                param,
                {
                    "weight_loader": _engram_head_shard_weight_loader,
                    "engram_vocab_start": self.vocab_start_idx,
                },
            )
        # dsv41 engram-mmap state: the two mapped tables, the host/device
        # staging and the identity index tensor the lookup kernel reads.
        self._mmap_tables: dict[str, EngramMmapTable] = {}
        self._mmap_staging_rows = 0
        self._mmap_idx_host: torch.Tensor | None = None
        self._mmap_raw_host: torch.Tensor | None = None
        self._mmap_scale_host: torch.Tensor | None = None
        self._mmap_raw_dev: torch.Tensor | None = None
        self._mmap_scale_dev: torch.Tensor | None = None
        self._mmap_ids: torch.Tensor | None = None
        # calls, rows, seconds waiting for the ids (GPU drain) and seconds
        # of host gather + H2D + kernel launch; logged at DEBUG every 500.
        self.mmap_stats = {"calls": 0, "rows": 0, "sync_s": 0.0, "gather_s": 0.0}
        if table_mode == "mmap":
            set_weight_attrs(
                self.weight, {"engram_mmap_attach": self._attach_mmap_weight}
            )
            set_weight_attrs(
                self.weight_scale_inv,
                {"engram_mmap_attach": self._attach_mmap_scale},
            )
            logger.info(
                "Engram table in mmap mode: rows %d..%d of %d (%.2f GiB of "
                "checkpoint per rank, page-cache backed, nothing pinned)",
                self.vocab_start_idx,
                self.vocab_end_idx,
                num_embeddings,
                self.part_num_embeddings * (dim + dim // block_size) / 1024**3,
            )
        if cpu_offload:
            logger.info(
                "Engram table offloaded to pinned host memory: %d rows x %d, "
                "%.2f GiB per rank",
                self.part_num_embeddings,
                dim,
                self.part_num_embeddings * (dim + dim // block_size) / 1024**3,
            )

    # ---- dsv41 engram-mmap -------------------------------------------------

    def _attach_mmap_weight(self, ref: SafetensorsMmapRef) -> None:
        self._attach_mmap("weight", ref, self.dim)

    def _attach_mmap_scale(self, ref: SafetensorsMmapRef) -> None:
        self._attach_mmap("scale", ref, self.dim // self.block_size)

    def _attach_mmap(self, kind: str, ref: SafetensorsMmapRef, row_bytes: int) -> None:
        """Map this rank's rows of one checkpoint table (called by the weight
        loader with the lazy ref the safetensors iterator produced)."""
        if len(ref.shape) != 2 or ref.shape[1] != row_bytes:
            raise ValueError(
                f"engram {kind} table {ref.name}: shape {ref.shape}, expected "
                f"[rows, {row_bytes}] (1 byte per element)"
            )
        if ref.nbytes != ref.shape[0] * row_bytes:
            raise ValueError(f"engram {kind} table {ref.name} is not 1 byte/element")
        if ref.shape[0] < self.vocab_end_idx:
            raise ValueError(
                f"engram {kind} table {ref.name} has {ref.shape[0]} rows, rank "
                f"needs rows up to {self.vocab_end_idx}"
            )
        old = self._mmap_tables.pop(kind, None)
        if old is not None:
            old.close()
        self._mmap_tables[kind] = EngramMmapTable.from_ref(
            ref, self.vocab_start_idx, self.part_num_embeddings
        )
        logger.info(
            "Engram %s table mapped from %s (rows %d..%d)",
            kind,
            ref.path,
            self.vocab_start_idx,
            self.vocab_end_idx,
        )

    def _mmap_table_pair(self) -> tuple[EngramMmapTable, EngramMmapTable]:
        """The mapped tables, or anonymous zero tables under a dummy load
        (`--load-format dummy` never calls the weight loader)."""
        if "weight" not in self._mmap_tables or "scale" not in self._mmap_tables:
            for kind in ("weight", "scale"):
                if kind in self._mmap_tables:
                    continue
                row_bytes = (
                    self.dim if kind == "weight" else self.dim // self.block_size
                )
                self._mmap_tables[kind] = EngramMmapTable.anonymous(
                    self.part_num_embeddings, row_bytes
                )
            logger.warning(
                "Engram mmap tables were never attached (dummy load?); using "
                "anonymous zero-filled tables for the host gather path"
            )
        return self._mmap_tables["weight"], self._mmap_tables["scale"]

    def configure_mmap_staging(self, max_tokens: int) -> None:
        """Size the staging for `max_tokens` tokens x local heads rows. Both
        buffers are consumed only inside the eager segment, so resizing them
        later (a larger batch than planned) does not invalidate captured
        graphs; the kernel output `out` is the caller's static buffer."""
        rows = max(1, max_tokens) * self.part_n_hash_cols
        if rows <= self._mmap_staging_rows:
            return
        max_tokens = max(1, max_tokens)
        self._mmap_staging_rows = rows
        # Explicit device: model init runs under a `torch.device("cuda")`
        # context, and only dense CPU tensors can be pinned.
        host = {"device": "cpu", "pin_memory": True}
        self._mmap_idx_host = torch.empty(
            max_tokens, self.part_n_hash_cols, dtype=torch.int32, **host
        )
        self._mmap_raw_host = torch.empty(rows, self.dim, dtype=torch.uint8, **host)
        self._mmap_scale_host = torch.empty(
            rows, self.dim // self.block_size, dtype=torch.uint8, **host
        )
        device = torch.accelerator.current_device_index()
        self._mmap_raw_dev = torch.empty(
            rows, self.dim, dtype=torch.uint8, device=f"cuda:{device}"
        )
        self._mmap_scale_dev = torch.empty(
            rows,
            self.dim // self.block_size,
            dtype=torch.uint8,
            device=f"cuda:{device}",
        )
        # Identity indices: staging row r = token r // local_heads, local
        # head r % local_heads. Laid out as [tokens, n_hash_cols] so the
        # lookup kernel reads them exactly like real hash ids; heads beyond
        # this rank's real heads (TP padding) never exist in the tensor and
        # are masked by the kernel like in the other modes.
        local_cols = self._mmap_local_cols()
        ids = torch.zeros(max_tokens, self.n_hash_cols, dtype=torch.int32, device="cpu")
        ids[:, self.head_start : self.head_start + local_cols] = (
            torch.arange(max_tokens, dtype=torch.int32)[:, None] * self.part_n_hash_cols
            + torch.arange(local_cols, dtype=torch.int32)[None, :]
        )
        self._mmap_ids = ids.to(f"cuda:{device}")

    def _mmap_local_cols(self) -> int:
        """Real (non-padded) hash heads on this rank."""
        return max(0, min(self.part_n_hash_cols, self.n_hash_cols - self.head_start))

    def stage_from_mmap(self, indices: torch.Tensor, out: torch.Tensor) -> None:
        """mmap-mode lookup of [T, heads] into `out` [T, local_heads, dim].

        Under a breakable cudagraph capture this becomes an eager segment
        (the host gather can never be captured); `out` must be a static
        buffer, `indices` is weak-referenced like the attention breaks do.
        """
        capture = BreakableCUDAGraphCapture.current()
        if capture is not None and capture._capturing:
            ids, dst = weak_ref_tensor(indices), weak_ref_tensor(out)
            capture.add_eager(lambda: self._stage_from_mmap_eager(ids, dst))
            return
        if torch.cuda.is_current_stream_capturing():
            if self.full_graph_prefetch:
                # FULL graph capture: the runner stages `out` before each
                # replay, so nothing is recorded here.
                return
            raise RuntimeError(
                "engram table_mode='mmap' cannot run inside a plain cudagraph "
                "capture; use breakable cudagraphs or --enforce-eager"
            )
        self._stage_from_mmap_eager(indices, out)

    def _stage_from_mmap_eager(self, indices: torch.Tensor, out: torch.Tensor) -> None:
        num_tokens = indices.shape[0]
        if num_tokens == 0:
            return
        t0 = time.perf_counter()
        self.configure_mmap_staging(num_tokens)
        assert self._mmap_idx_host is not None and self._mmap_ids is not None
        assert self._mmap_raw_host is not None and self._mmap_scale_host is not None
        assert self._mmap_raw_dev is not None and self._mmap_scale_dev is not None
        local_cols = self._mmap_local_cols()
        rows = num_tokens * self.part_n_hash_cols
        stream = torch.cuda.current_stream()
        # 1. indices to the host (the hash kernel is the single source of
        #    truth for them; the sync also retires last step's H2D copies out
        #    of the pinned staging before it is rewritten below).
        idx_host = self._mmap_idx_host[:num_tokens]
        if local_cols:
            idx_host[:, :local_cols].copy_(
                indices[:, self.head_start : self.head_start + local_cols],
                non_blocking=True,
            )
        stream.synchronize()
        t1 = time.perf_counter()
        idx = idx_host.numpy().reshape(-1).astype(np.int64, copy=False)
        owned = (idx >= self.vocab_start_idx) & (idx < self.vocab_end_idx)
        if local_cols < self.part_n_hash_cols:
            owned.reshape(num_tokens, self.part_n_hash_cols)[:, local_cols:] = False
        local = np.where(owned, idx - self.vocab_start_idx, 0)
        # 2. host gather of raw fp8 rows + ue8m0 scale rows; rows another
        #    rank owns are zero, which the kernel dequantises to 0 exactly
        #    like its own masked loads do.
        raw = self._mmap_raw_host[:rows].numpy()
        scale = self._mmap_scale_host[:rows].numpy()
        if owned.any():
            weight_table, scale_table = self._mmap_table_pair()
            weight_table.gather(local, raw)
            scale_table.gather(local, scale)
            if not owned.all():
                unowned = ~owned
                raw[unowned] = 0
                scale[unowned] = 0
        else:
            # Nothing owned this step (or a TP rank with only padded heads).
            raw[:] = 0
            scale[:] = 0
        # 3. H2D into the device staging, then the shared dequant kernel with
        #    identity indices over the staged rows.
        self._mmap_raw_dev[:rows].copy_(self._mmap_raw_host[:rows], non_blocking=True)
        self._mmap_scale_dev[:rows].copy_(
            self._mmap_scale_host[:rows], non_blocking=True
        )
        grid = min(triton.cdiv(rows, 16), self._num_sms)
        _engram_lookup_kernel[(grid,)](
            self._mmap_raw_dev,
            self._mmap_scale_dev,
            self._mmap_ids,
            out,
            0,
            rows,
            rows,
            self._mmap_ids.stride(0),
            self._mmap_ids.stride(1),
            HEAD_START=self.head_start,
            LOCAL_HEADS=self.part_n_hash_cols,
            TOTAL_HEADS=self.n_hash_cols,
            DIM=self.dim,
            QUANT_BLOCK=self.block_size,
            BLOCK_R=16,
            GRID=grid,
        )
        stats = self.mmap_stats
        stats["calls"] += 1
        stats["rows"] += rows
        stats["sync_s"] += t1 - t0
        stats["gather_s"] += time.perf_counter() - t1
        if stats["calls"] % 500 == 0:
            logger.debug(
                "engram mmap gather: %d calls, %.1f rows/call, %.3f ms/call "
                "waiting for ids (GPU drain), %.3f ms/call host gather + H2D",
                stats["calls"],
                stats["rows"] / stats["calls"],
                1e3 * stats["sync_s"] / stats["calls"],
                1e3 * stats["gather_s"] / stats["calls"],
            )

    def _storage(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Parameters when resident, else cached UVA views of the pinned shard.

        Rebuilt if anything swaps `.data`, so a stale device pointer cannot
        survive silently.
        """
        if self.table_mode == "mmap":
            raise RuntimeError("engram mmap mode has no device-readable table")
        if not self.cpu_offload:
            return self.weight.data, self.weight_scale_inv.data
        src = (self.weight.data_ptr(), self.weight_scale_inv.data_ptr())
        if self._view_src != src:
            self._views = (
                get_accelerator_view_from_cpu_tensor(self.weight.data),
                get_accelerator_view_from_cpu_tensor(self.weight_scale_inv.data),
            )
            self._view_src = src
        assert self._views is not None
        return self._views

    def lookup(
        self, indices: torch.Tensor, out: torch.Tensor, background: bool = False
    ) -> None:
        """Look up local heads of [T, heads] into [T, local_heads, dim] bf16.

        `background` limits the grid to leave SMs for concurrent work.
        """
        rows = indices.shape[0] * self.part_n_hash_cols
        if not rows:
            return
        if self.table_mode == "mmap":
            # dsv41 engram-mmap: host gather (eager break under capture).
            self.stage_from_mmap(indices, out)
            return
        weight, scales = self._storage()
        # The table dwarfs TLB reach, so a persistent grid near the SM count
        # beats one program per row; halve it to leave SMs for the main stream.
        tiles = triton.cdiv(rows, 16)
        grid = min(tiles, self._num_sms // 2 if background else self._num_sms)
        _engram_lookup_kernel[(grid,)](
            weight.view(torch.uint8),
            scales,
            indices,
            out,
            self.vocab_start_idx,
            self.vocab_end_idx,
            rows,
            indices.stride(0),
            indices.stride(1),
            HEAD_START=self.head_start,
            LOCAL_HEADS=self.part_n_hash_cols,
            TOTAL_HEADS=self.n_hash_cols,
            DIM=self.dim,
            QUANT_BLOCK=self.block_size,
            BLOCK_R=16,
            GRID=grid,
        )

    def forward(self, indices: torch.Tensor) -> torch.Tensor:
        """indices: [num_tokens, n_hash_cols] -> [num_tokens, n_hash_cols, dim]
        bf16, gathered from all TP shards."""
        out = torch.empty(
            (indices.shape[0], self.part_n_hash_cols, self.dim),
            dtype=torch.bfloat16,
            device=indices.device,
        )
        self.lookup(indices, out)
        if self.tp_size > 1:
            out = tensor_model_parallel_all_gather(out, dim=1)
            out = out[:, : self.n_hash_cols]
        return out


@triton.jit(do_not_specialize=["num_kv_tokens"])
def _fused_engram_post_wkv_kernel(
    hidden_states,
    kv,
    q_weight,
    k_weight,
    token_mask,
    output,
    num_kv_tokens,
    hidden_stride_t,
    hidden_stride_h,
    hidden_stride_d,
    kv_stride_t,
    kv_stride_d,
    q_stride_h,
    q_stride_d,
    k_stride_h,
    k_stride_d,
    mask_stride,
    output_stride_t,
    output_stride_h,
    output_stride_d,
    eps,
    clamp_value,
    DIM: tl.constexpr,
    HC_MULT: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    HAS_MASK: tl.constexpr,
):
    program_idx = tl.program_id(0)
    token_idx = program_idx // HC_MULT
    hc_idx = program_idx % HC_MULT
    token_idx = token_idx.to(tl.int64)
    source_idx = token_idx
    source_valid = source_idx < num_kv_tokens

    dim_offsets = tl.arange(0, BLOCK_SIZE)
    dim_valid = dim_offsets < DIM
    hidden = tl.load(
        hidden_states
        + token_idx * hidden_stride_t
        + hc_idx * hidden_stride_h
        + dim_offsets * hidden_stride_d,
        mask=dim_valid,
        other=0.0,
    ).to(tl.float32)
    key = tl.load(
        kv + source_idx * kv_stride_t + (hc_idx * DIM + dim_offsets) * kv_stride_d,
        mask=source_valid & dim_valid,
        other=0.0,
    ).to(tl.float32)
    q = tl.load(
        q_weight + hc_idx * q_stride_h + dim_offsets * q_stride_d,
        mask=dim_valid,
        other=0.0,
    ).to(tl.float32)
    k = tl.load(
        k_weight + hc_idx * k_stride_h + dim_offsets * k_stride_d,
        mask=dim_valid,
        other=0.0,
    ).to(tl.float32)

    hidden_rms = tl.rsqrt(tl.sum(hidden * hidden, axis=0) / DIM + eps)
    key_rms = tl.rsqrt(tl.sum(key * key, axis=0) / DIM + eps)
    dot = tl.sum(hidden * q * k * key, axis=0)
    dot *= hidden_rms * key_rms * tl.rsqrt(DIM * 1.0)
    gate_input = tl.sqrt(tl.maximum(tl.abs(dot), clamp_value))
    gate_input = tl.where(dot < 0.0, -gate_input, gate_input)
    gate = tl.sigmoid(gate_input)
    if HAS_MASK:
        active = tl.load(
            token_mask + source_idx * mask_stride,
            mask=source_valid,
            other=0,
        )
        gate = tl.where(active, gate, 0.0)

    value = tl.load(
        kv + source_idx * kv_stride_t + (HC_MULT * DIM + dim_offsets) * kv_stride_d,
        mask=source_valid & dim_valid,
        other=0.0,
    ).to(tl.float32)
    tl.store(
        output
        + token_idx * output_stride_t
        + hc_idx * output_stride_h
        + dim_offsets * output_stride_d,
        hidden + gate * value,
        mask=dim_valid,
    )


class Engram(nn.Module):
    """Writes an n-gram lookup into the residual stream, gated by how well it
    matches that stream.

    The hash ids fetch `n_hash_cols` rows; `wkv` turns them into one key per
    hc copy plus a shared value. The gate is a normalized dot product of the
    stream against the key, signed-sqrt'ed before the sigmoid (matching the
    training kernel).
    """

    def __init__(
        self,
        config,
        quant_config: QuantizationConfig | None,
        layout: EngramLayout,
        layer_hash_index: int,
        use_sequence_parallel: bool,
        prefix: str,
    ) -> None:
        super().__init__()
        self.layer_hash_index = layer_hash_index
        self.dim = config.hidden_size
        self.hc_mult = config.hc_mult
        self.eps = config.rms_norm_eps
        self.clamp_value = 1e-6
        self.use_sequence_parallel = use_sequence_parallel

        # Named ``embed_tokens`` so the checkpoint's ``engram.embed.weight``
        # survives the mapper's ``embed.weight`` -> ``embed_tokens.weight``
        # suffix rule.
        engram_config = get_current_vllm_config().engram_config
        self.embed_tokens = ParallelEngramEmbedding(
            layout.num_embeddings[layer_hash_index],
            layout.head_dim,
            tuple(size for order in layout.primes[layer_hash_index] for size in order),
            cpu_offload=engram_config.cpu_offload if engram_config else True,
            # dsv41 engram-mmap: "pinned" / "resident" / "mmap".
            table_mode=engram_config.resolved_table_mode if engram_config else None,
        )
        n_hash_cols = (layout.max_ngram_size - 1) * layout.n_heads
        self.wkv = ReplicatedLinear(
            n_hash_cols * layout.head_dim,
            self.dim * (self.hc_mult + 1),
            bias=False,
            quant_config=quant_config,
            return_bias=False,
            prefix=f"{prefix}.wkv",
        )
        self.q_weight = nn.Parameter(
            torch.empty(self.hc_mult, self.dim, dtype=torch.bfloat16),
            requires_grad=False,
        )
        self.k_weight = nn.Parameter(
            torch.empty(self.hc_mult, self.dim, dtype=torch.bfloat16),
            requires_grad=False,
        )

        max_tokens = get_current_vllm_config().scheduler_config.max_num_batched_tokens
        # Keep lookup results alive across breakable graph segments.
        self.staged_rows = torch.empty(
            max_tokens,
            self.embed_tokens.part_n_hash_cols,
            layout.head_dim,
            dtype=torch.bfloat16,
        )
        if self.embed_tokens.table_mode == "mmap":
            # dsv41 engram-mmap: pinned + device staging for the host gather.
            self.embed_tokens.configure_mmap_staging(max_tokens)

    def prepare_embeddings(self, hash_ids: torch.Tensor) -> None:
        """Gather this layer's rows on the main stream before decoder layers."""
        self.embed_tokens.lookup(hash_ids, self.staged_rows[: hash_ids.shape[0]])

    def embed(self, hash_ids: torch.Tensor) -> torch.Tensor:
        """Gather heads, returning only local tokens when SP is enabled."""
        rows = self.staged_rows[: hash_ids.shape[0]]
        if self.embed_tokens.tp_size == 1:
            return rows
        rows = tensor_model_parallel_all_gather(rows, dim=1)
        rows = rows[:, : self.embed_tokens.n_hash_cols]
        return sp_shard(rows) if self.use_sequence_parallel else rows

    def forward(
        self,
        hidden_states: torch.Tensor,
        hash_ids: torch.Tensor,
        token_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """hidden_states: [T, hc_mult, dim]; hash_ids: [T, n_hash_cols] (all
        tokens, pre sequence-parallel shard); token_mask: [T], False shuts
        the gate so those positions pass through untouched."""
        kv = self.wkv(self.embed(hash_ids).flatten(-2))
        num_kv_tokens = hash_ids.shape[0]
        assert token_mask is None or token_mask.shape == (num_kv_tokens,)
        if self.use_sequence_parallel:
            tp_size = get_tensor_model_parallel_world_size()
            tp_rank = get_tensor_model_parallel_rank()
            shard_size = (num_kv_tokens + tp_size - 1) // tp_size
            assert hidden_states.shape[0] == shard_size
            start = min(tp_rank * shard_size, num_kv_tokens)
            num_kv_tokens = min(shard_size, num_kv_tokens - start)
            if token_mask is not None:
                token_mask = token_mask[start : start + num_kv_tokens]

        num_tokens, hc_mult, dim = hidden_states.shape
        assert hc_mult == self.hc_mult and dim == self.dim
        assert kv.ndim == 2 and kv.shape[1] == (hc_mult + 1) * dim
        output = torch.empty_like(hidden_states)
        if num_tokens == 0:
            return output

        block_size = triton.next_power_of_2(dim)
        num_warps = 8 if block_size >= 2048 else 4
        mask = token_mask if token_mask is not None else hidden_states
        _fused_engram_post_wkv_kernel[(num_tokens * hc_mult,)](
            hidden_states,
            kv,
            self.q_weight,
            self.k_weight,
            mask,
            output,
            num_kv_tokens,
            hidden_states.stride(0),
            hidden_states.stride(1),
            hidden_states.stride(2),
            kv.stride(0),
            kv.stride(1),
            self.q_weight.stride(0),
            self.q_weight.stride(1),
            self.k_weight.stride(0),
            self.k_weight.stride(1),
            token_mask.stride(0) if token_mask is not None else 0,
            output.stride(0),
            output.stride(1),
            output.stride(2),
            self.eps,
            self.clamp_value,
            DIM=dim,
            HC_MULT=hc_mult,
            BLOCK_SIZE=block_size,
            HAS_MASK=token_mask is not None,
            num_warps=num_warps,
        )
        return output
