#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""dsv41 engram-mmap: check the mmap table mode against a REAL checkpoint shard.

Opens one engram shard (`model-00047-of-00048.safetensors` holds
`layers.1.engram.embed.{weight,scale}`, shard 48 holds layer 14) in mmap mode
exactly like the model does (this rank's row range of the table, lazy
safetensors ref -> `ParallelEngramEmbedding` weight loader), gathers random
rows through the host path and compares them, bit for bit, with a direct
safetensors slice read of the same rows dequantised with plain torch on the
GPU. Then times cold (page cache dropped with posix_fadvise DONTNEED) and
warm gathers at prefill and decode sizes to put a number on the NVMe cost.

Run inside the serving image with the worktree overlay, e.g. on x299:

    podman run --rm --entrypoint bash --device nvidia.com/gpu=all \\
      -e CUDA_VISIBLE_DEVICES=0 -v ~/dsv41-test/src:/src:ro \\
      -v ~/.cache/huggingface/hub:/hf:ro -w /src <image> -c '<overlay prelude>;
      python3 tools/dsv41/engram_mmap_check.py --shard /hf/models--deepseek-ai--\\
      DeepSeek-V4.1-Flash/snapshots/<rev>/model-00047-of-00048.safetensors'

No model, no tokenizer, no distributed init: TP is emulated by patching the
rank accessors, like tests/kernels/test_engram.py does.
"""

import argparse
import json
import os
import time
from types import SimpleNamespace

import numpy as np
import torch
from safetensors import safe_open

from vllm.model_executor.model_loader.weight_utils import (
    read_safetensors_header,
    safetensors_mmap_ref,
)
from vllm.models.deepseek_v4_1.common import engram as engram_ops
from vllm.models.deepseek_v4_1.common.engram import (
    EngramLayout,
    ParallelEngramEmbedding,
)


def find_engram_config(config: dict) -> SimpleNamespace:
    """The (possibly nested) HF config dict carrying the engram fields."""
    stack = [config]
    while stack:
        node = stack.pop()
        if "engram_layer_ids" in node:
            return SimpleNamespace(
                **{k: v for k, v in node.items() if k.startswith("engram")}
            )
        stack.extend(v for v in node.values() if isinstance(v, dict))
    raise SystemExit("config.json has no engram_layer_ids")


def dequant_reference(weight_u8: torch.Tensor, scale_u8: torch.Tensor, block: int):
    """The torch expression `_engram_lookup_kernel` is bit-exact with
    (tests/kernels/test_engram.py::_reference_lookup, minus the masking)."""
    values = weight_u8.view(torch.float8_e4m3fn).float()
    scales = (scale_u8.to(torch.int32) << 23).view(torch.float32)
    values = values.unflatten(-1, (-1, block)) * scales.unsqueeze(-1)
    return values.flatten(-2).to(torch.bfloat16)


def drop_page_cache(path: str) -> None:
    fd = os.open(path, os.O_RDONLY)
    try:
        os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
    finally:
        os.close(fd)


def timed_gather(layer, ids, out, label, cold_path=None):
    if cold_path is not None:
        drop_page_cache(cold_path)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    layer._stage_from_mmap_eager(ids, out)
    torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    rows = ids.shape[0] * layer.part_n_hash_cols
    print(
        f"  {label:<34} {rows:>7d} rows  {1e3 * dt:9.2f} ms  "
        f"{1e6 * dt / rows:8.2f} us/row"
    )
    return dt


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--shard", required=True, help="engram safetensors shard")
    ap.add_argument("--config", help="config.json (default: next to the shard)")
    ap.add_argument("--tp-size", type=int, default=1)
    ap.add_argument("--rank", type=int, default=0)
    ap.add_argument("--rows", type=int, default=2000, help="rows for the bit check")
    ap.add_argument("--tokens", type=int, default=4096, help="prefill-sized timing")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--threads", type=int, help="override the host gather pool size")
    ap.add_argument("--no-cold", action="store_true", help="skip the cold timings")
    args = ap.parse_args()

    if args.threads is not None:
        engram_ops._MMAP_GATHER_THREADS = args.threads
    config_path = args.config or os.path.join(
        os.path.dirname(args.shard), "config.json"
    )
    with open(config_path) as f:
        layout = EngramLayout(find_engram_config(json.load(f)))
    header, _ = read_safetensors_header(args.shard)
    weight_name = next(n for n in header if n.endswith("engram.embed.weight"))
    scale_name = weight_name[: -len("weight")] + "scale"
    layer_id = int(weight_name.split(".")[1])
    layer_index = layout.layer_ids.index(layer_id)
    head_sizes = tuple(size for order in layout.primes[layer_index] for size in order)

    engram_ops.get_tensor_model_parallel_world_size = lambda: args.tp_size
    engram_ops.get_tensor_model_parallel_rank = lambda: args.rank
    with torch.device("cuda"):
        layer = ParallelEngramEmbedding(
            layout.num_embeddings[layer_index],
            layout.head_dim,
            head_sizes,
            table_mode="mmap",
        )
    weight_ref = safetensors_mmap_ref(args.shard, weight_name)
    scale_ref = safetensors_mmap_ref(args.shard, scale_name)
    layer.weight.weight_loader(layer.weight, weight_ref)
    layer.weight_scale_inv.weight_loader(layer.weight_scale_inv, scale_ref)
    cols = layer.part_n_hash_cols
    print(
        f"layer {layer_id}: table {weight_ref.shape} fp8 + {scale_ref.shape} ue8m0, "
        f"rank {args.rank}/{args.tp_size} owns rows {layer.vocab_start_idx}.."
        f"{layer.vocab_end_idx} ({layer.part_num_embeddings * 264 / 1024**3:.2f} GiB), "
        f"{cols} local hash heads"
    )

    # 1. bit check: random owned rows through the host path vs a slice read.
    rng = np.random.default_rng(args.seed)
    num_tokens = -(-args.rows // cols)
    ids_np = rng.integers(
        layer.vocab_start_idx, layer.vocab_end_idx, (num_tokens, cols)
    )
    ids = torch.tensor(ids_np, dtype=torch.int32, device="cuda")
    out = torch.full(
        (num_tokens, cols, layer.dim), 7.0, dtype=torch.bfloat16, device="cuda"
    )
    layer.lookup(ids, out)
    torch.cuda.synchronize()
    unique_rows = np.unique(ids_np.reshape(-1))
    with safe_open(args.shard, framework="pt") as f:
        weight_slice = f.get_slice(weight_name)
        scale_slice = f.get_slice(scale_name)
        ref_weight = torch.cat([weight_slice[int(r) : int(r) + 1] for r in unique_rows])
        ref_scale = torch.cat([scale_slice[int(r) : int(r) + 1] for r in unique_rows])
    ref_rows = dequant_reference(
        ref_weight.view(torch.uint8).cuda(),
        ref_scale.view(torch.uint8).cuda(),
        layer.block_size,
    )
    position = {int(r): i for i, r in enumerate(unique_rows)}
    gather = torch.tensor([position[int(r)] for r in ids_np.reshape(-1)], device="cuda")
    expected = ref_rows[gather].view(num_tokens, cols, layer.dim)
    identical = torch.equal(out, expected)
    nonzero = int(torch.count_nonzero(expected))
    print(
        f"bit check: {num_tokens * cols} rows ({len(unique_rows)} unique): "
        f"{'IDENTICAL' if identical else 'MISMATCH'} "
        f"({nonzero} non-zero elements, max |x| = {expected.float().abs().max():.4g})"
    )
    if not identical:
        diff = (out != expected).any(-1)
        print("  mismatching (token, head):", diff.nonzero().tolist()[:10])
        raise SystemExit(1)

    # 2. cold vs warm timings.
    print(f"timings (gather pool {engram_ops._MMAP_GATHER_THREADS} threads):")
    cold = None if args.no_cold else args.shard
    for tokens in (args.tokens, 8, 1):
        ids = torch.tensor(
            rng.integers(layer.vocab_start_idx, layer.vocab_end_idx, (tokens, cols)),
            dtype=torch.int32,
            device="cuda",
        )
        out = torch.empty(tokens, cols, layer.dim, dtype=torch.bfloat16, device="cuda")
        layer.configure_mmap_staging(tokens)
        timed_gather(layer, ids, out, f"{tokens} tokens cold (cache dropped)", cold)
        timed_gather(layer, ids, out, f"{tokens} tokens warm (same rows)")
        fresh = torch.tensor(
            rng.integers(layer.vocab_start_idx, layer.vocab_end_idx, (tokens, cols)),
            dtype=torch.int32,
            device="cuda",
        )
        timed_gather(layer, fresh, out, f"{tokens} tokens fresh rows (no drop)")


if __name__ == "__main__":
    main()
