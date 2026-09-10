# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""dsv41 engram-mmap: the mmap table mode must be bit-identical to the
pinned/UVA mode (same kernel, rows gathered on the host instead of read
over UVA), survive breakable cudagraph replay as an eager segment, and be
fed by the safetensors iterator without reading the table."""

import json
import struct

import numpy as np
import pytest
import torch

from vllm.model_executor.model_loader.weight_utils import (
    SafetensorsMmapRef,
    safetensors_mmap_ref,
    safetensors_weights_iterator,
)
from vllm.models.deepseek_v4_1.common import engram as engram_ops
from vllm.models.deepseek_v4_1.common.engram import (
    Engram,
    ParallelEngramEmbedding,
    is_engram_table_weight,
)
from vllm.platforms import current_platform

HEAD_SIZES = (17, 19, 23, 29, 31, 37)
DIM = 256
WEIGHT_NAME = "layers.1.engram.embed.weight"
SCALE_NAME = "layers.1.engram.embed.scale"


def _write_safetensors(
    path, tensors: dict[str, tuple[str, tuple[int, ...], bytes]]
) -> None:
    """Minimal safetensors writer (dtype tag, shape, raw bytes), so the test
    does not depend on the installed safetensors' e8m0 support."""
    header, blobs, offset = {}, [], 0
    for name, (dtype, shape, blob) in tensors.items():
        header[name] = {
            "dtype": dtype,
            "shape": list(shape),
            "data_offsets": [offset, offset + len(blob)],
        }
        offset += len(blob)
        blobs.append(blob)
    encoded = json.dumps(header).encode()
    encoded += b" " * ((8 - len(encoded) % 8) % 8)
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(encoded)))
        f.write(encoded)
        for blob in blobs:
            f.write(blob)


def _random_table(num_rows: int):
    torch.manual_seed(0)
    weight = (torch.randn(num_rows, DIM) * 4).to(torch.float8_e4m3fn)
    scales = torch.randint(120, 134, (num_rows, DIM // 32), dtype=torch.uint8)
    return weight, scales


@pytest.fixture
def table_file(tmp_path):
    num_rows = sum(HEAD_SIZES) + 7
    weight, scales = _random_table(num_rows)
    path = tmp_path / "model-00047-of-00048.safetensors"
    _write_safetensors(
        path,
        {
            WEIGHT_NAME: (
                "F8_E4M3",
                tuple(weight.shape),
                weight.view(torch.uint8).numpy().tobytes(),
            ),
            SCALE_NAME: ("F8_E8M0", tuple(scales.shape), scales.numpy().tobytes()),
            "layers.1.engram.q_weight": (
                "BF16",
                (4, 8),
                torch.zeros(4, 8, dtype=torch.bfloat16)
                .view(torch.uint8)
                .numpy()
                .tobytes(),
            ),
        },
    )
    return str(path), weight, scales


def _make_layer(table_mode, num_rows, tp_size, rank, monkeypatch):
    monkeypatch.setattr(
        engram_ops, "get_tensor_model_parallel_world_size", lambda: tp_size
    )
    monkeypatch.setattr(engram_ops, "get_tensor_model_parallel_rank", lambda: rank)
    with torch.device("cuda"):
        return ParallelEngramEmbedding(num_rows, DIM, HEAD_SIZES, table_mode=table_mode)


def test_lazy_iterator_yields_refs_for_tables_only(table_file):
    path, weight, scales = table_file
    seen = dict(
        safetensors_weights_iterator(
            [path], use_tqdm_on_load=False, lazy_mmap_names=is_engram_table_weight
        )
    )
    assert isinstance(seen[WEIGHT_NAME], SafetensorsMmapRef)
    assert isinstance(seen[SCALE_NAME], SafetensorsMmapRef)
    assert isinstance(seen["layers.1.engram.q_weight"], torch.Tensor)
    ref = seen[WEIGHT_NAME]
    assert ref.shape == tuple(weight.shape)
    assert ref.dtype == torch.float8_e4m3fn
    assert ref.nbytes == weight.numel()
    assert ref == safetensors_mmap_ref(path, WEIGHT_NAME)
    with open(path, "rb") as f:
        f.seek(ref.data_begin)
        first_row = np.frombuffer(f.read(DIM), dtype=np.uint8)
    assert np.array_equal(first_row, weight[0].view(torch.uint8).numpy())
    assert is_engram_table_weight("model.layers.14.engram.embed.scale")
    assert not is_engram_table_weight("layers.1.engram.wkv.weight")


@pytest.mark.skipif(not current_platform.is_cuda(), reason="CUDA required")
@pytest.mark.parametrize("tp_size", [1, 2, 4])
@pytest.mark.parametrize("num_tokens", [1, 5, 700])
def test_engram_mmap_matches_pinned(table_file, tp_size, num_tokens, monkeypatch):
    """Same rows, same kernel: mmap and pinned outputs are bit-identical on
    every rank, including rows other ranks own (zero) and TP head padding.
    700 tokens x 6 heads exercises the threaded host gather."""
    path, weight, scales = table_file
    num_rows = weight.shape[0]
    torch.manual_seed(1)
    # Real hash ids of head c always fall in head c's bucket range; keep that
    # (so every rank owns something) but scatter the first token over the
    # whole table so out-of-range rows are exercised too.
    ids = torch.empty(num_tokens, len(HEAD_SIZES), dtype=torch.int32, device="cuda")
    start = 0
    for head, size in enumerate(HEAD_SIZES):
        ids[:, head].random_(start, start + size)
        start += size
    if num_tokens > 1:
        ids[0].random_(0, num_rows)
    for rank in range(tp_size):
        pinned = _make_layer("pinned", num_rows, tp_size, rank, monkeypatch)
        pinned.weight.weight_loader(pinned.weight, weight)
        pinned.weight_scale_inv.weight_loader(
            pinned.weight_scale_inv, scales.view(torch.float8_e8m0fnu)
        )
        mapped = _make_layer("mmap", num_rows, tp_size, rank, monkeypatch)
        assert mapped.weight.numel() == 0 and mapped.weight_scale_inv.numel() == 0
        mapped.weight.weight_loader(
            mapped.weight, safetensors_mmap_ref(path, WEIGHT_NAME)
        )
        mapped.weight_scale_inv.weight_loader(
            mapped.weight_scale_inv, safetensors_mmap_ref(path, SCALE_NAME)
        )
        assert mapped.part_num_embeddings == pinned.part_num_embeddings
        shape = (num_tokens, pinned.part_n_hash_cols, DIM)
        expected = torch.full(shape, 7.0, dtype=torch.bfloat16, device="cuda")
        actual = torch.full(shape, 7.0, dtype=torch.bfloat16, device="cuda")
        pinned.lookup(ids, expected)
        mapped.lookup(ids, actual)
        torch.cuda.synchronize()
        assert torch.equal(actual, expected), f"rank {rank} of {tp_size}"
        # Something must be non-zero for the comparison to mean anything
        # (a TP=4 rank with only padded heads owns no rows at all).
        assert (torch.count_nonzero(expected) > 0) == (pinned.part_num_embeddings > 0)
        if tp_size == 1:  # forward() all-gathers above that, needs a TP group
            assert torch.equal(mapped(ids), pinned(ids))


@pytest.mark.skipif(not current_platform.is_cuda(), reason="CUDA required")
def test_engram_mmap_dummy_load_uses_anonymous_zero_table(monkeypatch):
    """`--load-format dummy` never attaches a file; the gather path must
    still run (anonymous zero table) and produce zeros."""
    num_rows = sum(HEAD_SIZES) + 7
    layer = _make_layer("mmap", num_rows, 1, 0, monkeypatch)
    ids = torch.randint(0, num_rows, (9, len(HEAD_SIZES)), dtype=torch.int32)
    out = torch.full((9, len(HEAD_SIZES), DIM), 3.0, dtype=torch.bfloat16)
    layer.lookup(ids.cuda(), out.cuda())
    torch.cuda.synchronize()
    assert layer._mmap_tables["weight"].source == "anonymous"


@pytest.mark.skipif(not current_platform.is_cuda(), reason="CUDA required")
def test_engram_mmap_eager_break_survives_replay(table_file, monkeypatch):
    """Inside a breakable capture the host gather becomes an eager segment
    reading the weak-referenced ids; replays with fresh ids must match the
    pinned path computed eagerly."""
    from vllm.compilation.breakable_cudagraph import BreakableCUDAGraphCapture

    path, weight, scales = table_file
    num_rows = weight.shape[0]
    pinned = _make_layer("pinned", num_rows, 1, 0, monkeypatch)
    pinned.weight.weight_loader(pinned.weight, weight)
    pinned.weight_scale_inv.weight_loader(
        pinned.weight_scale_inv, scales.view(torch.float8_e8m0fnu)
    )
    mapped = _make_layer("mmap", num_rows, 1, 0, monkeypatch)
    mapped.weight.weight_loader(mapped.weight, safetensors_mmap_ref(path, WEIGHT_NAME))
    mapped.weight_scale_inv.weight_loader(
        mapped.weight_scale_inv, safetensors_mmap_ref(path, SCALE_NAME)
    )
    num_tokens, cols = 64, len(HEAD_SIZES)
    engram = Engram.__new__(Engram)
    torch.nn.Module.__init__(engram)
    engram.embed_tokens = mapped
    engram.use_sequence_parallel = False
    engram.staged_rows = torch.empty(
        num_tokens, cols, DIM, dtype=torch.bfloat16, device="cuda"
    )
    mapped.configure_mmap_staging(num_tokens)
    # Non-contiguous per-layer slice, like the model's hash tensor.
    hashes = torch.randint(
        0, num_rows, (num_tokens, 2, cols), dtype=torch.int32, device="cuda"
    )
    src = hashes[:, 1]
    out = torch.empty(num_tokens, cols, DIM, dtype=torch.bfloat16, device="cuda")

    def step():
        engram.prepare_embeddings(src)
        out.copy_(engram.embed(src))

    warmup = torch.cuda.Stream()
    warmup.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(warmup):
        step()
    torch.cuda.current_stream().wait_stream(warmup)
    graph = BreakableCUDAGraphCapture()
    with torch.cuda.stream(warmup), graph:
        step()
    torch.cuda.current_stream().wait_stream(warmup)
    assert graph.num_eager_breaks == 1 and graph.num_graphs == 2

    expected = torch.empty_like(out)
    for _ in range(3):
        hashes.random_(0, num_rows)
        graph.replay()
        pinned.lookup(src, expected)
        torch.cuda.synchronize()
        assert torch.equal(out, expected)
        assert torch.count_nonzero(expected) > 0


@pytest.mark.skipif(not current_platform.is_cuda(), reason="CUDA required")
def test_engram_mmap_plain_capture_needs_prefetch(monkeypatch):
    """A plain (non-breakable) capture, as the V2 runner's FULL graphs use,
    is refused unless the runner promised to stage the rows itself; then
    the captured lookup is a no-op and the graph reads the static buffer."""
    num_rows = sum(HEAD_SIZES) + 7
    layer = _make_layer("mmap", num_rows, 1, 0, monkeypatch)
    ids = torch.randint(0, num_rows, (4, len(HEAD_SIZES)), dtype=torch.int32).cuda()
    out = torch.full((4, len(HEAD_SIZES), DIM), 3.0, dtype=torch.bfloat16).cuda()
    layer.lookup(ids, out)  # warm-up outside capture (anonymous table)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with (
        pytest.raises(RuntimeError, match="plain cudagraph capture"),
        torch.cuda.graph(graph),
    ):
        layer.lookup(ids, out)
    layer.full_graph_prefetch = True
    out.fill_(5.0)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        layer.lookup(ids, out)
    graph.replay()
    torch.cuda.synchronize()
    assert torch.all(out == 5.0)  # untouched: the runner's prefetch fills it
