# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""dsv41 engram-mmap: the mmap table mode must be bit-identical to the
pinned/UVA mode (same kernel, rows gathered on the host instead of read
over UVA), survive breakable cudagraph replay as an eager segment, and be
fed by the safetensors iterator without reading the table."""

import ctypes
import json
import mmap
import os
import struct

import numpy as np
import pytest
import torch

from vllm.model_executor.model_loader.weight_utils import (
    SafetensorsMmapRef,
    WeightPageCacheDropper,
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


# ---- dsv41 engram-warm ---------------------------------------------------

WARM_HEAD_SIZES = (250_000,) * 4  # 1M rows: 256 MiB of fp8 + 8 MiB of scales


@pytest.fixture(scope="module")
def big_table_file(tmp_path_factory):
    """A few hundred MB table so the warm walks several 64 MiB steps."""
    num_rows = sum(WARM_HEAD_SIZES) + 3
    gen = torch.Generator().manual_seed(2)
    # Finite e4m3 bit patterns only (0x7f / 0xff are NaN).
    weight = torch.randint(0, 0x7F, (num_rows, DIM), dtype=torch.uint8, generator=gen)
    scales = torch.randint(
        120, 134, (num_rows, DIM // 32), dtype=torch.uint8, generator=gen
    )
    path = tmp_path_factory.mktemp("warm") / "model-00048-of-00048.safetensors"
    _write_safetensors(
        path,
        {
            WEIGHT_NAME: ("F8_E4M3", tuple(weight.shape), weight.numpy().tobytes()),
            SCALE_NAME: ("F8_E8M0", tuple(scales.shape), scales.numpy().tobytes()),
        },
    )
    return str(path), weight.view(torch.float8_e4m3fn), scales


def _make_clean(path: str) -> None:
    """Rewrite a file the test just wrote through O_DIRECT, so no dirty page
    of it is left in the page cache: the test container's overlayfs is
    mounted volatile, where fsync is a no-op and DONTNEED keeps the dirty
    pages a buffered write leaves behind (eviction then depends on the
    background writeback timing, which made the warm test flaky)."""
    with open(path, "rb") as f:
        data = f.read()
    size, page = len(data), mmap.PAGESIZE
    padded = -(-size // page) * page
    buf = mmap.mmap(-1, padded)  # page aligned, as O_DIRECT requires
    buf[:size] = data
    fd = os.open(path, os.O_WRONLY | os.O_TRUNC | os.O_DIRECT)
    try:
        written = os.write(fd, memoryview(buf))
        assert written == padded, (written, padded)
        os.ftruncate(fd, size)
    finally:
        os.close(fd)
        buf.close()


def _drop_page_cache(path: str) -> None:
    _make_clean(path)
    fd = os.open(path, os.O_RDONLY)
    try:
        os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
    finally:
        os.close(fd)


def _resident_pages(table) -> tuple[int, int]:
    """(resident, total) pages of the table's mapping, via mincore(2)."""
    page = mmap.PAGESIZE
    addr = table.rows.ctypes.data
    addr -= addr % page
    total = -(-table.mapped_bytes // page)
    vec = (ctypes.c_ubyte * total)()
    libc = ctypes.CDLL(None, use_errno=True)
    rc = libc.mincore(ctypes.c_void_p(addr), ctypes.c_size_t(total * page), vec)
    assert rc == 0, os.strerror(ctypes.get_errno())
    return sum(b & 1 for b in vec), total


def _attach(layer, path):
    layer.weight.weight_loader(layer.weight, safetensors_mmap_ref(path, WEIGHT_NAME))
    layer.weight_scale_inv.weight_loader(
        layer.weight_scale_inv, safetensors_mmap_ref(path, SCALE_NAME)
    )


@pytest.mark.skipif(not current_platform.is_cuda(), reason="CUDA required")
@pytest.mark.parametrize("mode", ["sync", "async"])
def test_engram_mmap_warm(big_table_file, mode, monkeypatch):
    """The warm reads every page of the rank's slices of both tables into
    the page cache (mincore), in both modes, and the gather afterwards is
    bit-identical to the unwarmed mmap path and to the pinned path."""
    path, weight, scales = big_table_file
    num_rows = weight.shape[0]
    tp_size, rank = 2, 1  # rank 1: the slice starts mid-file (page slack)

    def make(table_mode):
        monkeypatch.setattr(
            engram_ops, "get_tensor_model_parallel_world_size", lambda: tp_size
        )
        monkeypatch.setattr(engram_ops, "get_tensor_model_parallel_rank", lambda: rank)
        with torch.device("cuda"):
            return ParallelEngramEmbedding(
                num_rows, DIM, WARM_HEAD_SIZES, table_mode=table_mode
            )

    # Evict the file before anything maps it (DONTNEED skips mapped pages).
    _drop_page_cache(path)
    cold = make("mmap")
    _attach(cold, path)
    assert cold.mmap_warm == "none"
    assert cold.warm_mmap() is None and cold.mmap_warm_stats is None

    warm = make("mmap")
    _attach(warm, path)
    warm.mmap_warm = mode
    tables = [warm._mmap_tables["weight"], warm._mmap_tables["scale"]]
    expected_bytes = sum(t.mapped_bytes for t in tables)
    assert expected_bytes >= warm.part_num_embeddings * (DIM + DIM // 32)
    assert expected_bytes < warm.part_num_embeddings * (DIM + DIM // 32) + 2 * 4096
    before = sum(_resident_pages(t)[0] for t in tables)
    thread = warm.warm_mmap()
    assert thread is not None
    if mode == "sync":
        assert not thread.is_alive()
    else:
        thread.join(120)
        assert not thread.is_alive()
    assert warm.warm_mmap() is thread  # idempotent
    assert warm.mmap_warm_stats is not None
    assert warm.mmap_warm_stats["bytes"] == expected_bytes
    assert warm.mmap_warm_stats["seconds"] > 0
    resident = [_resident_pages(t) for t in tables]
    assert all(r == n for r, n in resident), (before, resident)
    assert before < sum(n for _, n in resident), f"nothing evicted: {before}"
    # The final pass (end of init) is a separate, idempotent thread that
    # reads the same bytes again and keeps its own stats.
    assert warm.mmap_warm_final_stats is None
    final = warm.warm_mmap(final=True)
    assert final is not None and final is not thread
    final.join(120)
    assert not final.is_alive()
    assert warm.warm_mmap(final=True) is final
    assert warm.warm_mmap() is thread
    assert warm.mmap_warm_final_stats is not None
    assert warm.mmap_warm_final_stats["bytes"] == expected_bytes
    assert all(_resident_pages(t) == (n, n) for t, (_, n) in zip(tables, resident))

    pinned = make("pinned")
    pinned.weight.weight_loader(pinned.weight, weight)
    pinned.weight_scale_inv.weight_loader(
        pinned.weight_scale_inv, scales.view(torch.float8_e8m0fnu)
    )
    num_tokens = 700
    torch.manual_seed(3)
    ids = torch.randint(
        0, num_rows, (num_tokens, len(WARM_HEAD_SIZES)), dtype=torch.int32
    )
    ids = ids.cuda()
    shape = (num_tokens, warm.part_n_hash_cols, DIM)
    outs = []
    for layer in (cold, warm, pinned):
        out = torch.full(shape, 7.0, dtype=torch.bfloat16, device="cuda")
        layer.lookup(ids, out)
        torch.cuda.synchronize()
        outs.append(out)
    assert torch.equal(outs[0], outs[1]) and torch.equal(outs[1], outs[2])
    assert torch.count_nonzero(outs[1]) > 0
    for layer in (cold, warm):  # unmap so the next run can evict the file
        for table in layer._mmap_tables.values():
            table.close()


@pytest.mark.skipif(not current_platform.is_cuda(), reason="CUDA required")
def test_engram_mmap_warm_skips_anonymous_tables(monkeypatch):
    """A dummy load has no file to read: the warm is a logged no-op both
    before and after the anonymous tables materialise."""
    num_rows = sum(HEAD_SIZES) + 7
    layer = _make_layer("mmap", num_rows, 1, 0, monkeypatch)
    layer.mmap_warm = "sync"
    assert layer.warm_mmap() is None
    ids = torch.randint(0, num_rows, (3, len(HEAD_SIZES)), dtype=torch.int32).cuda()
    out = torch.empty((3, len(HEAD_SIZES), DIM), dtype=torch.bfloat16).cuda()
    layer.lookup(ids, out)
    torch.cuda.synchronize()
    assert layer._mmap_tables["weight"].source == "anonymous"
    assert layer._mmap_tables["weight"].warm() == 0
    assert layer.warm_mmap() is None and layer.mmap_warm_stats is None
    assert layer.warm_mmap(final=True) is None
    pinned = _make_layer("pinned", num_rows, 1, 0, monkeypatch)
    pinned.mmap_warm = "sync"
    assert pinned.warm_mmap() is None  # other table modes: nothing to warm
    assert pinned.warm_mmap(final=True) is None


# ---- dsv41 engram-warm: weight page-cache drop ---------------------------

WEIGHT_TENSOR_NAME = "layers.0.attn.wq_a.weight"


@pytest.fixture
def weight_file(tmp_path):
    """A 64 MiB ordinary weight shard, sorted before the table shard."""
    path = tmp_path / "model-00001-of-00048.safetensors"
    blob = torch.ones(32 << 20, dtype=torch.bfloat16).view(torch.uint8).numpy()
    _write_safetensors(
        path, {WEIGHT_TENSOR_NAME: ("BF16", (32 << 20,), blob.tobytes())}
    )
    return str(path)


def _make_clean_and_resident(path: str) -> None:
    """Clean pages only (see `_make_clean`), then read every byte so the
    whole file is in the page cache."""
    _make_clean(path)
    with open(path, "rb") as f:
        while f.read(1 << 20):
            pass


def _file_resident_pages(path: str) -> tuple[int, int]:
    """(resident, total) pages of a file via a throwaway mapping and
    mincore(2); unmapped again on return so a later DONTNEED can evict."""
    page = mmap.PAGESIZE
    total = -(-os.path.getsize(path) // page)
    fd = os.open(path, os.O_RDONLY)
    try:
        mapping = mmap.mmap(fd, 0, flags=mmap.MAP_SHARED, prot=mmap.PROT_READ)
    finally:
        os.close(fd)
    try:
        view = np.frombuffer(mapping, dtype=np.uint8)
        addr = view.ctypes.data
        del view  # release the buffer export so the mapping can close
        vec = (ctypes.c_ubyte * total)()
        libc = ctypes.CDLL(None, use_errno=True)
        rc = libc.mincore(ctypes.c_void_p(addr), ctypes.c_size_t(total * page), vec)
        assert rc == 0, os.strerror(ctypes.get_errno())
        return sum(b & 1 for b in vec), total
    finally:
        mapping.close()


def test_weight_page_cache_drop(weight_file, tmp_path):
    """`WeightPageCacheDropper.drop` evicts every resident page of a clean,
    unmapped file and reports its size; a missing file is a 0-byte no-op."""
    _make_clean_and_resident(weight_file)
    resident, total = _file_resident_pages(weight_file)
    assert resident == total and total > 16000
    assert WeightPageCacheDropper.drop(weight_file) == os.path.getsize(weight_file)
    resident, total = _file_resident_pages(weight_file)
    assert resident < total // 100, (resident, total)
    assert WeightPageCacheDropper.drop(str(tmp_path / "missing.safetensors")) == 0


def test_weight_page_cache_drop_skips_engram_shard(weight_file, table_file):
    """Streaming a weight shard and the engram table shard through the
    iterator drops the weight shard's pages (one file behind, once its
    tensors are consumed) and never the table shard's; `finish` counts."""
    table_path = table_file[0]
    for path in (weight_file, table_path):
        _make_clean_and_resident(path)
    dropper = WeightPageCacheDropper()
    seen = []
    for name, loaded in safetensors_weights_iterator(
        [weight_file, table_path],
        use_tqdm_on_load=False,
        lazy_mmap_names=is_engram_table_weight,
        page_cache_drop=dropper,
    ):
        seen.append(name)
        if isinstance(loaded, torch.Tensor):
            loaded.sum()  # fault the pages in through the private mapping
        del loaded
    assert seen[0] == WEIGHT_TENSOR_NAME  # the weight shard sorts first
    assert set(seen[1:]) == {WEIGHT_NAME, SCALE_NAME, "layers.1.engram.q_weight"}
    assert dropper.files == {weight_file: False, table_path: True}
    # The lagged per-file drop ran when the table shard finished.
    resident, total = _file_resident_pages(weight_file)
    assert resident < total // 100, (resident, total)
    resident, total = _file_resident_pages(table_path)
    assert resident == total, (resident, total)
    dropper.finish()
    assert (dropper.dropped_files, dropper.kept_files) == (1, 1)
    assert dropper.dropped_bytes == os.path.getsize(weight_file)
    resident, total = _file_resident_pages(table_path)
    assert resident == total, "engram shard must never be dropped"
