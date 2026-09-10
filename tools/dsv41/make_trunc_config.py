#!/usr/bin/env python3
"""Write a truncated DeepSeek-V4.1-Flash model directory for dummy-weight boots.

Only ``json`` / ``argparse`` are needed. The output directory gets a
``config.json`` derived from the real one (num_hidden_layers cut to a small N,
fewer experts, tiny engram tables, fewer vision layers) plus the tokenizer files
and a minimal chat template, so ``vllm serve <out> --load-format dummy`` can boot
it on one GPU.

Layer-topology rules enforced here (from vllm/models/deepseek_v4_1/*):

* ``compress_ratios`` has one entry per layer INCLUDING the MTP/DSpark layers
  (attention.py: ``layer_id < len(compress_ratios)``, layers past the list are
  ratio 0). Values must be 0 (pure SWA), 1 (full compressed cache) or 2.
* A layer with ratio > 0 takes ``max(s for s in kv_source_layer_ids if s <= id)``
  as its KV source and the same over ``index_source_layer_ids`` as its index
  source, so the first compressed layer must itself be a kv source AND an
  index source, and both lists must be non-empty when any ratio > 0 exists.
* A kv source owns the compressor, the compressed KV cache and the indexer K
  cache; a non-kv index source borrows the K cache of ``<kv source>.indexer``
  (attention.py ~L375), which only exists when that kv source is also an index
  source. Hence kv_source_layer_ids must be a subset of index_source_layer_ids.
* Consumers read the source's compressed cache (``_compressed_kv_cache``), whose
  ``tokens_per_state`` is the source's ratio, so a consumer's ratio must equal
  its kv source's ratio (a ratio change happens only at a new kv source).
* ``candidate_source_layer_id`` must be an index source (only an indexer writes
  candidates); every later index source masks with those candidate blocks and
  must share the candidate source's ratio (blocks are compressed positions).
* PP cannot split inside a kv-sharing group (source and consumers on one rank).
* Engram (common/engram.py): ``len(engram_layer_ids) == len(engram_num_embeddings)``;
  the injection runs only from the second layer on, so engram ids must be >= 1;
  each layer's table needs ``sum(24 primes drawn upward from engram_vocab_size-1,
  never reused across layers) <= num_embeddings``; ``engram_compressed_vocab_size``
  is asserted against the tokenizer (99092) and must stay as-is.
* DSpark (nvidia/dspark.py): MTP layers are ids ``N + i`` for
  ``i < num_nextn_predict_layers``; their MoE uses ``dspark_n_routed_experts`` /
  ``dspark_num_experts_per_tok``; ``main_proj`` is sized by
  ``len(dspark_target_layer_ids)`` (backbone layers whose hidden states the
  drafter concatenates; the real checkpoint uses the last three).
* MoE: ``topk_method=noaux_tc`` with no ``n_group`` (defaults to 1), so the only
  rule is ``num_experts_per_tok <= n_routed_experts``.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
DEFAULT_CONFIG = HERE / "v41-config.json"
TOKENIZER_FILES = ("tokenizer.json", "tokenizer_config.json", "chat_template.jinja")


# ----------------------------------------------------------------------------
# Layer selection
# ----------------------------------------------------------------------------


def priority_layers(text: dict) -> list[int]:
    """Original layer ids in the order they earn a slot in the truncated model.

    Every prefix of this list is a valid topology (sources precede consumers),
    so ``sorted(priority[:N])`` is the kept set for N layers.
    """
    n = text["num_hidden_layers"]
    ratios = [int(r) for r in text["compress_ratios"][:n]]
    kv_src = [int(s) for s in text.get("kv_source_layer_ids", [])]
    idx_src = [int(s) for s in text.get("index_source_layer_ids", [])]
    cand = int(text.get("candidate_source_layer_id", -1))

    order: list[int] = []

    def add(layer: int) -> None:
        if 0 <= layer < n and layer not in order:
            order.append(layer)

    # 1. the leading pure-SWA layers (0, 1 in the real config)
    for layer in range(n):
        if ratios[layer] != 0:
            break
        add(layer)
    def plain_consumer(src: int) -> int | None:
        nxt = src + 1
        if nxt < n and ratios[nxt] == ratios[src] and nxt not in idx_src:
            return nxt
        return None

    # 2. the first kv source of each ratio; for the first ratio also its plain
    #    consumer, for the candidate source the first non-kv index source after
    #    it (K-cache borrowing + candidate filtering), then its plain consumer
    seen_ratios: set[int] = set()
    for src in kv_src:
        if ratios[src] in seen_ratios:
            continue
        seen_ratios.add(ratios[src])
        add(src)
        if len(seen_ratios) == 1 and plain_consumer(src) is not None:
            add(plain_consumer(src))
        if src == cand:
            for ix in idx_src:
                if ix > cand and ix not in kv_src:
                    add(ix)
                    break
        if plain_consumer(src) is not None:
            add(plain_consumer(src))
    # 3. candidate source if it was not a first-of-ratio kv source
    if cand >= 0:
        add(cand)
        for ix in idx_src:
            if ix > cand and ix not in kv_src:
                add(ix)
                break
    # 4. remaining kv sources (each followed by its plain consumer), then the
    #    remaining index sources, then everything else in order
    for src in kv_src:
        add(src)
        if src + 1 < n and ratios[src + 1] == ratios[src]:
            add(src + 1)
    for src in idx_src:
        add(src)
    for layer in range(n):
        add(layer)
    return order


def remap(ids: list[int], keep: list[int]) -> list[int]:
    pos = {orig: new for new, orig in enumerate(keep)}
    return [pos[i] for i in ids if i in pos]


# ----------------------------------------------------------------------------
# Engram table sizing (mirrors common/engram.py EngramLayout + find_next_prime)
# ----------------------------------------------------------------------------


def _is_prime(x: int) -> bool:
    if x < 2:
        return False
    if x % 2 == 0:
        return x == 2
    f = 3
    while f * f <= x:
        if x % f == 0:
            return False
        f += 2
    return True


def find_next_prime(start: int, seen: set[int]) -> int:
    candidate = start + 1
    while not _is_prime(candidate) or candidate in seen:
        candidate += 1
    return candidate


def engram_table_sizes(vocab_size: int, n_layers: int, max_ngram: int, n_heads: int) -> list[int]:
    """sum(head_sizes) per engram layer exactly as EngramLayout draws them."""
    seen: set[int] = set()
    sums = []
    for _ in range(n_layers):
        total = 0
        for _ in range(max_ngram - 1):
            current = vocab_size - 1
            for _ in range(n_heads):
                current = find_next_prime(current, seen)
                seen.add(current)
                total += current
        sums.append(total)
    return sums


def pick_engram_vocab(
    num_embeddings: int, n_layers: int, max_ngram: int, n_heads: int, preferred: int | None = None
) -> tuple[int, list[int]]:
    """Largest engram_vocab_size whose prime buckets fit in num_embeddings rows."""
    n_cols = (max_ngram - 1) * n_heads
    vocab = max(2, num_embeddings // n_cols)
    if preferred is not None:
        # Keep the checkpoint's vocab when its buckets (almost) fit; the caller
        # pads each table to its exact bucket sum (the real config does too).
        sums = engram_table_sizes(preferred, n_layers, max_ngram, n_heads)
        if max(sums) <= num_embeddings * 1.001:
            return preferred, sums
    while vocab > 2:
        sums = engram_table_sizes(vocab, n_layers, max_ngram, n_heads)
        if max(sums) <= num_embeddings:
            return vocab, sums
        vocab = int(vocab * 0.99) - 1
    raise SystemExit("engram_num_embeddings too small for even a tiny vocab")


# ----------------------------------------------------------------------------
# Validation of the resulting topology (same rules the model code applies)
# ----------------------------------------------------------------------------


def validate(text: dict) -> list[str]:
    n = text["num_hidden_layers"]
    nextn = text.get("num_nextn_predict_layers", 0)
    ratios = [int(r) for r in text["compress_ratios"]]
    kv_src = list(text.get("kv_source_layer_ids", []))
    idx_src = list(text.get("index_source_layer_ids", []))
    cand = int(text.get("candidate_source_layer_id", -1))
    errors = []
    if len(ratios) != n + nextn:
        errors.append(f"compress_ratios has {len(ratios)} entries, expected {n + nextn}")
    if any(r not in (0, 1, 2) for r in ratios):
        errors.append("compress_ratios must be 0, 1 or 2")
    if any(s >= n for s in kv_src + idx_src):
        errors.append("source layer ids must be backbone layers (< num_hidden_layers)")
    if not set(kv_src) <= set(idx_src):
        errors.append("kv_source_layer_ids must be a subset of index_source_layer_ids")
    for layer, r in enumerate(ratios[:n]):
        if r == 0:
            continue
        kv = [s for s in kv_src if s <= layer]
        ix = [s for s in idx_src if s <= layer]
        if not kv or not ix:
            errors.append(f"layer {layer} (ratio {r}) has no kv/index source at or below it")
            continue
        if ratios[max(kv)] != r:
            errors.append(f"layer {layer} ratio {r} != kv source {max(kv)} ratio {ratios[max(kv)]}")
        if ratios[max(ix)] != r:
            errors.append(f"layer {layer} ratio {r} != index source {max(ix)} ratio {ratios[max(ix)]}")
    if cand >= 0:
        if cand not in idx_src:
            errors.append("candidate_source_layer_id must be an index source")
        for s in idx_src:
            if s > cand and ratios[s] != ratios[cand]:
                errors.append(f"index source {s} uses candidates but ratio differs from source {cand}")
    for layer in text.get("engram_layer_ids", []):
        if not 1 <= layer < n:
            errors.append(f"engram layer {layer} must be in [1, num_hidden_layers)")
    if len(text.get("engram_layer_ids", [])) != len(text.get("engram_num_embeddings", [])):
        errors.append("engram_layer_ids / engram_num_embeddings length mismatch")
    for layer in text.get("dspark_target_layer_ids", []):
        if not 0 <= layer < n:
            errors.append(f"dspark target layer {layer} must be a backbone layer")
    if text["num_experts_per_tok"] > text["n_routed_experts"]:
        errors.append("num_experts_per_tok > n_routed_experts")
    if text.get("dspark_num_experts_per_tok", 0) > text.get("dspark_n_routed_experts", 0):
        errors.append("dspark_num_experts_per_tok > dspark_n_routed_experts")
    return errors


# ----------------------------------------------------------------------------
# Layer table
# ----------------------------------------------------------------------------


def layer_table(text: dict, keep: list[int]) -> str:
    n = text["num_hidden_layers"]
    nextn = text.get("num_nextn_predict_layers", 0)
    ratios = text["compress_ratios"]
    kv_src = text.get("kv_source_layer_ids", [])
    idx_src = text.get("index_source_layer_ids", [])
    cand = text.get("candidate_source_layer_id", -1)
    engram = set(text.get("engram_layer_ids", []))
    targets = set(text.get("dspark_target_layer_ids", []))
    rows = [("new", "orig", "ratio", "kv_src", "idx_src", "role", "engram", "dspark")]
    for layer in range(n + nextn):
        r = ratios[layer]
        orig = str(keep[layer]) if layer < n else f"mtp{layer - n}"
        if r > 0 and layer < n:
            kv = max((s for s in kv_src if s <= layer), default=None)
            ix = max((s for s in idx_src if s <= layer), default=None)
        else:
            kv = ix = None
        roles = []
        if layer in kv_src:
            roles.append("kv-source")
        if layer in idx_src:
            roles.append("index-source")
        if layer == cand:
            roles.append("candidate-source")
        elif layer in idx_src and 0 <= cand < layer:
            roles.append("uses-candidates")
        if r == 0 and layer < n:
            roles.append("swa-only")
        if layer >= n:
            roles.append("dspark-layer")
        rows.append(
            (
                str(layer),
                orig,
                str(r),
                "self" if layer in kv_src else ("-" if kv is None else str(kv)),
                "self" if layer in idx_src else ("-" if ix is None else str(ix)),
                ",".join(roles) or "consumer",
                "yes" if layer in engram else "",
                "target" if layer in targets else ("draft" if layer >= n else ""),
            )
        )
    widths = [max(len(row[i]) for row in rows) for i in range(len(rows[0]))]
    return "\n".join("  ".join(cell.ljust(widths[i]) for i, cell in enumerate(row)) for row in rows)


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------


def parse_int_list(value: str | None) -> list[int] | None:
    if value is None:
        return None
    return [int(v) for v in value.split(",") if v.strip()]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--config", type=Path, default=DEFAULT_CONFIG, help="real config.json")
    ap.add_argument("--out", type=Path, required=True, help="output model directory")
    ap.add_argument("--layers", type=int, default=6, help="backbone layers to keep (N)")
    ap.add_argument("--keep", type=parse_int_list, help="explicit original layer ids to keep (overrides --layers)")
    ap.add_argument("--experts", type=int, default=16, help="n_routed_experts")
    ap.add_argument("--experts-per-tok", type=int, default=6, help="num_experts_per_tok")
    ap.add_argument("--dspark-experts", type=int, default=8, help="dspark_n_routed_experts (MTP layers)")
    ap.add_argument("--nextn", type=int, default=1, help="num_nextn_predict_layers (DSpark stages)")
    ap.add_argument("--dspark-targets", type=parse_int_list, help="dspark_target_layer_ids (new ids); default = last len(orig) backbone layers")
    ap.add_argument("--engram-layers", type=parse_int_list, help="engram_layer_ids (new ids); default = kept originals, else new layer 1")
    ap.add_argument("--engram-num-embeddings", type=int, default=1_000_000, help="rows per engram table")
    ap.add_argument("--vision-layers", type=int, default=2, help="vision_config.num_hidden_layers")
    ap.add_argument("--max-position-embeddings", type=int, help="override max_position_embeddings (rope tables are max_pos x 64 fp32)")
    ap.add_argument("--bf16", action="store_true", help="drop quantization_config (bf16 dummy weights)")
    args = ap.parse_args()

    full = json.loads(args.config.read_text())
    text = dict(full["text_config"])
    n_orig = text["num_hidden_layers"]

    keep = sorted(args.keep) if args.keep else sorted(priority_layers(text)[: args.layers])
    if any(k >= n_orig for k in keep):
        raise SystemExit(f"--keep ids must be < {n_orig}")
    n = len(keep)

    ratios_orig = [int(r) for r in text["compress_ratios"]]
    text["num_hidden_layers"] = n
    text["compress_ratios"] = [ratios_orig[k] for k in keep] + [0] * args.nextn
    text["kv_source_layer_ids"] = remap(text.get("kv_source_layer_ids", []), keep)
    text["index_source_layer_ids"] = remap(text.get("index_source_layer_ids", []), keep)
    cand = text.get("candidate_source_layer_id", -1)
    text["candidate_source_layer_id"] = remap([cand], keep)[0] if cand in keep else -1

    # experts
    text["n_routed_experts"] = args.experts
    text["num_experts_per_tok"] = args.experts_per_tok
    text["dspark_n_routed_experts"] = args.dspark_experts

    # DSpark / MTP
    n_targets = len(text.get("dspark_target_layer_ids", [])) or 3
    text["num_nextn_predict_layers"] = args.nextn
    text["dspark_target_layer_ids"] = args.dspark_targets or list(range(max(0, n - n_targets), n))

    # engram
    if args.engram_layers is not None:
        engram_layers = args.engram_layers
    else:
        engram_layers = remap(text.get("engram_layer_ids", []), keep)
        engram_layers = [e for e in engram_layers if e >= 1] or ([1] if n > 1 else [])
    text["engram_layer_ids"] = engram_layers
    text["engram_num_embeddings"] = [args.engram_num_embeddings] * len(engram_layers)
    if engram_layers:
        vocab, sums = pick_engram_vocab(
            args.engram_num_embeddings,
            len(engram_layers),
            text["engram_max_ngram_size"],
            text["engram_n_heads"],
            preferred=text.get("engram_vocab_size"),
        )
        text["engram_vocab_size"] = vocab
        text["engram_num_embeddings"] = [max(args.engram_num_embeddings, s) for s in sums]
    else:
        sums = []

    if args.max_position_embeddings:
        text["max_position_embeddings"] = args.max_position_embeddings

    errors = validate(text)
    if errors:
        print("INVALID TOPOLOGY:", file=sys.stderr)
        for e in errors:
            print("  -", e, file=sys.stderr)
        return 1

    out_cfg = dict(full)
    out_cfg["text_config"] = text
    vision = dict(full.get("vision_config", {}))
    vision["num_hidden_layers"] = args.vision_layers
    out_cfg["vision_config"] = vision
    if args.bf16:
        out_cfg.pop("quantization_config", None)

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "config.json").write_text(json.dumps(out_cfg, indent=2) + "\n")
    for name in TOKENIZER_FILES:
        src = HERE / name
        if src.exists():
            shutil.copy2(src, args.out / name)
        else:
            print(f"warning: {src} missing, not copied", file=sys.stderr)

    print(f"wrote {args.out / 'config.json'}")
    print(f"kept original layers: {keep}")
    print(
        f"experts={args.experts} top{args.experts_per_tok}, dspark experts={args.dspark_experts} "
        f"top{text['dspark_num_experts_per_tok']}, nextn={args.nextn}, vision layers={args.vision_layers}, "
        f"quant={'bf16 (none)' if args.bf16 else out_cfg['quantization_config']['quant_method'] + '/' + out_cfg['quantization_config']['expert_dtype'] + ' experts'}"
    )
    if engram_layers:
        print(
            f"engram layers={engram_layers} num_embeddings={text['engram_num_embeddings']} "
            f"vocab_size={text['engram_vocab_size']} (prime buckets sum to {sums}) "
            f"compressed_vocab_size={text['engram_compressed_vocab_size']} (tokenizer-derived, unchanged)"
        )
    print()
    print(layer_table(text, keep))
    return 0


if __name__ == "__main__":
    sys.exit(main())
