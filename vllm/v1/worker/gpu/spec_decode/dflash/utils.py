# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import json
import os

import torch
import torch.nn as nn

from vllm.config import ModelConfig, VllmConfig, replace
from vllm.distributed.parallel_state import get_pp_group
from vllm.logger import init_logger
from vllm.model_executor.model_loader import get_model
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.v1.worker.gpu.spec_decode.eagle.utils import (
    _should_share,
    get_target_lm_head,
)

logger = init_logger(__name__)

_TARGET_EMBED_NAMES = (
    "model.embed_tokens.weight",
    "model.language_model.embed_tokens.weight",
    "language_model.model.embed_tokens.weight",
)


def _read_target_embedding(model_config: ModelConfig) -> torch.Tensor:
    """Read the target's input embedding from its safetensors checkpoint."""
    from huggingface_hub import hf_hub_download
    from safetensors import safe_open

    def resolve(filename: str) -> str:
        if os.path.isdir(model_config.model):
            return os.path.join(model_config.model, filename)
        return hf_hub_download(
            model_config.model, filename, revision=model_config.revision
        )

    try:
        index_path = resolve("model.safetensors.index.json")
        with open(index_path) as f:
            weight_map: dict[str, str] = json.load(f)["weight_map"]
    except (OSError, KeyError):
        weight_map = {name: "model.safetensors" for name in _TARGET_EMBED_NAMES}
    for name in _TARGET_EMBED_NAMES:
        if name not in weight_map:
            continue
        with safe_open(resolve(weight_map[name]), framework="pt") as f:
            if name in f.keys():  # noqa: SIM118
                return f.get_tensor(name)
    raise RuntimeError(
        f"no input embedding ({', '.join(_TARGET_EMBED_NAMES)}) in the target "
        f"checkpoint {model_config.model}"
    )


def _load_target_embedding_into_draft(
    draft_embed: nn.Module, model_config: ModelConfig
) -> None:
    """Give a PP-stage drafter the target embedding it cannot share.

    A drafter checkpoint without its own embedding relies on the target's, but
    under PP the drafting (last) stage holds no target embedding: it lives on
    the first stage. Left alone, the draft's embedding stays uninitialized and
    every anchor/mask token embeds to garbage; acceptance collapses with no
    error (the same failure GLM-5.3-Flash MTP had, c10a067fe2).
    """
    weight = _read_target_embedding(model_config)
    param = draft_embed.weight
    getattr(param, "weight_loader", default_weight_loader)(param, weight)
    logger.info(
        "Loaded the target input embedding %s into the PP drafter",
        tuple(weight.shape),
    )


def load_dflash_model(target_model: nn.Module, vllm_config: VllmConfig) -> nn.Module:
    from vllm.compilation.backends import set_model_tag
    from vllm.model_executor.models.qwen3_dflash import (
        dflash_has_any_non_causal,
    )

    speculative_config = vllm_config.speculative_config
    assert speculative_config is not None
    draft_model_config = speculative_config.draft_model_config
    # Select an attention backend that supports the drafter's attention: mixing
    # a non-causal layer onto a causal-only backend would fail.
    draft_vllm_config = replace(
        vllm_config,
        attention_config=replace(
            vllm_config.attention_config,
            use_non_causal=dflash_has_any_non_causal(draft_model_config.hf_config),
            backend=speculative_config.attention_backend,
        ),
        cache_config=(
            replace(
                vllm_config.cache_config,
                cache_dtype=speculative_config.kv_cache_dtype,
            )
            if speculative_config.kv_cache_dtype is not None
            else vllm_config.cache_config
        ),
    )
    with set_model_tag("dflash_head"):
        dflash_model = get_model(
            vllm_config=draft_vllm_config, model_config=draft_model_config
        )

    target_language_model = (
        target_model.get_language_model()
        if hasattr(target_model, "get_language_model")
        else target_model
    )
    # MuseGlimmerForCausalLM marks its inner MuseGlimmerModel as the language
    # model, so get_language_model() already returns the inner module and has
    # no .model of its own.
    target_inner = getattr(target_language_model, "model", target_language_model)
    draft_inner = dflash_model.model

    # Under PP the target embedding lives on the first stage, so a drafter
    # without its own embedding loads the target's from the checkpoint.
    if get_pp_group().world_size > 1:
        draft_embed = getattr(draft_inner, "embed_tokens", None)
        if draft_embed is not None and not getattr(
            dflash_model, "has_own_embed_tokens", False
        ):
            _load_target_embedding_into_draft(draft_embed, vllm_config.model_config)
    else:
        target_embed = getattr(target_inner, "embed_tokens", None) or getattr(
            target_inner, "embedding", None
        )
        draft_embed = getattr(draft_inner, "embed_tokens", None)
        if target_embed is not None and _should_share(
            dflash_model, "has_own_embed_tokens", draft_embed, target_embed
        ):
            if draft_embed is not None:
                del draft_inner.embed_tokens
            draft_inner.embed_tokens = target_embed

    target_lm_head = get_target_lm_head(target_model, target_language_model)
    draft_lm_head = getattr(dflash_model, "lm_head", None)
    if target_lm_head is not None and _should_share(
        dflash_model, "has_own_lm_head", draft_lm_head, target_lm_head
    ):
        if draft_lm_head is not None:
            del dflash_model.lm_head
        dflash_model.lm_head = target_lm_head

    return dflash_model
