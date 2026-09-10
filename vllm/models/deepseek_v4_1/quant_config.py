# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Quantization config for DeepSeek V4."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import torch

from vllm.config import get_current_vllm_config
from vllm.model_executor.layers.fused_moe import (
    RoutedExperts,
    UnquantizedFusedMoEMethod,
)
from vllm.model_executor.layers.linear import LinearBase, UnquantizedLinearMethod
from vllm.model_executor.layers.quantization import QuantizationMethods
from vllm.model_executor.layers.quantization.fp8 import Fp8Config
from vllm.model_executor.layers.quantization.modelopt import (
    ModelOptMxFp8LinearMethod,
)
from vllm.model_executor.layers.quantization.mxfp4 import Mxfp4MoEMethod
from vllm.model_executor.layers.quantization.utils.mxfp8_utils import (
    MXFP8_BLOCK_SIZE,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    is_layer_skipped,
)

_DEEPSEEK_V4_EXPERT_DTYPES = ("fp4", "fp8")

if TYPE_CHECKING:
    from vllm.model_executor.layers.quantization.modelopt import (
        ModelOptNvFp4Config,
    )


class DeepseekV4Mxfp8LinearMethod(ModelOptMxFp8LinearMethod):
    """MXFP8 linear method for native DeepSeek V4.1 ``[32, 32]`` checkpoints.

    Upstream (#56201) serves these layers through ``ModelOptLinearMethod``
    with ``CkptCtx(scale_block_size=(32, 32))``, which this tree predates.
    The checkpoint stores one ue8m0 exponent per 32x32 block of the
    ``[N, K]`` e4m3 weight, i.e. a ``[N/32, K/32]`` scale tensor, while
    ``ModelOptMxFp8LinearMethod`` and every kernel behind it (Marlin on
    sm8x) take one scale per 32 elements of K for every output row,
    ``[N, K/32]``. Mirroring upstream's ``KMxfp8Static`` scale loader, the
    checkpoint scale is row-repeated 32x along N before the layer's regular
    sharded loader runs, so TP and fused-shard offsets (expressed in output
    elements) apply unchanged.
    """

    def __init__(self, quant_config: DeepseekV4FP8Config) -> None:
        block_rows, block_cols = quant_config.weight_block_size
        if block_rows < 1 or block_cols != MXFP8_BLOCK_SIZE:
            raise NotImplementedError(
                f"MXFP8 checkpoint scale block {quant_config.weight_block_size} "
                "is unsupported"
            )
        self.scale_block_rows = block_rows
        super().__init__(quant_config)  # type: ignore[arg-type]

    def create_weights(
        self,
        layer: torch.nn.Module,
        input_size_per_partition: int,
        output_partition_sizes: list[int],
        input_size: int,
        output_size: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ) -> None:
        super().create_weights(
            layer,
            input_size_per_partition,
            output_partition_sizes,
            input_size,
            output_size,
            params_dtype,
            **extra_weight_attrs,
        )
        # The attention layer keys its o_proj recipe off this
        # (``DeepseekV4Attention._o_proj_block_size``).
        layer.weight_block_size = [1, MXFP8_BLOCK_SIZE]
        if self.scale_block_rows > 1:
            layer.weight_scale.weight_loader = self._block_scale_loader(
                layer.weight_scale.weight_loader
            )

    def _block_scale_loader(self, weight_loader):
        block_rows = self.scale_block_rows

        def loader(param, loaded_weight: torch.Tensor, *args, **kwargs):
            assert loaded_weight.dtype in (torch.uint8, torch.float8_e8m0fnu), (
                f"expected e8m0 block scales, got {loaded_weight.dtype}"
            )
            # Raw exponent bytes: view, never convert (2^-7 would round to 0).
            loaded_weight = loaded_weight.view(torch.uint8).repeat_interleave(
                block_rows, dim=0
            )
            return weight_loader(param, loaded_weight, *args, **kwargs)

        return loader


class DeepseekV4FP8Config(Fp8Config):
    """FP8 config for DeepSeek V4 with expert-dtype-aware MoE dispatch.

    DeepSeek V4 checkpoints always use FP8 block quantization for
    linear/attention layers. The MoE expert weights vary by checkpoint:
    - ``expert_dtype="fp4"`` (e.g. DeepSeek-V4-Flash): MXFP4 experts
      with ue8m0 (e8m0fnu) FP8 linear scales.
    - ``expert_dtype="fp8"`` (e.g. DeepSeek-V4-Flash-Base): FP8 block
      experts with float32 FP8 linear scales.

    The dispatch and the linear scale dtype are both keyed off
    ``expert_dtype`` from the model's hf_config; missing values default
    to ``"fp4"`` so existing FP4 checkpoints stay unchanged.

    NOTE: ``expert_dtype`` is resolved lazily because this config is
    constructed during VllmConfig setup, before ``set_current_vllm_config``
    is active. Reading hf_config eagerly in ``__init__`` would always see
    the default ``"fp4"`` and silently misroute Flash-Base checkpoints.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._resolved_expert_dtype: str | None = None
        self._resolved_moe_quant_algo: str | None = None
        self._nvfp4_config: ModelOptNvFp4Config | None = None
        # ``is_scale_e8m0`` is a property that resolves on first read,
        # by which time the current vllm_config has been set.

    @property
    def expert_dtype(self) -> str:
        if self._resolved_expert_dtype is None:
            try:
                hf_config = get_current_vllm_config().model_config.hf_config
            except Exception:
                # vllm_config not yet set; defer the decision until a
                # later call lands inside set_current_vllm_config.
                return "fp4"
            expert_dtype = getattr(hf_config, "expert_dtype", "fp4")
            if expert_dtype not in _DEEPSEEK_V4_EXPERT_DTYPES:
                raise ValueError(
                    f"Unsupported DeepSeek V4 expert_dtype={expert_dtype!r}; "
                    f"expected one of {_DEEPSEEK_V4_EXPERT_DTYPES}."
                )
            self._resolved_expert_dtype = expert_dtype
            from vllm.logger import init_logger

            init_logger(__name__).info_once(
                "DeepSeek V4 expert_dtype resolved to %r", expert_dtype
            )
        return self._resolved_expert_dtype

    @property
    def is_scale_e8m0(self) -> bool:
        # FP4 checkpoints store FP8 linear scales as e8m0fnu; FP8 expert
        # checkpoints (Flash-Base) store them as float32.
        return self.expert_dtype == "fp4"

    @property
    def is_checkpoint_mxfp8_serialized(self) -> bool:
        # Read by ModelOptMxFp8LinearMethod through DeepseekV4Mxfp8LinearMethod.
        return self.weight_block_size == [32, 32] and self.is_scale_e8m0

    def _resolve_moe_overrides(self) -> None:
        if self._resolved_moe_quant_algo is not None:
            return
        try:
            hf_config = get_current_vllm_config().model_config.hf_config
        except Exception:
            return
        quant_cfg = getattr(hf_config, "quantization_config", None) or {}
        algo = (quant_cfg.get("moe_quant_algo") or "").upper() or None
        self._resolved_moe_quant_algo = algo or ""

    @property
    def moe_quant_algo(self) -> str:
        self._resolve_moe_overrides()
        return self._resolved_moe_quant_algo or ""

    def _get_nvfp4_config(self) -> ModelOptNvFp4Config:
        if self._nvfp4_config is None:
            from vllm.model_executor.layers.quantization.modelopt import (
                ModelOptNvFp4Config,
            )

            self._nvfp4_config = ModelOptNvFp4Config(
                is_checkpoint_nvfp4_serialized=True,
                kv_cache_quant_algo=None,
                exclude_modules=[],
                group_size=16,
            )
        return self._nvfp4_config

    @classmethod
    def get_name(cls) -> QuantizationMethods:
        return "deepseek_v4_fp8"

    @staticmethod
    def _is_quark_mxfp4_ocp(hf_quant_cfg: dict) -> bool:
        """True for AMD-Quark exports whose global scheme is MXFP4."""
        weight = (hf_quant_cfg.get("global_quant_config") or {}).get("weight")
        # A non-dict weight (e.g. a list of multiple specs) means not an OCP
        # MXFP4 scheme (e.g. NVFP4 with 2-level scale).
        if not isinstance(weight, dict):
            return False
        return (
            weight.get("dtype") == "fp4"
            and weight.get("qscheme") == "per_group"
            and weight.get("group_size") == 32
        )

    @classmethod
    def override_quantization_method(
        cls, hf_quant_cfg, user_quant, hf_config=None
    ) -> QuantizationMethods | None:
        if not (
            isinstance(hf_quant_cfg, dict)
            and (
                hf_quant_cfg.get("quant_method") in ("fp8", "deepseek_v4_fp8")
                or (
                    hf_quant_cfg.get("quant_method") == "quark"
                    and cls._is_quark_mxfp4_ocp(hf_quant_cfg)
                )
            )
        ):
            return None
        model_type = getattr(hf_config, "model_type", None)
        if (
            model_type
            in (
                "deepseek_v4",
                "deepseek_v4_text",
                "deepseek_v41",
                "deepseek_v41_text",
            )
            or user_quant == "deepseek_v4_fp8"
        ):
            return "deepseek_v4_fp8"
        return None

    @classmethod
    def from_config(cls, config: dict) -> DeepseekV4FP8Config:
        # Reroute AMD-Quark fused shared expert MXFP4 checkpoints onto the fp8
        # path: the runtime layout matches the DeepSeek-native fp8 checkpoint,
        # so translate the schema into format Fp8Config.from_config expects.
        if config.get("quant_method") == "quark":
            quark_exclude = config.get("exclude") or []
            config = {
                "quant_method": "fp8",
                "activation_scheme": "dynamic",
                "fmt": "e4m3",
                "scale_fmt": "ue8m0",
                "weight_block_size": [128, 128],
                "ignored_layers": [
                    name for name in quark_exclude if isinstance(name, str)
                ],
            }
        return cast("DeepseekV4FP8Config", super().from_config(config))

    def get_quant_method(self, layer, prefix):
        if (
            isinstance(layer, LinearBase)
            and self.weight_block_size == [32, 32]
            and self.is_scale_e8m0
        ):
            if is_layer_skipped(
                prefix=prefix,
                ignored_layers=self.ignored_layers,
                fused_mapping=self.packed_modules_mapping,
                match_mode=self.ignored_layers_match_mode,
            ):
                return UnquantizedLinearMethod()
            # Upstream returns ModelOptLinearMethod(kMxfp8Static, kMxfp8Dynamic,
            # CkptCtx(scale_block_size=(32, 32))), which this tree lacks;
            # DeepseekV4Mxfp8LinearMethod (above) feeds the same kernels (Marlin
            # on sm8x) through ModelOptMxFp8LinearMethod with the block-scale
            # expansion loader.
            return DeepseekV4Mxfp8LinearMethod(self)
        if isinstance(layer, RoutedExperts):
            if is_layer_skipped(
                prefix=prefix,
                ignored_layers=self.ignored_layers,
                fused_mapping=self.packed_modules_mapping,
            ):
                return UnquantizedFusedMoEMethod(layer.moe_config)
            if self.expert_dtype == "fp4":
                if self.moe_quant_algo == "NVFP4":
                    from vllm.model_executor.layers.quantization.modelopt import (
                        ModelOptNvFp4FusedMoE,
                    )

                    return ModelOptNvFp4FusedMoE(
                        quant_config=self._get_nvfp4_config(),
                        moe_config=layer.moe_config,
                    )
                return Mxfp4MoEMethod(layer.moe_config)
            # expert_dtype == "fp8": fall through to Fp8Config which
            # returns Fp8MoEMethod with block-wise float32 scales.
        return super().get_quant_method(layer, prefix)
