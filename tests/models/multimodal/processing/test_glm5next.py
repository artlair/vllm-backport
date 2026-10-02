# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Device-side normalisation checks for GLM-5.3-Flash processing."""

import pytest
import torch
from PIL import Image

from vllm.model_executor.layers.fusion.mm_input_norm import build_mm_input_norm
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.platforms import current_platform

from ...utils import build_model_context


@pytest.mark.usefixtures("default_vllm_config")
def test_mm_device_do_normalize():
    device = current_platform.device_type
    ctx = build_model_context(
        "zai-org/GLM-5.3-Flash",
        limit_mm_per_prompt={"image": 2},
    )
    assert ctx.model_config.multimodal_config.mm_device_do_normalize

    ctx.model_config.multimodal_config.mm_device_do_normalize = False
    processor = MULTIMODAL_REGISTRY.create_processor(ctx.model_config)
    images = [
        Image.new("RGB", (310, 470), color=(17, 89, 231)),
        Image.new("RGB", (480, 320), color=(201, 13, 127)),
    ]
    prompt = " IMAGE_PLACEHOLDER" * len(images)
    mm_items = processor.info.parse_mm_data({"image": images})

    normalized_inputs = processor(prompt, mm_items=mm_items)
    normalized_values = normalized_inputs["mm_kwargs"].get_data()["pixel_values"]

    ctx.model_config.multimodal_config.mm_device_do_normalize = True
    raw_inputs = processor(prompt, mm_items=mm_items)
    raw_values = raw_inputs["mm_kwargs"].get_data()["pixel_values"]
    assert raw_values.dtype == torch.uint8

    input_norm = build_mm_input_norm(ctx.model_config).to(device)
    output = input_norm(raw_values.to(device), normalized_values.dtype)
    torch.testing.assert_close(
        output, normalized_values.to(device), rtol=1e-5, atol=1e-6
    )
