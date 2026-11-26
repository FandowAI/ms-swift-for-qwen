# Copyright (c) ModelScope Contributors. All rights reserved.
"""Formal FP8 policy and grouped-GEMM telemetry for Megatron training."""

from __future__ import annotations

import collections
import importlib
import logging
from collections.abc import Iterable
from megatron.core.extensions.transformer_engine import TEGroupedLinear, TEQuantizationParams, TEQuantizationRecipe

logger = logging.getLogger(__name__)


def _is_expert_module(name: str, module) -> bool:
    return isinstance(module, TEGroupedLinear) and '.experts.' in name


def _expert_only_recipe():
    recipe = TEQuantizationRecipe(
        fp8_quantization_recipe=None,
        fp4_quantization_recipe=None,
        override_quantized_autocast=True,
    )
    return TEQuantizationParams(training_recipe=recipe, evaluation_recipe=recipe)


def configure_expert_fp8(models: Iterable[object], enabled: bool = False) -> dict[str, int]:
    """Keep delayed FP8 on MoE experts and disable it for non-expert TE layers."""
    if not enabled:
        return {'experts': 0, 'non_experts': 0}

    forced = _expert_only_recipe()
    experts = 0
    non_experts = 0
    for model in models:
        for name, module in model.named_modules():
            if not hasattr(module, 'te_quant_params'):
                continue
            if _is_expert_module(name, module):
                experts += 1
                continue
            if module.te_quant_params is not None:
                raise RuntimeError(f'refusing to overwrite an existing TE recipe: {name}')
            module.te_quant_params = forced
            non_experts += 1

    if experts == 0:
        raise RuntimeError('fp8_expert_only requires at least one TEGroupedLinear expert module')
    logger.info('FP8 policy: experts=%d, non_expert_te_layers=%d', experts, non_experts)
    return {'experts': experts, 'non_experts': non_experts}


class GroupedGemmTelemetry:
    """Count native FP8 grouped GEMM calls without changing their results."""

    def __init__(self):
        self.counts = collections.Counter()
        self._installed = False
        self._last_snapshot = {}

    def install(self) -> None:
        if self._installed:
            return
        gemm = importlib.import_module('transformer_engine.pytorch.cpp_extensions.gemm')
        storage_type = importlib.import_module(
            'transformer_engine.pytorch.tensor.storage.float8_tensor_storage').Float8TensorStorage
        original = gemm.tex.te_general_grouped_gemm
        counts = self.counts

        def observed(A, transa, B, transb, *args, **kwargs):
            output = original(A, transa, B, transb, *args, **kwargs)
            if isinstance(A, (list, tuple)) and isinstance(B, (list, tuple)):
                fp8_dtypes = (7, 8)
                has_fp8_pair = any(
                    isinstance(x, storage_type) and isinstance(y, storage_type) and int(x._fp8_dtype) in fp8_dtypes
                    and int(y._fp8_dtype) in fp8_dtypes for x, y in zip(A, B))
                if has_fp8_pair:
                    counts[(bool(transa), bool(transb))] += 1
            return output

        gemm.tex.te_general_grouped_gemm = observed
        self._installed = True

    def take_delta(self) -> dict[str, int]:
        current = {f'transa_{a}_transb_{b}': count for (a, b), count in self.counts.items()}
        keys = set(current) | set(self._last_snapshot)
        delta = {key: current.get(key, 0) - self._last_snapshot.get(key, 0) for key in keys}
        self._last_snapshot = current
        return delta
