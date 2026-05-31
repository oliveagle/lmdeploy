# Copyright (c) OpenMMLab. All rights reserved.
"""Qwen3 model configuration builder with DFlash support."""

from .builder import AutoModelConfigBuilder
from .default import DefaultModelConfigBuilder


class Qwen3ModelConfigBuilder(AutoModelConfigBuilder):

    @classmethod
    def condition(cls, hf_config):
        """config."""
        return hf_config.model_type == 'qwen3'

    @classmethod
    def build(cls,
              hf_config,
              model_path: str = None,
              tp: int = 1,
              is_draft_model: bool = False,
              spec_method: str = None,
              num_spec_tokens: int = 0,
              **kwargs):
        """build."""
        cfg = DefaultModelConfigBuilder.build(hf_config, model_path, tp=tp, **kwargs)

        # for spec decoding
        if spec_method is not None and spec_method == 'dflash':
            # DFlash uses 5 layers evenly distributed across all layers
            num_layers = cfg.num_layers
            layer_indices = []
            # Select 5 layers: first, 25%, 50%, 75%, last
            for i in [0, num_layers // 4, num_layers // 2, num_layers * 3 // 4, num_layers - 1]:
                layer_indices.append(i)
            hf_config.aux_hidden_state_layers = tuple(layer_indices[:5])

        # draft model cfg
        if is_draft_model:
            # DFlash draft model keeps its own architecture
            # Remove auto_map to prevent loading issues
            if hasattr(hf_config, 'auto_map'):
                del hf_config.auto_map

        return cfg
