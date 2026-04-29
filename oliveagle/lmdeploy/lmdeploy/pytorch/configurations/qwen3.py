# Copyright (c) OpenMMLab. All rights reserved.
"""Qwen3 model config builder for DFlash/EAGLE3 speculative decoding."""

from .builder import AutoModelConfigBuilder
from .default import DefaultModelConfigBuilder


class Qwen3ModelConfigBuilder(AutoModelConfigBuilder):

    @classmethod
    def condition(cls, hf_config):
        """Check if config is Qwen3."""
        return hf_config.model_type == 'qwen3'

    @classmethod
    def build(cls, hf_config, model_path: str = None, **kwargs):
        """Build Qwen3 model config."""
        from lmdeploy.utils import is_bf16_supported

        cfg = DefaultModelConfigBuilder.build(hf_config, model_path, **kwargs)

        # Set dtype
        torch_dtype = 'bfloat16' if is_bf16_supported() else 'float16'
        if getattr(hf_config, 'bf16', False) and is_bf16_supported():
            torch_dtype = 'bfloat16'
        elif getattr(hf_config, 'fp16', False):
            torch_dtype = 'float16'
        hf_config.torch_dtype = torch_dtype

        # Set aux_hidden_state_layers for DFlash/EAGLE3
        # These are the layers where intermediate hidden states are captured
        # Default: layers at approximately 1/4, 1/2, and 3/4 of the model
        spec_method = kwargs.get('spec_method', None)
        from lmdeploy.utils import get_logger
        logger = get_logger('lmdeploy')
        logger.info(f'Qwen3ModelConfigBuilder: spec_method={spec_method}, num_layers={cfg.num_layers}')

        if spec_method == 'dflash':
            # DFlash uses multiple layers for hidden states
            # These should match the target_layer_ids in the draft model config
            num_layers = cfg.num_layers
            # Use layers similar to draft model's [1, 9, 17, 25, 33] for 36-layer model
            # Scale to actual num_layers
            if num_layers == 36:
                hf_config.aux_hidden_state_layers = (1, 9, 17, 25, 33)
            elif num_layers == 28:
                hf_config.aux_hidden_state_layers = (1, 7, 13, 19, 25)
            elif num_layers == 32:
                hf_config.aux_hidden_state_layers = (1, 8, 15, 22, 29)
            else:
                # Default: spread evenly across the model
                layer_stride = max(1, num_layers // 6)
                hf_config.aux_hidden_state_layers = tuple(
                    i * layer_stride for i in range(1, min(6, num_layers))
                )
            logger.info(f'Qwen3ModelConfigBuilder: Set aux_hidden_state_layers={hf_config.aux_hidden_state_layers} for DFlash')
        elif spec_method == 'eagle3':
            # EAGLE3 uses 3 layers
            num_layers = cfg.num_layers
            hf_config.aux_hidden_state_layers = (2, num_layers // 2, num_layers - 3)

        # Set model_paradigm for speculative decoding methods
        if spec_method is not None:
            cfg.model_paradigm = 'ar_spec'

        return cfg
