# Copyright (c) OpenMMLab. All rights reserved.
from .builder import AutoModelConfigBuilder
from .default import DefaultModelConfigBuilder


class Qwen3_5ModelConfigBuilder(AutoModelConfigBuilder):

    @classmethod
    def condition(cls, hf_config):
        """config."""
        model_type = getattr(hf_config, 'model_type', None)
        return model_type in ['qwen3_5', 'qwen3_5_text', 'qwen3_5_moe']

    @classmethod
    def build(cls, hf_config, model_path: str = None, **kwargs):
        """build."""
        # Qwen3.5 uses nested text_config
        if hasattr(hf_config, 'text_config'):
            text_config = hf_config.text_config
            # Copy quantization_config to text_config if needed
            if hasattr(hf_config, 'quantization_config') and not hasattr(text_config, 'quantization_config'):
                setattr(text_config, 'quantization_config', hf_config.quantization_config)
            # Copy text_config attributes to top-level for model code compatibility
            for attr in ['vocab_size', 'hidden_size', 'num_hidden_layers', 'num_attention_heads',
                         'num_key_value_heads', 'head_dim', 'rms_norm_eps', 'pad_token_id',
                         'tie_word_embeddings', 'max_position_embeddings', 'rope_theta',
                         'rope_scaling', 'sliding_window', 'use_sliding_window',
                         'attention_bias', 'intermediate_size', 'hidden_act', 'bos_token_id',
                         'eos_token_id']:
                if not hasattr(hf_config, attr):
                    val = getattr(text_config, attr, None)
                    if val is not None:
                        setattr(hf_config, attr, val)
            cfg = DefaultModelConfigBuilder.build(text_config, model_path, **kwargs)
            setattr(hf_config, 'dtype', getattr(text_config, 'dtype', 'float16'))
        else:
            cfg = DefaultModelConfigBuilder.build(hf_config, model_path, **kwargs)

        # Fix missing attributes for Qwen3.5
        if not hasattr(hf_config, 'pad_token_id') or hf_config.pad_token_id is None:
            hf_config.pad_token_id = getattr(hf_config, 'eos_token_id', None)
        if not hasattr(hf_config, 'tie_word_embeddings'):
            hf_config.tie_word_embeddings = False

        # Set aux_hidden_state_layers for DFlash/EAGLE3
        spec_method = kwargs.get('spec_method', None)
        from lmdeploy.utils import get_logger
        logger = get_logger('lmdeploy')
        logger.info(f'Qwen3_5ModelConfigBuilder: spec_method={spec_method}, num_layers={cfg.num_layers}')

        if spec_method == 'dflash':
            # DFlash uses multiple layers for hidden states
            num_layers = cfg.num_layers
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
            logger.info(f'Qwen3_5ModelConfigBuilder: Set aux_hidden_state_layers={hf_config.aux_hidden_state_layers} for DFlash')
        elif spec_method == 'eagle3':
            # EAGLE3 uses 3 layers
            num_layers = cfg.num_layers
            hf_config.aux_hidden_state_layers = (2, num_layers // 2, num_layers - 3)

        # Set model_paradigm for speculative decoding methods
        if spec_method is not None:
            cfg.model_paradigm = 'ar_spec'

        cfg.hf_config = hf_config
        return cfg
