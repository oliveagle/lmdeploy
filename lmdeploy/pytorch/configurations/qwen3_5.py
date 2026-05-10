# Copyright (c) OpenMMLab. All rights reserved.
import torch

from lmdeploy.pytorch import envs as _envs
from lmdeploy.utils import is_bf16_supported

from .builder import AutoModelConfigBuilder
from .default import DefaultModelConfigBuilder
from .qwen3_next import _check_env_qwen3_next


class Qwen3_5ModelConfigBuilder(AutoModelConfigBuilder):

    @classmethod
    def condition(cls, hf_config):
        """config."""
        return hf_config.model_type in ['qwen3_5', 'qwen3_5_moe']

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
        text_config = hf_config.text_config
        # propagate quantization_config from top-level hf_config into text_config
        quantization_config = getattr(hf_config, 'quantization_config', None)
        if quantization_config is not None and not hasattr(text_config, 'quantization_config'):
            text_config.quantization_config = quantization_config
        cfg = DefaultModelConfigBuilder.build(text_config, model_path, tp=tp, **kwargs)

        if getattr(hf_config.text_config, 'attn_output_gate', False):
            cfg.num_attention_heads *= 2
        # update num layers
        num_layers = cfg.num_layers
        layer_types = text_config.layer_types
        num_delta_layers = sum([1 for lt in layer_types if lt == 'linear_attention'])
        num_full_layers = num_layers - num_delta_layers
        cfg.num_layers = num_full_layers

        # set state shapes
        head_k_dim = text_config.linear_key_head_dim
        head_v_dim = text_config.linear_value_head_dim
        num_v_heads = text_config.linear_num_value_heads // tp
        num_k_heads = text_config.linear_num_key_heads // tp
        key_dim = head_k_dim * num_k_heads
        value_dim = head_v_dim * num_v_heads
        conv_dim = key_dim * 2 + value_dim
        conv_kernel_size = text_config.linear_conv_kernel_dim + num_spec_tokens
        conv_state_shape = (num_delta_layers, conv_dim, conv_kernel_size)

        # for spec decoding
        if num_spec_tokens > 0:
            recurrent_state_shape = (num_delta_layers, 1 + num_spec_tokens, num_v_heads, head_k_dim, head_v_dim)
        else:
            recurrent_state_shape = (num_delta_layers, num_v_heads, head_k_dim, head_v_dim)

        # 强制使用 float16 以避免 TileLang kernel 类型不匹配问题
        dtype = torch.float16
        ssm_dtype = dtype if not _envs.fp32_mamba_ssm_dtype else torch.float32
        cfg.states_shapes = [(conv_state_shape, dtype), (recurrent_state_shape, ssm_dtype)]
        cfg.is_gated_delta = True
        cfg.check_env_func = _check_env_qwen3_next

        cfg.use_mrope = True

        # 修复 Qwen3.5 EOS token 问题: 使用 tokenizer 的 eos_token_id 而非 config 中的
        # config.text_config.eos_token_id = 248044 (PAD), tokenizer.eos_token_id = 248046 (<|im_end|>)
        if model_path is not None:
            from transformers import AutoTokenizer
            try:
                tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
                cfg.eos_token_id = tokenizer.eos_token_id
            except:
                pass  # 使用默认值

        # for spec decoding
        if spec_method is not None:
            if spec_method == 'dflash':
                # DFlash uses 5 layers from full_attention layers only
                # Distribute layers evenly, avoiding linear_attention layers
                layer_indices = []
                full_count = 0
                for idx, layer_type in enumerate(layer_types):
                    if layer_type == 'full_attention':
                        full_count += 1
                        # Select 5 layers evenly distributed: 1st, 25%, 50%, 75%, last
                        if full_count in [1, full_count // 4 or 1, full_count // 2,
                                          full_count * 3 // 4, full_count]:
                            layer_indices.append(idx)
                # Ensure we have exactly 5 layers
                while len(layer_indices) < 5 and layer_indices:
                    # Duplicate some layers if we don't have enough full_attention layers
                    layer_indices.append(layer_indices[-1])
                hf_config.aux_hidden_state_layers = tuple(layer_indices[:5])
            elif spec_method == 'qwen3_5_mtp':
                assert spec_method == 'qwen3_5_mtp'
                cfg.model_paradigm = 'ar_spec'

        # draft model cfg
        if is_draft_model:
            original_arch = hf_config.architectures[0]
            if original_arch == 'DFlashDraftModel':
                # Keep DFlash architecture, don't remap to MTP
                cfg.model_paradigm = 'ar_spec'
                cfg.states_shapes = []
            else:
                hf_config.architectures[0] = 'Qwen3_5MTPModel'
                # remove for correct mapping when building the patched model
                if hasattr(hf_config, 'auto_map'):
                    del hf_config.auto_map

                cfg.model_paradigm = 'ar_spec'
                cfg.num_layers = text_config.mtp_num_hidden_layers
                cfg.states_shapes = []

        return cfg
