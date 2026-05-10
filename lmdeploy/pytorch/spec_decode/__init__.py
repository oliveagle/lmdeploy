# Copyright (c) OpenMMLab. All rights reserved.

from ..config import BackendConfig, MiscConfig, SpecDecodeConfig
from ..distributed import DistContext


def build_spec_agent(specdecode_config: SpecDecodeConfig,
                     backend_config: BackendConfig,
                     dist_ctx: DistContext,
                     inputs_strategy,
                     agent_strategy,
                     misc_config: MiscConfig,
                     device: str = 'cuda'):
    """Build spec agent."""
    # 检查是否是 EP 模式
    dist_config = dist_ctx.dist_config
    is_ep_mode = dist_config.ep > 1

    if is_ep_mode:
        # 在 EP 模式下，只在 rank 0 上启用 speculative decoding
        enable = dist_ctx.rank == 0 and specdecode_config is not None
    else:
        # 在非 EP 模式下，使用原来的逻辑：每个 TP 组的第一个 rank
        enable = dist_ctx.rank % dist_config.attn_tp == 0 and specdecode_config is not None

    if enable:
        from .spec_agent import SpecModelAgent
        return SpecModelAgent(specdecode_config,
                              backend_config,
                              inputs_strategy,
                              agent_strategy,
                              misc_config,
                              device=device)
    else:
        from .base import BaseSpecModelAgent
        return BaseSpecModelAgent(specdecode_config)


__all__ = ['build_spec_agent']
