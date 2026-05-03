# Copyright (c) OpenMMLab. All rights reserved.
from collections.abc import Callable

import torch
from torch import nn

from lmdeploy.pytorch.backends import OpType, get_backend
from lmdeploy.pytorch.distributed import get_dist_manager, get_ep_world_rank, get_tp_world_rank

from .base import FusedMoEBase, MoeType, moe_gather_inputs, moe_reduce, update_dims


class AwqLinearWeights(nn.Module):
    """AWQ quantized MoE linear weights."""

    def __init__(self,
                 num_experts: int,
                 in_features: int,
                 out_features: int,
                 w_bit: int,
                 group_size: int,
                 weight_type: str,
                 device: torch.device,
                 bias: bool = False,
                 expert_list: list[int] | None = None):
        super().__init__()
        self.num_experts = num_experts
        self.in_features = in_features
        self.out_features = out_features
        self.w_bit = w_bit
        self.group_size = group_size
        self.ep = expert_list is not None
        self.expert_list = expert_list
        self.weight_type = weight_type
        self.half_out = out_features // 2

        elem_per_int = 32 // w_bit
        grouped_in_feats = in_features // group_size
        quant_out_feats = out_features // elem_per_int

        # Create empty buffers for AWQ quantized weights
        # Shape: (num_experts, in_features, quant_out_feats)
        qweight_shape = (num_experts, in_features, quant_out_feats)
        self.qweight = nn.Parameter(torch.zeros(qweight_shape, dtype=torch.int32, device=device), requires_grad=False)

        # Shape: (num_experts, grouped_in_feats, out_features)
        scales_shape = (num_experts, grouped_in_feats, out_features)
        self.scales = nn.Parameter(torch.zeros(scales_shape, dtype=torch.float16, device=device), requires_grad=False)

        # Shape: (num_experts, grouped_in_feats, quant_out_feats)
        qzeros_shape = (num_experts, grouped_in_feats, quant_out_feats)
        self.qzeros = nn.Parameter(torch.zeros(qzeros_shape, dtype=torch.int32, device=device), requires_grad=False)

        if bias:
            bias_shape = (num_experts, out_features)
            self.bias = nn.Parameter(torch.zeros(bias_shape, dtype=torch.float16, device=device), requires_grad=False)
        else:
            self.register_parameter('bias', None)

        self.setup_weight_loader()

    def setup_weight_loader(self):
        """Setup weight loader."""
        if self.expert_list is not None:
            from collections import defaultdict
            self.expert_map = defaultdict(list)
            for idx, eid in enumerate(self.expert_list):
                self.expert_map[eid].append(idx)
            self.qweight.weight_loader = self.weight_loader_ep
            self.scales.weight_loader = self.weight_loader_ep
            self.qzeros.weight_loader = self.weight_loader_ep
            if self.bias is not None:
                self.bias.weight_loader = self.weight_loader_ep
        else:
            self.qweight.weight_loader = self.weight_loader_tp
            self.scales.weight_loader = self.weight_loader_tp
            self.qzeros.weight_loader = self.weight_loader_tp
            if self.bias is not None:
                self.bias.weight_loader = self.weight_loader_tp

    def weight_loader_tp(self, param: nn.Parameter, loaded_weight: torch.Tensor, expert_id: int, shard_id: str):
        """Weight loader for tensor parallelism."""
        world_size, rank = get_tp_world_rank('moe')
        if shard_id == 'gate':
            param_data = param.data[expert_id, :, :self.half_out]
            weight = loaded_weight.chunk(world_size, dim=-1)[rank]
        elif shard_id == 'up':
            param_data = param.data[expert_id, :, self.half_out:]
            weight = loaded_weight.chunk(world_size, dim=-1)[rank]
        elif shard_id == 'down':
            param_data = param.data[expert_id]
            weight = loaded_weight
            if weight.dim() > 1:
                weight = weight.chunk(world_size, dim=0)[rank]
        else:
            raise RuntimeError(f'Unknown shard_id: {shard_id}')
        param_data.copy_(weight)

    def weight_loader_ep(self, param: nn.Parameter, loaded_weight: torch.Tensor, expert_id: int, shard_id: str):
        """Weight loader for expert parallelism."""
        expert_list = self.expert_list
        if expert_id not in expert_list:
            return

        expert_map = self.expert_map
        param_ids = expert_map[expert_id]
        for param_id in param_ids:
            if shard_id == 'gate':
                param_data = param.data[param_id, :, :self.half_out]
            elif shard_id == 'up':
                param_data = param.data[param_id, :, self.half_out:]
            elif shard_id == 'down':
                param_data = param.data[param_id]
            else:
                raise RuntimeError(f'Unknown shard_id: {shard_id}')
            param_data.copy_(loaded_weight)


class FusedMoEAWQ(FusedMoEBase):
    """Fused MoE with AWQ quantization."""

    def __init__(self,
                 hidden_dim: int,
                 ffn_dim: int,
                 num_experts: int,
                 top_k: int,
                 w_bit: int,
                 group_size: int,
                 bias: bool = False,
                 renormalize: bool = False,
                 dtype: torch.dtype | None = None,
                 device: torch.device | None = None,
                 all_reduce: bool = True,
                 layer_idx: int = 0,
                 act_func: Callable = None):

        device = device or torch.device('cpu')
        dtype = dtype or torch.float16
        # init distributed tp arguments
        self.init_dist_args(all_reduce)

        super().__init__(
            tp=self.tp,
            tp_mode=self.tp_mode,
            do_renormalize=renormalize,
        )

        # create implementation
        dist_ctx = get_dist_manager().current_context()
        self.ep_size, rank = get_ep_world_rank()
        impl_builder = get_backend().get_layer_impl_builder(OpType.FusedMoE)
        self.impl = impl_builder.build(
            top_k,
            num_experts,
            renormalize,
            hidden_dim=hidden_dim,
            ep_size=self.ep_size,
            ep_group=dist_ctx.ep_gpu_group,
            layer_idx=layer_idx,
        )

        # create weights
        if self.ep_size > 1:
            expert_list = self.impl.ep_expert_list(self.ep_size, rank)
            num_experts = len(expert_list)
        else:
            hidden_dim, ffn_dim = update_dims(hidden_dim, ffn_dim)
            expert_list = None
        self.expert_list = expert_list

        self.gate_up = AwqLinearWeights(
            num_experts=num_experts,
            in_features=hidden_dim,
            out_features=ffn_dim * 2,
            w_bit=w_bit,
            group_size=group_size,
            weight_type='gate_up',
            device=device,
            bias=bias,
            expert_list=expert_list,
        )

        self.down = AwqLinearWeights(
            num_experts=num_experts,
            in_features=ffn_dim,
            out_features=hidden_dim,
            w_bit=w_bit,
            group_size=group_size,
            weight_type='down',
            device=device,
            bias=bias,
            expert_list=expert_list,
        )

        self.hidden_dim = hidden_dim
        self.ffn_dim = ffn_dim
        self.num_experts = num_experts
        self.w_bit = w_bit
        self.group_size = group_size
        self.dtype = dtype
        self.device = device
        self.act_func = act_func

    def before_dispatch(self, state):
        """Before dispatch."""
        return state

    def dispatch(self, state: dict):
        """dispatch."""
        moe_type = state['moe_type']
        if moe_type == MoeType.Default or (isinstance(moe_type, str) and moe_type == 'Default'):
            hidden_states, topk_weights, topk_idx = moe_gather_inputs(state['hidden_states'],
                                                                      state['topk_weights'],
                                                                      state['topk_idx'],
                                                                      group=self.gather_group)
            recv_state = {
                'hidden_states': hidden_states,
                'topk_idx': topk_idx,
                'topk_weights': topk_weights,
                'moe_type': moe_type
            }
        else:
            raise NotImplementedError(f'Not supported moe type: {moe_type}')
        return recv_state

    def gemm(self, state: dict):
        """gemm."""
        from lmdeploy.pytorch.kernels.cuda.fused_moe import fused_moe

        moe_type = state['moe_type']
        if moe_type == MoeType.Default or (isinstance(moe_type, str) and moe_type == 'Default'):
            hidden_states = state['hidden_states']
            topk_weights = state['topk_weights']
            topk_ids = state['topk_idx']

            # Reshape to 2D if 3D (like Qwen3MoeSparseMoeBlock does)
            was_3d = hidden_states.dim() == 3
            if was_3d:
                batch_size, seq_len, hidden_dim = hidden_states.shape
                hidden_states = hidden_states.view(-1, hidden_dim)
                topk_weights = topk_weights.view(-1, topk_weights.size(-1))
                topk_ids = topk_ids.view(-1, topk_ids.size(-1))

            # Find which experts are actually used in this batch
            unique_experts = torch.unique(topk_ids).cpu().tolist()

            # Dequantize only the experts that are actually used
            # Shape: (num_experts, in_features, quant_out_feats) -> (active_experts, out_features, in_features)
            gate_up_weights = self._dequantize_gate_up_weights(unique_experts)
            down_weights = self._dequantize_down_weights(unique_experts)

            # Remap topk_ids to the new expert indices
            expert_id_map = {old_id: new_id for new_id, old_id in enumerate(unique_experts)}
            remapped_topk_ids = topk_ids.clone()
            for old_id, new_id in expert_id_map.items():
                remapped_topk_ids[topk_ids == old_id] = new_id

            # Calculate expert_offset and num_experts for EP
            expert_offset = 0
            num_experts = None
            if self.expert_list is not None and len(self.expert_list) != self.num_experts:
                expert_offset = self.expert_list[0]
                num_experts = self.num_experts

            # Use standard fused_moe kernel
            hidden_states = fused_moe(
                hidden_states,
                gate_up_weights,
                down_weights,
                topk_weights=topk_weights,
                topk_ids=remapped_topk_ids,
                topk=self.impl.top_k,
                w1_bias=self.gate_up.bias,
                w2_bias=self.down.bias,
                expert_offset=expert_offset,
                num_experts=len(unique_experts),
                renormalize=self.do_renormalize,
                act_func=self.act_func,
            )

            # Reshape back to 3D if it was 3D
            if was_3d:
                hidden_states = hidden_states.view(batch_size, seq_len, -1)

            gemm_state = {'hidden_states': hidden_states, 'moe_type': moe_type}
        else:
            raise NotImplementedError(f'Not supported moe type: {moe_type}')
        return gemm_state

    def _dequantize_gate_up_weights(self, expert_indices):
        """Dequantize gate_up weights for specified experts only.

        Args:
            expert_indices: List of expert indices to dequantize

        Returns:
            Dequantized weights, shape (len(expert_indices), out_features, in_features)
        """
        from lmdeploy.pytorch.kernels.cuda.awq_kernels import awq_dequant_weights_single_expert

        results = []
        for e in expert_indices:
            expert_qw = self.gate_up.qweight[e]
            expert_s = self.gate_up.scales[e]
            expert_qz = self.gate_up.qzeros[e]

            result = awq_dequant_weights_single_expert(
                expert_qw, expert_s, expert_qz, self.w_bit, self.group_size
            )
            results.append(result.t())

        return torch.stack(results, dim=0)

    def _dequantize_down_weights(self, expert_indices):
        """Dequantize down weights for specified experts only.

        Args:
            expert_indices: List of expert indices to dequantize

        Returns:
            Dequantized weights, shape (len(expert_indices), out_features, in_features)
        """
        from lmdeploy.pytorch.kernels.cuda.awq_kernels import awq_dequant_weights_single_expert

        results = []
        for e in expert_indices:
            expert_qw = self.down.qweight[e]
            expert_s = self.down.scales[e]
            expert_qz = self.down.qzeros[e]

            result = awq_dequant_weights_single_expert(
                expert_qw, expert_s, expert_qz, self.w_bit, self.group_size
            )
            results.append(result.t())

        return torch.stack(results, dim=0)

    def combine(self, state: dict):
        """combine."""
        moe_type = state['moe_type']
        if moe_type == MoeType.Default:
            if self.all_reduce:
                state['hidden_states'] = moe_reduce(state['hidden_states'],
                                                    rank=self.tp_rank,
                                                    tp_mode=self.tp_mode,
                                                    group=self.tp_group)
            out_state = {'hidden_states': state['hidden_states'], 'moe_type': moe_type}
        else:
            raise NotImplementedError(f'Not supported moe type: {moe_type}')
        return out_state

    def wait(self, state):
        """wait."""
        return False
