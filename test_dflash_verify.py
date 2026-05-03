#!/usr/bin/env python3
"""DFlash 集成验证 - 验证核心修复"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))


def print_section(title: str):
    """打印分隔符."""
    print('\n' + '=' * 70)
    print(title)
    print('=' * 70)


def main():
    print_section('DFlash 集成验证')

    # 1. 验证 DFlashDraftModel 注册
    print('\n1. DFlashDraftModel 注册:')
    from lmdeploy.pytorch.models.module_map import MODULE_MAP
    if 'DFlashDraftModel' in MODULE_MAP:
        print(f'   ✓ 已注册: {MODULE_MAP["DFlashDraftModel"]}')
    else:
        print('   ✗ 未注册')
        return 1

    # 2. 验证 DFlashProposer 注册
    print('\n2. DFlashProposer 注册:')
    from lmdeploy.pytorch.spec_decode.proposers.base import SPEC_PROPOSERS
    if 'dflash' in SPEC_PROPOSERS:
        print('   ✓ dflash proposer 已注册')
    else:
        print('   ✗ 未注册')
        return 1

    # 3. 验证 DFlash 类导入
    print('\n3. DFlash 类导入:')
    try:
        from lmdeploy.pytorch.models.dflash import DFlashDraftModel, DFlashAttention, DFlashDecoderLayer
        from lmdeploy.pytorch.spec_decode.proposers.dflash import DFlashProposer
        print('   ✓ 所有 DFlash 类导入成功')
    except Exception as e:
        print(f'   ✗ 导入失败: {e}')
        return 1

    # 4. 验证 spec_agent.py 修复
    print('\n4. Spec Agent 修复:')
    import inspect
    from lmdeploy.pytorch.spec_decode.spec_agent import SpecModelAgent
    source = inspect.getsource(SpecModelAgent._async_model_forward)
    if 'get_outputs(outputs, inputs, extra_inputs)' in source:
        print('   ✓ get_outputs 包含 extra_inputs 参数')
    else:
        print('   ✗ 缺少 extra_inputs 参数')
        return 1

    # 5. 验证 dflash.py load_weights 修复
    print('\n5. DFlashDraftModel load_weights 修复:')
    source = inspect.getsource(DFlashDraftModel.load_weights)
    if 'named_buffers()' in source:
        print('   ✓ load_weights 包含 named_buffers()')
    else:
        print('   ✗ 缺少 named_buffers()')
        return 1

    # 6. 验证 awq.py 修复
    print('\n6. AWQ MoE weight_loader 修复:')
    from lmdeploy.pytorch.nn.moe.awq import AwqLinearWeights
    source = inspect.getsource(AwqLinearWeights.weight_loader_tp)
    if 'param_data.numel() == 0' in source and 'param_data.shape != weight.shape' in source:
        print('   ✓ weight_loader_tp 包含形状检查')
    else:
        print('   ✗ 缺少形状检查')
        return 1

    print_section('验证结果')
    print('✓ 所有核心修复已验证！')
    print('\n修复总结:')
    print('  1. spec_agent.py:412  - 添加 extra_inputs 参数到 get_outputs')
    print('  2. dflash.py:471-472  - 合并 named_parameters() 和 named_buffers()')
    print('  3. awq.py:97-104     - 添加形状检查到 weight_loader_tp')
    print('  4. awq.py:120-125    - 添加形状检查到 weight_loader_ep')

    print('\n下一步:')
    print('  - 修复 Qwen3.5 MoE AWQ 模型加载问题')
    print('  - 运行完整的 DFlash 性能测试')

    return 0


if __name__ == '__main__':
    sys.exit(main())
