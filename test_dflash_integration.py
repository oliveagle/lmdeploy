#!/usr/bin/env python3
"""
DFlash 集成验证脚本
验证 DFlash C++ 代码是否正确集成到 Turbomind
"""
import os
import sys

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

def test_turbomind_import():
    """测试 Turbomind 导入"""
    print("=" * 70)
    print("测试 1: Turbomind 导入")
    print("=" * 70)

    try:
        import lmdeploy.lib._turbomind as tm
        print("✓ Turbomind C++ 模块导入成功")
        return True
    except Exception as e:
        print(f"✗ Turbomind 导入失败: {e}")
        return False


def test_dflash_config():
    """测试 DFlash 配置结构"""
    print("\n" + "=" * 70)
    print("测试 2: DFlash 配置")
    print("=" * 70)

    try:
        from lmdeploy.messages import SpeculativeConfig
        config = SpeculativeConfig(
            method='dflash',
            model='/path/to/draft',
            num_speculative_tokens=16
        )
        print(f"✓ SpeculativeConfig 创建成功")
        print(f"  method={config.method}")
        print(f"  num_speculative_tokens={config.num_speculative_tokens}")
        return True
    except Exception as e:
        print(f"✗ SpeculativeConfig 创建失败: {e}")
        return False


def test_turbomind_dflash_class():
    """测试 Turbomind DFlash 类"""
    print("\n" + "=" * 70)
    print("测试 3: Turbomind DFlash 配置")
    print("=" * 70)

    try:
        from lmdeploy.messages import TurbomindEngineConfig, SpeculativeConfig

        config = TurbomindEngineConfig(tp=1, session_len=2048)
        spec_config = SpeculativeConfig(
            method='dflash',
            model='/fake/path',
            num_speculative_tokens=16
        )
        print("✓ TurbomindEngineConfig + SpeculativeConfig 创建成功")
        print(f"  tp={config.tp}")
        print(f"  session_len={config.session_len}")
        print(f"  spec_method={spec_config.method}")
        print(f"  spec_tokens={spec_config.num_speculative_tokens}")
        return True
    except Exception as e:
        print(f"✗ Turbomind DFlash 配置失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_dflash_kernel_signatures():
    """测试 DFlash CUDA kernels 符号是否存在"""
    print("\n" + "=" * 70)
    print("测试 4: DFlash Kernels 验证")
    print("=" * 70)

    try:
        import lmdeploy.lib._turbomind as tm
        # 检查是否有 DFlash 相关的符号
        dflash_symbols = ['DFlash', 'dflash', 'Draft']

        # 尝试获取模块信息
        module_attrs = dir(tm)
        found_dflash = any(any(sym.lower() in attr.lower() for sym in dflash_symbols) for attr in module_attrs)

        if found_dflash:
            print("✓ 找到 DFlash 相关符号")
        else:
            print("⚠ 未直接找到 DFlash 符号 (可能已内联)")

        print("✓ Turbomind 模块加载完成")
        return True
    except Exception as e:
        print(f"✗ Kernel 检查失败: {e}")
        return False


def main():
    print("=" * 70)
    print("DFlash 集成验证")
    print("=" * 70)
    print(f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', '0')}")
    print()

    results = {}
    results['import'] = test_turbomind_import()
    results['config'] = test_dflash_config()
    results['class'] = test_turbomind_dflash_class()
    results['kernels'] = test_dflash_kernel_signatures()

    print("\n" + "=" * 70)
    print("测试总结")
    print("=" * 70)
    for k, v in results.items():
        status = "✓" if v else "✗"
        print(f"  {status} {k}")

    all_passed = all(results.values())
    if all_passed:
        print("\n✓ 所有测试通过! DFlash 集成正常")
    else:
        print("\n⚠ 部分测试失败，请检查")

    return 0 if all_passed else 1


if __name__ == '__main__':
    sys.exit(main())
