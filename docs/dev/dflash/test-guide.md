# DFlash 测试指南

## 测试脚本清单

### 核心测试
| 文件 | 用途 | 说明 |
|------|------|------|
| `test_dflash_integration.py` | 集成验证 (10/10 检查) | 无需 GPU |

### EP=4 测试
| 文件 | 用途 |
|------|------|
| `test_dflash_ep4_simple.py` | Baseline vs DFlash 对比 |
| `test_dflash_ep4_benchmark.py` | 性能基准测试 |
| `test_dflash_ep4_complete.py` | 完整 EP=4 多配置测试 |

### 调试测试
| 文件 | 用途 |
|------|------|
| `test_dflash_simple.py` | 最小化功能验证 |
| `test_dflash_load.py` | 模型加载测试 |
| `test_dflash_no_ep.py` | 无 EP 模式的基线测试 |

## 运行方式

### 1. 集成验证（无需 GPU）

```bash
source ~/venvs/lmdeploy/bin/activate
python test_dflash_integration.py
```

### 2. 最小化测试（4K, spec=5）

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 \
python -m torch.distributed.run --nproc_per_node=4 \
test_dflash_simple.py
```

### 3. 性能基准测试

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 \
python -m torch.distributed.run --nproc_per_node=4 \
test_dflash_ep4_benchmark.py --quick
```

生成 `dflash_ep4_benchmark_results.json`。

### 4. 完整对比测试

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 \
python -m torch.distributed.run --nproc_per_node=4 \
test_dflash_ep4_complete.py
```

## 硬件要求

| 测试 | 最低 GPU |
|------|----------|
| 集成验证 | 无需 GPU |
| 最小化测试 | V100 32GB 或 A100 40GB |
| 基准测试 | A100 40GB+ 推荐 |

## 预期输出

每个测试会显示：
- ✓ 模型加载时间
- ✓ Prefill 时间和速度
- ✓ Decode 时间和速度
- ✓ 加速比（相对于 Baseline）
- ✓ 接受率估计
