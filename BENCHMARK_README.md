# DFlash Benchmark 使用说明

## 两个真正的 Benchmark 脚本

| 文件 | 用途 | 运行时长 |
|------|------|----------|
| `benchmark_normal.py` | 普通推理性能测试 | 约 2 分钟 |
| `benchmark_dflash.py` | DFlash 推理性能测试 | 约 2 分钟 |

## 使用方法

### 1. 测试普通推理
```bash
python benchmark_normal.py
```

### 2. 测试 DFlash 推理
```bash
python benchmark_dflash.py
```

## 配置

- **运行时长**: 120 秒 (2 分钟)
- **输出 Token 数**: max_new_tokens = 128
- **采样方式**: greedy (do_sample=False)
- **Prompt 循环**: 5 个简单问题循环使用

## 输出指标

每个脚本会输出：
- 每次请求的耗时、输出 Token 数、速度 (tokens/s)
- 每 10 次请求的实时平均
- 最终统计：
  - 总请求数
  - 总 Tokens
  - 总耗时
  - 平均耗时/请求
  - 平均速度 (tokens/s)

## 对比方法

1. 先运行 `benchmark_normal.py`，记录结果
2. 再运行 `benchmark_dflash.py`，记录结果
3. 对比两个的平均速度和平均耗时

注意：两个脚本是独立的，避免同时运行导致 OOM。