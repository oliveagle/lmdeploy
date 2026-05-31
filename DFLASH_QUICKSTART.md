# DFlash 快速开始指南

> **状态**: ✅ 可用 | **更新**: 2026-05-11

## 快速开始

```bash
export LD_LIBRARY_PATH=/home/oliveagle/opt/lmdeploy/lmdeploy/build/lib:$LD_LIBRARY_PATH
```

## Demo 文件

| 文件 | 用途 |
|------|------|
| `test_dflash_simple.py` | 简单测试 |
| `demo_normal_inference.py` | 普通推理演示 |
| `demo_dflash_inference.py` | DFlash 推理演示 |

## Benchmark (性能测试)

| 文件 | 用途 | 运行时长 |
|------|------|----------|
| `benchmark_normal.py` | 普通推理 benchmark | 约 2 分钟 |
| `benchmark_dflash.py` | DFlash 推理 benchmark | 约 2 分钟 |

**使用方法**:
```bash
python benchmark_normal.py   # 先跑，记录结果
python benchmark_dflash.py   # 再跑，对比结果
```

## 代码示例

```python
from lmdeploy import pipeline, TurbomindEngineConfig, SpeculativeConfig, GenerationConfig

target_model = "/home/oliveagle/.cache/modelscope/hub/models/tclf90/Qwen3.5-9B-AWQ"
draft_model = "/home/oliveagle/.cache/huggingface/hub/models--z-lab--Qwen3.5-9B-DFlash/snapshots/492f4b532a957a50561e1418e5a3f31690f127f4"

speculative_config = SpeculativeConfig(
    method='dflash', model=draft_model,
    num_speculative_tokens=8, quant_policy=0
)

tm_config = TurbomindEngineConfig(
    model_format='awq', tensor_parallel=1,
    cache_max_entry_count=0.2, quant_policy=8, session_len=8192
)

pipe = pipeline(target_model, backend_config=tm_config, speculative_config=speculative_config)

gen_config = GenerationConfig(max_new_tokens=256, do_sample=False)
messages = [{"role": "user", "content": "人工智能是什么？"}]
response = pipe(messages, gen_config=gen_config,
                sequence_start=True, sequence_end=True,
                chat_template_kwargs={'enable_thinking': False})
print(response.text)
```

## 相关文档

- `DFLASH_INTEGRATION_SUMMARY.md` - 集成总结
- `DFLASH_COMPARISON.md` - 与 lucebox 对比
- `DFLASH_PERFORMANCE_NOTES.md` - 性能分析
- `BENCHMARK_README.md` - Benchmark 使用说明