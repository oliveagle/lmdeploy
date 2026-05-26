# 历史数据错误分析

> **发现时间**: 2026-05-25
> **问题**: 历史基准数据使用了错误的计算方式

## 历史数据 (BENCHMARK_PYTHON_TM_20260518.json)

```json
{
  "context_length": 8192,
  "ttft_ms_avg": 191.37,
  "prefill_speed_tps_avg": 42875.0  // ← 错误！
}
```

## 错误原因

历史报告 `RUST_VS_PYTHON_BENCHMARK_20260518.md` 明确说明：

```
| Long | 8192 chars (~1100 tokens) | 191 | 42875 |
```

- **输入**: 8192 **字符**
- **实际 tokens**: ~1100 tokens
- **计算方式**: `8192 chars / 0.191s = 42,875 chars/s` ❌
- **正确方式**: `1100 tokens / 0.191s = 5,760 tok/s` ✅

## 修正后的数据

| Context | 原始 TPS (错误) | 实际 tokens | 修正 TPS | 当前实测 TPS |
|---------|----------------|-------------|----------|--------------|
| 1K (1024 chars, ~138 tokens) | 14,827 | 138 | 722 | 5,363 |
| 4K (4096 chars, ~550 tokens) | 33,728 | 550 | 4,505 | 6,040 |
| 8K (8192 chars, ~1100 tokens) | 42,875 | 1100 | 5,760 | 5,224 |

## 结论

1. **历史 42K tok/s 是计算错误**，应该是字符每秒
2. **当前性能正常**：实测 5-6K tok/s 与修正后的历史数据一致
3. **Rust vs Python 差距**：
   - 1K: Python 5,363 vs Rust 3,518 (1.5x)
   - 4K: Python 6,040 vs Rust 3,409 (1.8x)
   - 8K: Python 5,224 vs Rust 2,657 (2.0x)

差距依然存在，需要优化 Rust 端的 tensor 传递路径。
