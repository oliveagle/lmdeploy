# Rust + C++ 性能瓶颈分析

## 问题总结

用户报告：
- Python + C++ 的预填充速度好几万 tok/s
- Rust + PythonBridge + C++ 的整体吞吐量超过 600 tok/s
- Rust + C++ 的性能表现不佳

## 根本原因分析

### 1. **异步模型差异**

**Python 的实现：**
```python
# Python 使用真正的异步 semaphore
sem = StreamingSemaphore()
signal_cb = partial(self.async_signal_cb, sem)
outputs = self.model_inst.forward(..., signal_cb)

while True:
    await sem.acquire()  # 真正的异步等待，C++ 回调触发
    state = shared_state.consume()
    # 处理 token
```

**Rust 的实现：**
```rust
// Rust 使用轮询模式
tokio::task::spawn_blocking(move || {
    request.forward_async(...)?;
    loop {
        std::thread::sleep(std::time::Duration::from_millis(1));  // 轮询！
        let (status, _) = request.get_streaming_state()?;
        if status == FINISH { break; }
    }
});
```

**问题：**
- Python 的 `await sem.acquire()` 是事件驱动的，C++ 引擎完成 token 生成后通过回调唤醒
- Rust 的 `sleep(1ms) + get_streaming_state()` 是主动轮询，浪费 CPU 且延迟更高

### 2. **spawn_blocking 的线程池竞争**

```rust
tokio::task::spawn_blocking(move || {
    let (_permit, mut request) = pool.acquire_blocking();  // 可能阻塞
    // ... 所有工作都在这里
});
```

**问题：**
- `spawn_blocking` 使用专用的阻塞线程池（默认 512 线程）
- 如果所有线程都在 `acquire_blocking()` 等待，新请求会被拒绝
- 即使有线程可用，`acquire_blocking()` 本身也是同步阻塞调用

### 3. **Request Pool 的 slot 选择问题**

```rust
fn acquire_blocking(&self) {
    let permit = block_in_place(|| {
        block_on(self.semaphore.acquire())  // 获取许可
    });
    let active = self.slots.len() - self.semaphore.available_permits() - 1;
    let idx = active % self.slots.len();  // slot 选择
    let guard = self.slots[idx].blocking_lock();  // 可能再次等待
}
```

**问题：**
- `available_permits()` 在 `permit` 获取后才计算，与 `active` 计算之间存在竞态
- `blocking_lock()` 可能需要等待其他请求释放锁
- 多个请求可能选择同一个 slot（因为计算基于动态的 `available_permits()`）

### 4. **token_callback 的性能问题**

```rust
extern "C" fn token_callback(token_id: c_int, _seq_len: c_int, user_data: *mut c_void) {
    unsafe {
        let ctx = &*(user_data as *const StreamContext);
        let token_ids: Vec<u32> = vec![token_id as u32];  // 每次分配！
        let token_str = match ctx.tokenizer.decode(&token_ids, true) {
            Ok(s) if !s.is_empty() => s,
            _ => return,
        };
        let _ = ctx.tx.try_send(token_str);  // try_send 可能丢失 token
    }
}
```

**问题：**
- 每次 callback 都分配新的 `Vec<u32>`（即使是单个 token）
- `decode()` 调用是同步的，会阻塞 C++ 的 callback 线程
- `try_send()` 在 channel 满时会丢弃 token，导致输出不完整

### 5. **配置差异**

**Rust 的引擎配置：**
```rust
engine_config.set_session_len(65536);
engine_config.set_max_prefill_token_num(8192);
engine_config.set_max_batch_size(max_batch_size);  // GPU-adaptive
engine_config.set_async(1);
```

**Python 的配置：**
```python
TurbomindEngineConfig(
    session_len=2048,  # 默认更小
    max_batch_size=32,
    cache_block_seq_len=64,
    tp=1,
)
```

**差异：**
- Rust 使用更大的 `session_len`（65536 vs 2048），可能影响内存分配和缓存
- Rust 使用 GPU-adaptive 的 `max_batch_size`（A100=384, H100=1024），但实际只有单请求
- Python 的配置更保守，可能有不同的内部优化路径

## 统一口径对比测试方案

### 测试目标

1. **验证 Python Direct 的基准性能**
   - TTFT (Time To First Token)
   - 预填充速度 (tok/s)
   - 解码速度 (tok/s)
   - 整体吞吐量

2. **对比 Rust C++ 的性能**
   - 使用相同模型
   - 使用相同配置参数
   - 测量相同指标

3. **分析瓶颈点**
   - spawn_blocking vs async/await
   - 轮询 vs 回调
   - Request Pool 的锁竞争

### 测试脚本

```python
#!/usr/bin/env python3
"""
统一口径性能测试：Python Direct vs Rust C++
"""
import asyncio
import time
import json
import sys
from pathlib import Path

sys.path.insert(0, "/mnt/data/lmdeploy")

from lmdeploy.messages import GenerationConfig, TurbomindEngineConfig
from lmdeploy.turbomind import TurboMind
from lmdeploy.tokenizer import Tokenizer

MODEL_PATH = "/mnt/eaget-4tb/data/llm_server/models/Qwen3.5-9B"
CONTEXT_LENGTHS = [128, 256, 512, 1024, 2048, 4096, 8192]
OUTPUT_TOKENS = 128
RUNS = 5

def test_python_direct():
    """测试 Python Direct API"""
    tm = TurboMind(
        model_path=MODEL_PATH,
        engine_config=TurbomindEngineConfig(
            session_len=65536,  # 与 Rust 保持一致
            max_batch_size=128,   # 与 Rust 默认保持一致
            cache_block_seq_len=64,
            tp=1,
            enable_prefix_caching=False,
        ),
        trust_remote_code=True,
    )
    tokenizer = Tokenizer(MODEL_PATH, trust_remote_code=True)
    
    results = {}
    
    for ctx_len in CONTEXT_LENGTHS:
        print(f"\n=== Context: {ctx_len} tokens ===")
        
        prompt = "The quick brown fox " * (ctx_len // 6)
        input_ids = tokenizer.encode(prompt)
        actual_ctx_len = len(input_ids)
        
        run_results = []
        for run in range(RUNS):
            start = time.perf_counter()
            first_token_time = None
            token_count = 0
            
            gen_cfg = GenerationConfig(
                max_new_tokens=OUTPUT_TOKENS,
                temperature=0.7,
            )
            
            async for output in tm.async_stream_infer(
                session_id=run,
                input_ids=input_ids,
                gen_config=gen_cfg,
                sequence_start=True,
                sequence_end=True,
                stream_output=True,
            ):
                if output.status.value in (1, 2):  # SUCCESS or FINISH
                    elapsed = (time.perf_counter() - start) * 1000
                    if first_token_time is None:
                        first_token_time = elapsed
                    token_count += len(output.token_ids)
            
            total_ms = (time.perf_counter() - start) * 1000
            
            ttft_ms = first_token_time or 0
            prefill_tps = (actual_ctx_len / ttft_ms * 1000) if ttft_ms > 0 else 0
            decode_ms = total_ms - ttft_ms
            decode_tps = (token_count / decode_ms * 1000) if decode_ms > 0 else 0
            
            print(f"  Run {run+1}: TTFT={ttft_ms:.1f}ms, "
                  f"Prefill={prefill_tps:.0f} tps, Decode={decode_tps:.0f} tps, "
                  f"Total={total_ms:.1f}ms, Tokens={token_count}")
            
            run_results.append({
                "ttft_ms": ttft_ms,
                "prefill_tps": prefill_tps,
                "decode_tps": decode_tps,
                "total_ms": total_ms,
                "token_count": token_count,
            })
        
        # 计算平均值
        avg_ttft = sum(r["ttft_ms"] for r in run_results) / RUNS
        avg_prefill = sum(r["prefill_tps"] for r in run_results) / RUNS
        avg_decode = sum(r["decode_tps"] for r in run_results) / RUNS
        avg_total = sum(r["total_ms"] for r in run_results) / RUNS
        
        print(f"  Average: TTFT={avg_ttft:.1f}ms, "
              f"Prefill={avg_prefill:.0f} tps, Decode={avg_decode:.0f} tps")
        
        results[actual_ctx_len] = {
            "avg_ttft_ms": avg_ttft,
            "avg_prefill_tps": avg_prefill,
            "avg_decode_tps": avg_decode,
            "avg_total_ms": avg_total,
        }
    
    tm.close()
    return results

if __name__ == "__main__":
    print("="*60)
    print("Python Direct API 性能测试")
    print("="*60)
    print(f"Model: {MODEL_PATH}")
    print(f"Context lengths: {CONTEXT_LENGTHS}")
    print(f"Output tokens: {OUTPUT_TOKENS}")
    print(f"Runs per context: {RUNS}")
    
    results = test_python_direct()
    
    # 保存结果
    with open("/mnt/data/lmdeploy/lmdeploy-rust-server/python_baseline.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print("\n" + "="*60)
    print("测试完成！结果已保存到 python_baseline.json")
    print("="*60)
```

## 建议的修复方案

### 1. 使用真正的异步等待（替代轮询）

在 C++ FFI 中暴露一个等待完成的异步函数，或者使用 `futures::executor::block_on` 包装 future。

### 2. 优化 Request Pool

使用无锁队列或 channel 替代 semaphore + mutex 组合。

### 3. 优化 token_callback

- 预分配 token buffer
- 批量发送 token（而不是逐个发送）
- 在独立线程中解码 token（避免阻塞 C++ callback 线程）

### 4. 统一配置

确保 Rust 和 Python 使用相同的引擎配置参数。

## 下一步

1. 运行 Python 基准测试，获取真实性能数据
2. 使用相同配置测试 Rust C++ 引擎
3. 对比分析瓶颈点
4. 实施优化方案
