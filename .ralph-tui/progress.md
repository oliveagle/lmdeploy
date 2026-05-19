# Ralph Progress Log

This file tracks progress across iterations. Agents update this file
after each iteration and it is included in prompts for context.

## 2026-05-20 - lmdeploy-7kt
- **Status**: Verified and closed - Pure Rust + C++ inference path complete
- **Files changed**:
  - `src/turbomind/capi/turbomind_c.h` - Fixed `TM_RequestStatus` enum forward declaration
  - C++ library rebuilt with streaming forward symbols exported
- **Verified Components**:
  - `TurboMindCEngine` - Pure C++ model loading and inference
  - `LMTokenizer` - Pure Rust tokenizer (HuggingFace tokenizers crate)
  - `ModelEngine` enum - PythonBridge vs PureCpp unified dispatch
  - `generate()` - Synchronous text generation
  - `generate_stream()` - Token-by-token streaming output
  - `generate_with_metrics()` - Generation with timing metrics
  - `reload()` - Hot model reloading
  - AWQ quantization detection and support
- **Test Results**:
  - 52 Rust tests passed
  - 3 Python integration tests passed
  - C++ streaming symbols exported: `TM_ModelRequest_ForwardAsync`, `TM_ModelRequest_GetStreamToken`, `TM_ModelRequest_GetStreamingState`
- **Learnings**:
  - `TM_RequestStatus` enum must be declared before functions that use it as parameter (forward declaration issue)
  - C++ library must be rebuilt after header changes to export new symbols
  - Pure C++ path has no Python dependency, suitable for production deployment
  - Python bridge path remains available for compatibility and testing

---

## Codebase Patterns

### C++ Streaming Forward Pattern

**问题**: C++ `TM_ModelRequest_Forward` 是同步阻塞的，无法实现流式输出。

**解决方案**: 添加非阻塞 `TM_ModelRequest_ForwardAsync` 和轮询机制：

```cpp
// C API: 提交异步推理请求
int TM_ModelRequest_ForwardAsync(
    TM_ModelRequest* req,
    TM_TensorMap* input_tensors,
    const TM_SessionParam* session,
    const TM_GenerationConfig* gen_cfg,
    bool stream_output,  // true = 启用逐 token 更新
    bool enable_metrics);

// 获取流式输出状态（原子交换，只返回一次）
int TM_ModelRequest_GetStreamingState(
    TM_ModelRequest* req,
    TM_RequestStatus* out_status,
    int* out_seq_len);

// 读取 output_ids tensor
int TM_ModelRequest_GetStreamToken(
    TM_ModelRequest* req,
    void** out_data,
    size_t* out_count);
```

**Rust 层实现**:
```rust
// 轮询循环在 spawn_blocking 线程中运行
loop {
    sleep(Duration::from_millis(5));  // 避免忙等待
    let (status, seq_len) = req.get_streaming_state()?;

    // 只处理新 token
    if seq_len > prev_seq_len {
        let (ptr, count) = req.get_stream_token()?;
        let tokens = unsafe { slice::from_raw_parts(ptr, count) };
        let new_tokens = &tokens[prev_seq_len..seq_len];
        tx.blocking_send(tokenizer.decode(new_tokens)?);
        prev_seq_len = seq_len;
    }

    if matches!(status, TM_STATUS_FINISH | TM_STATUS_CANCEL | ...) {
        break;
    }
}
```

**关键点**:
- `stream_output=true` 使 C++ 引擎在每 token 后调用 `UpdateState(r, Request::kOk, seq_len)`
- `AtomicRequestState::exchange(nullptr)` 原子交换只返回一次状态
- Rust 轮询检测 `seq_len` 增加，只解码新生成的 token
- `tokio_stream::wrappers::ReceiverStream` 将 `mpsc::Receiver` 转为 `Stream<Item = String>`

---
