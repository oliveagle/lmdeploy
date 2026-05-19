# Ralph Progress Log

This file tracks progress across iterations. Agents update this file
after each iteration and it is included in prompts for context.

## 2026-05-20 - lmdeploy-0lz
- **Issue**: `TM_TurboMind_InitFromPath` in `src/turbomind/capi/turbomind_c.cc` had `ContextGuard` created but going out of scope immediately after creation, before `LoadWeightsFromSafetensors` was called
- **Fix**: Moved `ctx_guard` creation to line 1315, before GPU allocations. The guard now covers all GPU tensor allocations (`tok_emb_param.alloc()`, `LoadWeightsFromSafetensors`) through the entire weight loading process until `ProcessWeights`/`CreateEngine`
- **Files changed**:
  - `src/turbomind/capi/turbomind_c.cc` — moved `ctx_guard` from between model creation and GPU allocations to right before the first GPU allocation
- **Learnings:**
  - `ContextGuard` is a RAII wrapper that pushes CUDA context + allocator on construction and pops on destruction
  - If guard goes out of scope before GPU operations, those operations run without the correct CUDA context
  - Pattern: Build module tree (CPU-only) first, then create ContextGuard, then do GPU allocations under guard
  - `ProcessWeights` creates its own guard internally, so only the C API path was affected
---

## 2026-05-20 - lmdeploy-j8d
- **Issue**: config/default.toml had `engine_type = "python_bridge"`, causing server to use python_bridge instead of pure_cpp
- **Fix**: Changed both `config/default.toml` and `default_engine_type()` / `default_model_engine_type()` in `src/config.rs` to use `"pure_cpp"`
- **Files changed**:
  - `lmdeploy-rust-server/config/default.toml` — `engine_type = "pure_cpp"`
  - `lmdeploy-rust-server/src/config.rs` — `default_engine_type()` and `default_model_engine_type()` return `"pure_cpp"`, updated comments
- **Verified**: 52 Rust tests pass, config loads correctly
- **Learnings:**
  - Config loading order: hardcoded defaults → `/etc/lmdeploy/config.toml` → bundled `default.toml` → env vars
  - `AppConfig::load()` uses `include_str!("../config/default.toml")` so the bundled TOML must match desired defaults
  - Both `default_engine_type()` and `default_model_engine_type()` default functions must be consistent with the TOML file

---

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

### CUDA ContextGuard RAII Pattern

**问题**: C++ `TM_TurboMind_InitFromPath` 中 `ContextGuard` 创建后立即超出作用域，导致 GPU 内存分配时没有正确的 CUDA 上下文。

**解决方案**: 确保 `ContextGuard` 覆盖所有 GPU 操作：

```cpp
// ❌ 错误: guard 立即超出作用域
auto ctx_guard = model_root->context();  // Line 1314
// ... 大量代码 ...
LoadWeightsFromSafetensors(...);  // Line 1731 - GPU 分配时 guard 已销毁!

// ✅ 正确: guard 在所有 GPU 操作前创建，在函数结束时销毁
auto* model_weight = model_root->text_model_ptr();

// 创建 guard - 覆盖后续所有 GPU 操作
auto ctx_guard = model_root->context();

// 1. GPU tensor 分配
tok_emb_param.alloc(shape, dtype);

// 2. 加载权重到 GPU
LoadWeightsFromSafetensors(model_weight, path, config);  // 内部创建 GPU tensor

// 3. ProcessWeights 内部创建自己的 guard
tm->instance->ProcessWeights(index);
```

**关键点**:
- `ContextGuard` 是 RAII 包装器：构造时 push CUDA context + allocator，析构时 pop
- 模块树构建 (CPU-only) 可以在 guard 外进行
- **所有** GPU 内存分配必须在 guard 作用域内
- `LoadWeightsFromSafetensors` 内部分配 GPU tensor，必须被 guard 覆盖

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
