# Rust C++ 性能瓶颈分析报告

## 问题回顾

用户报告的性能数据：
- Python + C++ 预填充速度：好几万 tok/s
- Rust + PythonBridge + C++ 整体吞吐量：超过 600 tok/s
- Rust + C++ 性能：表现不佳（越做越差）

## 关键瓶颈分析

### 1. 轮询 vs 回调（核心问题）

**Python 的实现（事件驱动）：**
```python
# Python 使用 semaphore + callback 模式
sem = StreamingSemaphore()
signal_cb = partial(self.async_signal_cb, sem)

# forward() 提交后立即返回，C++ 引擎在后台处理
outputs = self.model_inst.forward(..., signal_cb)

# 事件循环等待 C++ 回调唤醒
while True:
    await sem.acquire()  # 零 CPU 占用，事件驱动
    state = shared_state.consume()
    # 处理 token
```

**Rust 的实现（轮询）：**
```rust
// Rust 使用 spawn_blocking + 轮询模式
tokio::task::spawn_blocking(move || {
    request.forward_async(...)?;  // 提交异步请求
    
    // 轮询状态（CPU 浪费！）
    loop {
        std::thread::sleep(std::time::Duration::from_millis(1));
        let (status, _) = request.get_streaming_state()?;
        if status == FINISH { break; }
    }
});
```

**问题：**
- Python 的 `await sem.acquire()` 是零 CPU 占用的事件驱动模式
- Rust 的 `sleep(1ms) + get_streaming_state()` 每秒轮询 1000 次
- 轮询引入了 0-1ms 的额外延迟，影响 TTFT

### 2. spawn_blocking 线程池瓶颈

```rust
tokio::task::spawn_blocking(move || {
    // 所有工作都在这里完成：
    // 1. acquire_blocking（可能阻塞）
    // 2. tensor 准备
    // 3. FFI 调用
    // 4. 轮询等待
});
```

**问题：**
- `spawn_blocking` 使用专用的阻塞线程池（默认 512 线程）
- 如果所有线程都在 `acquire_blocking()` 等待，新请求会被拒绝
- 即使有线程可用，`acquire_blocking()` 本身是同步阻塞调用

### 3. Request Pool 的锁竞争

```rust
fn acquire_blocking(&self) {
    let permit = block_in_place(|| {
        block_on(self.semaphore.acquire())  // 获取许可
    });
    let active = self.slots.len() - self.semaphore.available_permits() - 1;
    let idx = active % self.slots.len();
    let guard = self.slots[idx].blocking_lock();  // 可能再次等待
}
```

**问题：**
- `available_permits()` 在获取 permit 后计算，与 `active` 计算不同步
- `blocking_lock()` 可能需要等待其他请求释放锁
- 多个请求可能选择同一个 slot（因为计算基于动态值）

### 4. token_callback 的性能问题

```rust
extern "C" fn token_callback(token_id: c_int, _seq_len: c_int, user_data: *mut c_void) {
    unsafe {
        let ctx = &*(user_data as *const StreamContext);
        let token_ids: Vec<u32> = vec![token_id as u32];  // 每次分配！
        let token_str = match ctx.tokenizer.decode(&token_ids, true) {
            Ok(s) if !s.is_empty() => s,
            _ => return,
        };
        let _ = ctx.tx.try_send(token_str);  // 可能丢失 token
    }
}
```

**问题：**
- 每次 callback 都分配新的 `Vec<u32>`（即使是单个 token）
- `decode()` 是同步调用，会阻塞 C++ callback 线程
- `try_send()` 在 channel 满时会丢弃 token，导致输出不完整

### 5. 同步 vs 异步 forward 调用

**非流式路径使用 `forward()`（同步）：**
```rust
// generate_with_metrics 使用同步 forward
match request.forward(
    &mut input_tensors,
    &session,
    &gen_cfg,
    false,  // stream_output=false
    true,   // enable_metrics
    &mut output_tensors,
) {
    // 这是一个阻塞调用，一次性完成所有推理
}
```

**流式路径使用 `forward_async()` + 轮询：**
```rust
// generate_stream_impl 使用异步 forward + 轮询
if let Err(e) = request.forward_async(
    &mut input_tensors,
    &session,
    &gen_cfg,
    true,   // stream_output=true
    false,  // enable_metrics=false
) {
    // 轮询等待
    loop {
        std::thread::sleep(std::time::Duration::from_millis(1));
        let (status, _) = request.get_streaming_state()?;
        if status == FINISH { break; }
    }
}
```

**关键差异：**
- `forward()` 是同步阻塞调用，一次性完成推理，性能最优
- `forward_async()` + 轮询引入了额外开销

## 性能数据对比

根据用户报告：
- Python + C++ 预填充：几万 tok/s（使用 `forward()` 同步调用）
- Rust + C++ 预填充：未知（但表现不佳）
- Rust + PythonBridge + C++ 整体吞吐：600+ tok/s

## 修复方案

### 1. 实现真正的异步等待（替代轮询）

在 C++ FFI 中暴露一个等待完成的函数：

```rust
// 在 turbomind_c.rs 中添加
extern "C" {
    fn TM_ModelRequest_WaitForCompletion(request: *mut TM_ModelRequest) -> c_int;
}

// 在 Rust 中使用
request.forward_async(...)?;
TM_ModelRequest_WaitForCompletion(request.as_ptr())?;
```

### 2. 优化 Request Pool 使用无锁队列

```rust
struct RequestPool {
    slots: Vec<ModelRequest>,
    available: crossbeam::queue::SegQueue<usize>,  // 无锁队列
}

async fn acquire(&self) -> ModelRequest {
    loop {
        if let Some(idx) = self.available.pop() {
            return self.slots[idx].lock().await;
        }
        tokio::time::sleep(Duration::from_micros(100)).await;
    }
}
```

### 3. 优化 token_callback

```rust
// 使用预分配的 buffer
thread_local! {
    static TOKEN_BUFFER: RefCell<Vec<u32>> = RefCell::new(Vec::with_capacity(128));
}

extern "C" fn token_callback(token_id: c_int, _seq_len: c_int, user_data: *mut c_void) {
    unsafe {
        TOKEN_BUFFER.with(|buf| {
            buf.borrow_mut().push(token_id as u32);
        });
        // 批量发送，而不是每次都发送
    }
}
```

### 4. 非流式路径使用同步 forward

对于不需要流式输出的场景，直接使用同步 `forward()` 调用：

```rust
pub async fn generate(&self, prompt: &str, params: GenerationParams) -> String {
    // ... tokenization ...
    
    // 使用同步 forward，不使用 spawn_blocking
    match request.forward(
        &mut input_tensors,
        &session,
        &gen_cfg,
        false,  // stream_output=false
        false,  // enable_metrics=false
        &mut output_tensors,
    ) {
        Ok(_) => { /* decode and return */ }
        Err(e) => { /* handle error */ }
    }
}
```

## 建议的测试方案

1. **Python 基准测试**：使用相同配置获取真实性能数据
2. **Rust C++ 对比测试**：使用相同配置测试 Rust C++ 引擎
3. **火焰图分析**：对比 Python vs Rust 的调用栈
4. **统一配置**：确保两种模式使用相同的 session_len、batch_size 等参数
