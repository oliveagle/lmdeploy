# InitFromPath Hang Investigation Guide

## Overview

This document provides guidance for debugging hangs in `TM_TurboMind_InitFromPath` when using the C++ TurboMind C API.

## InitFromPath Call Sequence

The `TM_TurboMind_InitFromPath` function performs the following steps:

1. **CreateContext** - Creates CUDA context
2. **CreateRoot** - Creates ModelRoot sentinel
3. **ParseHfConfig** - Parses HuggingFace `config.json`
4. **Build Module Tree** - Creates ModelWeight module hierarchy
5. **LoadWeightsFromSafetensors** - Loads weights from `.safetensors` files
6. **ProcessWeights** - Calls `weights_[index]->prepare()` - moves weights to GPU
7. **CreateEngine** - Creates inference engine and starts background threads

## Potential Hang Points

### 1. ProcessWeights (Line 2237 in turbomind_c.cc)

```cpp
tm->instance->ProcessWeights(index);
```

**What it does**: Calls `prepare()` on the ModelWeight, which:
- Allocates GPU memory for all weights
- Transfers weights from CPU to GPU
- Calls `prepare()` on each module (Attention, FFN, etc.)

**Potential hang causes**:
- CUDA stream synchronization waiting for kernel completion
- CUDA out-of-memory handling (cudaMallocManaged fallback)
- ContextGuard allocator initialization

**Key files**:
- `src/turbomind/turbomind.cc:90-98` - ProcessWeights implementation
- `src/turbomind/models/model_weight.h` - ModelWeight::prepare()

### 2. CreateEngine (Line 2240 in turbomind_c.cc)

```cpp
tm->instance->CreateEngine(index);
```

**What it does**:
1. Creates `LanguageModel` object
2. Creates `Engine` object
3. Calls `engine.Start()` which spawns:
   - `internal_thread_` - runs `InternalThreadEntry()` infinite loop
   - `executor_.Start()` - spawns ModelExecutor thread

**Potential hang causes**:
- **InternalThreadEntry** enters infinite loop with `gateway_.pop(blocking=true)`
- Thread synchronization (tp_group_->Sync, dp_group_->Sync)
- Gateway queue initialization

**Key files**:
- `src/turbomind/turbomind.cc:264-298` - CreateEngine implementation
- `src/turbomind/engine/engine.cc:100-104` - Engine::Impl::Start()
- `src/turbomind/engine/engine.cc:813-900` - Engine::Impl::InternalThreadEntry()
- `src/turbomind/engine/request_queue.h:42-64` - RequestQueue::pop()

## The Blocking Gateway Pop Issue

The main issue is in `Engine::Impl::InternalThreadEntry()`:

```cpp
const int  n_free   = param_.max_batch_size - st.size() + st.finish;
const bool blocking = n_free == param_.max_batch_size;  // TRUE when empty!

gateway_.pop(rs->infer, rs->kill, n_free, blocking, rs->abort, dp_group_, queue_id_);
```

When `st.size() == 0` (no active requests), `blocking == true`, causing:
```cpp
cv_.wait(lock, [this] { return !(queue_.empty() && kill_.empty()) || closed_; });
```

This **blocks indefinitely** waiting for a request to be pushed to the queue.

**Important**: This is NOT a hang - it's the normal idle state. The thread is waiting for inference requests.

## Debugging Tools

### 1. Step-by-Step Test Program

```bash
# Build the step-by-step test
cd build
cmake .. -DBUILD_C_FFI=ON
make test_initfrompath_stepby

# Run the test
./src/turbomind/capi/test_initfrompath_stepby /path/to/model
```

The test logs each step with timestamps to `/tmp/initfrompath_stepby.log`.

### 2. GDB Thread Analysis

```bash
# Run the hanging process in background
./lmdeploy-server &
PID=$!

# After it hangs, attach gdb
sudo ./scripts/debug_initfrompath_hang.sh $PID gdb_threads.txt

# Analyze the output
cat gdb_threads.txt
```

**What to look for**:
- Thread in `RequestQueue::pop` at `pthread_cond_wait` - Normal idle state
- Thread in `InternalThreadEntry` - Normal (waiting for requests)
- Thread stuck in CUDA operation - Actual hang
- Thread in `ModelWeight::prepare()` - Weight processing hang

### 3. Strace for System Calls

```bash
# Trace system calls and signals
strace -p $PID -f -e trace=file,network,process -o strace.log

# Check for futex calls (mutex/condition variable waits)
grep futex strace.log | head -20
```

### 4. CUDA Profiling

```bash
# Check CUDA operations
nvprof ./test_initfrompath_stepby /path/to/model

# Or use Nsight Systems for detailed timeline
nsys profile --stats=true ./test_initfrompath_stepby /path/to/model
```

## Expected Behavior

After `CreateEngine` returns:
1. The `internal_thread_` should be running in `InternalThreadEntry()`
2. The `executor_` thread should be running
3. Both threads will be **blocked** waiting for requests
4. This is **normal** - the engine is ready to accept inference requests

The threads are NOT hung - they're waiting in:
- `gateway_.pop()` - RequestQueue condition variable wait
- `inbound_.pop()` - ModelExecutor queue wait

## Verifying Engine is Ready

After `InitFromPath` returns successfully, verify:

```c
// Check if engine is created
TM_TurboMind_GetScheduleMetrics(tm, index,
    &total_seqs, &active_seqs, &waiting_seqs,
    &total_blocks, &active_blocks, &cached_blocks, &free_blocks);

// Should return 0 (success) even with no active requests
// Metrics should show 0 active/total sequences
```

## Common Issues and Solutions

### Issue 1: Actual Hang in ProcessWeights

**Symptoms**: Never returns from `ProcessWeights`

**Diagnosis**:
```bash
gdb -batch -p $PID -ex "thread apply all bt"
# Look for thread stuck in prepare() or CUDA operations
```

**Solution**: Check GPU memory, CUDA driver version, and model weight files.

### Issue 2: Gateway Not Initialized

**Symptoms**: `InternalThreadEntry` crashes or gateway is null

**Diagnosis**: Check `TurboMind::Impl` constructor and gateway initialization

**Solution**: Ensure gateway is created before `CreateEngine` is called.

### Issue 3: Missing Queue Binding

**Symptoms**: `gateway_.pop()` aborts with "queue not found"

**Diagnosis**: Check `queue_id_` matches a valid queue in gateway

**Solution**: Verify queue registration in gateway constructor.

## Files Modified for This Investigation

1. `scripts/debug_initfrompath_hang.sh` - GDB automation script
2. `src/turbomind/capi/test_initfrompath_stepby.cc` - Step-by-step test program
3. `src/turbomind/capi/CMakeLists.txt` - Build configuration for test program

## Related Beads

- `lmdeploy-v3k` - InitFromPath hang investigation
