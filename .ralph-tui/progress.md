# Ralph Progress Log

This file tracks progress across iterations. Agents update this file
after each iteration and it's included in prompts for context.

## Codebase Patterns (Study These First)

*Add reusable patterns discovered during development here.*

- **TurboMind Engine Thread Lifecycle**: `CreateEngine` → `engines_[index].Start()` → spawns `internal_thread_` (infinite loop in `InternalThreadEntry`) + `executor_.Start()`. The main thread calls `gateway_.pop(blocking=true)` which waits on `pthread_cond_wait` until requests arrive. This is **normal idle state**, NOT a hang.
- **WarmUp is gated by `need_warm_up_` flag** (default=1): If enabled, `WarmUp` sends synthetic requests to the gateway and waits for completion via `std::promise::get_future().wait()`. If the gateway is not properly initialized, this could hang.
- **RequestQueue blocking pop**: `cv_.wait(lock, [this] { return !(queue_.empty() && kill_.empty()) || closed_; });` - blocks indefinitely until a request is pushed or queue is closed.
- **Gateway constructor** spawns `signal_thread_` which runs `signal_thread_entry()` - also uses `take_all(abort)` with blocking wait.

---

## 2026-05-22 - lmdeploy-v3k
- Investigated C++ InitFromPath hang with gdb
- Created debug tools: `scripts/debug_initfrompath_hang.sh`, `src/turbomind/capi/test_initfrompath_stepby.cc`
- Added build target for step-by-step test in CMakeLists.txt
- Created investigation document: `docs/debug_initfrompath_hang_20260522.md`
- **Key Finding**: `CreateEngine` → `Start()` → `InternalThreadEntry` → `gateway_.pop(blocking=true)` is NORMAL idle behavior (pthread_cond_wait). The engine threads are designed to block waiting for requests.
- **Potential actual hang points**:
  1. `ProcessWeights` → `prepare()` (CUDA operations)
  2. `WarmUp` (if enabled) → sends test request via gateway, waits on promise
  3. `CreateEngine` itself - unlikely, but could hang if `h_comm->Sync()` deadlocks
- **Files changed**: `scripts/debug_initfrompath_hang.sh`, `src/turbomind/capi/test_initfrompath_stepby.cc`, `src/turbomind/capi/CMakeLists.txt`, `docs/debug_initfrompath_hang_20260522.md`
---