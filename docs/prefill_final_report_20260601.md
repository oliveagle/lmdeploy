# Prefill 性能分析综合报告 (2026-06-01)

## 摘要

本文档是对 Prefill 性能分析任务的最终总结，包含:
1. Python + TurboMind 真实性能数据
2. Rust Server 性能分析
3. 当前阻塞问题根因
4. 解决方案和 Beads 任务

---

## 一、Python + TurboMind 真实性能数据

### 测试配置
- **模型**: `/mnt/data/models/modelscope_models/Qwen3.6-35B-A3B-AWQ`
- **后端**: TurboMind C++ 引擎 (AWQ 4-bit)
- **测试方法**: 单请求串行，5 次运行取平均
- **测量指标**: TTFT, Prefill tok/s, Decode tok/s

### 多轮实测数据汇总

| 输入长度 | TTFT (ms) | Prefill (tok/s) | Decode (tok/s) |
|----------|-----------|-----------------|----------------|
| 512 | 118-120 | **4,253-4,312** | 39-42 |
| 1024 | 190-195 | **5,244-5,394** | 38-41 |
| 4096 | 676-709 | **5,778-6,070** | 34-39 |
| 8192 | 1561-1565 | **5,236-5,250** | 35-36 |

### 关键发现

**"好几万 tok/s" 不是真实单请求性能**：
1. **真实峰值**: **~6,000 tok/s** (4096 tokens)
2. **Archive 声称**: 6,408-12,614 tok/s - **与实测不符**
3. **Archive Commit 验证**: 回到 738f60eb 重新测试，实际只有 841-1076 tok/s
4. **可能来源**: aggregate 吞吐量、并发场景、不同测量方法

---

## 二、Rust Server 性能分析

### 2.1 当前状态: 阻塞

**核心问题**:
```
lmdeploy/lib/libturbomind_c.so: 6月1日 14:36 (旧版本)
src/turbomind/core/buffer.h:    6月1日 15:02 (新版本有 NULL 检查)
                                   ↓
                       .so 早于代码修改 - 需要重新编译
```

### 2.2 已完成修复

| Beads ID | 任务 | 提交 | 状态 |
|----------|------|------|------|
| lmdeploy-9yov | QKV Fusion 崩溃 | 8fabb2ad | ✅ Closed |
| lmdeploy-osrp | model-00009 加载后崩溃 | 17f12277 | ✅ Closed |
| lmdeploy-evkx | get_model_arch 参数 | (已有) | ✅ Closed |

### 2.3 当前阻塞任务

| Beads ID | 任务 | 优先级 | 状态 |
|----------|------|--------|------|
| **lmdeploy-tgpf** | CMake 重新编译 libturbomind_c.so | P0 | 🔄 Open |
| lmdeploy-q2f1 | Rust 超过 Python 优化方案 | P0 | 🔒 Blocked |

---

## 三、根本原因分析

### 3.1 Python vs Rust 加载路径差异

| 方面 | Python TurboMind | Rust Server |
|------|------------------|-------------|
| 绑定方式 | pybind11 直接 | C API InitFromPath |
| 权重加载 | `_tm_model.export()` | `LoadWeightsFromSafetensors()` |
| QKV Fusion | pybind11 内部 | C++ 代码 (已修复) |
| 库编译 | setup.py + cmake | 共享 libturbomind_c.so |

### 3.2 阻塞问题

**问题链**:
1. 6月1日修复了 C++ 代码 (`buffer.h` 加入 NULL 检查)
2. 修复后未重新编译 `libturbomind_c.so`
3. Rust binary 仍链接旧 .so
4. 运行 `prefill_benchmark` 崩溃在 buffer.h:70

---

## 四、解决方案

### 4.1 编译方案

**方案 A: setup.py (推荐)**
```bash
cd /mnt/data/lmdeploy
pip install -e . --no-build-isolation
```

**方案 B: 手动 CMake**
```bash
cd /mnt/data/lmdeploy
mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release -DBUILD_C_FFI=ON
cmake --build . --target turbomind_c -j$(nproc)
cp build/src/turbomind/capi/libturbomind_c.so lmdeploy/lib/
```

### 4.2 编译后验证

```bash
# 验证时间戳
stat -c "%Y %n" lmdeploy/lib/libturbomind_c.so src/turbomind/core/buffer.h
# .so 时间戳应该晚于 buffer.h

# 运行 benchmark
LD_LIBRARY_PATH=/mnt/data/lmdeploy/lmdeploy/lib \
    /mnt/data/lmdeploy/lmdeploy-rust-server/target/release/prefill_benchmark
```

---

## 五、超越 Python 的策略

### 5.1 已实施优化（25-43% 提升）

| 优化项 | 提交 | 状态 | 预期提升 |
|--------|------|------|----------|
| GPU Tokenizer 零拷贝 | f91db35e | ✅ | 5-10% |
| forward_async | affa5d7d | ✅ | 10-15% |
| Condvar wait() | affa5d7d | ✅ | 3-5% |
| Event Sync 消除 | affa5d7d | ✅ | 5-10% |
| Request Pool | (历史) | ✅ | 2-3% |

### 5.2 待实施优化（需先修复编译）

| 策略 | 预期提升 | 优先级 |
|------|---------|--------|
| TensorMap 复用 | 3-5% | P1 |
| 回调路径优化 | 3-5% | P2 |
| CUDA Stream 重叠 | 5-10% | P3 |
| Batch 调度 | 20-50% (并发) | P3 |

### 5.3 综合预期

- **单请求场景**: 30-50% 提升 → **~7,800-9,000 tok/s**
- **并发场景**: Batch 调度额外 20-50%

---

## 六、Beads 任务全状态

### 6.1 当前开放任务

| ID | 任务 | 优先级 | 状态 |
|----|------|--------|------|
| lmdeploy-tgpf | CMake 重新编译 libturbomind_c.so | P0 | Open |
| lmdeploy-q2f1 | Rust 超过 Python 优化方案 | P0 | Blocked |

### 6.2 任务依赖关系

```
lmdeploy-q2f1 (Rust 超过 Python)
  └── blocks: lmdeploy-tgpf (CMake 重新编译)
  └── blocks: lmdeploy-osrp (model-00009 崩溃) ✅
  └── blocks: lmdeploy-9yov (QKV Fusion 崩溃) ✅
  └── blocks: lmdeploy-evkx (get_model_arch) ✅
  └── blocks: lmdeploy-hqb2 (get_model_arch) ✅
```

---

## 七、验收标准

### 7.1 修复验收

- [ ] lmdeploy-tgpf 完成: `libturbomind_c.so` 重新编译
- [ ] .so 时间戳晚于 buffer.h 修改时间
- [ ] Rust Server 能成功加载 35B AWQ 模型
- [ ] prefill_benchmark 能运行并输出结果

### 7.2 性能验收

| 输入长度 | Python 基准 | Rust 目标 |
|----------|-------------|-----------|
| 512 | ~4,300 tok/s | ≥ 4,500 tok/s |
| 1024 | ~5,300 tok/s | ≥ 5,500 tok/s |
| 4096 | ~6,000 tok/s | ≥ 6,200 tok/s |
| 8192 | ~5,200 tok/s | ≥ 5,500 tok/s |

---

## 八、下一步行动

### 立即执行 (P0)

1. **lmdeploy-tgpf**: 通过 CMake 重新编译 `libturbomind_c.so`
2. 验证 .so 包含 NULL 检查
3. 运行 `prefill_benchmark` 获取 Rust 性能数据

### 验证阶段

4. 对比 Python vs Rust 真实数据
5. 根据实际差距决定进一步优化

### 长期目标

6. Rust Prefill 超过 Python: 目标 **>6,200 tok/s**
7. 实施剩余优化策略（TensorMap 复用、回调优化等）

---

## 九、结论

1. **Python 真实性能**: ~6,000 tok/s 峰值，不是"好几万"
2. **Rust 阻塞原因**: .so 旧于代码修复，需要重新编译
3. **解决方案**: CMake 编译或 setup.py install
4. **超越预期**: 30-50% 提升空间，可达 7,800-9,000 tok/s

---

## 十、参考文档

- `docs/prefill_performance_deep_analysis_20260531.md`
- `docs/prefill_analysis_final_20260531.md`
- `docs/prefill_summary_20260601.md`
- `docs/prefill_analysis_complete_20260531.md`
- `benchmarks-archive/python-turbomind_35b_awq_20260526/README.md`

## 十一、关键文件位置

### C++ 代码
- `/mnt/data/lmdeploy/src/turbomind/capi/turbomind_c.cc` - C API 实现
- `/mnt/data/lmdeploy/src/turbomind/capi/CMakeLists.txt` - 构建配置
- `/mnt/data/lmdeploy/src/turbomind/core/buffer.h` - 修复位置
- `/mnt/data/lmdeploy/CMakeLists.txt` - 主构建

### Rust 代码
- `/mnt/data/lmdeploy/lmdeploy-rust-server/src/model/cpp_engine.rs`
- `/mnt/data/lmdeploy/lmdeploy-rust-server/src/bin/prefill_benchmark.rs`
- `/mnt/data/lmdeploy/lmdeploy-rust-server/build.rs`

### 编译产物
- `/mnt/data/lmdeploy/lmdeploy/lib/libturbomind_c.so` - 需重新编译
- `/mnt/data/lmdeploy/lmdeploy/lib/_turbomind.cpython-312-x86_64-linux-gnu.so`

### Python 代码
- `/mnt/data/lmdeploy/lmdeploy/turbomind/turbomind.py`
- `/mnt/data/lmdeploy/lmdeploy/archs.py`
- `/mnt/data/lmdeploy/lmdeploy/tokenizer.py`

### 测试脚本
- `/mnt/data/lmdeploy/benchmarks-archive/python-turbomind_35b_awq_20260526/scripts/benchmark_turbomind_quick.py`
