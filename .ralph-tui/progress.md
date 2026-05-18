## Codebase Patterns
- **TurboMind C API 不支持 HuggingFace safetensors**: `turbomind_c.cc` 中的 safetensors reader 缺少 AWQ 量化解包和反量化逻辑
- **Python 自动 HF→TM 转换**: Python API 通过 `get_tm_config()` → `Qwen3_5Model.model()` → `ModelLoader.export()` 自动转换，但只加载到 GPU 内存，不写入磁盘 `.bin` 文件
- **Rust 自动转换检测**: `engine.rs` 检测 HuggingFace safetensors 格式，自动调用 Python 脚本转换为 TurboMind 格式
- **AWQ 自动检测**: 从 config.json 检测 `quantization_config.quant_method == "awq"` 并设置 `quant_policy=4`
- **Python 推理桥接**: `python_inference_bridge.py` 使用 LMDeploy `Pipeline` API 提供 stdin/stdout JSON 接口，支持 HuggingFace 格式模型
- **C API 模型树构建问题**: `InitFromPath` 只创建空 `ModelWeight`，没有 Python 中的 `TextModelBuilder` 等模型树构建逻辑
- **GPU 内存限制**: Qwen3.6-35B-A3B-AWQ 在 32GB V100 上 session_len=2048 可行，4096 OOM
- **模型路径**: Qwen3.6-35B-A3B-AWQ 位于 `/mnt/eaget-4tb/modelscope_models/tclf90/` (不是 tclf00)
- **Cargo 编译极慢**: 本地 cargo 检查需要很长时间（>2分钟），预留 3-5 分钟编译时间

## 2026-05-18 - lmdeploy-ghi
- **生成 Rust vs Python 性能对比报告**: 汇总了 Python LMDeploy 的性能测试结果，分析 Rust Server 的阻塞原因
- **修改的文件**:
  - `BENCHMARK_RUST_VS_PYTHON_20260518.md`: 新增性能对比报告
    - Python TurboMind 基准测试结果（1K/4K/8K context）
    - Decode 速度稳定在 ~41 tokens/s
    - Prefill 速度随 context 增长（14K → 42K tokens/s）
    - TTFT 随 context 线性增长（70ms → 191ms）
    - Rust Server 阻塞原因分析（C API 无法加载 HF 格式）
    - 改进建议和下一步计划
- **Learnings**:
  - Python TurboMind 的 prefill 速度随 context 增长而提高（更好的 GPU 利用）
  - Decode 速度受显存带宽限制，几乎不随 context 变化
  - ITL (Inter-Token Latency) 稳定在 ~24ms，与 decode 速度一致
  - 历史数据对比显示不同测试间差异 <2%，性能稳定
  - Rust Server 需要预转换模型或完善 C API HF 支持才能执行基准测试

---

## 2026-05-18 - lmdeploy-vh5
- **创建 Rust 基准测试框架**: 实现了完整的性能基准测试工具，支持 TTFT、Prefill 速度、Decode 速度、多场景测试
- **修改的文件**:
  - `lmdeploy-rust-server/src/model/benchmark.rs`: 新增基准测试模块
    - `BenchmarkConfig`: 可配置 context lengths (1024/4096/8192)、output length、迭代次数
    - `BenchmarkResult`: 单次测试结果，包含 TTFT、prefill/decode 速度
    - `BenchmarkSummary`: 按 context length 聚合统计
    - `BenchmarkRunner`: 直接调用 engine 的基准测试
    - `HttpBenchmarkRunner`: 通过 HTTP API 进行基准测试
  - `lmdeploy-rust-server/src/model/mod.rs`: 添加 benchmark 模块导出
  - `lmdeploy-rust-server/Cargo.toml`: 添加 criterion 依赖和 benchmark target
  - `lmdeploy-rust-server/benches/benchmark.rs`: criterion 基准测试套件
    - prefill: 1024/4096/8192 context lengths
    - decode: 128/256/512/1024 output lengths
    - full_request: 4 种场景组合
    - http_api: 序列化/反序列化开销
  - `lmdeploy-rust-server/scripts/run_benchmark.sh`: 一键运行脚本
- **Learnings**:
  - Criterion 基准测试需要 `harness = false` 才能与 async tokio runtime 配合使用
  - 基准测试支持两种模式：直接调用 engine 和通过 HTTP API
  - TTFT 测量需要 stream API 才能精确记录第一个 token 到达时间
  - 非流式 API 的 TTFT 只能估算（~20% prefill time）

## 2026-05-18 - lmdeploy-k7d
- **实现自动 HF→TM 模型转换**: 在 `engine.rs` 中添加了 HuggingFace 格式检测和自动转换功能
- **创建 Python 转换辅助脚本**: `scripts/convert_hf_to_turbomind.py` 用于执行实际的模型转换
- **添加 AWQ 量化检测**: 自动检测 config.json 中的 AWQ 配置并设置正确的 quant_policy
- **修改的文件**:
  - `lmdeploy-rust-server/src/model/engine.rs`:
    - 添加 `has_hf_safetensors()` 检测 HF 格式
    - 添加 `is_turbomind_workspace()` 检测 TM 格式
    - 添加 `convert_hf_to_turbomind()` 调用 Python 转换脚本
    - 添加 `detect_awq_quantization()` 检测 AWQ 量化
    - 修改 `init()` 方法在加载前自动检测并转换模型
  - `lmdeploy-rust-server/scripts/convert_hf_to_turbomind.py`:
    - Python 脚本用于将 HuggingFace 模型转换为 TurboMind 格式
    - 支持 AWQ 量化模型的自动配置
- **工作流程**:
  1. 检测模型是否为 TurboMind 格式（有 config.yaml 或 triton_models）
  2. 如果不是，检测是否为 HuggingFace safetensors 格式
  3. 如果是 HF 格式，调用 Python 转换脚本转换为 workspace
  4. 检测 AWQ 量化并设置 quant_policy=4
  5. 使用转换后的 workspace 路径初始化 TurboMind C API
- **Learnings**:
  - Python TurboMind API 的转换逻辑封装在 `TurboMind.__init__()` → `_from_hf()` → `ModelLoader.export()` 中
  - C API 的 `InitFromPath` 期望 TurboMind 格式，不能直接加载 HF safetensors
  - AWQ 模型需要设置 `quant_policy=4` 才能正确加载

---

## 2026-05-18 - lmdeploy-v8o
- **创建 Python LMDeploy 基准测试工具**: 实现了与 Rust 基准测试同口径的 Python 性能测试工具
- **修改的文件**:
  - `examples/python_benchmark.py`: 新增 Python 基准测试脚本
    - 配置与 Rust benchmark 一致: 1K/4K/8K context lengths, 512 output tokens, 3 次迭代
    - 支持 stream 和 non-stream 两种模式（stream 用于精确 TTFT 测量）
    - 自动检测 AWQ 量化并设置 quant_policy=4
    - 记录 TTFT、prefill 速度、decode 速度、GPU 内存占用
    - 结果输出到 JSON 文件，格式与 Rust benchmark 兼容
- **关键 API 发现**:
  - `GenerationConfig` 使用 `max_new_tokens` 而非 `max_tokens`
  - `async_engine.tokenizer` 用于准确 token 计数（而非 hf_tokenizer）
  - `stream_infer` 返回迭代器，但具体行为取决于输入是单个 prompt 还是 prompt 列表
  - `infer` 返回单个 Response 或 Response 列表
- **Learnings**:
  - Python LMDeploy pipeline 初始化时间约 53 秒（包含权重加载）
  - stream_infer 返回类型复杂：可能是 Response 对象或 Response 迭代器，需要运行时判断
  - 使用 `getattr(response, "text", "")` 安全访问 Response.text 属性
  - 估算 TTFT：stream 模式可精确测量，非 stream 模式约为总时间的 20%

---

## 2026-05-18 - lmdeploy-xt9
- **完成 Rust 基准测试工具**: 创建了完整的性能基准测试 CLI 工具
- **修改的文件**:
  - `lmdeploy-rust-server/examples/benchmark.rs`: 新增基准测试示例程序
    - 使用 `BenchmarkRunner` 执行 1K/4K/8K context 场景测试
    - 每次场景运行 3 次迭代取平均
    - 记录 TTFT、prefill 速度、decode 速度、GPU 内存占用
    - 结果输出到 JSON 文件
  - `lmdeploy-rust-server/Cargo.toml`: 添加 chrono 依赖用于时间戳
  - `lmdeploy-rust-server/scripts/run_benchmark.sh`: 修正模型路径为 tclf90
  - `lmdeploy-rust-server/scripts/convert_hf_to_turbomind.py`: 修复无效的 cache_max_entry_count=0.0
  - `lmdeploy-rust-server/src/model/benchmark.rs`: 修复 Serialize/Deserialize derives
- **Learnings**:
  - 模型路径修正: `/mnt/eaget-4tb/modelscope_models/tclf00/` → `/mnt/eaget-4tb/modelscope_models/tclf90/`
  - C API `InitFromPath` 无法直接加载 HuggingFace safetensors，期望 TurboMind 格式的 `.bin` 文件
  - Python `TurboMind.__init__()` 虽然能做 HF→TM 转换，但只加载到内存，不写入磁盘
  - Rust Server 需要预转换模型或等待 C API HF 支持完善
  - 基准测试工具已就绪，但模型转换问题是阻塞项

---

## 2026-05-18 - lmdeploy-ah9
- **实现 Python 推理桥接**: 创建了基于 LMDeploy Pipeline API 的 Python 子进程桥接，支持 HuggingFace 格式模型的真实加载和推理
- **修改的文件**:
  - `lmdeploy-rust-server/scripts/python_inference_bridge.py`: 新增 Python 推理桥接脚本
    - 使用 `Pipeline` API 代替直接调用 TurboMind C API
    - 支持 stdin/stdout JSON 协议与 Rust 通信
    - 自动检测 AWQ 量化并配置 quant_policy=4
    - 成功加载 Qwen3.6-35B-A3B-AWQ 模型并生成文本
  - `lmdeploy-rust-server/scripts/convert_hf_to_turbomind.py`: 原转换脚本保留但不再使用（OOM 问题）
  - `test_bridge.py`: 直接测试脚本（验证 Pipeline API 可用性）
- **关键发现**:
  - **C API 的根本问题**: TurboMind C API 的 `InitFromPath` 只创建空的 `ModelWeight`，没有构建模型树
  - **Python API 的优势**: Python 中的 `Qwen3_5Model.model()` 方法通过 `TextModelBuilder` 构建完整的 C++ 模型树，然后通过 `ModelLoader.export()` 从 safetensors 加载权重到 GPU 内存
  - **AWQ 量化**: Qwen3.6-35B-A3B-AWQ 使用 4-bit AWQ 量化，需要在 `TurbomindEngineConfig` 中设置 `quant_policy=4`
  - **GPU 内存限制**: session_len=4096 时 OOM，session_len=2048 可正常工作（约 40 秒加载时间）
  - **Pipeline.infer() API**: 非流式 API 返回 `Response` 对象，直接访问 `response.text` 获取生成文本
- **Python 桥接验证成功**:
  - 模型加载: ~40 秒（40 个层，每层 ~1 秒）
  - 推理成功: 输入 "Hello, what is 2+2?" 成功生成回答
  - 协议验证: stdin/stdout JSON 通信正常工作
- **下一步**:
  - 修改 Rust engine.rs 使用 Python 桥接替代 C API
  - 或者直接使用 Python LMDeploy 作为推理后端（Rust 只负责 HTTP 服务）

---
- **完成 AWQ 模型加载失败原因分析**: 根本原因是 C API `InitFromPath` 期望 TurboMind `.bin` 格式，不能直接加载 HuggingFace safetensors。Python API 有自动 HF→TM 转换，但 C API 没有。
- **分析文件**: `AWQ_MODEL_LOAD_ANALYSIS_20260518.md`
- **关键发现**:
  1. `turbomind_c.cc` 的 safetensors reader 缺少 AWQ 量化权重的解包和反量化逻辑
  2. Rust `engine.rs` 的 `init()` 方法定义了 `set_quant_policy()` 但未调用
  3. `InitFromHF()` 通过 Python 脚本桥接但标记为 `TM_ERR_NOT_IMPLEMENTED`
- **修复方案** (文档中已详细列出):
  - A: 预转换模型到 TurboMind 格式（推荐，零代码改动）
  - B: Rust Server 启动时自动检测并调用 Python 转换
  - C: 扩展 C API 原生支持 AWQ safetensors（长期）

---
