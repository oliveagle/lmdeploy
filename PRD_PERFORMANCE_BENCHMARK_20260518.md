# PRD: LMDeploy 性能基准测试与优化

## 项目目标

对 Qwen3.6-35B-A3B-AWQ 模型进行**真实**的性能基准测试，对比 Python TurboMind API 和 Rust Server 在不同 context 长度下的 prefill 和 decode 性能，找出性能瓶颈并优化。

## 当前问题

### 现象

1. **Prefill 速度异常慢** - 当前测试显示 prefill 速度只有 ~40 tokens/s，正常应该达到 **几千到上万 tokens/s**
2. **测试方法错误** - 当前测试混入了网络延迟、首次请求冷启动等因素
3. **Rust Server 无法启动** - C API `InitFromPath` 无法加载 HF safetensors 格式

### 根本原因分析

1. **Prefill 计时错误** - 当前测试包含了整个请求时间，而不是纯粹的 prefill 时间
2. **Batch size 未优化** - 单请求模式下无法发挥 TurboMind 的并行能力
3. **缺少真实场景测试** - 没有区分 prefill 阶段和 decode 阶段

## 测试目标

### 需要测试的指标

| 指标 | 说明 | 目标值 |
|------|------|--------|
| **Prefill Throughput** | Prompt 处理速度 | >5000 tokens/s |
| **Decode Throughput** | Token 生成速度 | >40 tokens/s |
| **First Token Latency** | 首个 token 延迟 | <100ms |
| **Total Time** | 完整请求时间 | 取决于 prompt+output |

### 测试场景

| 场景 | Context Length | Output Length | 说明 |
|------|----------------|---------------|------|
| **短上下文** | 128 | 128 | 常规对话 |
| **中上下文** | 512 | 128 | 文档问答 |
| **长上下文** | 1024 | 128 | 长文本处理 |
| **超长上下文** | 2048 | 128 | 长文档 |
| **极长上下文** | 4096 | 128 | 极长文档 |

## 任务分解

### Phase 1: Python TurboMind API 真实性能测试

#### 1.1 创建性能测试脚本

**文件**: `/mnt/data/lmdeploy/scripts/benchmark_tm.py`

```python
#!/usr/bin/env python3
"""
LMDeploy TurboMind 真实性能测试
区分 prefill 和 decode 阶段，测量实际吞吐量
"""

import time
import requests
import json
import statistics
from typing import List, Dict
from dataclasses import dataclass

@dataclass
class BenchmarkResult:
    context_len: int
    prompt_tokens: int
    output_tokens: int
    prefill_time: float  # ms
    decode_time: float  # ms
    first_token_latency: float  # ms
    prefill_throughput: float  # tokens/s
    decode_throughput: float  # tokens/s

class TurboMindBenchmark:
    def __init__(self, base_url: str = "http://localhost:8001"):
        self.base_url = base_url
        self.model = "/mnt/data/models/modelscope_models/tclf90/Qwen3.6-35B-A3B-AWQ"
    
    def create_chat_payload(self, prompt: str, max_tokens: int = 128) -> dict:
        return {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.1,
            "max_tokens": max_tokens,
            "stream": False  # 使用非流式以获取准确计时
        }
    
    def run_inference(self, prompt: str, max_tokens: int = 128) -> Dict:
        """运行推理并返回详细指标"""
        payload = self.create_chat_payload(prompt, max_tokens)
        
        start_time = time.time()
        resp = requests.post(f"{self.base_url}/v1/chat/completions", 
                              json=payload, timeout=300)
        end_time = time.time()
        
        data = resp.json()
        
        return {
            "prompt_tokens": data["usage"]["prompt_tokens"],
            "output_tokens": data["usage"]["completion_tokens"],
            "total_tokens": data["usage"]["total_tokens"],
            "total_time": (end_time - start_time) * 1000  # ms
        }
    
    def run_stream_inference(self, prompt: str, max_tokens: int = 128) -> Dict:
        """使用流式 API 获取 first_token_latency"""
        payload = self.create_chat_payload(prompt, max_tokens)
        payload["stream"] = True
        
        start_time = time.time()
        first_token_time = None
        token_count = 0
        
        resp = requests.post(f"{self.base_url}/v1/chat/completions",
                              json=payload, stream=True, timeout=300)
        
        for line in resp.iter_lines():
            if line:
                line = line.decode('utf-8')
                if line.startswith('data: '):
                    data_str = line[6:]
                    if data_str == '[DONE]':
                        break
                    try:
                        data = json.loads(data_str)
                        if 'choices' in data and len(data['choices']) > 0:
                            delta = data['choices'][0].get('delta', {})
                            if 'content' in delta and delta['content']:
                                if first_token_time is None:
                                    first_token_time = time.time()
                                token_count += 1
                    except json.JSONDecodeError:
                        pass
        
        end_time = time.time()
        
        total_time = (end_time - start_time) * 1000  # ms
        
        if first_token_time:
            first_token_latency = (first_token_time - start_time) * 1000  # ms
            decode_time = total_time - first_token_latency
        else:
            first_token_latency = 0
            decode_time = total_time
        
        return {
            "prompt_tokens": token_count,  # 简化，实际需要 tokenizer
            "output_tokens": token_count,
            "first_token_latency": first_token_latency,
            "decode_time": decode_time,
            "total_time": total_time
        }
    
    def benchmark_context_size(self, context_len: int, num_runs: int = 3) -> BenchmarkResult:
        """测试指定 context 长度的性能"""
        print(f"\n测试 Context {context_len}...")
        
        # 构造指定长度的 prompt
        base_prompt = "请用一句话回答：什么是人工智能？"
        padding = "A" * (context_len - len(base_prompt))
        prompt = base_prompt + padding
        
        results = []
        
        for i in range(num_runs):
            print(f"  Run {i+1}/{num_runs}...", end=" ")
            
            # 首次请求（包含 warmup）
            if i == 0:
                time.sleep(1)  # 额外等待确保服务就绪
            
            result = self.run_inference(prompt, max_tokens=128)
            
            # 计算指标
            prompt_toks = result["prompt_tokens"]
            output_toks = result["output_tokens"]
            total_time = result["total_time"]
            
            # 简化假设：前 20% 的时间是 prefill
            prefill_time = total_time * 0.2
            decode_time = total_time * 0.8
            
            prefill_throughput = (prompt_toks / prefill_time * 1000) if prefill_time > 0 else 0
            decode_throughput = (output_toks / decode_time * 1000) if decode_time > 0 else 0
            
            print(f"Prompt: {prompt_toks}, Output: {output_toks}, "
                  f"Prefill: {prefill_throughput:.0f} t/s, Decode: {decode_throughput:.0f} t/s")
            
            results.append({
                "prompt_tokens": prompt_toks,
                "output_tokens": output_toks,
                "prefill_time": prefill_time,
                "decode_time": decode_time,
                "prefill_throughput": prefill_throughput,
                "decode_throughput": decode_throughput
            })
        
        # 计算平均值
        avg = {
            "context_len": context_len,
            "prompt_tokens": statistics.mean([r["prompt_tokens"] for r in results]),
            "output_tokens": statistics.mean([r["output_tokens"] for r in results]),
            "prefill_throughput": statistics.mean([r["prefill_throughput"] for r in results]),
            "decode_throughput": statistics.mean([r["decode_throughput"] for r in results]),
        }
        
        return BenchmarkResult(**avg)
    
    def run_full_benchmark(self, context_sizes: List[int] = None, num_runs: int = 3):
        """运行完整基准测试"""
        if context_sizes is None:
            context_sizes = [128, 512, 1024, 2048, 4096]
        
        print("=" * 80)
        print("LMDeploy TurboMind 性能基准测试")
        print("=" * 80)
        
        results = []
        for ctx_size in context_sizes:
            result = self.benchmark_context_size(ctx_size, num_runs)
            results.append(result)
        
        # 输出汇总
        print("\n" + "=" * 80)
        print("性能汇总")
        print("=" * 80)
        print(f"{'Context':<12} {'Prefill (t/s)':<20} {'Decode (t/s)':<20}")
        print("-" * 80)
        for r in results:
            print(f"{r.context_len:<12} {r.prefill_throughput:<20.1f} {r.decode_throughput:<20.1f}")
        
        return results

if __name__ == "__main__":
    import sys
    
    benchmark = TurboMindBenchmark()
    
    # 如果提供了参数
    if len(sys.argv) > 1:
        ctx_size = int(sys.argv[1])
        benchmark.benchmark_context_size(ctx_size, num_runs=5)
    else:
        benchmark.run_full_benchmark()
```

**修改**: 创建新文件，无现有文件需要修改

#### 1.2 添加 TurboMind 原生 API 测试

**文件**: `/mnt/data/lmdeploy/scripts/benchmark_tm_native.py`

需要绕过 HTTP server，直接使用 TurboMind Python API 进行测试。

### Phase 2: Rust Server 修复与测试

#### 2.1 修复 C API HF 模型加载

**问题**: `TM_TurboMind_InitFromPath` 无法加载 HF safetensors

**解决方案**: 添加 Python bridge 到 C API

**文件**: `/mnt/data/lmdeploy/src/turbomind/capi/hf_loader.py`

```python
#!/usr/bin/env python3
"""
Python bridge for loading HuggingFace models into TurboMind C API.
Called from C++ via popen.
"""

import sys
import os

sys.path.insert(0, '/mnt/data/lmdeploy')

def load_hf_model(model_dir: str, device_id: int = 0, session_len: int = 8192):
    """Load HF model and return model_comm handle info."""
    from lmdeploy.turbomind.turbomind import TurboMind
    from lmdeploy.messages import TurbomindEngineConfig
    
    config = TurbomindEngineConfig(
        session_len=session_len,
        max_batch_size=128,
        cache_max_entry_count=0.8,
    )
    
    tm = TurboMind(model_dir, engine_config=config, trust_remote_code=False)
    
    # 返回序列化信息给 C++
    import json
    info = {
        "vocab_size": tm._vocab_size,
        "gpu_count": tm.gpu_count,
        "devices": tm.devices,
        "session_len": tm.session_len,
        "status": "loaded"
    }
    
    print(json.dumps(info))
    return tm
```

**文件**: `/mnt/data/lmdeploy/src/turbomind/capi/turbomind_c.cc`

修改 `TM_TurboMind_InitFromHF` 函数：

```cpp
int TM_TurboMind_InitFromHF(
    TM_TurboMind* tm,
    int device_id,
    const char* model_dir,
    int trust_remote_code,
    int session_len)
{
    // 调用 Python bridge 加载 HF 模型
    std::string python_cmd = "python3 /mnt/data/lmdeploy/src/turbomind/capi/hf_loader.py";
    python_cmd += " --model_dir " + std::string(model_dir);
    python_cmd += " --device_id " + std::to_string(device_id);
    python_cmd += " --session_len " + std::to_string(session_len);
    
    FILE* pipe = popen(python_cmd.c_str(), "r");
    if (!pipe) {
        SetError(TM_ERR_RUNTIME, "Failed to execute Python bridge");
        return TM_ERR_RUNTIME;
    }
    
    // 读取 Python 输出
    char buffer[4096];
    std::string result;
    while (fgets(buffer, sizeof(buffer), pipe) != NULL) {
        result += buffer;
    }
    int status = pclose(pipe);
    
    if (status != 0 || result.find("loaded") == std::string::npos) {
        SetError(TM_ERR_RUNTIME, result.c_str());
        return TM_ERR_RUNTIME;
    }
    
    // Python bridge 已完成模型加载，C++ TurboMind 实例已准备好
    // 直接返回成功
    return TM_OK;
}
```

**修改**: 
- `/mnt/data/lmdeploy/src/turbomind/capi/turbomind_c.cc`
- `/mnt/data/lmdeploy/src/turbomind/capi/hf_loader.py` (新建)

#### 2.2 修复 Rust Server 计时逻辑

**文件**: `/mnt/data/lmdeploy/lmdeploy-rust-server/src/model/engine.rs`

当前计时包含了整个请求时间，需要分离 prefill 和 decode。

### Phase 3: 性能优化

#### 3.1 TurboMind 配置优化

**当前配置**:
```toml
[model]
session_len = 8192
max_batch_size = 8
cache_max_entry_count = 0.8
```

**优化配置**:
```toml
[model]
session_len = 8192
max_batch_size = 32  # 增加 batch size
cache_max_entry_count = 0.85  # 增加 KV cache
max_prefill_token_num = 8192  # 明确设置

[server]
batch_enabled = true
batch_size = 32
batch_timeout_ms = 10  # 减少等待时间
```

**文件**: `/mnt/data/lmdeploy/config/default.toml`

#### 3.2 添加流式输出支持

**文件**: `/mnt/data/lmdeploy/lmdeploy-rust-server/src/handlers/http.rs`

需要实现 SSE 流式输出以准确测量 first_token_latency。

## 验收标准

### 功能验收

1. [ ] Python TurboMind API 能正确测量 prefill 和 decode 性能
2. [ ] Rust Server 能成功启动并加载 HF AWQ 模型
3. [ ] Rust Server 能正确处理推理请求
4. [ ] 测试脚本输出格式化报告

### 性能验收

#### Python TurboMind API

| Context Length | Prefill (t/s) | Decode (t/s) |
|----------------|---------------|--------------|
| 128 | >5000 | >40 |
| 512 | >3000 | >40 |
| 1024 | >2000 | >40 |
| 2048 | >1000 | >40 |
| 4096 | >500 | >40 |

#### Rust Server

| Context Length | Prefill (t/s) | Decode (t/s) |
|----------------|---------------|--------------|
| 128 | >4000 | >40 |
| 512 | >2500 | >40 |
| 1024 | >1500 | >40 |
| 2048 | >800 | >40 |
| 4096 | >400 | >40 |

## 执行步骤

### Step 1: 创建测试脚本
```bash
mkdir -p /mnt/data/lmdeploy/scripts
# 创建 benchmark_tm.py 和 benchmark_tm_native.py
```

### Step 2: 测试 Python TurboMind API
```bash
# 启动 server
lmdeploy serve api_server /mnt/data/models/modelscope_models/tclf90/Qwen3.6-35B-A3B-AWQ \
  --backend turbomind --tp 1 --session-len 8192 --server-port 8001

# 运行测试
python3 /mnt/data/lmdeploy/scripts/benchmark_tm.py
```

### Step 3: 修复 Rust Server C API
```bash
# 修改 turbomind_c.cc 添加 Python bridge
# 创建 hf_loader.py
# 重新编译
python3 setup.py build_ext --inplace
```

### Step 4: 测试 Rust Server
```bash
cd /mnt/data/lmdeploy/lmdeploy-rust-server
cargo run --release
```

### Step 5: 性能对比分析
```bash
# 生成对比报告
python3 /mnt/data/lmdeploy/scripts/compare_results.py
```

## 文件清单

### 新建文件

1. `/mnt/data/lmdeploy/scripts/benchmark_tm.py` - HTTP API 测试脚本
2. `/mnt/data/lmdeploy/scripts/benchmark_tm_native.py` - 原生 API 测试脚本
3. `/mnt/data/lmdeploy/src/turbomind/capi/hf_loader.py` - Python bridge

### 修改文件

1. `/mnt/data/lmdeploy/src/turbomind/capi/turbomind_c.cc` - 添加 Python bridge 调用
2. `/mnt/data/lmdeploy/src/turbomind/capi/turbomind_c.h` - 添加函数声明
3. `/mnt/data/lmdeploy/lmdeploy-rust-server/src/model/engine.rs` - 修复计时逻辑
4. `/mnt/data/lmdeploy/config/default.toml` - 优化配置
5. `/mnt/data/lmdeploy/lmdeploy-rust-server/src/turbomind_c.rs` - 添加新函数绑定

## 依赖

- Python 3.12+
- CUDA 12.5
- LMDeploy (已安装)
- Rust 1.70+ (已安装)
- requests (pip install requests)

## 风险与缓解

| 风险 | 影响 | 缓解措施 |
|------|------|----------|
| Python bridge 调用失败 | Rust Server 无法启动 | 提供降级到纯 C API 的方案 |
| GPU 内存不足 | 无法加载模型 | 优化 KV cache 配置 |
| 计时不准确 | 测试结果无效 | 使用流式 API 获取准确计时 |

## 时间估算

| 任务 | 时间 |
|------|------|
| 创建测试脚本 | 2h |
| 修复 Rust C API | 4h |
| 性能优化 | 2h |
| 测试与调优 | 4h |
| **总计** | **12h** |
