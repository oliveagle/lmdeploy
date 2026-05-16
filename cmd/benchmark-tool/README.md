# LMDeploy Benchmark Tool

通用的 LLM 推理性能压测工具，支持测试 Prefill 和 Decode 阶段在不同上下文大小和并发下的性能。

## 功能特性

- **Prefill 性能测试**: 测试预填充阶段吞吐量 (tokens/s)
- **Decode 性能测试**: 测试解码阶段吞吐量 (tokens/s)
- **TTFT 测试**: 测试首 token 延迟 (Time To First Token)
- **并发测试**: 支持不同并发级别
- **多场景配置**: 灵活配置上下文大小和输出长度

## 编译

```bash
cd cmd/benchmark-tool
go build -o lmdeploy-benchmark .
```

## 使用方式

### 方式一：命令行参数

```bash
./lmdeploy-benchmark \
  --url http://localhost:8000 \
  --model Qwen3.6-35B-A3B-AWQ \
  --contexts 512,1024,2048,4096,8192 \
  --outputs 128,256,512 \
  --concurrency 1,2,4,8 \
  --requests 10 \
  --output results.json
```

### 方式二：配置文件

```bash
./lmdeploy-benchmark --config config.json --output results.json
```

## 配置文件格式

```json
{
  "base_url": "http://localhost:8000",
  "model": "Qwen3.6-35B-A3B-AWQ",
  "context_sizes": [512, 1024, 2048, 4096, 8192, 16384],
  "output_lens": [128, 256, 512],
  "concurrency": [1, 2, 4, 8, 16],
  "num_requests": 10,
  "timeout_sec": 300
}
```

## 参数说明

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--url` | LMDeploy 服务器地址 | http://localhost:8000 |
| `--model` | 模型名称 | Qwen3.6-35B-A3B-AWQ |
| `--contexts` | 上下文大小 (tokens) | 512,1024,2048,4096,8192,16384 |
| `--outputs` | 输出长度 (tokens) | 128,256,512 |
| `--concurrency` | 并发数 | 1,2,4,8,16 |
| `--requests` | 每场景请求数 | 5 |
| `--timeout` | 请求超时 (秒) | 300 |
| `--api-key` | API 密钥 | 空 |
| `--output` | 输出 JSON 文件 | benchmark_results.json |
| `--config` | 配置文件 | 无 |

## 输出示例

```
=== PREFILL PERFORMANCE (tokens/second) ===
Scenario        | C=1        | C=2        | C=4        | C=8        | C=16       
------------------------------------------------------------------------------------------
C512_O128       | 38000      | 35000      | 32000      | 28000      | 24000     
C1024_O256      | 36000      | 33000      | 30000      | 26000      | 22000     
...

=== DECODE PERFORMANCE (tokens/second) ===
Scenario        | C=1        | C=2        | C=4        | C=8        | C=16       
------------------------------------------------------------------------------------------
C512_O128       | 42.5       | 38.2       | 32.1       | 25.8       | 18.3      
C1024_O256      | 41.8       | 37.5       | 31.2       | 24.9       | 17.5      
...

=== TIME TO FIRST TOKEN (milliseconds) ===
Scenario        | C=1        | C=2        | C=4        | C=8        | C=16       
------------------------------------------------------------------------------------------
C512_O128       | 13.5       | 15.2       | 18.3       | 22.1       | 28.7      
C1024_O256      | 28.4       | 32.1       | 38.5       | 46.2       | 58.9      
...
```

## 测试 LMDeploy

### 启动 LMDeploy 服务器

```bash
# 使用 TurboMind 后端
lmdeploy serve api_server \
  /mnt/eaget-4tb/modelscope_models/tclf90/Qwen3___6-35B-A3B-AWQ \
  --model-format awq \
  --tp 1 \
  --cache-max-entry-count 0.85
```

### 运行基准测试

```bash
./lmdeploy-benchmark \
  --url http://localhost:8000 \
  --model Qwen3.6-35B-A3B-AWQ \
  --contexts 1024,2048,4096,8192 \
  --outputs 256,512 \
  --concurrency 1,2,4,8 \
  --requests 10
```

## 性能指标说明

1. **Prefill TPS**: 预填充阶段每秒处理的 token 数，越高越好
2. **Decode TPS**: 解码阶段每秒生成的 token 数，越高越好
3. **TTFT**: 首个 token 的延迟时间，越低越好
4. **Overall TPS**: 整体吞吐量（包含预填充和解码）

## 注意事项

- 确保服务器正在运行并可访问
- 预留足够的测试时间（可能需要数小时）
- 大并发数可能需要调整服务器的 `max_batch_size`
- 输出 JSON 文件可用于进一步分析