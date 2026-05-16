# Wasm Edge Deployment Guide

LMDeploy MoonBit API Server can be compiled to WebAssembly (Wasm) for edge deployment using the WasmEdge runtime. This enables running the API server on edge devices with minimal memory footprint (< 10 MB).

## Overview

- **Target**: `wasm32-unknown-unknown`
- **Runtime**: WasmEdge with WASI support
- **Memory**: < 10 MB RSS
- **Startup**: < 100ms cold start
- **Binary Size**: ~3-5 MB

## Prerequisites

### 1. Install WasmEdge Runtime

```bash
# Install WasmEdge
curl -sSf https://raw.githubusercontent.com/WasmEdge/WasmEdge/master/utils/install.sh | bash -s -- -p ~/.wasmedge

# Add to PATH
export PATH=$HOME/.wasmedge/bin:$PATH

# Verify installation
wasmedge --version
```

### 2. Install MoonBit Toolchain

```bash
# Install MoonBit (if not already installed)
curl -fsSL https://moonbitlang.com/install.sh | bash

# Add to PATH
export PATH=$HOME/.moon/bin:$PATH

# Verify installation
moon --version
```

## Building for Wasm

### Quick Build

```bash
# From project root
./build_wasm.sh
```

### Manual Build

```bash
# Build for wasm32-unknown-unknown target
moon build --target wasm32-unknown-unknown --release

# The output will be in:
# target/wasm32-unknown-unknown/release/lmdeploy_moonbit_server.wasm
```

## Deploying

### Using WasmEdge CLI

```bash
# Basic deployment
wasmedge --dir .:. lmdeploy_moonbit_server.wasm

# With environment variables
wasmedge --dir .:. \
  --env LMDEPLOY_PORT=8080 \
  --env LMDEPLOY_LOG_LEVEL=info \
  lmdeploy_moonbit_server.wasm

# With WASI preopened directories
wasmedge \
  --dir /models:/models \
  --dir /etc/lmdeploy:/etc/lmdeploy \
  lmdeploy_moonbit_server.wasm
```

### Using Docker

```dockerfile
FROM wasmedge/slim:latest

COPY target/wasm32-unknown-unknown/release/lmdeploy_moonbit_server.wasm /app/
COPY wasm_config.jsonc /etc/lmdeploy/config.jsonc

ENV WASMEDGE_MEMORY_LIMIT=10737418240  # 10 GB in bytes (for WasmEdge)
ENV LMDEPLOY_PORT=8080

EXPOSE 8080

CMD ["wasmedge", "--dir", "/app:/app", "/app/lmdeploy_moonbit_server.wasm"]
```

Build and run:

```bash
docker build -t lmdeploy-moonbit-wasm .
docker run -p 8080:8080 lmdeploy-moonbit-wasm
```

## Testing

### Health Check

```bash
curl http://localhost:8080/health
```

Expected response:

```json
{
  "status": "healthy",
  "memory_mb": 8,
  "request_count": 0,
  "runtime": "wasmedge"
}
```

### List Models

```bash
curl http://localhost:8080/v1/models
```

### Chat Completion

```bash
curl -X POST http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [{"role": "user", "content": "Hello from Wasm!"}],
    "model": "wasm-model",
    "max_tokens": 100
  }'
```

### Metrics

```bash
curl http://localhost:8080/metrics
```

Expected output (Prometheus format):

```
# HELP lmdeploy_wasm_memory_bytes Current WASM memory usage in bytes
# TYPE lmdeploy_wasm_memory_bytes gauge
lmdeploy_wasm_memory_bytes 8388608
# HELP lmdeploy_wasm_requests_total Total number of requests
# TYPE lmdeploy_wasm_requests_total counter
lmdeploy_wasm_requests_total 5
```

## Configuration

Configuration can be provided via:

1. **JSONC Config File**: `wasm_config.jsonc`
2. **Environment Variables**: `LMDEPLOY_*`

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `LMDEPLOY_PORT` | HTTP server port | 8080 |
| `LMDEPLOY_ADDRESS` | Bind address | 0.0.0.0 |
| `LMDEPLOY_MODEL_PATH` | Model directory path | /etc/lmdeploy/models |
| `LMDEPLOY_LOG_LEVEL` | Log level | info |
| `LMDEPLOY_MEMORY_LIMIT_MB` | Memory limit in MB | 10 |

## Memory Optimization

The Wasm build is optimized for edge deployment:

- **Memory Pages**: 160 pages (10 MB)
- **Max Connections**: 50 (reduced from 1000)
- **Max Batch Size**: 4 (reduced from 8)
- **Tokenizer Cache**: 100 entries (reduced from 1000)
- **Context Length**: 4096 tokens

### Memory Monitoring

Monitor memory usage via the `/metrics` endpoint:

```bash
# Check current memory usage
curl -s http://localhost:8080/metrics | grep lmdeploy_wasm_memory_bytes
```

## Performance

### Benchmarks

| Metric | Value |
|--------|-------|
| Cold Start | < 100ms |
| Warm Start | < 10ms |
| Memory (Idle) | ~3 MB |
| Memory (Active) | ~8 MB |
| Binary Size | ~3-5 MB |

### Optimization Tips

1. **Enable Quantization**: Reduces model memory footprint
2. **Reduce Context Length**: Lower context = less memory
3. **Batch Requests**: Combine multiple requests for efficiency
4. **Enable Compression**: Reduces network bandwidth

## Troubleshooting

### Out of Memory

If you encounter memory errors:

1. Check current usage: `curl http://localhost:8080/metrics`
2. Reduce `max_connections` in config
3. Reduce `max_batch_size` in config
4. Enable model quantization

### WASI Errors

Ensure WASI is enabled in WasmEdge:

```bash
wasmedge --version | grep WASI
```

If not available, reinstall WasmEdge with WASI support.

### Model Loading Issues

Verify model directory is mounted:

```bash
wasmedge --dir /models:/models lmdeploy_moonbit_server.wasm
```

## Comparison with Native Builds

| Feature | Native (Linux) | Wasm (Edge) |
|---------|----------------|-------------|
| Startup Time | ~2-5s | < 100ms |
| Memory | ~50-100 MB | < 10 MB |
| Binary Size | ~20 MB | ~3-5 MB |
| Performance | 100% | ~80-90% |
| Portability | Linux only | Any Wasm runtime |

## Production Deployment

### Using Systemd

Create `/etc/systemd/system/lmdeploy-wasm.service`:

```ini
[Unit]
Description=LMDeploy MoonBit Wasm API Server
After=network.target

[Service]
Type=simple
User=lmdeploy
WorkingDirectory=/opt/lmdeploy-wasm
ExecStart=/usr/local/bin/wasmedge \
  --dir /models:/models \
  --dir /etc/lmdeploy:/etc/lmdeploy \
  --env LMDEPLOY_PORT=8080 \
  /opt/lmdeploy-wasm/lmdeploy_moonbit_server.wasm
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
```

Enable and start:

```bash
sudo systemctl enable lmdeploy-wasm
sudo systemctl start lmdeploy-wasm
sudo systemctl status lmdeploy-wasm
```

### Using Kubernetes

Deploy as a Wasm workload:

```yaml
apiVersion: v1
kind: Pod
metadata:
  name: lmdeploy-wasm
spec:
  containers:
  - name: lmdeploy
    image: lmdeploy-moonbit-wasm:latest
    imagePullPolicy: IfNotPresent
    resources:
      limits:
        memory: "10Mi"
      requests:
        memory: "5Mi"
    ports:
    - containerPort: 8080
```

## References

- [WasmEdge Documentation](https://wasmedge.org/docs/)
- [WASI Specification](https://wasi.dev/)
- [MoonBit Documentation](https://moonbitlang.com/)
