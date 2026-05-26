# LMDeploy TurboMind 性能测试报告

**测试日期**: 2026-05-26  
**模型**: Qwen3.6-35B-A3B-AWQ  
**后端**: TurboMind (AWQ 4-bit)  
**GPU**: Tesla PG503-216 (32GB)  
**LMDeploy 版本**: 0.13.0

---

## 测试方法

### 测试脚本
- `scripts/run_benchmark.sh`: 主测试脚本
- `benchmark/profile_throughput.py`: LMDeploy 官方性能测试脚本

### 测试配置
- **并发度**: 1 (串行测试)
- **请求数**: 5 (每个场景)
- **输出长度**: 512 tokens
- **输入长度**: 512, 1024, 4096, 8192 tokens

---

## 实测数据 (来源: CSV 文件)

### 原始数据

| 输入长度 | 输出长度 | TTFT (ms) | TPOT (ms) | E2E (ms) |
|----------|----------|-----------|-----------|----------|
| 512 | 512 | 79.9 | 23.5 | 4,112 |
| 1024 | 512 | 139.2 | 23.7 | 7,886 |
| 4096 | 512 | 402.0 | 24.7 | 9,550 |
| 8192 | 512 | 649.4 | 25.1 | 6,803 |

### 计算得出的吞吐量

| 输入长度 | Prefill (tok/s) | Decode (tok/s) |
|----------|-----------------|----------------|
| 512 | **6,408** | **42.6** |
| 1024 | **7,356** | **42.2** |
| 4096 | **10,190** | **40.5** |
| 8192 | **12,614** | **39.8** |

**计算公式**:
- Prefill (tok/s) = input_len / (ttft_ms / 1000)
- Decode (tok/s) = 1000 / tpot_ms

---

## 关键发现

1. **Prefill 性能**: 6,400 - 12,600 tok/s，随上下文长度增加而提高
2. **Decode 性能**: 40 - 43 tok/s，相对稳定
3. **Prefill/Decode 比例**: 约 150:1 - 300:1
4. **TTFT**: 随输入长度线性增长

---

## 文件说明

```
turbomind_35b_awq_20260526/
├── README.md                              # 本文件
├── scripts/
│   └── run_benchmark.sh                   # 测试脚本
└── results/
    ├── profile_throughput_35b_random_512_512.csv
    ├── profile_throughput_35b_random_1024_512.csv
    ├── profile_throughput_35b_random_4096_512.csv
    └── profile_throughput_35b_random_8192_512.csv
```

---

## 测试命令

```bash
cd /mnt/data/lmdeploy
bash benchmarks-archive/turbomind_35b_awq_20260526/scripts/run_benchmark.sh
```

---

**备注**: 本报告固化了 2026-05-26 的测试结果，所有数据来源于实际测试 CSV 文件。