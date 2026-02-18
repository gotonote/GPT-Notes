# vLLM 部署指南

## 概述

vLLM 是一个高性能的 LLM 推理框架，由加州大学伯克利分校开发。它采用 PagedAttention 技术，能够显著提升大语言模型的推理效率，降低显存占用。本章将详细介绍 vLLM 的原理、部署步骤和 API 调用方法。

## 1. vLLM 核心原理

### 1.1 PagedAttention 机制

传统的 Attention 机制在处理长序列时，需要将所有 KV Cache 存储在显存中，导致显存占用巨大。vLLM 引入的 PagedAttention 借鉴了操作系统的分页管理思想，将 KV Cache 分成固定大小的页面进行管理。

```
┌─────────────────────────────────────────────────────────────┐
│                    PagedAttention 原理                       │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  传统方式 (Continuous KV Cache):                           │
│  ┌─────────────────────────────────────────────────────┐    │
│  │ [Token 1] [Token 2] [Token 3] ... [Token n]        │    │
│  └─────────────────────────────────────────────────────┘    │
│  ❌ 必须连续存储，显存碎片化，动态扩展困难                   │
│                                                             │
│  PagedAttention (分页管理):                                 │
│  ┌──────┐ ┌──────┐ ┌──────┐ ┌──────┐ ┌──────┐              │
│  │Page 1│ │Page 2│ │Page 3│ │Page 4│ │ ...  │              │
│  └──────┘ └──────┘ └──────┘ └──────┘ └──────┘              │
│  ✅ 非连续存储，按需分配，显存利用率高                       │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 1.2 核心技术优势

| 特性 | 描述 | 提升效果 |
|------|------|----------|
| PagedAttention | 分页式 KV Cache 管理 | 显存利用率提升 2-4x |
| Continuous Batching | 连续批处理机制 | 吞吐量提升 2-3x |
| CUDA Kernel Optimization | 深度优化的 CUDA 内核 | 推理速度提升 1.5-2x |
| Speculative Decoding | 推测解码（可选） | 延迟降低 30-50% |

### 1.3 架构图

```
┌─────────────────────────────────────────────────────────────────┐
│                        vLLM 架构图                               │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐         │
│  │   Client    │───▶│  HTTP Server │───▶│   Engine    │         │
│  │  (OpenAI API)│    │  (FastAPI)   │    │  (Core)     │         │
│  └─────────────┘    └─────────────┘    └──────┬──────┘         │
│                                                │                 │
│                        ┌───────────────────────┼───────────┐    │
│                        │                       ▼           │    │
│                        │  ┌─────────────────────────────────┐ │    │
│                        │  │      Scheduler                  │ │    │
│                        │  │  ┌─────────────────────────┐   │ │    │
│                        │  │  │   Block Manager         │   │ │    │
│                        │  │  │   (PagedAttention)      │   │ │    │
│                        │  │  └─────────────────────────┘   │ │    │
│                        │  └─────────────────────────────────┘ │    │
│                        │                       │               │    │
│                        ▼                       ▼               │    │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │                    GPU Worker Pool                       │   │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────┐               │   │
│  │  │  GPU 0   │  │  GPU 1   │  │  GPU N   │               │   │
│  │  │ KV Cache │  │ KV Cache │  │ KV Cache │               │   │
│  │  │ (Pages)  │  │ (Pages)  │  │ (Pages)  │               │   │
│  │  └──────────┘  └──────────┘  └──────────┘               │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

## 2. 环境准备

### 2.1 系统要求

- **操作系统**: Ubuntu 18.04+ / CentOS 7+
- **GPU**: NVIDIA GPU with CUDA 11.8+ (推荐 CUDA 12.1+)
- **显存**: 至少 16GB (推荐 24GB+)
- **Python**: 3.8+

### 2.2 安装 vLLM

```bash
# 创建虚拟环境
conda create -n vllm python=3.10
conda activate vllm

# 安装 PyTorch (根据 CUDA 版本选择)
# CUDA 12.1
pip install torch==2.1.2 torchvision==0.16.2 --index-url https://download.pytorch.org/whl/cu121

# 安装 vLLM
pip install vllm

# 验证安装
python -c "import vllm; print(vllm.__version__)"
```

### 2.3 Docker 部署（推荐）

```bash
# 拉取官方镜像
docker pull vllm/vllm:latest

# 运行容器
docker run --gpus all \
    -v /models:/models \
    -p 8000:8000 \
    --ipc=host \
    vllm/vllm:latest \
    --model /models/llama-2-7b-hf \
    --tensor-parallel-size 1
```

## 3. 快速启动

### 3.1 命令行启动

```bash
# 单卡部署
vllm serve Qwen/Qwen2-7B-Instruct \
    --dtype half \
    --port 8000

# 多卡部署 ( tensor-parallel-size = GPU数量)
vllm serve Qwen/Qwen2-7B-Instruct \
    --tensor-parallel-size 2 \
    --dtype half \
    --port 8000

# 指定 GPU
CUDA_VISIBLE_DEVICES=0,1 vllm serve Qwen/Qwen2-7B-Instruct
```

### 3.2 Python API 启动

```python
from vllm import LLM, SamplingParams

# 初始化引擎
llm = LLM(
    model="Qwen/Qwen2-7B-Instruct",
    tensor_parallel_size=1,  # GPU 数量
    dtype="half",            # 数据类型
    max_model_len=4096,      # 最大序列长度
    gpu_memory_utilization=0.9,  # GPU 显存利用率
)

# 定义采样参数
sampling_params = SamplingParams(
    temperature=0.7,
    max_tokens=512,
    top_p=0.9,
)

# 推理调用
prompts = [
    "请介绍一下人工智能的发展历史：",
    "什么是机器学习？",
]

outputs = llm.generate(prompts, sampling_params)

# 输出结果
for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt}")
    print(f"Generated: {generated_text}")
    print("-" * 50)
```

### 3.3 启动 OpenAI 兼容 API

```bash
# 启动服务
vllm serve Qwen/Qwen2-7B-Instruct \
    --dtype half \
    --port 8000 \
    --api-key token-vllm

# 测试 API
curl http://localhost:8000/v1/models
```

## 4. OpenAI API 调用

### 4.1 Chat Completions API

```python
import openai

# 配置客户端
client = openai.OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="token-vllm"  # 与启动时的 --api-key 一致
)

# 流式调用
response = client.chat.completions.create(
    model="Qwen/Qwen2-7B-Instruct",
    messages=[
        {"role": "system", "content": "你是一个专业的AI助手。"},
        {"role": "user", "content": "请解释一下什么是大语言模型？"}
    ],
    temperature=0.7,
    max_tokens=1024,
    stream=True  # 流式输出
)

# 处理流式响应
for chunk in response:
    if chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="", flush=True)
```

### 4.2 非流式调用

```python
import openai

client = openai.OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="token-vllm"
)

response = client.chat.completions.create(
    model="Qwen/Qwen2-7B-Instruct",
    messages=[
        {"role": "user", "content": "用Python实现一个快速排序算法"}
    ],
    temperature=0.7,
    max_tokens=2048,
    stream=False
)

print(response.choices[0].message.content)
```

### 4.3 Embeddings API

```python
import openai

client = openai.OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="token-vllm"
)

response = client.embeddings.create(
    model="BAAI/bge-large-zh-v1.5",
    input=["你好，世界！", "人工智能是未来的趋势"]
)

for embedding in response.data:
    print(f"Embedding shape: {len(embedding.embedding)}")
    print(f"First 5 values: {embedding.embedding[:5]}")
```

## 5. 高级配置

### 5.1 多卡部署配置

```python
from vllm import LLM

# 2卡部署
llm = LLM(
    model="Qwen/Qwen2-70B-Instruct",
    tensor_parallel_size=2,  # 使用2张GPU
    dtype="half",
    max_model_len=4096,
    gpu_memory_utilization=0.95,
)

# 4卡部署
llm = LLM(
    model="Qwen/Qwen2-70B-Instruct",
    tensor_parallel_size=4,
    dtype="half",
    max_model_len=4096,
    gpu_memory_utilization=0.9,
)
```

### 5.2 量化配置

```bash
# AWQ 量化模型
vllm serve llama-2-70b-awq --quantization awq

# GPTQ 量化模型  
vllm serve llama-2-70b-gptq --quantization gptq
```

```python
# Python API 使用量化模型
llm = LLM(
    model="TheBloke/Llama-2-70B-Chat-AWQ",
    quantization="awq",
    dtype="half",
)
```

### 5.3 批量推理优化

```python
from vllm import LLM, SamplingParams

llm = LLM(
    model="Qwen/Qwen2-7B-Instruct",
    dtype="half",
)

sampling_params = SamplingParams(
    temperature=0.8,
    max_tokens=512,
    top_p=0.95,
)

# 批量 prompts
prompts = [
    f"请解释一下概念{i}：" for i in range(100)
]

# 批量推理 - 高效利用 GPU
outputs = llm.generate(prompts, sampling_params)

for i, output in enumerate(outputs):
    print(f"Prompt {i}: {output.outputs[0].text[:100]}...")
```

### 5.4 LoRA 适配器配置

```python
from vllm import LLM, SamplingParams

# 加载基础模型 + LoRA 适配器
llm = LLM(
    model="Qwen/Qwen2-7B-Instruct",
    enable_lora=True,
    lora_configs=[
        {
            "lora_name": "math_adapter",
            "lora_path": "/path/to/math_lora",
        }
    ]
)

sampling_params = SamplingParams(
    max_tokens=512,
)

# 使用 LoRA 推理
outputs = llm.generate(
    ["请计算 123 * 456 = "],
    sampling_params,
    lora_request="math_adapter"
)
```

## 6. 性能调优

### 6.1 关键参数调优

| 参数 | 说明 | 推荐值 |
|------|------|--------|
| `gpu_memory_utilization` | GPU 显存利用率 | 0.9-0.95 |
| `max_model_len` | 最大序列长度 | 根据模型和显存调整 |
| `max_num_seqs` | 最大并发序列数 | 16-256 |
| `max_num_batched_tokens` | 批处理最大 token 数 | 8192-16384 |

### 6.2 性能基准测试

```python
import time
from vllm import LLM, SamplingParams

llm = LLM(
    model="Qwen/Qwen2-7B-Instruct",
    dtype="half",
)

sampling_params = SamplingParams(
    temperature=0.7,
    max_tokens=512,
)

# 预热
llm.generate(["warmup"], sampling_params)

# 性能测试
num_requests = 100
prompts = ["测试prompt"] * num_requests

start_time = time.time()
outputs = llm.generate(prompts, sampling_params)
end_time = time.time()

total_tokens = sum(len(o.outputs[0].token_ids) for o in outputs)
latency = (end_time - start_time) / num_requests
throughput = total_tokens / (end_time - start_time)

print(f"总请求数: {num_requests}")
print(f"平均延迟: {latency:.3f}s")
print(f"吞吐量: {throughput:.2f} tokens/s")
```

### 6.3 显存优化技巧

```python
# 1. 启用 KV Cache 优化
llm = LLM(
    model="Qwen/Qwen2-7B-Instruct",
    dtype="half",
    enforce_eager=False,  # 启用 CUDA graph 优化
)

# 2. 使用更小的数据类型
llm = LLM(
    model="Qwen/Qwen2-7B-Instruct", 
    dtype="float16",  # 或 "bfloat16"
)

# 3. 限制最大序列长度
llm = LLM(
    model="Qwen/Qwen2-7B-Instruct",
    max_model_len=2048,  # 减少显存占用
)
```

## 7. 常见问题

### 7.1 CUDA 版本不匹配

```bash
# 检查 CUDA 版本
nvcc --version

# 如果版本不匹配，安装对应版本的 PyTorch
pip install torch==2.1.2 torchvision==0.16.2 --index-url https://download.pytorch.org/whl/cu118
```

### 7.2 显存不足

```python
# 减小 gpu_memory_utilization
llm = LLM(
    model="Qwen/Qwen2-7B-Instruct",
    gpu_memory_utilization=0.7,  # 降低显存占用
)

# 或者减小 max_model_len
llm = LLM(
    model="Qwen/Qwen2-7B-Instruct",
    max_model_len=2048,
)
```

### 7.3 模型加载慢

```bash
# 使用离线模型
vllm serve /path/to/local/model --dtype half

# 或者设置环境变量加速
export VLLM_ATTENTION_BACKEND=FLASH_ATTN
```

## 8. 总结

vLLM 通过 PagedAttention 技术和连续批处理机制，为 LLM 推理提供了高效的解决方案。本章我们学习了：

1. **vLLM 核心原理**：PagedAttention 分页管理机制
2. **环境安装**：pip/Docker 多种安装方式
3. **快速启动**：命令行和 Python API
4. **API 调用**：OpenAI 兼容的 Chat 和 Embeddings 接口
5. **高级配置**：多卡部署、量化、批量推理
6. **性能调优**：关键参数和优化技巧

vLLM 的设计理念是让 LLM 推理更加高效和易用，是生产环境部署的理想选择。
