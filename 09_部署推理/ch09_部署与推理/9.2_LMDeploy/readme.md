# LMDeploy 部署指南

## 概述

LMDeploy 是由上海人工智能实验室开发的高效 LLM 推理部署框架。它支持多种大语言模型的量化推理，提供 Python、C++、RESTful 等多种接口，具有高性能、低显存占用的特点。本章将详细介绍 LMDeploy 的使用方法和最佳实践。

## 1. LMDeploy 核心特性

### 1.1 主要特性

| 特性 | 描述 | 优势 |
|------|------|------|
| **TurboMind 引擎** | 自研高性能推理引擎 | 推理速度提升 2-3x |
| **动态Batch** | 动态批处理机制 | 吞吐量提升显著 |
| **量化推理** | W4A16/AWQ量化 | 显存降低 60-70% |
| **多模态支持** | VL 模型支持 | 视觉语言模型部署 |
| **OpenAI 兼容** | RESTful API | 快速迁移现有应用 |

### 1.2 架构图

```
┌─────────────────────────────────────────────────────────────────┐
│                      LMDeploy 架构图                            │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │                    Python SDK                            │    │
│  │  ┌───────────┐  ┌───────────┐  ┌───────────┐            │    │
│  │  │ Pipeline  │  │  Turbomind│  │  Quantize │            │    │
│  │  │  (高层API) │  │  (推理引擎)│  │  (量化工具) │            │    │
│  │  └───────────┘  └───────────┘  └───────────┘            │    │
│  └─────────────────────────────────────────────────────────┘    │
│                              │                                   │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │                    C++ Runtime                           │    │
│  │  ┌───────────┐  ┌───────────┐  ┌───────────┐            │    │
│  │  │  Kernel  │  │  Memory   │  │  Scheduler│            │    │
│  │  │ (CUDA)   │  │  Manager  │  │           │            │    │
│  │  └───────────┘  └───────────┘  └───────────┘            │    │
│  └─────────────────────────────────────────────────────────┘    │
│                              │                                   │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │                    服务层                                │    │
│  │  ┌───────────┐  ┌───────────┐  ┌───────────┐            │    │
│  │  │RESTful API│  │  gRPC    │  │  WebUI   │            │    │
│  │  └───────────┘  └───────────┘  └───────────┘            │    │
│  └─────────────────────────────────────────────────────────┘    │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

## 2. 环境安装

### 2.1 系统要求

- **操作系统**: Ubuntu 18.04+ / Windows 10+ / macOS 12+
- **GPU**: NVIDIA GPU (Compute Capability 7.0+)
- **CUDA**: 11.3+
- **Python**: 3.8+

### 2.2 安装 LMDeploy

```bash
# 创建虚拟环境
conda create -n lmdeploy python=3.10
conda activate lmdeploy

# 安装 LMDeploy
pip install lmdeploy

# 安装 pytorch (如需要)
pip install torch torchvision

# 验证安装
lmdeploy --version
```

### 2.3 Docker 安装

```bash
# 拉取镜像
docker pull openmmlab/lmdeploy:latest

# 运行容器
docker run --gpus all -it \
    -v /models:/models \
    -p 33333:33333 \
    openmmlab/lmdeploy:latest \
    /bin/bash
```

## 3. 快速开始

### 3.1 命令行推理

```bash
# 基础推理
lmdeploy chat /path/to/llama-2-7b-chat

# 指定 GPU
CUDA_VISIBLE_DEVICES=0 lmdeploy chat /path/to/llama-2-7b-chat

# 指定参数
lmdeploy chat /path/to/llama-2-7b-chat \
    --temperature 0.8 \
    --max_new_tokens 1024
```

### 3.2 Python API

```python
from lmdeploy import pipeline, TurbomindEngineConfig

# 方式1: 使用 pipeline (推荐)
pipe = pipeline("Qwen/Qwen2-7B-Instruct")

response = pipe([
    "你好，请介绍一下自己",
    "什么是人工智能？"
])

for res in response:
    print(f"Response: {res.text}")
```

```python
# 方式2: 使用 TurbomindEngineConfig 精细控制
from lmdeploy import pipeline, TurbomindEngineConfig

# 配置引擎
config = TurbomindEngineConfig(
    session_len=4096,
    max_batch_size=16,
    tp=1,              # tensor parallel
    cache_max_entry_count=0.8,
)

pipe = pipeline("Qwen/Qwen2-7B-Instruct", config=config)

# 推理
response = pipe([
    {"role": "user", "content": "用Python实现快速排序"}
])

print(response[0].text)
```

### 3.3 Chat 接口

```python
from lmdeploy import ChatTemplate

# 使用聊天模板
chat_template = ChatTemplate.from_pretrained("Qwen/Qwen2-7B-Instruct")

# 方式1: 流式输出
for text in chat_template.stream_async("请解释什么是机器学习"):
    print(text, end="", flush=True)

# 方式2: 非流式
response = chat_template("请解释什么是深度学习")
print(response)
```

## 4. 模型量化

### 4.1 量化类型介绍

| 量化方式 | 描述 | 显存降低 | 精度损失 |
|----------|------|----------|----------|
| W4A16 | Weight 4bit + Activation fp16 | ~60% | <2% |
| AWQ | Activation-aware Weight Quantization | ~70% | <3% |
| KV Cache 8bit | KV Cache 量化 | ~30% | <1% |

### 4.2 AWQ 量化

```bash
# 在线量化并启动服务
lmdeploy serve api_server \
    Qwen/Qwen2-7B-Instruct \
    --quant-policy 4 \
    --cache-max-entry-count 0.8
```

```python
# Python 中使用量化模型
from lmdeploy import pipeline, TurbomindEngineConfig

config = TurbomindEngineConfig(
    quant_policy=4,    # AWQ 量化
    kv_cache_quant_bit=8,  # KV Cache 8bit 量化
)

pipe = pipeline("Qwen/Qwen2-7B-Instruct", config=config)

response = pipe(["你好，请介绍一下自己"])
print(response[0].text)
```

### 4.3 离线量化

```bash
# 将模型量化并保存到指定目录
lmdeploy convert qlama-2-7b-chat \
    /path/to/llama-2-7b-chat \
    --quant-policy 4 \
    --output-dir /path/to/quantized_model
```

```python
# 使用量化后的模型
pipe = pipeline("/path/to/quantized_model")
response = pipe(["你好"])
print(response[0].text)
```

## 5. 服务部署

### 5.1 启动 RESTful API

```bash
# 启动 API 服务
lmdeploy serve api_server \
    Qwen/Qwen2-7B-Instruct \
    --server-name 0.0.0.0 \
    --server-port 33333 \
    --tp 1
```

```bash
# 测试 API
curl http://localhost:33333/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen2-7B-Instruct",
    "messages": [
      {"role": "user", "content": "你好"}
    ],
    "max_tokens": 512
  }'
```

### 5.2 Python 客户端调用

```python
from lmdeploy.serve.openai.api_client import APIClient

# 连接 API 服务
api_client = APIClient("http://localhost:33333")

# 列出可用模型
models = api_client.list_models()
print("Available models:", models)

# 创建对话
response = api_client.chat.completions.create(
    model="Qwen/Qwen2-7B-Instruct",
    messages=[
        {"role": "user", "content": "什么是大语言模型？"}
    ],
    temperature=0.7,
    max_tokens=1024
)

print(response.choices[0].message.content)
```

### 5.3 流式输出

```python
import asyncio
from lmdeploy.serve.openai.api_client import APIClient

api_client = APIClient("http://localhost:33333")

# 流式调用
for chunk in api_client.chat.completions.create(
    model="Qwen/Qwen2-7B-Instruct",
    messages=[
        {"role": "user", "content": "写一首关于春天的诗"}
    ],
    stream=True,
    max_tokens=512
):
    if chunk.choices and chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="", flush=True)
```

### 5.4 多卡部署

```bash
# 2卡部署
lmdeploy serve api_server \
    Qwen/Qwen2-70B-Instruct \
    --tp 2

# 4卡部署
lmdeploy serve api_server \
    Qwen/Qwen2-70B-Instruct \
    --tp 4
```

```python
# Python 多卡配置
from lmdeploy import pipeline, TurbomindEngineConfig

config = TurbomindEngineConfig(
    tp=2,  # 使用2张GPU
    session_len=8192,
)

pipe = pipeline("Qwen/Qwen2-70B-Instruct", config=config)
```

## 6. 多模态模型部署

### 6.1 视觉语言模型

```python
from lmdeploy import pipeline, TurbomindEngineConfig

# 加载多模态模型
pipe = pipeline("liuhaotian/llava-v1.6-7b")

# 图文对话
response = pipe(
    prompt="请描述这张图片的内容",
    images="path/to/image.jpg"
)
print(response[0].text)
```

### 6.2 多图输入

```python
from lmdeploy import pipeline

pipe = pipeline("liuhaotian/llava-v1.6-7b")

# 多图对话
response = pipe(
    prompt="比较这两张图片的差异",
    images=["path/to/image1.jpg", "path/to/image2.jpg"]
)
print(response[0].text)
```

## 7. 性能优化

### 7.1 关键参数配置

```python
from lmdeploy import pipeline, TurbomindEngineConfig

# 优化配置
config = TurbomindEngineConfig(
    session_len=8192,              # 最大序列长度
    max_batch_size=32,             # 最大批处理大小
    tp=1,                          # Tensor Parallel
    cache_max_entry_count=0.8,    # KV Cache 缓存比例
    cache_block_seq_len=128,       # Cache 块大小
    quant_policy=0,                # 量化策略 (0:无, 4:AWQ)
    kv_cache_quant_bit=8,          # KV Cache 量化位数
)

pipe = pipeline("Qwen/Qwen2-7B-Instruct", config=config)
```

### 7.2 性能基准测试

```python
import time
from lmdeploy import pipeline, TurbomindEngineConfig

# 初始化
config = TurbomindEngineConfig(session_len=4096)
pipe = pipeline("Qwen/Qwen2-7B-Instruct", config=config)

# 预热
pipe(["warmup"])

# 测试数据
prompts = ["测试prompt"] * 100
max_tokens = 512

# 性能测试
start = time.time()
responses = pipe(prompts, max_new_tokens=max_tokens)
elapsed = time.time() - start

# 计算指标
total_tokens = sum(len(r.text) for r in responses)
throughput = total_tokens / elapsed
avg_latency = elapsed / len(prompts)

print(f"总请求数: {len(prompts)}")
print(f"总生成 token 数: {total_tokens}")
print(f"平均延迟: {avg_latency:.3f}s")
print(f"吞吐量: {throughput:.2f} tokens/s")
```

### 7.3 动态 Batch 优化

```python
from lmdeploy import pipeline, TurbomindEngineConfig

# 启用动态 Batch
config = TurbomindEngineConfig(
    max_batch_size=64,
    dynamic_batch=True,  # 启用动态批处理
)

pipe = pipeline("Qwen/Qwen2-7B-Instruct", config=config)

# 批量请求 - 自动优化调度
prompts = [f"问题{i}：" for i in range(50)]
responses = pipe(prompts, max_new_tokens=256)
```

## 8. 与 LangChain 集成

### 8.1 LangChain LLM

```python
from langchain.llms import LMDeploy

# 初始化 LangChain LLM
llm = LMDeploy(
    model_name="Qwen/Qwen2-7B-Instruct",
    endpoint="http://localhost:33333",
    temperature=0.7,
    max_tokens=1024,
)

# 使用 LangChain
from langchain.prompts import PromptTemplate

template = """请回答以下问题：
问题: {question}
回答: """

prompt = PromptTemplate(template=template, input_variables=["question"])
chain = prompt | llm

response = chain.invoke({"question": "什么是人工智能？"})
print(response)
```

### 8.2 ChatGLM 类

```python
from lmdeploy.serve.langchain import ChatGLM

llm = ChatGLM(
    model_name="Qwen/Qwen2-7B-Instruct",
    api_url="http://localhost:33333",
    temperature=0.7,
)

response = llm("你好，请介绍一下自己")
print(response)
```

## 9. 常见问题

### 9.1 显存不足

```python
# 降低显存占用
config = TurbomindEngineConfig(
    cache_max_entry_count=0.5,  # 减少缓存
    session_len=2048,           # 减少最大长度
)

pipe = pipeline("Qwen/Qwen2-7B-Instruct", config=config)
```

### 9.2 推理速度慢

```bash
# 使用量化模型
lmdeploy serve api_server Qwen/Qwen2-7B-Instruct --quant-policy 4

# 增加批处理大小
# Python 中设置
config = TurbomindEngineConfig(max_batch_size=64)
```

### 9.3 模型加载失败

```bash
# 检查模型路径
ls -la /path/to/model

# 手动指定模型格式
lmdeploy serve api_server /path/to/model --model-format hf
```

## 10. 总结

本章我们全面学习了 LMDeploy 的使用方法：

1. **核心特性**：TurboMind 引擎、动态Batch、量化推理、多模态支持
2. **环境安装**：pip 和 Docker 两种方式
3. **快速开始**：命令行和 Python API
4. **模型量化**：W4A16/AWQ/KV Cache 量化
5. **服务部署**：RESTful API、多卡部署
6. **多模态**：视觉语言模型部署
7. **性能优化**：参数调优、基准测试
8. **LangChain 集成**：与现有生态集成

LMDeploy 作为国产开源的高性能 LLM 推理框架，为大语言模型的部署提供了简洁高效的解决方案。
