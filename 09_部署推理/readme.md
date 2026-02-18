# 模型量化技术

## 概述

模型量化是降低大语言模型显存占用和提升推理速度的关键技术。本章将详细介绍三种主流的量化方法：GPTQ、AWQ 和 GGUF，帮助你理解量化原理并掌握实际应用。

## 1. 量化基础

### 1.1 什么是模型量化

模型量化是将模型参数从高精度（如 FP32、FP16）转换为低精度（如 INT8、INT4）的技术，可以显著减少模型存储和推理时的显存占用。

```
┌─────────────────────────────────────────────────────────────────┐
│                        量化精度对比                              │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  FP32 (32位浮点)                                               │
│  ┌─────────────────────────────────────┐                        │
│  │ S EEEEEEEE EMMMMMMM EMMMMMMM EMMMMMMM│ 32 bits             │
│  └─────────────────────────────────────┘                        │
│  存储空间: 4 bytes/参数                                         │
│  精度: 最高                                                     │
│                                                                 │
│  FP16/BF16 (16位浮点)                                          │
│  ┌─────────────────────────────────────┐                        │
│  │ S EEEEEEE EMMMMMMM                 │ 16 bits               │
│  └─────────────────────────────────────┘                        │
│  存储空间: 2 bytes/参数                                         │
│  精度: 较高                                                     │
│                                                                 │
│  INT8 (8位整数)                                                 │
│  ┌─────────────┐                                               │
│  │ S DDDDDDDD  │ 8 bits                                       │
│  └─────────────┘                                               │
│  存储空间: 1 byte/参数                                          │
│  精度: 中等                                                     │
│                                                                 │
│  INT4 (4位整数)                                                 │
│  ┌─────────┐                                                   │
│  │ DDDD    │ 4 bits                                          │
│  └─────────┘                                                   │
│  存储空间: 0.5 bytes/参数                                      │
│  精度: 较低                                                     │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 1.2 量化类型

| 类型 | 描述 | 优点 | 缺点 |
|------|------|------|------|
| 训练后量化 (PTQ) | 训练后直接量化 | 简单快速 | 精度可能有损失 |
| 量化感知训练 (QAT) | 训练中模拟量化 | 精度更好 | 训练时间长 |
| 动态量化 | 推理时动态量化 | 无需重训练 | 效果有限 |

## 2. GPTQ 量化

### 2.1 原理介绍

GPTQ (GPTQ: Accurate Post-Training Quantization for LLMs) 是一种基于分层贪婪算法的 INT4 量化方法，它通过最小化量化误差来保持模型性能。

```
┌─────────────────────────────────────────────────────────────────┐
│                     GPTQ 量化原理                                │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  原始权重矩阵 W (FP16)                                          │
│  ┌────────────────────────────────────────────────────────┐     │
│  │  1.23  -0.45   2.67   0.89  -1.23   3.45   ...        │     │
│  │ -0.78   1.56  -2.34   0.12   2.45  -0.67   ...        │     │
│  │  2.34  -1.23   0.78   1.45  -2.67   0.89   ...        │     │
│  │  ...                                                        │     │
│  └────────────────────────────────────────────────────────┘     │
│                          │                                      │
│                          ▼                                      │
│  ┌────────────────────────────────────────────────────────┐     │
│  │  1.0   -0.5   3.0   1.0  -1.0   3.0   ...            │     │
│  │ -1.0   2.0   -2.0   0.0   2.0  -1.0   ...            │     │
│  │  2.0   -1.0   1.0   1.0  -3.0   1.0   ...            │     │
│  │  ...                                                        │     │
│  └────────────────────────────────────────────────────────┘     │
│  量化后权重矩阵 W_Q (INT4, 存储为 INT8)                         │
│                                                                 │
│  核心思想：逐列量化，最小化重建误差                               │
│  min ||W - W_Q||_2^2                                            │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 2.2 核心算法

```python
"""
GPTQ 量化算法伪代码
"""
def gptq_quantize(W, bits=4, perchannel=True):
    """
    W: 原始权重矩阵
    bits: 量化位数
    """
    # 1. 计算量化 scale 和 zero point
    if perchannel:
        # 按通道计算（每列独立）
        scales = W.abs().max(dim=0)[0] / (2**(bits-1) - 1)
    else:
        # 全局计算
        scales = W.abs().max() / (2**(bits-1) - 1)
    
    # 2. 量化
    W_q = torch.round(W / scales)
    
    # 3. 重建误差最小化 (Greedy 优化)
    # 逐列调整以减少重建误差
    for col in range(W.shape[1]):
        # 计算当前列的原始值和量化值
        w_col = W[:, col]
        wq_col = W_q[:, col] * scales[col]
        
        # 计算误差
        error = w_col - wq_col
        
        # 将误差按比例分配给其他列
        # (简化说明，实际算法更复杂)
        pass
    
    return W_q, scales
```

### 2.3 安装和使用

```bash
# 安装 GPTQ 量化工具
pip install optimum[exporters] transformers
pip install auto-gptq

# 或使用 GPTQ 官方仓库
git clone https://github.com/IST-DASLab/gptq.git
cd gptq
pip install -r requirements.txt
```

### 2.4 命令行量化

```bash
# 量化模型
python -m gptq.main \
    --model meta-llama/Llama-2-7b-hf \
    --bits 4 \
    --group-size 128 \
    --save quantized-model \
    --dataset wikitext2
```

### 2.5 Python 代码量化

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from optimum.gptq import GPTQQuantizer

# 加载模型
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-hf",
    torch_dtype="auto",
    device_map="auto"
)

# 配置量化
quantizer = GPTQQuantizer(
    bits=4,              # 量化位数
    group_size=128,      # 分组大小
    desc_act=False,      # 是否使用 desc_act
    dataset="wikitext2"  # 校准数据集
)

# 执行量化
quantized_model = quantizer.quantize_model(model, tokenizer)

# 保存量化模型
quantized_model.save_pretrained("llama2-7b-gptq")
tokenizer.save_pretrained("llama2-7b-gptq")
```

### 2.6 使用量化模型推理

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

# 加载量化模型
model = AutoModelForCausalLM.from_pretrained(
    "llama2-7b-gptq",
    device_map="auto"
)

tokenizer = AutoTokenizer.from_pretrained("llama2-7b-gptq")

# 推理
prompt = "什么是人工智能？"
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

outputs = model.generate(**inputs, max_new_tokens=512)
print(tokenizer.decode(outputs[0]))
```

## 3. AWQ 量化

### 3.1 原理介绍

AWQ (Activation-aware Weight Quantization) 是一种考虑激活值分布的权重量化方法，它通过"保护"重要权重来减少量化误差。

```
┌─────────────────────────────────────────────────────────────────┐
│                      AWQ 量化原理                                │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  核心思想：不是所有权重都同等重要                                  │
│                                                                 │
│  ┌────────────────────────────────────────────────────────┐     │
│  │            权重重要性分布 (按激活值加权)                 │     │
│  │                                                        │     │
│  │   高重要性权重 ─────────────────────────────────────▶  │     │
│  │   (被保护，不被量化或使用更高精度)                      │     │
│  │                                                        │     │
│  │   低重要性权重 (可以使用更激进的量化)                   │     │
│  │                                                        │     │
│  └────────────────────────────────────────────────────────┘     │
│                                                                 │
│  AWQ vs GPTQ:                                                  │
│  - GPTQ: 逐列优化，忽略激活分布                                  │
│  - AWQ: 考虑激活分布，保护高影响力权重                          │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 3.2 量化流程

```python
"""
AWQ 量化流程
"""
def awq_quantize(model, tokenizer, bits=4):
    # 1. 收集激活值统计
    # 运行校准数据集，记录各层的激活分布
    activations = {}
    for name, module in model.named_modules():
        if 'attention' in name.lower() or 'mlp' in name.lower():
            activations[name] = collect_activations(module)
    
    # 2. 计算权重重要性
    # 基于激活值计算每个权重的重要性分数
    weight_importance = {}
    for name, act in activations.items():
        weight_importance[name] = compute_importance(act)
    
    # 3. 搜索最优保护阈值
    # 找到需要保护的权重比例 (通常 1-10%)
    scales = search_best_scales(weight_importance)
    
    # 4. 应用量化
    # 保护重要权重，对其他权重进行量化
    quantized_model = apply_quantization(model, scales, bits)
    
    return quantized_model
```

### 3.3 安装和使用

```bash
# 安装 AWQ
pip install awq
```

### 3.4 Python 代码量化

```python
from awq import AutoAWQ

# 加载模型
model_path = "meta-llama/Llama-2-7b-hf"
quantizer = AutoAWQ.from_pretrained(model_path)

# 量化配置
quantizer.quantize(
    quant_bits=4,           # 量化位数
    quant_group=128,        # 分组大小
    quant_method="awq",     # 量化方法
    calibration_data=[      # 校准数据
        "The history of artificial intelligence...",
        "Machine learning is a subset of...",
    ],
)

# 保存量化模型
quantizer.save_quantized(model_path, "llama2-7b-awq")
```

### 3.5 使用 vLLM 部署 AWQ 模型

```bash
# 使用 vLLM 加载 AWQ 量化模型
vllm serve TheBloke/Llama-2-7B-Chat-AWQ \
    --quantization awq \
    --dtype half
```

```python
# Python API
from vllm import LLM, SamplingParams

llm = LLM(
    model="TheBloke/Llama-2-7B-Chat-AWQ",
    quantization="awq",
    dtype="half"
)

outputs = llm.generate(["你好"], SamplingParams(max_tokens=512))
print(outputs[0].outputs[0].text)
```

## 4. GGUF 量化

### 4.1 原理介绍

GGUF 是 GGML (Georgi Gerganov's Machine Learning) 库中的量化格式，专门为 CPU+GPU 混合推理设计，支持多种精度选项。

```
┌─────────────────────────────────────────────────────────────────┐
│                      GGUF 量化格式                               │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  GGUF 支持的量化类型:                                            │
│                                                                 │
│  ┌──────────┬───────────┬────────────────────────────────┐     │
│  │ 类型      │ 每参数位数 │ 描述                           │     │
│  ├──────────┼───────────┼────────────────────────────────┤     │
│  │ Q4_0     │ 4 bits    │ 经典量化                        │     │
│  │ Q4_1     │ 4 bits    │ 更平滑的量化                    │     │
│  │ Q5_0     │ 5 bits    │ 5位量化                         │     │
│  │ Q5_1     │ 5 bits    │ 更平滑的5位量化                 │     │
│  │ Q8_0     │ 8 bits    │ 8位量化，接近FP16               │     │
│  │ Q8_1     │ 8 bits    │ 8位高精度                       │     │
│  │ F16      │ 16 bits   │ 半精度浮点                      │     │
│  │ F32      │ 32 bits   │ 单精度浮点                      │     │
│  └──────────┴───────────┴────────────────────────────────┘     │
│                                                                 │
│  量化方法:                                                       │
│  - 静态量化：使用固定 scale                                      │
│  - 动态量化：在推理时计算 scale                                  │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 4.2 llama.cpp 量化

```bash
# 安装 llama.cpp
git clone https://github.com/ggerganov/llama.cpp.git
cd llama.cpp
make

# 下载并转换模型为 GGUF 格式
# 1. 将模型转换为 GGML 格式
python scripts/convert.py /path/to/llama-2-7b --outfile models/llama-7b.gguf

# 2. 量化模型
./quantize models/llama-7b.gguf models/llama-7b-q4_0.bin q4_0
```

### 4.3 使用 Python 调用

```python
from llama_cpp import Llama

# 加载量化模型
llm = Llama(
    model_path="./models/llama-2-7b-q4_0.bin",
    n_ctx=4096,           # 上下文长度
    n_threads=4,          # CPU 线程数
    n_gpu_layers=0,       # GPU 层数 (0=纯CPU)
)

# 推理
output = llm(
    "什么是人工智能？",
    max_tokens=512,
    temperature=0.7,
)

print(output['choices'][0]['text'])
```

### 4.4 多线程和 GPU 加速

```python
from llama_cpp import Llama

# CPU 多线程推理
llm = Llama(
    model_path="./models/llama-2-7b-q4_0.bin",
    n_ctx=4096,
    n_threads=8,           # 使用8个CPU线程
    n_threads_batch=8,    // 批量处理线程
)

# GPU 加速
llm = Llama(
    model_path="./models/llama-2-7b-q4_0.bin",
    n_ctx=4096,
    n_gpu_layers=35,      // 将35层加载到GPU
)
```

### 4.5 LangChain 集成

```python
from langchain.llms import LlamaCpp
from langchain.prompts import PromptTemplate

# 初始化
llm = LlamaCpp(
    model_path="./models/llama-2-7b-q4_0.bin",
    n_ctx=2048,
    n_threads=4,
)

# 使用 LangChain
template = """问题: {question}

请给出详细回答: """

prompt = PromptTemplate(template=template, input_variables=["question"])
chain = prompt | llm

result = chain.invoke({"question": "深度学习和机器学习有什么区别？"})
print(result)
```

## 5. 量化对比

### 5.1 性能对比

```
┌─────────────────────────────────────────────────────────────────┐
│                    量化方法性能对比                              │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  模型: LLaMA-2-7B                                               │
│                                                                 │
│  ┌────────────┬───────────┬───────────┬──────────────────┐   │
│  │ 量化方法    │ 模型大小   │ 显存占用  │ 推理速度 (token/s)│   │
│  ├────────────┼───────────┼───────────┼──────────────────┤   │
│  │ FP16       │ 14GB      │ ~16GB     │ ~30              │   │
│  │ INT8       │ 7GB       │ ~10GB     │ ~25              │   │
│  │ GPTQ INT4  │ 3.5GB     │ ~6GB      │ ~20              │   │
│  │ AWQ INT4   │ 3.5GB     │ ~6GB      │ ~22              │   │
│  │ GGUF Q4_0  │ 3.9GB     │ ~5GB(CPU) │ ~15 (CPU)        │   │
│  │ GGUF Q4_0  │ 3.9GB     │ ~6GB(GPU) │ ~25 (GPU)        │   │
│  └────────────┴───────────┴───────────┴──────────────────┘   │
│                                                                 │
│  * 实际性能取决于硬件配置和具体模型                               │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 5.2 精度对比 (Perplexity)

```
┌─────────────────────────────────────────────────────────────────┐
│              在 WikiText-2 上的 Perplexity 对比                 │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌────────────┬────────────────┬────────────────┐              │
│  │ 量化方法    │ 7B 模型        │ 13B 模型       │              │
│  ├────────────┼────────────────┼────────────────┤              │
│  │ FP16       │ 5.09           │ 4.56           │              │
│  │ INT8       │ 5.21           │ 4.68           │              │
│  │ GPTQ INT4  │ 5.56           │ 4.82           │              │
│  │ AWQ INT4   │ 5.34           │ 4.71           │              │
│  │ GGUF Q4_0  │ 5.49           │ 4.79           │              │
│  └────────────┴────────────────┴────────────────┘              │
│                                                                 │
│  * Perplexity 越低越好                                          │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

## 6. 量化实战

### 6.1 选择量化方法

```python
"""
根据场景选择量化方法
"""

def choose_quantization_method(use_case, hardware, precision_required=True):
    """
    use_case: "production" | "research" | "personal"
    hardware: "gpu" | "cpu" | "hybrid"
    precision_required: True | False
    """
    
    if hardware == "cpu":
        # CPU 场景：推荐 GGUF
        return {
            "method": "GGUF",
            "format": "Q4_0" if not precision_required else "Q5_1",
            "library": "llama.cpp",
            "pros": ["CPU 推理优化", "无需 GPU", "部署简单"],
        }
    
    elif hardware == "gpu":
        # GPU 场景：推荐 AWQ 或 GPTQ
        if precision_required:
            return {
                "method": "AWQ",
                "format": "INT4",
                "library": "vllm/awq",
                "pros": ["精度更好", "推理速度快"],
            }
        else:
            return {
                "method": "GPTQ",
                "format": "INT4", 
                "library": "transformers",
                "pros": ["量化速度快", "兼容性好"],
            }
    
    else:  # hybrid
        return {
            "method": "GGUF",
            "format": "Q4_K_M",  // 支持 GPU offload
            "library": "llama.cpp",
            "pros": ["灵活分配", "CPU+GPU 混合"],
        }
```

### 6.2 完整量化流程示例

```python
"""
完整模型量化流程
"""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from optimum.gptq import GPTQQuantizer
from awq import AutoAWQ
import os

class ModelQuantizer:
    def __init__(self, model_path, output_dir):
        self.model_path = model_path
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
    
    def quantize_gptq(self, bits=4, group_size=128):
        """GPTQ 量化"""
        print(f"开始 GPTQ {bits}bit 量化...")
        
        # 加载模型
        model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        
        # 量化
        quantizer = GPTQQuantizer(
            bits=bits,
            group_size=group_size
        )
        
        quantized_model = quantizer.quantize_model(model, tokenizer)
        
        # 保存
        output_path = os.path.join(self.output_dir, f"gptq-{bits}bit")
        quantized_model.save_pretrained(output_path)
        tokenizer.save_pretrained(output_path)
        
        print(f"GPTQ 量化完成，保存至: {output_path}")
        return output_path
    
    def quantize_awq(self, bits=4):
        """AWQ 量化"""
        print(f"开始 AWQ {bits}bit 量化...")
        
        quantizer = AutoAWQ.from_pretrained(self.model_path)
        
        # 量化
        quantizer.quantize(
            quant_bits=bits,
            quant_group=128,
            quant_method="awq",
            calibration_data=["sample text 1", "sample text 2"]
        )
        
        # 保存
        output_path = os.path.join(self.output_dir, f"awq-{bits}bit")
        quantizer.save_quantized(output_path)
        
        print(f"AWQ 量化完成，保存至: {output_path}")
        return output_path

# 使用示例
if __name__ == "__main__":
    quantizer = ModelQuantizer(
        model_path="meta-llama/Llama-2-7b-hf",
        output_dir="./quantized_models"
    )
    
    # 选择量化方法
    # quantizer.quantize_gptq(bits=4)
    # quantizer.quantize_awq(bits=4)
```

## 7. 常见问题

### 7.1 量化后模型精度下降过多

```python
# 解决思路：
# 1. 使用更温和的量化 (INT8 而不是 INT4)
# 2. 增加校准数据集大小
# 3. 使用 AWQ 而不是 GPTQ (通常精度更好)
# 4. 尝试不同的 group_size

# 示例: 使用 INT8 量化
quantizer = GPTQQuantizer(bits=8)  # 而非 bits=4
```

### 7.2 推理速度没有提升

```python
# 检查：
# 1. 模型是否真正被量化 (检查参数类型)
# 2. GPU 显存是否足够
# 3. Batch size 是否设置合理
# 4. 是否使用了正确的量化后端

# 验证量化
import torch
model = AutoModelForCausalLM.from_pretrained("quantized_model")
for name, param in model.named_parameters():
    print(f"{name}: {param.dtype}")
```

### 7.3 显存占用仍然很高

```bash
# 解决：
# 1. 使用更激进的量化 (Q4_0 而不是 Q5_1)
# 2. 减少 max_model_len / n_ctx
# 3. 使用 GGUF + CPU 推理
# 4. 开启 KV Cache 量化
```

## 8. 总结

本章我们详细介绍了三种主流的 LLM 量化技术：

1. **GPTQ**
   - 基于分层贪婪算法
   - 逐列量化，最小化重建误差
   - 适合 GPU 推理

2. **AWQ**
   - 考虑激活值分布
   - 保护重要权重
   - 精度通常优于 GPTQ

3. **GGUF**
   - CPU+GPU 混合推理
   - 多种精度选择
   - 部署灵活

量化是 LLM 部署中必不可少的技术，正确选择量化方法可以显著降低资源需求同时保持可接受的模型性能。
