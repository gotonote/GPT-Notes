# QLoRA 原理与实现

## 1. 什么是 QLoRA？

**QLoRA (Quantized LoRA)** 是由 Tim Dettmers 等人于 2023 年提出的一种高效的模型微调技术。它结合了**量化 (Quantization)** 和 **LoRA** 两种技术，使得在单个消费级 GPU 上能够微调超过 100B 参数的大模型。

### 1.1 QLoRA 的核心创新

| 技术 | 作用 |
|------|------|
| **4-bit 量化** | 将模型权重从 FP16 压缩到 4-bit，显存减少 4 倍 |
| **双量化 (Double Quantization)** | 进一步压缩量化后的缩放因子 |
| **分页优化器 (Paged Optimizers)** | 使用 CPU 内存管理梯度优化器状态 |
| **LoRA** | 仅训练少量低秩适配器参数 |

### 1.2 显存对比

```
┌─────────────────────────────────────────────────────────────────┐
│                      显存需求对比 (65B 模型)                     │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  全参数 FP16:       ████████████████████████████████  130GB    │
│  LoRA FP16:        ████████████                        ~65GB    │
│  QLoRA 4-bit:      █████                                 ~30GB  │
│  QLoRA 4-bit (24GB GPU):                              ████     │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

## 2. QLoRA 原理详解

### 2.1 量化基础

QLoRA 使用 **NF4 (Normalized Float 4-bit)** 量化格式，这是一种针对大模型权重分布优化的量化方法。

```python
import torch
import torch.nn.functional as F

class NF4Quantizer:
    """NF4 量化器实现"""
    
    # NF4 量化级别（针对大模型权重分布优化）
    QUANT_LEVELS = torch.tensor([
        -1.0, -0.6961928009986877, -0.5250732717514038,
        -0.39491748809814453, -0.2844413813495636,
        -0.18477358078956604, -0.09105003621578217,
        0.0, 0.09105003621578217, 0.18477358078956604,
        0.2844413813495636, 0.39491748809814453,
        0.5250732717514038, 0.6961928009986877, 1.0
    ])
    
    def __init__(self, bits=4):
        self.bits = bits
        self.quant_levels = self.QUANT_LEVELS.to(dtype=torch.float32)
    
    def quantize(self, weights: torch.Tensor):
        """量化权重到 NF4 格式"""
        # 归一化权重到 [-1, 1]
        max_val = weights.abs().max()
        normalized = weights / max_val
        
        # 量化到最近的 NF4 级别
        quantized = torch.zeros_like(normalized)
        for i, level in enumerate(self.quant_levels):
            mask = torch.abs(normalized - level).argmin(dim=-1)
            # 这里简化处理，实际实现更复杂
            pass
        
        # 存储缩放因子用于反量化
        return quantized, max_val
    
    def dequantize(quantized_weights, scale):
        """反量化"""
        return quantized_weights * scale
```

### 2.2 QLoRA 架构

```
┌────────────────────────────────────────────────────────────────────┐
│                        QLoRA 架构图                                 │
└────────────────────────────────────────────────────────────────────┘

                    ┌─────────────────────────────────┐
                    │   预训练模型 (FP16)              │
                    │   65B 参数                       │
                    └─────────────────────────────────┘
                                │
                                │ 4-bit 量化
                                ▼
                    ┌─────────────────────────────────┐
                    │   量化权重 (NF4)                │
                    │   存储: 4-bit                   │
                    │   计算: 16-bit                   │
                    └─────────────────────────────────┘
                                │
                    ┌───────────┴───────────┐
                    │                       │
                    ▼                       ▼
        ┌───────────────────┐   ┌───────────────────┐
        │  LoRA 分支 A      │   │  梯度计算         │
        │  (量化权重上)     │   │  (仅 LoRA 参数)   │
        └───────────────────┘   └───────────────────┘
                    │                       │
                    └───────────┬───────────┘
                                │
                                ▼
                    ┌─────────────────────────────────┐
                    │   输出: 16-bit                   │
                    │   (FP16 计算结果)               │
                    └─────────────────────────────────┘
```

### 2.3 核心组件

1. **4-bit NormalFloat4 量化**: 针对大模型权重分布优化
2. **双量化**: 对量化缩放因子再次量化
3. **Paged Optimizers**: 使用 CPU 分页内存管理优化器状态

```
┌─────────────────────────────────────────────────────────────────┐
│                     分页优化器机制                               │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  GPU 显存                                                      │
│  ┌─────────────────┐                                          │
│  │  模型权重 (Q)   │  ← 4-bit 量化存储                        │
│  │  LoRA 参数     │  ← FP16 (少量)                            │
│  │  激活值         │  ← 反量化到 FP16 计算                     │
│  └─────────────────┘                                          │
│          │                                                     │
│          ▼ 梯度                                                │
│  ┌─────────────────┐                                          │
│  │  优化器状态     │  ← CPU 分页内存                          │
│  │  (Adam 动量)   │     当 GPU 不足时自动使用                  │
│  └─────────────────┘                                          │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

## 3. 代码实现

### 3.1 使用 bitsandbytes 库

最简单的方式是使用 `bitsandbytes` 库配合 PEFT：

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model, TaskType
from bitsandbytes import BitsAndBytesConfig

# 1. 配置 4-bit 量化
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,                    # 4-bit 加载
    bnb_4bit_quant_type="nf4",            # NF4 量化类型
    bnb_4bit_compute_dtype=torch.float16, # 计算时使用 FP16
    bnb_4bit_use_double_quant=True,       # 启用双量化
)

# 2. 加载量化模型
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-13b-hf",          # 13B 模型
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True,
)
tokenizer = AutoTokenizer.from_pretrained(
    "meta-llama/Llama-2-13b-hf",
    trust_remote_code=True
)

# 3. 配置 LoRA
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.CAUSAL_LM,
)

# 4. 应用 LoRA
model = get_peft_model(model, lora_config)

# 5. 打印可训练参数
model.print_trainable_parameters()
# 输出: trainable params: 6,291,456 || all params: 13,502,048,256 || trainable%: 0.047
```

### 3.2 完整 QLoRA 训练脚本

```python
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model, TaskType
from bitsandbytes import BitsAndBytesConfig
from datasets import Dataset

# ==================== 配置 ====================
class QLoRAConfig:
    model_name = "facebook/opt-13b"  # 使用 OPT-13B 作为示例
    output_dir = "./qlora_output"
    lora_r = 16
    lora_alpha = 32
    lora_dropout = 0.05
    learning_rate = 3e-4
    num_epochs = 3
    per_device_train_batch_size = 4
    gradient_accumulation_steps = 2
    max_seq_length = 512


def get_bnb_config():
    """获取 BitsAndBytes 量化配置"""
    return BitsAndBytesConfig(
        # 4-bit 量化
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        
        # 计算时使用 FP16
        bnb_4bit_compute_dtype=torch.float16,
        
        # 双量化 - 进一步压缩
        bnb_4bit_use_double_quant=True,
        
        # 量化阈值 - NF4 建议使用
        bnb_4bit_quant_storage=None,  # 使用默认值
    )


def get_lora_config():
    """获取 LoRA 配置"""
    return LoraConfig(
        r=QLoRAConfig.lora_r,
        lora_alpha=QLoRAConfig.lora_alpha,
        lora_dropout=QLoRAConfig.lora_dropout,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj"  # 可选：也添加到 FFN
        ],
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )


def prepare_sample_data():
    """准备训练样本"""
    samples = [
        {
            "instruction": "什么是人工智能？",
            "output": "人工智能（Artificial Intelligence，AI）是计算机科学的一个分支，致力于开发能够模拟、延伸和扩展人类智能的理论、方法、技术及应用系统。"
        },
        {
            "instruction": "解释一下机器学习和深度学习的区别",
            "output": "机器学习是人工智能的一个子领域，专注于开发能够从数据中学习的算法。深度学习则是机器学习的一个分支，使用多层神经网络来学习数据的层次化表示。"
        },
        {
            "instruction": "请介绍一下 Python 的优势",
            "output": "Python 是一种高级编程语言，具有以下优势：1. 简洁易读的语法 2. 丰富的库支持 3. 广泛的应用领域 4. 强大的社区支持 5. 跨平台兼容性"
        },
    ]
    
    # 格式化
    formatted = []
    for s in samples:
        text = f"### 指令\n{s['instruction']}\n\n### 回答\n{s['output']}"
        formatted.append({"text": text})
    
    return formatted


def main():
    print("=" * 50)
    print("QLoRA 训练开始")
    print("=" * 50)
    
    # 1. 加载量化模型
    print("\n[1/5] 加载量化模型...")
    bnb_config = get_bnb_config()
    model = AutoModelForCausalLM.from_pretrained(
        QLoRAConfig.model_name,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )
    
    tokenizer = AutoTokenizer.from_pretrained(
        QLoRAConfig.model_name,
        trust_remote_code=True
    )
    tokenizer.pad_token = tokenizer.eos_token
    
    # 2. 应用 LoRA
    print("[2/5] 应用 LoRA...")
    lora_config = get_lora_config()
    model = get_peft_model(model, lora_config)
    
    # 3. 打印参数统计
    print("[3/5] 参数统计:")
    model.print_trainable_parameters()
    
    # 4. 准备数据
    print("[4/5] 准备训练数据...")
    train_data = prepare_sample_data()
    dataset = Dataset.from_list(train_data)
    
    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=QLoRAConfig.max_seq_length,
            padding="max_length",
        )
    
    dataset = dataset.map(tokenize_function, batched=True)
    
    # 5. 配置训练参数
    print("[5/5] 开始训练...")
    training_args = TrainingArguments(
        output_dir=QLoRAConfig.output_dir,
        num_train_epochs=QLoRAConfig.num_epochs,
        per_device_train_batch_size=QLoRAConfig.per_device_train_batch_size,
        gradient_accumulation_steps=QLoRAConfig.gradient_accumulation_steps,
        learning_rate=QLoRAConfig.learning_rate,
        fp16=True,
        logging_steps=10,
        save_strategy="steps",
        save_steps=50,
        save_total_limit=2,
        optim="paged_adamw_8bit",  # 使用分页优化器节省显存
        report_to="none",
    )
    
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )
    
    # 训练
    from transformers import Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=data_collator,
    )
    
    trainer.train()
    
    # 保存
    model.save_pretrained(f"{QLoRAConfig.output_dir}/final")
    tokenizer.save_pretrained(f"{QLoRAConfig.output_dir}/final")
    
    print("\n训练完成!")


if __name__ == "__main__":
    main()
```

### 3.3 模型推理

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# 加载基础模型（量化）
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
)

base_model = AutoModelForCausalLM.from_pretrained(
    "facebook/opt-13b",
    quantization_config=bnb_config,
    device_map="auto",
)

# 加载 LoRA 权重
model = PeftModel.from_pretrained(
    base_model,
    "./qlora_output/final"
)

tokenizer = AutoTokenizer.from_pretrained("facebook/opt-13b")

# 推理
def generate(prompt: str, max_length: int = 200):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    outputs = model.generate(
        **inputs,
        max_length=max_length,
        temperature=0.7,
        top_p=0.9,
    )
    
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# 测试
prompt = "### 指令\n什么是人工智能？\n\n### 回答\n"
response = generate(prompt)
print(response)
```

### 3.4 合并 LoRA 权重（可选）

```python
# 合并 LoRA 权重到量化模型
model = model.merge_and_unload()

# 保存合并后的模型
model.save_pretrained("merged_model")
tokenizer.save_pretrained("merged_model")
```

## 4. QLoRA 最佳实践

### 4.1 量化配置选择

| 场景 | 量化类型 | 显存 | 效果 |
|------|----------|------|------|
| 消费级 GPU (24GB) | 4-bit NF4 | ~30GB (65B) | 可接受 |
| 专业级 GPU (48GB) | 4-bit NF4 | ~40GB (65B) | 很好 |
| 高精度需求 | 8-bit | 翻倍 | 更好 |

### 4.2 LoRA 配置优化

```python
# 针对不同大小模型的推荐配置

# 小模型 (< 7B)
small_model_config = {
    "r": 8,
    "lora_alpha": 16,
    "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],
}

# 中等模型 (7B - 13B)
medium_model_config = {
    "r": 16,
    "lora_alpha": 32,
    "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],
}

# 大模型 (> 13B)
large_model_config = {
    "r": 16,
    "lora_alpha": 32,
    "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
}
```

### 4.3 训练技巧

1. **使用 paged optimizer**: `optim="paged_adamw_8bit"`
2. **梯度累积**: 弥补 batch size 的不足
3. **适当的序列长度**: 根据显存调整
4. **监控显存**: 使用 `nvidia-smi` 监控

## 5. QLoRA vs 其他技术对比

```
┌─────────────────────────────────────────────────────────────────┐
│                    大模型微调技术对比                            │
├──────────────────┬──────────┬──────────┬──────────┬────────────┤
│     技术         │  显存    │ 训练速度  │  效果    │   难度     │
├──────────────────┼──────────┼──────────┼──────────┼────────────┤
│ 全参数 FP16      │  130GB   │  快      │  最好    │   复杂     │
│ 全参数 FP32      │  260GB   │  快      │  最好    │   复杂     │
│ LoRA FP16        │  ~65GB   │  快      │   好     │   简单     │
│ QLoRA 4-bit     │  ~30GB   │  较慢    │   不错   │   中等     │
│ QLoRA 8-bit     │  ~50GB   │  中等    │   好     │   中等     │
└──────────────────┴──────────┴──────────┴──────────┴────────────┘
```

## 6. 常见问题

**Q: QLoRA 训练效果比 LoRA 差？**
- A: 4-bit 量化会引入一定的精度损失，但可以通过调整 r 值或使用 8-bit 量化来平衡

**Q: 如何选择量化位数？**
- A: 显存充足时使用 8-bit，追求最低显存时使用 4-bit

**Q: 训练时显存不足？**
- A: 1) 减小序列长度 2) 减小 batch size 3) 使用梯度累积 4) 考虑使用 4-bit

## 7. 总结

QLoRA 通过结合量化和 LoRA 技术，使得在消费级 GPU 上微调超大模型成为可能：

1. **4-bit NF4 量化**: 大幅减少显存占用
2. **双量化**: 进一步压缩
3. **分页优化器**: 智能管理显存
4. **LoRA**: 保持高效的参数更新

推荐在实际项目中使用 `bitsandbytes` + `peft` 的组合方案。
