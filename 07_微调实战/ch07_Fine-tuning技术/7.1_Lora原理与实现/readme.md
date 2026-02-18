# LoRA 原理与实现

## 1. 什么是 LoRA？

**LoRA (Low-Rank Adaptation)** 是一种高效的参数微调技术，由 Microsoft Research 于 2021 年提出。其核心思想是在预训练模型旁边添加**低秩分解矩阵**，通过训练少量参数来适应下游任务，而无需更新原始模型的所有参数。

### 1.1 LoRA 的核心优势

| 优势 | 说明 |
|------|------|
| **参数效率** | 只需训练原模型参数的 0.1%~1% |
| **内存效率** | 显著减少 GPU 显存占用 |
| **可插拔** | 可以轻松切换不同任务的 LoRA 权重 |
| **无推理延迟** | 推理时可以合并权重，无额外延迟 |

### 1.2 传统微调 vs LoRA

```
┌─────────────────────────────────────────────────────────────────┐
│                        传统全参数微调                            │
├─────────────────────────────────────────────────────────────────┤
│  Pre-trained Model (7B)  ──────▶  Fine-tuned Model (7B)        │
│                                                                 │
│  需要训练: 7,000,000,000 参数                                    │
│  GPU显存需求: ~14GB (FP16)                                      │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│                          LoRA 微调                              │
├─────────────────────────────────────────────────────────────────┤
│  Pre-trained Model (7B)  ──────▶  + LoRA Modules               │
│                                      ↓                          │
│                                  Output                         │
│                                                                 │
│  需要训练: ~10,000,000 参数 (r=8, rank=8)                      │
│  GPU显存需求: ~8GB (FP16)                                       │
└─────────────────────────────────────────────────────────────────┘
```

## 2. LoRA 原理详解

### 2.1 核心思想

LoRA 的核心思想源自于一个关键假设：**预训练模型在适应下游任务时，其参数变化矩阵具有低秩特性**。

对于预训练模型的权重矩阵 $W_0 \in \mathbb{R}^{d \times k}$，LoRA 保持 $W_0$ 不变，而是添加一个低秩更新：

$$W = W_0 + \Delta W$$

其中 $\Delta W$ 被分解为两个低秩矩阵的乘积：

$$\Delta W = BA$$

其中 $B \in \mathbb{R}^{d \times r}$，$A \in \mathbb{R}^{r \times k}$，$r \ll \min(d, k)$

### 2.2 架构图

```
┌────────────────────────────────────────────────────────────────────┐
│                        LoRA 架构图                                 │
└────────────────────────────────────────────────────────────────────┘

输入向量 x
    │
    ▼
┌─────────────────────┐
│   原始权重 W₀ (冻结) │
│   d × k             │
└─────────────────────┘
    │
    │  h = W₀x
    ▼
┌─────────────────────┐     ┌─────────────────────┐
│   LoRA 分支 A       │     │   LoRA 分支 B       │
│   r × k (下投影)    │────▶│   d × r (上投影)    │
└─────────────────────┘     └─────────────────────┘
    │                           ▲
    │         BAx              │
    │           │              │
    └───────────┼──────────────┘
                ▼
        h = W₀x + BAx
```

### 2.3 秩 (Rank) 的选择

| Rank (r) | 参数量 | 效果 |
|----------|--------|------|
| 1-4 | 极少 | 基础任务可用 |
| 8-16 | 少 | 推荐默认值，效果好 |
| 32-64 | 中等 | 复杂任务可用 |
| 128+ | 较多 | 接近全参数微调 |

典型的配置：
- `r = 8`: 最小配置
- `r = 16`: 推荐默认值
- `r = 32`: 复杂任务

### 2.4 哪些层使用 LoRA？

通常对以下注意力层应用 LoRA：

```
┌─────────────────────────────────────────────────────────┐
│                   Transformer 层的 LoRA                 │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  Input                                                  │
│    │                                                    │
│    ▼                                                    │
│  ┌─────────────────┐                                   │
│  │ Attention 层    │ ◀── LoRA 主要应用位置              │
│  │ ├─ Q_proj       │                                   │
│  │ ├─ K_proj       │  ← 通常对 Q, K, V, O 投影应用 LoRA │
│  │ ├─ V_proj       │                                   │
│  │ └─ out_proj     │                                   │
│  └─────────────────┘                                   │
│    │                                                    │
│    ▼                                                    │
│  ┌─────────────────┐                                   │
│  │ FFN 层          │ ◀── 可选：也可对 FFN 应用 LoRA     │
│  │ ├─ gate_proj    │                                   │
│  │ ├─ up_proj      │                                   │
│  │ └─ down_proj    │                                   │
│  └─────────────────┘                                   │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

## 3. 代码实现

### 3.1 基础 LoRA 实现

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

class LoRALayer(nn.Module):
    """基础 LoRA 层实现"""
    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int = 8,
        alpha: int = 16,
        dropout: float = 0.0,
        bias: bool = False
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        
        # 冻结原始权重
        self.weight = nn.Parameter(
            torch.empty(out_features, in_features)
        )
        self.weight.requires_grad = False
        
        # LoRA 矩阵 A 和 B
        # A: 将输入从 in_features 投影到 rank
        self.lora_A = nn.Parameter(
            torch.empty(rank, in_features)
        )
        # B: 将 rank 投影到 out_features
        self.lora_B = nn.Parameter(
            torch.empty(out_features, rank)
        )
        
        # Dropout
        self.dropout = nn.Dropout(p=dropout) if dropout > 0 else nn.Identity()
        
        # 偏置
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.bias = None
            
        self._init_parameters()
    
    def _init_parameters(self):
        """初始化参数"""
        # 原始权重使用 xavier 初始化
        nn.init.xavier_uniform_(self.weight)
        
        # LoRA A 使用随机初始化
        nn.init.normal_(self.lora_A, std=0.02)
        
        # LoRA B 初始化为零（训练初期无影响）
        nn.init.zeros_(self.lora_B)
        
        if self.bias is not None:
            nn.init.zeros_(self.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播"""
        # 原始权重计算
        output = F.linear(x, self.weight, self.bias)
        
        # LoRA 分支计算
        # x @ A.T @ B.T = x @ (B @ A).T
        lora_output = self.dropout(x) @ self.lora_A.T @ self.lora_B.T
        
        # 缩放 LoRA 输出
        return output + lora_output * self.scaling


class LoRALinear(nn.Module):
    """集成 LoRA 的线性层"""
    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int = 8,
        alpha: int = 16,
        dropout: float = 0.0,
        bias: bool = False,
        enable_lora: bool = True
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.enable_lora = enable_lora
        
        # 原始线性层
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        
        if enable_lora:
            self.lora = LoRALayer(
                in_features, out_features, rank, alpha, dropout, bias=False
            )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.enable_lora:
            return self.lora(self.linear(x))
        return self.linear(x)
    
    def merge_weights(self):
        """合并 LoRA 权重到原始权重"""
        if self.enable_lora:
            # 计算合并后的权重
            delta_weight = self.lora.lora_B @ self.lora.lora_A * self.lora.scaling
            merged_weight = self.linear.weight.data + delta_weight
            self.linear.weight.data = merged_weight
            # 禁用 LoRA
            self.lora = None
            self.enable_lora = False
```

### 3.2 在 Transformer 中应用 LoRA

```python
import math
from dataclasses import dataclass

@dataclass
class LoRAConfig:
    """LoRA 配置"""
    rank: int = 8
    alpha: int = 16
    dropout: float = 0.05
    target_modules: list = None
    
    def __post_init__(self):
        if self.target_modules is None:
            # 默认对 QKV 和输出投影应用 LoRA
            self.target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]


class LoRAAttention(nn.Module):
    """带 LoRA 的注意力机制"""
    
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        head_dim: int = None,
        lora_config: LoRAConfig = None
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = head_dim or hidden_size // num_heads
        self.lora_config = lora_config or LoRAConfig()
        
        # 投影层
        self.q_proj = nn.Linear(hidden_size, self.num_heads * self.head_dim)
        self.k_proj = nn.Linear(hidden_size, self.num_heads * self.head_dim)
        self.v_proj = nn.Linear(hidden_size, self.num_heads * self.head_dim)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, hidden_size)
        
        # 应用 LoRA
        self._apply_lora()
    
    def _apply_lora(self):
        """对指定模块应用 LoRA"""
        config = self.lora_config
        
        if "q_proj" in config.target_modules:
            self.q_proj = LoRALinear(
                self.hidden_size, self.num_heads * self.head_dim,
                rank=config.rank, alpha=config.alpha,
                dropout=config.dropout, bias=False
            )
        
        if "k_proj" in config.target_modules:
            self.k_proj = LoRALinear(
                self.hidden_size, self.num_heads * self.head_dim,
                rank=config.rank, alpha=config.alpha,
                dropout=config.dropout, bias=False
            )
        
        if "v_proj" in config.target_modules:
            self.v_proj = LoRALinear(
                self.hidden_size, self.num_heads * self.head_dim,
                rank=config.rank, alpha=config.alpha,
                dropout=config.dropout, bias=False
            )
        
        if "o_proj" in config.target_modules:
            self.o_proj = LoRALinear(
                self.num_heads * self.head_dim, self.hidden_size,
                rank=config.rank, alpha=config.alpha,
                dropout=config.dropout, bias=False
            )
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        batch_size = hidden_states.size(0)
        
        # 投影 Q, K, V
        q = self.q_proj(hidden_states).view(
            batch_size, -1, self.num_heads, self.head_dim
        ).transpose(1, 2)
        k = self.k_proj(hidden_states).view(
            batch_size, -1, self.num_heads, self.head_dim
        ).transpose(1, 2)
        v = self.v_proj(hidden_states).view(
            batch_size, -1, self.num_heads, self.head_dim
        ).transpose(1, 2)
        
        # 计算注意力分数
        attention_scores = torch.matmul(q, k.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.head_dim)
        
        if attention_mask is not None:
            attention_scores = attention_scores + attention_mask
        
        attention_probs = F.softmax(attention_scores, dim=-1)
        
        # 应用注意力
        context = torch.matmul(attention_probs, v)
        context = context.transpose(1, 2).contiguous().view(
            batch_size, -1, self.num_heads * self.head_dim
        )
        
        # 输出投影
        output = self.o_proj(context)
        
        return output
```

### 3.3 使用 PEFT 库 (推荐)

实际项目中推荐使用 `peft` 库，它提供了完整的 LoRA 实现：

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model, TaskType

# 加载基础模型
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-hf",
    device_map="auto",
    load_in_8bit=True
)
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")

# 配置 LoRA
lora_config = LoraConfig(
    r=16,                      # Rank
    lora_alpha=32,             # Alpha (缩放因子)
    target_modules=[           # 目标模块
        "q_proj", 
        "k_proj", 
        "v_proj", 
        "o_proj"
    ],
    lora_dropout=0.05,         # Dropout
    bias="none",               # 偏置设置
    task_type=TaskType.CAUSAL_LM
)

# 应用 LoRA
model = get_peft_model(model, lora_config)

# 打印可训练参数
model.print_trainable_parameters()
# 输出: trainable params: 4,194,304 || all params: 6,742,609,920 || trainable%: 0.062

# 训练
# ... 训练代码 ...

# 保存 LoRA 权重
model.save_pretrained("lora_weights/")

# 加载 LoRA 权重
from peft import PeftModel
base_model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf")
model = PeftModel.from_pretrained(base_model, "lora_weights/")
```

### 3.4 完整训练脚本

```python
import torch
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer, 
    TrainingArguments,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model
from datasets import Dataset
import json

# 1. 准备数据
def prepare_data():
    """准备训练数据"""
    data = [
        {"instruction": "请帮我写一首关于春天的诗", "output": "春风拂面万物苏，\n绿柳轻摇映碧湖。\n燕子归来寻旧垒，\n花开时节满山隅。"},
        {"instruction": "解释一下什么是机器学习", "output": "机器学习是人工智能的一个分支，它使计算机能够从数据中学习并改进性能，而无需被明确编程。"},
    ]
    
    # 格式化数据
    formatted_data = []
    for item in data:
        text = f"### 指令\n{item['instruction']}\n\n### 回答\n{item['output']}"
        formatted_data.append({"text": text})
    
    return formatted_data

# 2. 配置 LoRA
def get_lora_config():
    return LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )

# 3. 主训练流程
def main():
    # 模型和分词器
    model_name = "gpt2"  # 可换成更大的模型
    model = AutoModelForCausalLM.from_pretrained(
        model_name, 
        torch_dtype=torch.float16,
        device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # 配置 pad token
    tokenizer.pad_token = tokenizer.eos_token
    
    # 应用 LoRA
    lora_config = get_lora_config()
    model = get_peft_model(model, lora_config)
    
    print("可训练参数比例:")
    model.print_trainable_parameters()
    
    # 准备数据
    train_data = prepare_data()
    dataset = Dataset.from_list(train_data)
    
    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=512,
            padding="max_length"
        )
    
    dataset = dataset.map(tokenize_function, batched=True)
    
    # 训练参数
    training_args = TrainingArguments(
        output_dir="./lora_output",
        num_train_epochs=3,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=2,
        learning_rate=3e-4,
        fp16=True,
        logging_steps=10,
        save_strategy="epoch",
        save_total_limit=2,
    )
    
    # 数据整理器
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False  # 因果语言模型
    )
    
    # 训练器
    from transformers import Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=data_collator,
    )
    
    # 开始训练
    trainer.train()
    
    # 保存模型
    model.save_pretrained("./lora_finetuned")
    tokenizer.save_pretrained("./lora_finetuned")

if __name__ == "__main__":
    main()
```

## 4. LoRA 最佳实践

### 4.1 超参数推荐

```python
# 常见模型的最佳 LoRA 配置

# LLaMA 7B
lora_config_llama7b = {
    "r": 16,
    "alpha": 32,
    "dropout": 0.05,
    "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],
    "bias": "none",
}

# LLaMA 13B
lora_config_llama13b = {
    "r": 16,
    "alpha": 32,
    "dropout": 0.05,
    "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],
    "bias": "none",
}

# BLOOM
lora_config_bloom = {
    "r": 8,
    "lora_alpha": 16,
    "target_modules": ["query_key_value", "dense"],
    "bias": "none",
}

# StarCoder
lora_config_starcoder = {
    "r": 8,
    "lora_alpha": 16,
    "target_modules": ["c_attn", "c_proj"],
    "bias": "none",
}
```

### 4.2 训练技巧

1. **学习率**: 推荐使用 `3e-4` 到 `1e-3`，比全参数微调高 10-100 倍
2. **Epochs**: 通常 2-3 个 epoch 足够
3. **Batch Size**: 配合梯度累积，可以使用较大的 effective batch size
4. **Alpha 调优**: 通常设置为 rank 的 1-2 倍

### 4.3 常见问题

**Q: LoRA 训练效果不佳？**
- 检查 target_modules 是否正确
- 尝试增大 rank
- 检查数据质量

**Q: 推理时如何合并权重？**
```python
# 方法1: 合并权重
model.merge_weights()
# 之后推理与普通模型相同

# 方法2: 保持分离（推荐，便于切换任务）
# 推理时动态加载不同的 LoRA 权重
from peft import PeftModel
model = PeftModel.from_pretrained(base_model, "lora_checkpoint/")
```

## 5. LoRA 变体

### 5.1 LoRA+

```python
# LoRA+ 核心思想：对 A 和 B 使用不同的学习率
# B 的学习率是 A 的若干倍
```

### 5.2 LoRA 权重合并可视化

```
┌─────────────────────────────────────────────────────────────────┐
│                      权重合并流程                                │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  原始权重 W₀                                                    │
│     │                                                           │
│     ▼                                                           │
│  ┌──────────────────────┐                                      │
│  │  LoRA 权重 ΔW = BA   │                                      │
│  │  缩放因子: α/r       │                                      │
│  └──────────────────────┘                                      │
│     │                                                           │
│     ▼                                                           │
│  W_merged = W₀ + (α/r) * BA                                    │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

## 6. 总结

LoRA 是一种高效、实用的参数微调技术，通过低秩分解实现了：

1. **参数效率**: 仅需训练少量参数
2. **内存效率**: 大幅降低显存需求
3. **灵活性**: 可轻松切换不同任务
4. **实用性**: 已广泛用于实际项目

推荐在实际项目中使用 `peft` 库，它提供了稳定可靠的实现。
