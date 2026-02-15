# 全参数微调 (Full-Parameter Finetuning)

## 1. 什么是全参数微调？

全参数微调 (Full-Parameter Finetuning) 是指在微调过程中更新模型的所有参数。与参数高效微调（如 LoRA、Prefix-Tuning）不同，全参数微调需要调整预训练模型的每一个权重。

### 1.1 全参数微调的特点

| 特点 | 说明 |
|------|------|
| **参数更新** | 更新模型全部参数 |
| **显存需求** | 高（需要存储梯度和优化器状态） |
| **效果** | 通常是最好的 |
| **灵活性** | 最高，可完全适应新任务 |

### 1.2 显存需求分析

```
┌─────────────────────────────────────────────────────────────────┐
│                    全参数微调显存需求                            │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  模型参数 (FP16):          ████████████████████████  7B ≈ 14GB │
│  梯度 (FP16):             ████████████████████████  7B ≈ 14GB  │
│  优化器状态 (FP32):       ████████████████████████████████     │
│  动量/方差 (FP32):        ████████████████████████████████     │
│                                              共计: 约 80-120GB │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

## 2. 全参数微调原理

### 2.1 训练流程

```
┌─────────────────────────────────────────────────────────────────┐
│                      全参数微调流程图                            │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │                    预训练模型                              │  │
│  │                  θ (随机初始化)                           │  │
│  └──────────────────────────────────────────────────────────┘  │
│                           │                                     │
│                           ▼                                     │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │                    前向传播                               │  │
│  │        h = f(x; θ)  →  loss = L(h, y)                    │  │
│  └──────────────────────────────────────────────────────────┘  │
│                           │                                     │
│                           ▼                                     │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │                    反向传播                               │  │
│  │        ∇θ = ∂L/∂θ  (计算所有参数的梯度)                   │  │
│  └──────────────────────────────────────────────────────────┘  │
│                           │                                     │
│                           ▼                                     │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │                    参数更新                               │  │
│  │        θ_new = θ_old - lr * ∇θ                          │  │
│  │  (Adam 优化器会保留动量等状态)                            │  │
│  └──────────────────────────────────────────────────────────┘  │
│                           │                                     │
│                           └──────────┬──────────────┐            │
│                                      │              │            │
│                         ┌────────────▼─────┐ ┌─────▼────────┐   │
│                         │   更新优化器状态  │ │  更新模型参数 │   │
│                         │  m_t, v_t (FP32) │ │    θ (FP16)   │   │
│                         └──────────────────┘ └───────────────┘   │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 2.2 梯度计算

```python
import torch
import torch.nn as nn

class FullParameterFineTuner:
    """全参数微调管理器"""
    
    def __init__(self, model, learning_rate=1e-5):
        self.model = model
        self.learning_rate = learning_rate
        
        # 使用 AdamW 优化器（更新所有参数）
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=0.01,
            betas=(0.9, 0.999),
            eps=1e-8
        )
    
    def training_step(self, batch):
        """单步训练"""
        self.model.train()
        
        # 前向传播
        inputs = batch["input_ids"]
        labels = batch["labels"]
        outputs = self.model(inputs, labels=labels)
        loss = outputs.loss
        
        # 反向传播
        loss.backward()
        
        # 参数更新
        self.optimizer.step()
        self.optimizer.zero_grad()
        
        return loss.item()
```

## 3. 代码实现

### 3.1 使用 Transformers Trainer

```python
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
    Trainer
)
from datasets import Dataset

# 1. 加载模型
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-hf",
    torch_dtype=torch.float16,      # 使用 FP16
    device_map="auto",               # 自动设备映射
)
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")
tokenizer.pad_token = tokenizer.eos_token

# 2. 准备数据
def prepare_data():
    data = [
        {"text": "### 指令\n什么是人工智能？\n\n### 回答\n人工智能是..."},
        {"text": "### 指令\n解释机器学习\n\n### 回答\n机器学习是..."},
    ]
    return Dataset.from_list(data)

dataset = prepare_data()

def tokenize_function(examples):
    return tokenizer(
        examples["text"],
        truncation=True,
        max_length=512,
        padding="max_length"
    )

dataset = dataset.map(tokenize_function, batched=True)

# 3. 配置训练参数（全参数微调）
training_args = TrainingArguments(
    output_dir="./full_ft_output",
    num_train_epochs=3,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    learning_rate=1e-5,              # 全参数微调通常用较小的学习率
    warmup_ratio=0.1,
    fp16=True,                       # 混合精度
    logging_steps=10,
    save_strategy="epoch",
    save_total_limit=2,
    max_grad_norm=1.0,               # 梯度裁剪
    label_smoothing_factor=0.1,
    optim="adamw_torch",             # 使用 PyTorch 的 AdamW
    dataloader_num_workers=4,
)

# 4. 数据整理器
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False
)

# 5. 创建 Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    data_collator=data_collator,
)

# 6. 开始训练
trainer.train()

# 7. 保存模型
trainer.save_model("./full_ft_final")
tokenizer.save_pretrained("./full_ft_final")
```

### 3.2 使用 FSDP (Fully Sharded Data Parallel)

FSDP 是 PyTorch 原生的分布式训练方案，适合全参数微调：

```python
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
    Trainer
)
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    ShardingStrategy,
    MixedPrecision,
    AutoWrapPolicy,
)
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
import torch.nn as nn

# FSDP 配置
fsdp_config = {
    "backward_prefetch": "backward_pre",
    "forward_prefetch": False,
    "sharding_strategy": "FULL_SHARD",  # 或 "SHARD_GRAD_OP"
    "auto_wrap_policy": lambda m: len(list(m.modules())) > 100,
    "mixed_precision": {
        "param_dtype": torch.float16,
        "reduce_dtype": torch.float16,
        "buffer_dtype": torch.float16,
    }
}

# 训练参数
training_args = TrainingArguments(
    output_dir="./fsdp_output",
    num_train_epochs=3,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
    learning_rate=1e-5,
    fp16=True,
    fsdp="full_shard",               # 启用 FSDP
    fsdp_config=["fsdp_config.json"],
    logging_steps=10,
    save_strategy="steps",
    save_steps=100,
)

# 或者使用 transformers 原生 FSDP
training_args = TrainingArguments(
    output_dir="./fsdp_output",
    num_train_epochs=3,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
    learning_rate=1e-5,
    fp16=True,
    fsdp="full_shard_transformer",   # 自动 wrap transformer 层
    fsdp_config={
        "backward_prefetch": "backward_pre",
        "forward_prefetch": False,
    },
    logging_steps=10,
)
```

### 3.3 自定义训练循环

```python
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler

class FullParameterTrainer:
    """自定义全参数微调训练器"""
    
    def __init__(
        self,
        model,
        train_dataloader,
        learning_rate=1e-5,
        max_grad_norm=1.0,
        warmup_steps=100,
        total_steps=1000,
    ):
        self.model = model
        self.train_dataloader = train_dataloader
        self.max_grad_norm = max_grad_norm
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.global_step = 0
        
        # 优化器
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=0.01,
        )
        
        # 学习率调度器
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=total_steps,
            eta_min=learning_rate * 0.1,
        )
        
        # 混合精度训练
        self.scaler = GradScaler()
    
    def get_lr(self):
        """计算当前学习率（带 warmup）"""
        if self.global_step < self.warmup_steps:
            return self.global_step / self.warmup_steps
        else:
            progress = (self.global_step - self.warmup_steps) / (
                self.total_steps - self.warmup_steps
            )
            return 0.5 * (1 + torch.cos(torch.tensor(progress * 3.14159)))
    
    def training_step(self, batch):
        """单步训练"""
        self.model.train()
        
        # 混合精度前向传播
        with autocast(dtype=torch.float16):
            inputs = batch["input_ids"].to(self.model.device)
            labels = batch["labels"].to(self.model.device)
            outputs = self.model(inputs, labels=labels)
            loss = outputs.loss / self.model.config.gradient_accumulation_steps
        
        # 反向传播
        self.scaler.scale(loss).backward()
        
        # 梯度裁剪
        if (self.global_step + 1) % self.model.config.gradient_accumulation_steps == 0:
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.max_grad_norm
            )
            
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.optimizer.zero_grad()
            
            # 更新学习率
            self.scheduler.step()
        
        self.global_step += 1
        return loss.item() * self.model.config.gradient_accumulation_steps
    
    def train(self):
        """完整训练流程"""
        for epoch in range(10):
            for batch in self.train_dataloader:
                loss = self.training_step(batch)
                
                if self.global_step % 10 == 0:
                    print(f"Step {self.global_step}, Loss: {loss:.4f}")
                
                if self.global_step >= self.total_steps:
                    print("训练完成!")
                    return


# 使用示例
# trainer = FullParameterTrainer(
#     model=model,
#     train_dataloader=train_dataloader,
#     learning_rate=1e-5,
#     total_steps=1000,
# )
# trainer.train()
```

### 3.4 多机分布式训练

```bash
# 使用 torchrun 启动
torchrun --nproc_per_node=4 train.py

# 使用 torchrun 多节点
torchrun --nnodes=2 --nproc_per_node=8 train.py

# 使用 DeepSpeed 启动
deepspeed --num_gpus=8 train.py
```

## 4. 全参数微调最佳实践

### 4.1 学习率设置

| 模型大小 | 推荐学习率 | 备注 |
|----------|------------|------|
| 7B | 1e-5 ~ 2e-5 | |
| 13B | 5e-6 ~ 1e-5 | |
| 30B+ | 1e-6 ~ 5e-6 | 需要更小的学习率 |

### 4.2 训练配置模板

```python
# 全参数微调推荐配置
FULL_FT_CONFIG = {
    "learning_rate": 1e-5,           # 较小学习率
    "warmup_ratio": 0.1,             # 10% warmup
    "weight_decay": 0.01,            # 权重衰减
    "max_grad_norm": 1.0,            # 梯度裁剪
    "num_epochs": 2-3,               # 通常 2-3 个 epoch
    "batch_size": "根据显存调整",
    "gradient_accumulation": "保持 effective batch size = 32",
    "fp16": True,                    # 混合精度
    "logging_steps": 10,
    "save_steps": 500,
}
```

### 4.3 显存优化技巧

```python
# 1. 梯度检查点（降低显存）
model.gradient_checkpointing_enable()

# 2. 卸载优化器到 CPU
# 使用 DeepSpeed ZeRO Stage 2/3

# 3. 梯度累积
gradient_accumulation_steps = 8

# 4. 序列长度截断
max_seq_length = 1024
```

### 4.4 LoRA vs 全参数微调 对比

```
┌─────────────────────────────────────────────────────────────────┐
│                    微调方法对比                                  │
├──────────────────┬────────────────────┬────────────────────────┤
│     特性         │      LoRA          │     全参数微调          │
├──────────────────┼────────────────────┼────────────────────────┤
│ 显存需求         │   低 (~30GB/7B)    │    高 (~80GB/7B)       │
│ 训练时间         │   快               │    慢                  │
│ 参数量           │   0.1-1%           │    100%                │
│ 效果             │   好               │    最好                │
│ 灵活性           │   中               │    高                  │
│ 切换任务         │   快（加载权重）   │    需要重新训练         │
│ 适用场景         │   资源受限         │    追求最佳效果         │
└──────────────────┴────────────────────┴────────────────────────┘
```

## 5. 常见问题

**Q: 显存不足怎么办？**
- A: 1) 减小 batch size 2) 使用梯度累积 3) 开启梯度检查点 4) 使用 DeepSpeed/FSDP

**Q: 训练不稳定？**
- A: 1) 减小学习率 2) 增加 warmup 3) 使用梯度裁剪 4) 调整混合精度设置

**Q: 何时使用全参数微调？**
- A: 1) 任务与预训练差异大 2) 追求最佳效果 3) 资源充足 4) 数据量大

**Q: 如何选择微调方法？**

```python
def choose_finetune_method(gpu_memory, model_size, task_type):
    """选择微调方法"""
    
    # 模型参数量 (以 B 为单位)
    param_ratio = gpu_memory / model_size
    
    if param_ratio >= 10:
        # 显存充足，选择全参数微调
        return "full_parameter"
    elif param_ratio >= 4:
        # 显存适中，选择 LoRA
        return "lora"
    else:
        # 显存不足，选择 QLoRA
        return "qlora"
```

## 6. 总结

全参数微调是效果最好的微调方式，但需要较高的计算资源：

1. **显存需求**: 需要 ~80GB 显存 (7B 模型)
2. **学习率**: 通常使用 1e-5 级别
3. **训练技巧**: warmup、梯度裁剪、混合精度
4. **替代方案**: 资源不足时使用 LoRA 或 QLoRA

建议在实际项目中根据硬件条件和任务需求选择合适的微调方法。
