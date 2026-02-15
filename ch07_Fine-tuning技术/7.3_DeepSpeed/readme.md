# DeepSpeed 分布式训练

## 1. 什么是 DeepSpeed？

**DeepSpeed** 是 Microsoft 开发的深度学习优化库，专门用于大规模分布式训练。它通过创新性的 **ZeRO (Zero Redundancy Optimizer)** 技术，显著降低了显存占用，同时保持了高效的计算性能。

### 1.1 DeepSpeed 的核心特性

| 特性 | 说明 |
|------|------|
| **ZeRO** | 零冗余优化器，分片存储模型状态 |
| **3D 并行** | 数据 + 流水线 + 张量并行 |
| **混合精度** | FP16/BF16 自动训练 |
| **异步 I/O** | 高效的检查点保存/加载 |
| **Sparse Attention** | 高效的稀疏注意力机制 |

### 1.2 ZeRO 技术架构

```
┌─────────────────────────────────────────────────────────────────┐
│                     ZeRO 优化阶段图解                            │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Stage 0: 无优化 (DDP)                                          │
│  ┌────────┐ ┌────────┐ ┌────────┐ ┌────────┐                 │
│  │  Opt0  │ │  Opt1  │ │  Opt2  │ │  Opt3  │  全部复制          │
│  │  P0    │ │  P1    │ │  P2    │ │  P3    │                  │
│  └────────┘ └────────┘ └────────┘ └────────┘                  │
│                                                                 │
│  Stage 1: 优化器分片                                            │
│  ┌────────┐ ┌────────┐ ┌────────┐ ┌────────┐                 │
│  │  Opt0  │ │  Opt1  │ │  Opt2  │ │  Opt3  │  每卡一份         │
│  │  P_all │ │  P_all │ │  P_all │ │  P_all │                  │
│  └────────┘ └────────┘ └────────┘ └────────┘                  │
│                                                                 │
│  Stage 2: 优化器 + 梯度分片                                     │
│  ┌────────┐ ┌────────┐ ┌────────┐ ┌────────┐                 │
│  │  Opt0  │ │  Opt1  │ │  Opt2  │ │  Opt3  │                  │
│  │  G0    │ │  G1    │ │  G2    │ │  G3    │  梯度分片         │
│  │  P0    │ │  P1    │ │  P2    │ │  P3    │                  │
│  └────────┘ └────────┘ └────────┘ └────────┘                  │
│                                                                 │
│  Stage 3: 全部分片                                              │
│  ┌────────┐ ┌────────┐ ┌────────┐ ┌────────┐                 │
│  │  Opt0  │ │  Opt1  │ │  Opt2  │ │  Opt3  │  全部参数分片     │
│  │  G0    │ │  G1    │ │  G2    │ │  G3    │                  │
│  │  P0    │ │  P1    │ │  P2    │ │  P3    │                  │
│  └────────^┘ └────────^┘ └────────^┘ └────────^              │
│           │            │            │            │             │
│           └────────────┴────────────┴────────────┘             │
│                         Allgather (需要时)                       │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

## 2. DeepSpeed 原理

### 2.1 ZeRO Stage 详解

**Stage 1: 优化器状态分片**
- 只在每个 GPU 上保留 1/N 的优化器状态
- 显存节省约 4 倍

**Stage 2: 优化器状态 + 梯度分片**
- 优化器状态和梯度都分片
- 显存节省约 8 倍

**Stage 3: 全部参数分片**
- 优化器状态、梯度、模型参数全部
- 显存节省约 N 倍 (N = GPU 数量)

### 2.2 3D 并行

```
┌─────────────────────────────────────────────────────────────────┐
│                       3D 并行架构                               │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│                     数据并行 (Data Parallel)                    │
│                     ┌─────┬─────┬─────┬─────┐                  │
│          复制       │GPU 0│GPU 1│GPU 2│GPU 3│                  │
│                     └──┬──┘──┬──┘──┬──┘──┬──┘                  │
│                        │     │     │     │                     │
│        ┌──────────────┼─────┼─────┼─────┼──────────────┐     │
│        │              │     │     │     │              │     │
│        ▼              ▼     ▼     ▼     ▼              ▼     │
│  ┌───────────┐  ┌───────────┐  ┌───────────┐  ┌───────────┐  │
│  │ 流水线并行│  │ 流水线并行│  │ 流水线并行│  │ 流水线并行│  │
│  │ Layer 0-1│  │ Layer 2-3 │  │ Layer 4-5 │  │ Layer 6-7 │  │
│  └─────┬─────┘  └─────┬─────┘  └─────┬─────┘  └─────┬─────┘  │
│        │              │              │              │        │
│        └──────────────┴──────────────┴──────────────┘        │
│                        张量并行 (Tensor Parallel)             │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

## 3. 代码实现

### 3.1 安装 DeepSpeed

```bash
pip install deepspeed
```

验证安装：
```bash
deepspeed --version
```

### 3.2 基础配置

```json
// deepspeed_config.json
{
    "train_batch_size": 32,
    "train_micro_batch_size_per_gpu": 4,
    "gradient_accumulation_steps": 2,
    
    "fp16": {
        "enabled": true
    },
    
    "zero_optimization": {
        "stage": 2,
        "offload_optimizer": {
            "device": "cpu",
            "pin_memory": true
        },
        "allgather_partitions": true,
        "allgather_bucket_size": 2e8,
        "overlap_comm": true,
        "reduce_scatter": true,
        "reduce_bucket_size": 2e8,
        "contiguous_gradients": true
    },
    
    "gradient_clipping": 1.0,
    "steps_per_print": 10,
    "wall_clock_breakdown": false
}
```

### 3.3 训练脚本

```python
import torch
import deepspeed
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    DataCollatorForLanguageModeling
)
from datasets import Dataset

# ==================== DeepSpeed 配置 ====================
DEEPSPEED_CONFIG = {
    "train_batch_size": 32,
    "train_micro_batch_size_per_gpu": 4,
    "gradient_accumulation_steps": 2,
    
    # 混合精度
    "fp16": {
        "enabled": True
    },
    
    # ZeRO 优化
    "zero_optimization": {
        "stage": 2,                    # Stage 2: 优化器 + 梯度分片
        "offload_optimizer": {         # 将优化器状态卸载到 CPU
            "device": "cpu",
            "pin_memory": True
        },
        "allgather_partitions": True,
        "allgather_bucket_size": 2e8,
        "overlap_comm": True,
        "reduce_scatter": True,
        "reduce_bucket_size": 2e8,
        "contiguous_gradients": True,
    },
    
    # 梯度裁剪
    "gradient_clipping": 1.0,
    
    # 日志
    "steps_per_print": 10,
    "wall_clock_breakdown": False,
    
    # 通信
    "communication_data_type": "fp16"
}


def get_deepspeed_config():
    """返回 DeepSpeed 配置"""
    return DEEPSPEED_CONFIG


def prepare_data():
    """准备训练数据"""
    data = [
        {"text": "人工智能是计算机科学的一个分支。"},
        {"text": "机器学习是人工智能的核心技术。"},
        {"text": "深度学习使用神经网络模型。"},
    ]
    return Dataset.from_list(data)


def main():
    # 1. 初始化 DeepSpeed
    deepspeed.init_distributed()
    
    # 2. 加载模型
    model = AutoModelForCausalLM.from_pretrained(
        "gpt2",
        torch_dtype=torch.float16,
    )
    
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    
    # 3. 准备数据
    dataset = prepare_data()
    
    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=128,
            padding="max_length",
        )
    
    dataset = dataset.map(tokenize_function, batched=True)
    
    # 4. 配置训练参数
    training_args = TrainingArguments(
        output_dir="./deepspeed_output",
        num_train_epochs=3,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=2,
        learning_rate=3e-4,
        logging_steps=10,
        save_strategy="epoch",
        save_total_limit=2,
        save_steps=50,
        # DeepSpeed 相关
        deepspeed=get_deepspeed_config(),
    )
    
    # 5. 创建 Trainer
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )
    
    from transformers import Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=data_collator,
    )
    
    # 6. 训练
    trainer.train()
    
    # 7. 保存
    trainer.save_model("./deepspeed_final")
    tokenizer.save_pretrained("./deepspeed_final")


if __name__ == "__main__":
    main()
```

### 3.4 启动分布式训练

```bash
# 单机多卡
deepspeed --num_gpus=4 train.py

# 多机多卡
deepspeed --num_nodes=2 --num_gpus=8 train.py

# 指定配置文件
deepspeed --config deepspeed_config.json train.py
```

### 3.5 ZeRO Stage 3 配置（显存最优）

```json
{
    "train_batch_size": 64,
    "train_micro_batch_size_per_gpu": 1,
    "gradient_accumulation_steps": 64,
    
    "fp16": {
        "enabled": true
    },
    
    "zero_optimization": {
        "stage": 3,
        "offload_optimizer": {
            "device": "cpu",
            "pin_memory": true
        },
        "offload_param": {
            "device": "cpu",
            "pin_memory": true
        },
        "overlap_comm": true,
        "contiguous_gradients": true,
        "reduce_bucket_size": 1e6,
        "stage3_prefetch_bucket_size": 1e6,
        "stage3_param_persistence_threshold": 1e5,
        "stage3_max_live_parameters": 1e9,
        "stage3_max_reuse_distance": 1e9,
        "stage3_gather_16bit_weights_on_model_save": true
    }
}
```

### 3.6 流水线并行配置

```json
{
    "train_batch_size": 32,
    "train_micro_batch_size_per_gpu": 2,
    
    "pipeline": {
        "engine": "deepspeed",
        "num_layers": 24,
        "activation_checkpoint_interval": 1,
        "pipe_partitioned": true,
        "grad_partition": true,
        "partition_method": "uniform"
    },
    
    "zero_optimization": {
        "stage": 1
    }
}
```

### 3.7 自定义模型使用 DeepSpeed

```python
import torch.nn as nn
from deepspeed.runtime.zero.partition_parameters import zero3_init_flag

# 方式1: 使用装饰器
from deepspeed.runtime.zero.Stage3 import zero3_func

@zero3_init_flag
def create_model():
    return MyModel()


# 方式2: 在模型中集成
class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = nn.Linear(100, 100)
    
    def forward(self, x):
        return self.layer(x)


# 方式3: 使用 DeepSpeed 封装
model = MyModel()
model, optimizer, _, _ = deepspeed.initialize(
    model=model,
    optimizer=optimizer,
    config=DEEPSPEED_CONFIG,
)

# 训练循环
for batch in dataloader:
    loss = model(batch)
    model.backward(loss)
    model.step()
```

## 4. DeepSpeed 最佳实践

### 4.1 Stage 选择

| Stage | 显存节省 | 通信开销 | 适用场景 |
|-------|----------|----------|----------|
| Stage 1 | ~4x | 低 | 单卡显存足够 |
| Stage 2 | ~8x | 中 | 多卡训练，显存受限 |
| Stage 3 | ~Nx | 高 | 超大规模训练 |

### 4.2 配置模板

```json
// 单机 4 卡配置
{
    "train_batch_size": 16,
    "train_micro_batch_size_per_gpu": 2,
    "gradient_accumulation_steps": 2,
    
    "fp16": {
        "enabled": true,
        "loss_scale": 0,
        "loss_scale_window": 1000,
        "initial_scale_power": 16,
        "hysteresis": 2,
        "min_loss_scale": 1
    },
    
    "zero_optimization": {
        "stage": 2,
        "offload_optimizer": {
            "device": "cpu",
            "pin_memory": true
        }
    }
}
```

```json
// 多机配置
{
    "train_batch_size": "auto",
    "train_micro_batch_size_per_gpu": "auto",
    "gradient_accumulation_steps": "auto",
    
    "zero_optimization": {
        "stage": 3,
        "offload_optimizer": {
            "device": "cpu",
            "pin_memory": true
        },
        "offload_param": {
            "device": "cpu",
            "pin_memory": true
        }
    },
    
    "gradient_clipping": 1.0,
    "prescale_gradients": false,
    "bf16": {
        "enabled": true
    },
    
    "wall_clock_breakdown": false,
    "zero_allow_untested_optimizer": true
}
```

### 4.3 调试技巧

```python
# 启用调试模式
import os
os.environ["DEEPSPEED_DEBUG"] = "1"

# 查看内存使用
import deepspeed
# 在训练循环中添加
deepspeed.runtime.utils.logger.info(
    f"Memory allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB"
)
```

## 5. 与 LoRA 结合

### 5.1 DeepSpeed + LoRA 配置

```python
from peft import LoraConfig, get_peft_model

# LoRA 配置
lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

# 应用 LoRA
model = get_peft_model(model, lora_config)

# 转换为 DeepSpeed 格式
model = model.to_distributed()
```

### 5.2 ZeRO-3 + LoRA 完整示例

```python
# deepspeed_config_ds3_lora.json
{
    "train_batch_size": 32,
    "train_micro_batch_size_per_gpu": 4,
    "gradient_accumulation_steps": 2,
    
    "bf16": {
        "enabled": true
    },
    
    "zero_optimization": {
        "stage": 3,
        "offload_optimizer": {
            "device": "cpu",
            "pin_memory": true
        },
        "offload_param": {
            "device": "cpu",
            "pin_memory": true
        },
        "overlap_comm": true,
        "contiguous_gradients": true,
        "reduce_bucket_size": 1e6,
        "stage3_prefetch_bucket_size": 1e6,
        "stage3_param_persistence_threshold": 1e5,
        "stage3_max_live_parameters": 1e9,
        "stage3_max_reuse_distance": 1e9
    }
}
```

## 6. 常见问题

**Q: 显存不足？**
- 1) 使用 Stage 2 或 Stage 3
- 2) 开启 offload
- 3) 减小 batch size
- 4) 使用梯度累积

**Q: 通信瓶颈？**
- 1) 调整 `overlap_comm`
- 2) 增大 bucket size
- 3) 使用高速网络

**Q: 训练不稳定？**
- 1) 检查混合精度配置
- 2) 调整学习率
- 3) 使用 BF16 替代 FP16

## 7. 总结

DeepSpeed 是大规模模型训练的核心工具：

1. **ZeRO**: 大幅降低显存占用
2. **3D 并行**: 支持超大规模训练
3. **混合精度**: 提高训练效率
4. **易用性**: 与 Transformers 无缝集成

建议在多卡训练场景优先使用 DeepSpeed，配合 LoRA 可以实现高效的参数微调。
