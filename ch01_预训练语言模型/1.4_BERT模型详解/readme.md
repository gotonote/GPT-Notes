# 1.4 BERT 模型详解

BERT（Bidirectional Encoder Representations from Transformers）是 Google 于 2018 年发布的预训练语言模型，在 NLP 领域引发了革命性的变化。本章将深入解析 BERT 的结构、预训练任务和 Fine-tuning 方法。

## 一、BERT 架构概述

BERT 的核心创新在于**双向 Transformer Encoder** 和**预训练-微调范式**。

```
┌─────────────────────────────────────────────────────────────────────────┐
│                              BERT 整体架构                               │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  ┌─────────────────────────────────────────────────────────────────┐    │
│  │                    Pre-training（预训练阶段）                     │    │
│  │                                                                  │    │
│  │  大规模无标注文本 ──► BERT ──► MLM + NSP 损失                     │    │
│  │                         (14B+ 词)                                │    │
│  └─────────────────────────────────────────────────────────────────┘    │
│                                    │                                    │
│                                    ▼                                    │
│  ┌─────────────────────────────────────────────────────────────────┐    │
│  │                    Fine-tuning（微调阶段）                       │    │
│  │                                                                  │    │
│  │  特定任务数据 ──► BERT + 任务头 ──► 任务预测                      │    │
│  │  (较小数据集)                                                    │    │
│  └─────────────────────────────────────────────────────────────────┘    │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

## 二、BERT 模型结构

### 2.1 模型配置

| 配置 | BERT-Base | BERT-Large |
|------|-----------|------------|
| Transformer 层数 | 12 | 24 |
| 注意力头数 | 12 | 16 |
| 隐藏层维度 | 768 | 1024 |
| 总参数量 | 110M | 340M |
| FFN 中间维度 | 3072 | 4096 |

### 2.2 BERT 架构图

```
┌─────────────────────────────────────────────────────────────────┐
│                        BERT Encoder                              │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  输入序列: [CLS] The cat sat on the [MASK] . [SEP]             │
│            │                                                    │
│            ▼                                                    │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  Token Embedding + Position Embedding + Segment Embedding│   │
│  └─────────────────────────────────────────────────────────┘   │
│            │                                                    │
│            ▼                                                    │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │              Transformer Encoder Layer × 12              │   │
│  │  ┌─────────────────────────────────────────────────────┐ │   │
│  │  │  Multi-Head Self-Attention                          │ │   │
│  │  │      ↓                                              │ │   │
│  │  │  Add & LayerNorm                                    │ │   │
│  │  │      ↓                                              │ │   │
│  │  │  Feed Forward Network                               │ │   │
│  │  │      ↓                                              │ │   │
│  │  │  Add & LayerNorm                                    │ │   │
│  │  └─────────────────────────────────────────────────────┘ │   │
│  └─────────────────────────────────────────────────────────┘   │
│            │                                                    │
│            ▼                                                    │
│  输出序列: [CLS] The cat sat on the [MASK] . [SEP]             │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### 2.3 BERT 输入表示

BERT 的输入由三部分组成：

```
┌─────────────────────────────────────────────────────────────────┐
│                     BERT 输入表示                                │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Token Embedding                                                │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  the → [0.1, 0.2, 0.3, ...]  (768维)                    │   │
│  │  cat → [0.4, 0.1, 0.7, ...]                             │   │
│  │  sat → [0.2, 0.5, 0.1, ...]                             │   │
│  └─────────────────────────────────────────────────────────┘   │
│                           +                                     │
│  Position Embedding                                            │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  pos=0 → [1.0, 0.0, 0.0, ...]                           │   │
│  │  pos=1 → [0.0, 1.0, 0.0, ...]                           │   │
│  │  pos=2 → [0.0, 0.0, 1.0, ...]                           │   │
│  └─────────────────────────────────────────────────────────┘   │
│                           +                                     │
│  Segment Embedding                                             │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  sentence A → [1.0, 0.0]                                 │   │
│  │  sentence B → [0.0, 1.0]                                 │   │
│  └─────────────────────────────────────────────────────────┘   │
│                           =                                     │
│  Final Input Embedding                                         │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  [0.5, 0.3, 0.2, ...]  (768维)                          │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### 2.4 BERT 特殊 Token

| Token | 作用 |
|-------|------|
| `[CLS]` | Classification Token，位于句首，用于分类任务 |
| `[SEP]` | Separator Token，分隔句子对 |
| `[PAD]` | Padding Token，用于批处理填充 |
| `[MASK]` | Mask Token，用于 MLM 预训练 |

## 三、BERT 预训练任务

### 3.1 Masked Language Modeling (MLM)

MLM 是 BERT 最重要的预训练任务，也被称为"完形填空"任务。

```
┌─────────────────────────────────────────────────────────────────┐
│                    MLM 任务示例                                  │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  原始句子: "The cat sat on the mat."                           │
│                                                                  │
│  随机 Mask 15% 的 Token:                                        │
│  "The cat sat on the [MASK] ."                                 │
│                                                                  │
│  模型预测:                                                       │
│  [MASK] → mat (概率: 0.85)                                     │
│  [MASK] → floor (概率: 0.10)                                   │
│  [MASK] → rug (概率: 0.05)                                     │
│                                                                  │
│  训练目标: 最大化预测正确的概率                                   │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

**MLM 的 80-10-10 策略**：

```
┌─────────────────────────────────────────────────────────────────┐
│                    Mask 策略详解                                 │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  随机选择 15% 的 Token，其中:                                   │
│                                                                  │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  80% 替换为 [MASK]                                       │   │
│  │  "The cat sat on the [MASK] ."                         │   │
│  └─────────────────────────────────────────────────────────┘   │
│                              │                                   │
│                              ▼                                   │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  10% 替换为随机 Token                                    │   │
│  │  "The cat sat on the dog ."                            │   │
│  └─────────────────────────────────────────────────────────┘   │
│                              │                                   │
│                              ▼                                   │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  10% 保持不变（但仍参与预测）                             │   │
│  │  "The cat sat on the mat ."                            │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                  │
│  目的: 防止模型过度依赖 [MASK] Token                             │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### 3.2 Next Sentence Prediction (NSP)

NSP 任务帮助 BERT 理解句子间的关系。

```
┌─────────────────────────────────────────────────────────────────┐
│                    NSP 任务示例                                  │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  输入: [CLS] The cat is sleeping. [SEP] It looks tired. [SEP]  │
│        ↑                                                        │
│        IsNext = True                                           │
│                                                                  │
│  输入: [CLS] The cat is sleeping. [SEP] The sky is blue. [SEP] │
│        ↑                                                        │
│        IsNext = False                                          │
│                                                                  │
│  训练目标: 二分类 (IsNext / NotNext)                           │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### 3.3 预训练代码实现

```python
import torch
import torch.nn as nn
from transformers import BertModel, BertConfig, BertForPreTraining

# 方法1: 使用 Hugging Face Transformers
model = BertForPreTraining.from_pretrained('bert-base-uncased')

# 方法2: 自定义 BERT 预训练模型
class BERTPreTrainer(nn.Module):
    def __init__(self, vocab_size, d_model=768, num_heads=12, 
                 num_layers=12, max_len=512):
        super().__init__()
        self.bert = BertModel(BertConfig(
            vocab_size=vocab_size,
            hidden_size=d_model,
            num_attention_heads=num_heads,
            num_hidden_layers=num_layers,
            max_position_embeddings=max_len
        ))
        
        # MLM 预测头
        self.mlm_head = nn.Linear(d_model, vocab_size)
        self.mlm_activation = nn.GELU()
        self.mlm_layer_norm = nn.LayerNorm(d_model)
        
        # NSP 预测头
        self.nsp_head = nn.Linear(d_model, 2)
    
    def forward(self, input_ids, attention_mask=None, 
                token_type_ids=None, labels=None):
        # BERT 编码
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )
        
        sequence_output = outputs.last_hidden_state  # (batch, seq, d_model)
        pooled_output = outputs.pooler_output       # (batch, d_model)
        
        # MLM 预测
        mlm_logits = self.mlm_layer_norm(
            self.mlm_activation(self.mlm_head(sequence_output))
        )
        
        # NSP 预测
        nsp_logits = self.nsp_head(pooled_output)
        
        loss = None
        if labels is not None:
            # MLM 损失
            mlm_labels, nsp_labels = labels
            loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
            mlm_loss = loss_fct(
                mlm_logits.view(-1, mlm_logits.size(-1)),
                mlm_labels.view(-1)
            )
            
            # NSP 损失
            nsp_loss = loss_fct(nsp_logits, nsp_labels)
            
            # 总损失
            loss = mlm_loss + nsp_loss
        
        return {'loss': loss, 'mlm_logits': mlm_logits, 'nsp_logits': nsp_logits}
```

## 四、BERT Fine-tuning

### 4.1 Fine-tuning 概述

Fine-tuning 是将预训练的 BERT 应用于下游任务的关键步骤。

```
┌─────────────────────────────────────────────────────────────────┐
│                    BERT Fine-tuning 流程                         │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  步骤 1: 加载预训练 BERT                                   │   │
│  │                                                          │   │
│  │  model = BertModel.from_pretrained('bert-base-uncased') │   │
│  └─────────────────────────────────────────────────────────┘   │
│                              │                                   │
│                              ▼                                   │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  步骤 2: 添加任务输出层                                    │   │
│  │                                                          │   │
│  │  ┌────────────┐    ┌────────────┐    ┌────────────┐   │   │
│  │  │ 文本分类   │    │ 序列标注   │    │ 问答任务   │   │   │
│  │  │ [CLS] → FC │    │ Token → FC │    │ S/E → FC   │   │   │
│  │  └────────────┘    └────────────┘    └────────────┘   │   │
│  └─────────────────────────────────────────────────────────┘   │
│                              │                                   │
│                              ▼                                   │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  步骤 3: 在任务数据上训练                                  │   │
│  │                                                          │   │
│  │  for epoch in range(epochs):                            │   │
│  │      for batch in dataloader:                           │   │
│  │          loss = model(**batch)                          │   │
│  │          loss.backward()                                │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### 4.2 文本分类（Text Classification）

```
┌─────────────────────────────────────────────────────────────────┐
│                    BERT 文本分类                                  │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  输入: "This movie is amazing!"                                 │
│                                                                  │
│  [CLS] This movie is amazing! [SEP]                           │
│    │                                                          │
│    ▼                                                          │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │              BERT Encoder (12 layers)                   │   │
│  └─────────────────────────────────────────────────────────┘   │
│    │                                                          │
│    ▼                                                          │
│  [CLS] Representation (768维)                                  │
│    │                                                          │
│    ▼                                                          │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  Linear(768 → num_labels) + Softmax                     │   │
│  └─────────────────────────────────────────────────────────┘   │
│    │                                                          │
│    ▼                                                          │
│  输出: 正面(0.92), 负面(0.05), 中性(0.03)                       │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

**代码实现**：

```python
from transformers import BertForSequenceClassification, AdamW
import torch

# 加载预训练模型并添加分类层
model = BertForSequenceClassification.from_pretrained(
    'bert-base-uncased',
    num_labels=2
)

# 准备数据
from transformers import BertTokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# 编码文本
encoded = tokenizer(
    "This movie is amazing!",
    padding=True,
    truncation=True,
    max_length=128,
    return_tensors='pt'
)

# 前向传播
outputs = model(**encoded)
logits = outputs.logits  # (batch_size, num_labels)
predictions = torch.argmax(logits, dim=-1)

# 训练循环
optimizer = AdamW(model.parameters(), lr=2e-5)

# 假设有训练数据
input_ids = encoded['input_ids']
attention_mask = encoded['attention_mask']
labels = torch.tensor([1])  # 正面标签

# 计算损失
outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
loss = outputs.loss

# 反向传播
loss.backward()
optimizer.step()
```

### 4.3 命名实体识别（NER）

```
┌─────────────────────────────────────────────────────────────────┐
│                    BERT NER 任务                                  │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  输入: "John works at Google in California"                    │
│                                                                  │
│  [CLS] John works at Google in California [SEP]               │
│         B-PER  O    O   B-ORG   O  B-LOC                       │
│                                                                  │
│  标签体系:                                                       │
│  B-PER: 人名开始                                                │
│  I-PER: 人名中间                                                │
│  B-ORG: 组织开始                                                │
│  I-ORG: 组织中间                                                │
│  B-LOC: 地点开始                                                │
│  I-LOC: 地点中间                                                │
│  O: 非实体                                                      │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

**代码实现**：

```python
from transformers import BertForTokenClassification
import torch.nn as nn

# 加载 NER 模型
model = BertForTokenClassification.from_pretrained(
    'bert-base-uncased',
    num_labels=9,  # BIOES 标签数
    id2label={0: 'O', 1: 'B-PER', 2: 'I-PER', 3: 'B-ORG', 4: 'I-ORG',
              5: 'B-LOC', 6: 'I-LOC', 7: 'B-MISC', 8: 'I-MISC'},
    label2id={'O': 0, 'B-PER': 1, 'I-PER': 2, 'B-ORG': 3, 'I-ORG': 4,
              'B-LOC': 5, 'I-LOC': 6, 'B-MISC': 7, 'I-MISC': 8}
)

# Token 分类
outputs = model(**encoded)
logits = outputs.logits  # (batch_size, seq_len, num_labels)

# 解码预测
predictions = torch.argmax(logits, dim=-1)
```

### 4.4 问答任务（SQuAD）

```
┌─────────────────────────────────────────────────────────────────┐
│                    BERT 问答任务                                  │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  问题: "What is the capital of France?"                        │
│  文本: "France's capital is Paris, a beautiful city..."      │
│                                                                  │
│  输入: [CLS] What is the capital of France? [SEP]             │
│        France's capital is Paris, a beautiful city... [SEP]  │
│                                                                  │
│  输出:                                                          │
│  - Start Position: 指向 "Paris"                                │
│  - End Position: 指向 "Paris"                                  │
│                                                                  │
│  预测: "Paris"                                                 │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

**代码实现**：

```python
from transformers import BertForQuestionAnswering

# 加载问答模型
model = BertForQuestionAnswering.from_pretrained('bert-base-uncased')

# SQuAD 格式输入
question = "What is the capital of France?"
context = "France's capital is Paris, a beautiful city in Europe."

encoded = tokenizer(
    question,
    context,
    return_tensors='pt',
    max_length=384,
    truncation=True,
    return_token_type_ids=True
)

# 前向传播
outputs = model(**encoded)
start_logits = outputs.start_logits  # 每个位置作为起点的分数
end_logits = outputs.end_logits     # 每个位置作为终点的分数

# 预测答案范围
start_idx = torch.argmax(start_logits)
end_idx = torch.argmax(end_logits) + 1

# 解码答案
answer = tokenizer.decode(encoded['input_ids'][0][start_idx:end_idx])
```

### 4.5 Fine-tuning 超参数建议

```
┌─────────────────────────────────────────────────────────────────┐
│                BERT Fine-tuning 推荐超参数                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  参数                    推荐值             说明                  │
│  ───────────────────────────────────────────────────────────   │
│  Learning Rate         2e-5 ~ 5e-5      较预训练更小             │
│  Batch Size           16 ~ 32          根据显存调整             │
│  Epochs               2 ~ 5            数据集小时可适当增加     │
│  Warmup Steps         总步数的 10%      学习率预热               │
│  Weight Decay         0.01             防止过拟合               │
│  Max Sequence Length  128 ~ 512        根据任务调整             │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

## 五、BERT 的变体和发展

### 5.1 BERT 系列模型

```
┌─────────────────────────────────────────────────────────────────┐
│                    BERT 家族                                      │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐        │
│  │   BERT      │    │   RoBERTa   │    │   ALBERT    │        │
│  │  (2018)     │    │   (2019)    │    │   (2019)    │        │
│  │             │    │             │    │             │        │
│  │ 110M params │    │  125M params│    │   12M params│        │
│  │ + NSP任务   │    │ - NSP任务   │    │  跨层共享   │        │
│  │             │    │  动态Masking│    │  词向量分解 │        │
│  └─────────────┘    └─────────────┘    └─────────────┘        │
│         │                  │                  │                 │
│         ▼                  ▼                  ▼                 │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐        │
│  │  DistilBERT │    │    XLNet    │    │    ELECTRA  │        │
│  │   (2019)    │    │   (2019)    │    │   (2020)    │        │
│  │             │    │             │    │             │        │
│  │  60M params │    │  175M params│    │  110M params│        │
│  │  知识蒸馏   │    │  排列语言模型│    │  替换Token  │        │
│  └─────────────┘    └─────────────┘    └─────────────┘        │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### 5.2 主要变体对比

| 模型 | 参数量 | 预训练任务 | 特点 |
|------|--------|-----------|------|
| BERT-Base | 110M | MLM + NSP | 标准版本 |
| BERT-Large | 340M | MLM + NSP | 更大更强 |
| RoBERTa | 125M | MLM Only | 动态Masking，更大数据 |
| ALBERT | 12M | MLM + SOP | 参数共享，轻量级 |
| DistilBERT | 66M | MLM | 知识蒸馏，快速推理 |
| ELECTRA | 110M | RTD | 替换Token检测，效率高 |

## 六、实战：完整的 BERT Fine-tuning 示例

```python
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    BertTokenizer, 
    BertForSequenceClassification,
    AdamW,
    get_linear_schedule_with_warmup
)
import pandas as pd

# 1. 加载数据
class SentimentDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        
        encoding = self.tokenizer(
            text,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

# 2. 训练函数
def train_epoch(model, dataloader, optimizer, scheduler, device):
    model.train()
    total_loss = 0
    
    for batch in dataloader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        
        optimizer.zero_grad()
        
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
        
        loss = outputs.loss
        total_loss += loss.item()
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
    
    return total_loss / len(dataloader)

# 3. 主训练流程
def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 加载 tokenizer 和模型
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertForSequenceClassification.from_pretrained(
        'bert-base-uncased',
        num_labels=2
    ).to(device)
    
    # 准备数据（示例）
    texts = ["I love this!", "This is bad."]
    labels = [1, 0]
    
    dataset = SentimentDataset(texts, labels, tokenizer)
    dataloader = DataLoader(dataset, batch_size=2)
    
    # 优化器和学习率调度
    optimizer = AdamW(model.parameters(), lr=2e-5)
    total_steps = len(dataloader) * 3  # 3 epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(total_steps * 0.1),
        num_training_steps=total_steps
    )
    
    # 训练
    for epoch in range(3):
        loss = train_epoch(model, dataloader, optimizer, scheduler, device)
        print(f"Epoch {epoch + 1}, Loss: {loss:.4f}")

if __name__ == "__main__":
    main()
```

## 总结

本章详细介绍了 BERT 模型的核心内容：

1. **模型结构**：基于双向 Transformer Encoder，使用三类嵌入表示
2. **预训练任务**：MLM（完形填空）和 NSP（下一句预测）
3. **Fine-tuning**：针对不同任务的微调方法
4. **发展演进**：BERT 系列的变体模型

BERT 的成功在于其预训练-微调范式，极大降低了 NLP 任务的数据需求，成为了现代 NLP 的基础组件。

## 参考资料

- Devlin, J., et al. (2018). "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding"
- "Hugging Face Transformers 文档"
- "BERT Explained" by Jian-Huang
