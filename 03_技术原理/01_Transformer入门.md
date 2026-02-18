# 01 Transformer入门

> 理解革命性的架构设计，开启大语言模型之旅

## 📖 什么是 Transformer？

**Transformer** 是 2017 年由 Google 在论文《Attention Is All You Need》中提出的深度学习架构，它完全基于**注意力机制（Attention Mechanism）**，摒弃了传统的 RNN 和 CNN 结构。

### 🌟 核心特点

| 特点 | 说明 |
|------|------|
| **并行计算** | 支持 GPU 并行处理，大幅提升训练效率 |
| **长距离依赖** | 注意力机制直接建模任意位置的关系 |
| **可扩展性** | 模型规模可以 scaling up |
| **通用性强** | 适用于 NLP、CV、语音等多种任务 |

---

## 🏗️ 整体架构

Transformer 采用 **编码器-解码器（Encoder-Decoder）** 结构：

```
输入序列 → [编码器] → 上下文表示 → [解码器] → 输出序列
```

### 编码器（Encoder）

- 由 N 个相同的 **Transformer Block** 组成
- 每个 Block 包含：
  - 多头自注意力（Multi-Head Self-Attention）
  - 前馈神经网络（Feed Forward Network）
  - 残差连接 & 层归一化

### 解码器（Decoder）

- 同样由 N 个 Transformer Block 组成
- 每个 Block 包含：
  - 多头自注意力
  - 多头编码器-解码器注意力
  - 前馈神经网络

---

## 🔄 Transformer Block 详解

```python
import torch
import torch.nn as nn
import math

class TransformerBlock(nn.Module):
    """Transformer 编码器/解码器块"""
    
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()
        
        # 多头注意力
        self.attention = nn.MultiheadAttention(d_model, n_heads, dropout=dropout)
        # 层归一化
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        # 前馈网络
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, mask=None):
        # 注意力 + 残差
        attn_output, _ = self.attention(x, x, x, attn_mask=mask)
        x = self.norm1(x + self.dropout(attn_output))
        
        # 前馈 + 残差
        ffn_output = self.ffn(x)
        x = self.norm2(x + ffn_output)
        
        return x
```

---

## 📊 架构对比

| 模型 | 编码器 | 解码器 | 典型应用 |
|------|--------|--------|----------|
| **Transformer** | ✓ | ✓ | 机器翻译 |
| **BERT** | ✓ | ✗ | 文本分类/序列标注 |
| **GPT** | ✗ | ✓ | 文本生成 |
| **T5** | ✓ | ✓ | 文本到文本 |

---

## 💡 简单示例：使用 Transformers 库

```python
from transformers import AutoTokenizer, AutoModel

# 加载预训练模型
model_name = "bert-base-chinese"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

# 编码输入
text = "今天天气真好"
inputs = tokenizer(text, return_tensors="pt")

# 前向传播
outputs = model(**inputs)

print(f"输入: {text}")
print(f"隐藏层维度: {outputs.last_hidden_state.shape}")
# 输出: torch.Size([1, 5, 768])
```

---

## 📈 发展历程

```
2017 → Transformer 论文发表
   ↓
2018 → BERT (Google) - 预训练+微调范式
   ↓
2018 → GPT-1 (OpenAI) - 生成式预训练
   ↓
2019 → GPT-2 - 更大模型、零样本学习
   ↓
2020 → GPT-3 - few-shot 学习能力
   ↓
2022 → ChatGPT - 人类对齐
   ↓
2023 → GPT-4 - 多模态能力
```

---

## 🎯 小结

1. **Transformer** 是现代大语言模型的基础架构
2. 核心组件是**注意力机制**，可以并行处理序列
3. 衍生出 BERT（编码器）、GPT（解码器）等重要模型
4. 推动了 NLP 领域的范式转变

---

## 📚 延伸阅读

- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) - 原始论文
- [The Illustrated Transformer](http://jalammar.github.io/illustrated-transformer/) - 图解 Transformer
- [Harvard NLP Transformer](http://nlp.seas.harvard.edu/2018/04/03/attention.html) - 代码实现

---

*🔜 下一章：[02_注意力机制](./02_注意力机制.md)*
