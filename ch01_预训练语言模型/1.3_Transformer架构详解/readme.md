# 1.3 Transformer 架构详解

Transformer 是当前大语言模型的核心架构，最初由 Vaswani 等人在论文《Attention Is All You Need》中提出。本章将深入解析 Transformer 的原理、位置编码和 Multi-Head Attention 机制。

## 一、Transformer 整体架构

Transformer 采用的是 Encoder-Decoder 架构，完全基于注意力机制（Attention）实现，摒弃了传统的 RNN 和 CNN。

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           Transformer 架构                              │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│   Input Embedding                                                            │
│        │                                                                  │
│        ▼                                                                  │
│   ┌─────────────┐                                                           │
│   │  Positional │  ◄──── 位置编码                                          │
│   │  Encoding   │                                                          │
│   └─────────────┘                                                           │
│        │                                                                  │
│        ▼                                                                  │
│   ┌─────────────────────────────────────────────────────────────────┐    │
│   │                     Encoder Stack (N层)                          │    │
│   │  ┌─────────────────────────────────────────────────────────┐   │    │
│   │  │  Multi-Head Attention                                    │   │    │
│   │  │      │                                                    │   │    │
│   │  │  Add & Norm                                              │   │    │
│   │  │      │                                                    │   │    │
│   │  │  Feed Forward                                            │   │    │
│   │  │      │                                                    │   │    │
│   │  │  Add & Norm                                              │   │    │
│   │  └─────────────────────────────────────────────────────────┘   │    │
│   └─────────────────────────────────────────────────────────────────┘    │
│        │                                                                  │
│        ▼                                                                  │
│   ┌─────────────────────────────────────────────────────────────────┐    │
│   │                     Decoder Stack (N层)                          │    │
│   │  ┌─────────────────────────────────────────────────────────┐   │    │
│   │  │  Masked Multi-Head Attention                              │   │    │
│   │  │      │                                                    │   │    │
│   │  │  Add & Norm                                              │   │    │
│   │  │      │                                                    │   │    │
│   │  │  Multi-Head Attention (Encoder-Decoder)                   │   │    │
│   │  │      │                                                    │   │    │
│   │  │  Add & Norm                                              │   │    │
│   │  │      │                                                    │   │    │
│   │  │  Feed Forward                                            │   │    │
│   │  │      │                                                    │   │    │
│   │  │  Add & Norm                                              │   │    │
│   │  └─────────────────────────────────────────────────────────┘   │    │
│   └─────────────────────────────────────────────────────────────────┘    │
│        │                                                                  │
│        ▼                                                                  │
│   Linear + Softmax                                                          │
│        │                                                                  │
│        ▼                                                                  │
│   Output Probabilities                                                      │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### 1.1 Encoder 结构

每个 Encoder 层包含两个子层：
- **Multi-Head Self-Attention**：多头自注意力机制
- **Feed Forward Network**：前馈神经网络

每个子层都采用残差连接（Residual Connection）和层归一化（Layer Normalization）。

```python
class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.feed_forward = FeedForward(d_model, d_ff)
        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, mask=None):
        # Multi-Head Self-Attention with residual
        attn_output = self.self_attn(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_output))
        
        # Feed Forward with residual
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))
        
        return x
```

### 1.2 Decoder 结构

Decoder 与 Encoder 类似，但包含三个子层：
- **Masked Multi-Head Self-Attention**：掩盖未来的位置
- **Multi-Head Cross-Attention**：接收 Encoder 的输出
- **Feed Forward Network**：前馈神经网络

## 二、位置编码（Positional Encoding）

由于 Transformer 不使用 RNN 的顺序处理方式，需要显式地向输入添加位置信息。

### 2.1 正弦余弦位置编码

Transformer 采用正弦和余弦函数来编码位置：

```python
import numpy as np

def get_positional_encoding(d_model, max_len=5000):
    """生成位置编码矩阵
    
    Args:
        d_model: 模型的维度
        max_len: 最大序列长度
    
    Returns:
        pos_enc: 位置编码矩阵 (max_len, d_model)
    """
    position = np.arange(max_len).reshape(-1, 1)
    div_term = np.exp(np.arange(0, d_model, 2) * (-np.log(10000.0) / d_model))
    
    pos_enc = np.zeros((max_len, d_model))
    pos_enc[:, 0::2] = np.sin(position * div_term)  # 偶数维度
    pos_enc[:, 1::2] = np.cos(position * div_term)  # 奇数维度
    
    return pos_enc
```

### 2.2 位置编码的特性

```
┌─────────────────────────────────────────────────────────────────┐
│                    位置编码可视化 (d_model=4)                   │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Position 0: [0.000, 1.000, 0.000, 1.000]                       │
│  Position 1: [0.841, 0.540, 0.809, 0.587]                       │
│  Position 2: [0.909, 0.416, 0.141, 0.990]                      │
│  Position 3: [0.141, 0.990, 0.757, 0.654]                       │
│                                                                  │
│  特性:                                                           │
│  1. 每个位置有唯一的编码                                          │
│  2. 不同位置间的距离可以通过向量差值计算                           │
│  3. 正弦余弦编码支持相对位置学习                                  │
│  4. 编码是有界的，不会随序列长度爆炸                              │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### 2.3 可学习的位置编码

除了固定的位置编码，研究者还提出了可学习的位置编码：

```python
class LearnedPositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        self.position_embeddings = nn.Embedding(max_len, d_model)
    
    def forward(self, x):
        # x: (batch_size, seq_len, d_model)
        batch_size, seq_len, _ = x.shape
        position_ids = torch.arange(seq_len, dtype=torch.long, device=x.device)
        position_ids = position_ids.unsqueeze(0).expand(batch_size, -1)
        position_embeddings = self.position_embeddings(position_ids)
        return x + position_embeddings
```

## 三、Multi-Head Attention（多头注意力）

多头注意力是 Transformer 的核心组件，允许模型同时关注不同位置的不同表示子空间。

### 3.1 Scaled Dot-Product Attention

```
┌─────────────────────────────────────────────────────────────────┐
│                Scaled Dot-Product Attention                     │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│      Q (Query)                                                   │
│        │                                                         │
│        ▼                                                         │
│   ┌─────────┐      K (Key)      ┌─────────┐                     │
│   │ MatMul  │◄──────────────────│         │                     │
│   └────┬────┘                   └─────────┘                     │
│        │                                                      │
│        ▼                                                      │
│   ┌─────────┐                                                  │
│   │  Scale │  (÷ √d_k)                                        │
│   └────┬────┘                                                  │
│        │                                                      │
│        ▼                                                      │
│   ┌─────────┐      V (Value)      ┌─────────┐                 │
│   │ Softmax │◄───────────────────│         │                 │
│   └────┬────┘                   └─────────┘                 │
│        │                                                      │
│        ▼                                                      │
│   ┌─────────┐                                                  │
│   │ MatMul  │                                                  │
│   └────┬────┘                                                  │
│        │                                                      │
│        ▼                                                      │
│      Output                                                    │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### 3.2 数学公式

$$
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V
$$

其中：
- $Q \in \mathbb{R}^{n_q \times d_k}$：查询矩阵
- $K \in \mathbb{R}^{n_k \times d_k}$：键矩阵
- $V \in \mathbb{R}^{n_v \times d_v}$：值矩阵
- $d_k$：键向量的维度（用于缩放）

### 3.3 多头注意力实现

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.1):
        super().__init__()
        assert d_model % num_heads == 0
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads  # 每个头的维度
        
        # 线性投影层
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)
        
        # 1. 线性投影并分头
        Q = self.W_q(query).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_k(key).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_v(value).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        
        # 2. 计算注意力分数
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        # 3. 应用掩码（可选）
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        # 4. Softmax 归一化
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # 5. 加权求和
        output = torch.matmul(attn_weights, V)
        
        # 6. 合并多头输出
        output = output.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        output = self.W_o(output)
        
        return output
```

### 3.4 多头注意力的可视化

```
┌─────────────────────────────────────────────────────────────────┐
│                    多头注意力机制                                  │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  输入 X: (batch_size, seq_len, d_model)                         │
│                        │                                         │
│                        ▼                                         │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │  线性投影: X → Q, K, V                                   │    │
│  └─────────────────────────────────────────────────────────┘    │
│                        │                                         │
│                        ▼                                         │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │  分头: (batch, seq_len, d_model) → (batch, h, seq, d_k) │    │
│  └─────────────────────────────────────────────────────────┘    │
│                        │                                         │
│                        ▼                                         │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │  并行计算 h 个注意力头                                   │    │
│  │                                                          │    │
│  │  Head 1: 关注语法结构                                    │    │
│  │  Head 2: 关注语义关系                                    │    │
│  │  Head 3: 关注位置信息                                    │    │
│  │  ...                                                     │    │
│  │  Head h: 关注特定模式                                    │    │
│  └─────────────────────────────────────────────────────────┘    │
│                        │                                         │
│                        ▼                                         │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │  拼接输出: (batch, h, seq, d_k) → (batch, seq, d_model) │    │
│  └─────────────────────────────────────────────────────────┘    │
│                        │                                         │
│                        ▼                                         │
│  输出: (batch_size, seq_len, d_model)                           │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

## 四、完整 Transformer 编码器实现

```python
import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # 创建位置编码矩阵
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        # x: (batch_size, seq_len, d_model)
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        return self.linear2(self.dropout(F.gelu(self.linear1(x))))


class TransformerEncoder(nn.Module):
    def __init__(self, vocab_size, d_model=512, num_heads=8, 
                 num_layers=6, d_ff=2048, dropout=0.1, max_len=5000):
        super().__init__()
        
        self.d_model = d_model
        
        # 词嵌入和位置编码
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model, max_len, dropout)
        
        # 编码器层
        self.encoder_layers = nn.ModuleList([
            EncoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        
        self.norm = nn.LayerNorm(d_model)
    
    def forward(self, x, mask=None):
        # 词嵌入并缩放
        x = self.embedding(x) * math.sqrt(self.d_model)
        x = self.pos_encoding(x)
        
        # 逐层编码
        for layer in self.encoder_layers:
            x = layer(x, mask)
        
        return self.norm(x)


# 使用示例
if __name__ == "__main__":
    # 参数设置
    vocab_size = 30000
    d_model = 512
    num_heads = 8
    num_layers = 6
    max_len = 100
    
    # 创建模型
    model = TransformerEncoder(vocab_size, d_model, num_heads, num_layers)
    
    # 模拟输入: batch_size=2, seq_len=20
    x = torch.randint(0, vocab_size, (2, 20))
    
    # 前向传播
    output = model(x)
    print(f"输入形状: {x.shape}")
    print(f"输出形状: {output.shape}")
```

## 五、Transformer 的优势与局限

### 5.1 优势

| 特性 | 说明 |
|------|------|
| **并行计算** | 不依赖序列顺序，可并行处理所有位置 |
| **长距离依赖** | 注意力机制可直接建模任意距离的依赖 |
| **可解释性** | 注意力权重可可视化 |
| **可扩展性** | 可通过增加层数和宽度扩展模型容量 |

### 5.2 局限

| 特性 | 说明 |
|------|------|
| **计算复杂度** | O(n²) 的注意力计算复杂度 |
| **内存占用** | 注意力矩阵占用大量显存 |
| **序列长度限制** | 长序列处理困难 |

## 六、实战：构建简单的 Transformer 模型

```python
import torch
import torch.nn as nn

# 完整的 Transformer Encoder 示例
transformer = nn.TransformerEncoder(
    nn.TransformerEncoderLayer(
        d_model=256,
        nhead=8,
        dim_feedforward=1024,
        dropout=0.1,
        batch_first=True
    ),
    num_layers=6
)

# 测试
batch_size = 4
seq_len = 50
input_tensor = torch.randn(batch_size, seq_len, 256)

output = transformer(input_tensor)
print(f"输入: {input_tensor.shape}")
print(f"输出: {output.shape}")
```

## 总结

本章详细介绍了 Transformer 的核心组件：

1. **整体架构**：Encoder-Decoder 结构，完全基于注意力机制
2. **位置编码**：正弦余弦编码或可学习编码，为序列注入位置信息
3. **多头注意力**：并行计算多个注意力头，捕获不同类型的关系

这些组件构成了现代大语言模型的基础，后续的 BERT、GPT 等模型都是在 Transformer 基础上发展而来的。

## 参考资料

- Vaswani, A., et al. (2017). "Attention Is All You Need"
- "The Illustrated Transformer" by Jay Alammar
- "Transformer Anatomy" by Michael Phi
