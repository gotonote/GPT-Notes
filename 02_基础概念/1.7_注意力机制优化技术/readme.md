# 1.7 注意力机制优化技术

随着大语言模型规模的不断增长，标准注意力机制的计算复杂度和内存占用成为主要瓶颈。本章将介绍当前主流的注意力机制优化技术，帮助你理解现代大模型如何实现高效推理。

## 一、标准注意力的计算问题

### 1.1 时间与空间复杂度

标准自注意力的计算复杂度为 O(n²)，其中 n 是序列长度。这意味着：

- **计算复杂度**：O(n² × d) - d 是模型维度
- **内存复杂度**：O(n²) - 需要存储注意力矩阵

对于长序列（如 32K 或更长的上下文），这成为严重的瓶颈。

```python
# 标准自注意力的实现
class StandardAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
    
    def forward(self, x, mask=None):
        batch_size = x.size(0)
        
        # 1. 线性投影
        Q = self.W_q(x).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_k(x).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_v(x).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        
        # 2. 计算注意力分数 O(n² × d)
        # 这是主要的计算瓶颈
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        # 3. Softmax 归一化
        attn_weights = F.softmax(scores, dim=-1)
        
        # 4. 加权求和 O(n² × d)
        context = torch.matmul(attn_weights, V)
        
        # 5. 输出投影
        output = context.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        return self.W_o(output)
```

### 1.2 内存占用分析

对于一个 70B 参数的模型处理 32K 序列：
- 注意力矩阵：32K × 32K = 1B 个浮点数
- 单精度内存：4GB（仅注意力矩阵）
- 多头注意力：如果是 80 头，则需要 320GB

这显然是不可接受的，因此需要优化技术。

---

## 二、Flash Attention

### 2.1 核心思想

Flash Attention 由 Stanford 大学提出，通过**分块计算**和**核融合**技术，将注意力计算从 O(n²) 降低到近似 O(n)，同时大幅减少显存占用。

### 2.2 算法原理

核心创新在于**在线 softmax 技巧**：

1. **分块处理**：将长序列分成多个小块
2. **在线计算**：逐步计算 softmax 的归一化因子
3. **核融合**：将多个操作融合到一个 CUDA kernel 中

```python
# Flash Attention 的简化伪代码
def flash_attention(Q, K, V, block_size=128):
    """
    Flash Attention 核心算法
    
    传统方法：先计算完整的注意力矩阵，再 softmax
    Flash Attention：分块在线计算 softmax
    """
    n = Q.shape[1]  # 序列长度
    d = Q.shape[3]  # 头维度
    
    # 初始化输出和统计量
    output = torch.zeros_like(Q)
    
    # 逐块处理
    for i in range(0, n, block_size):
        # 1. 提取当前块
        Q_block = Q[:, i:i+block_size]
        K_block = K[:, i:i+block_size]
        V_block = V[:, i:i+block_size]
        
        # 2. 计算当前块的注意力
        # 使用在线 softmax 技巧
        exp_qk = torch.exp(Q_block @ K_block.transpose(-2, -1) / math.sqrt(d))
        
        # 3. 累加到全局统计量
        if i == 0:
            max_i = torch.max(exp_qk, dim=-1, keepdim=True)
            exp_qk_normalized = exp_qk / max_i
        else:
            # 在线更新最大值和归一化因子
            new_max = torch.maximum(old_max, torch.max(exp_qk, dim=-1, keepdim=True))
            exp_qk_normalized = exp_qk * torch.exp(old_max - new_max)
            old_max = new_max
        
        # 4. 累加输出
        output_block = exp_qk_normalized @ V_block
    
    return output
```

### 2.3 实际使用示例

```python
# 使用 xFormers 库的 Flash Attention
from xformers.ops import memory_efficient_attention

# 标准调用
output = memory_efficient_attention(
    query,  # [B, H, N, D]
    key,
    value,
    attn_bias=None  # 可以传入因果 mask
)

# 或者使用 FlashAttention 库
from flash_attn import flash_attn_func

output = flash_attn_func(
    query, 
    key, 
    value,
    causal=True  # 因果注意力
)
```

### 2.4 性能对比

| 序列长度 | 标准 Attention 显存 | Flash Attention 显存 | 加速比 |
|---------|---------------------|---------------------|--------|
| 2K      | 16GB                | 2GB                 | 2-3x   |
| 8K      | 256GB (OOM)         | 4GB                 | 10-20x |
| 32K     | OOM                 | 16GB                | 20-50x |

---

## 三、Multi-Query Attention (MQA)

### 3.1 核心思想

Multi-Query Attention 由 Google 提出，其核心创新是**多个 Query 头共享同一组 Key 和 Value**。

### 3.2 与标准注意力的对比

```python
class MultiQueryAttention(nn.Module):
    """
    Multi-Query Attention: 多个 Query 头共享 Key 和 Value
    
    参数量对比（以 LLaMA 7B 为例）：
    - 标准 MHA: num_heads * 3 * d_model * d_k + num_heads * d_model * d_k = 12 * 4096 * 128 * 3 + 12 * 4096 * 128
    - MQA: 3 * d_model * d_k + num_heads * d_model * d_k = 3 * 4096 * 128 + 32 * 4096 * 128
    
    显存节省：约 30-50%
    """
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        # Query: 每个头独立
        self.W_q = nn.Linear(d_model, d_model)
        
        # Key 和 Value: 所有头共享
        self.W_k = nn.Linear(d_model, self.d_k)  # 单个头维度
        self.W_v = nn.Linear(d_model, self.d_k)  # 单个头维度
        
        self.W_o = nn.Linear(d_model, d_model)
    
    def forward(self, x, mask=None):
        batch_size, seq_len, d_model = x.size(0), x.size(1), x.size(2)
        
        # Query: 保持多头结构
        Q = self.W_q(x).view(batch_size, seq_len, self.num_heads, self.d_k)
        Q = Q.transpose(1, 2)  # [B, H, N, D]
        
        # Key 和 Value: 广播到所有头
        K = self.W_k(x).unsqueeze(1)  # [B, 1, N, D]
        V = self.W_v(x).unsqueeze(1)  # [B, 1, N, D]
        
        # 计算注意力（K 和 V 自动广播）
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        attn_weights = F.softmax(scores, dim=-1)
        context = torch.matmul(attn_weights, V)  # [B, H, N, D]
        
        output = context.transpose(1, 2).contiguous().view(batch_size, seq_len, d_model)
        return self.W_o(output)
```

### 3.3 性能特点

- **推理速度提升**：2-3 倍（尤其在 KV 缓存场景）
- **显存减少**：KV 缓存大幅减小
- **质量损失**：通常 < 2% 性能下降
- **代表模型**：PaLM、Falcon、LLaMA 2 (部分配置)

---

## 四、Grouped-Query Attention (GQA)

### 4.1 核心思想

Grouped-Query Attention 是 MHA 和 MQA 的折中方案：将 Query 头分成 G 组，每组共享 Key 和 Value。

### 4.2 架构对比

```
标准 MHA:     Q0 Q1 Q2 Q3 Q4 Q5 Q6 Q7
              |  |  |  |  |  |  |  |
              K0 K1 K2 K3 K4 K5 K6 K7  (每对独立)
              V0 V1 V2 V3 V4 V5 V6 V7

GQA (G=2):    Q0 Q1 Q2 Q3 Q4 Q5 Q6 Q7
              |  |  |  |  |  |  |  |
              |  |  |  |  |  |  |  |
              K0 K1 K0 K1 K2 K3 K2 K3  (分组共享)
              V0 V1 V0 V1 V2 V3 V2 V3

MQA:          Q0 Q1 Q2 Q3 Q4 Q5 Q6 Q7
              |  |  |  |  |  |  |  |
              K  K  K  K  K  K  K  K   (全部共享)
              V  V  V  V  V  V  V  V
```

### 4.3 实现代码

```python
class GroupedQueryAttention(nn.Module):
    """
    Grouped-Query Attention (GQA)
    
    num_query_heads: Query 头的数量
    num_kv_heads: Key/Value 头的数量 (num_query_heads 的约数)
    """
    def __init__(self, d_model, num_query_heads, num_kv_heads):
        super().__init__()
        self.num_query_heads = num_query_heads
        self.num_kv_heads = num_kv_heads
        self.d_k = d_model // num_query_heads
        self.group_size = num_query_heads // num_kv_heads
        
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, num_kv_heads * self.d_k)
        self.W_v = nn.Linear(d_model, num_kv_heads * self.d_k)
        self.W_o = nn.Linear(d_model, d_model)
    
    def forward(self, x, mask=None):
        batch_size, seq_len = x.size(0), x.size(1)
        
        # Query: [B, N, H, D]
        Q = self.W_q(x).view(batch_size, seq_len, self.num_query_heads, self.d_k)
        Q = Q.transpose(1, 2)
        
        # Key/Value: [B, N, G, D] -> 广播 -> [B, H, N, D]
        K = self.W_k(x).view(batch_size, seq_len, self.num_kv_heads, self.d_k)
        K = K.transpose(1, 2)  # [B, G, N, D]
        K = K.repeat_interleave(self.group_size, dim=1)  # 广播到所有 Query 头
        
        V = self.W_v(x).view(batch_size, seq_len, self.num_kv_heads, self.d_k)
        V = V.transpose(1, 2)
        V = V.repeat_interleave(self.group_size, dim=1)
        
        # 后续计算与标准注意力相同
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        attn_weights = F.softmax(scores, dim=-1)
        context = torch.matmul(attn_weights, V)
        
        output = context.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        return self.W_o(output)
```

### 4.4 性能对比

| 架构 | 推理速度 | 显存占用 | 模型质量 |
|-----|---------|---------|---------|
| MHA | 1x      | 1x      | 100%   |
| GQA | 1.5-2x  | 0.5-0.7x | 98-99% |
| MQA | 2-3x    | 0.3-0.5x | 96-98% |

**代表模型**：LLaMA 2 (70B)、Mistral 7B、Qwen 系列

---

## 五、Sliding Window Attention

### 5.1 核心思想

Sliding Window Attention（滑动窗口注意力）基于一个关键洞察：远处的 token 对当前 token 的影响可以忽略不计。因此，只计算固定窗口内的注意力。

### 5.2 算法图示

```
序列位置:    0   1   2   3   4   5   6   7   8   9  10
窗口大小:    3
   
位置 3 的注意力范围:
                [1   2   3]
                    ↑
                只能关注 window_size 范围内的 token

位置 8 的注意力范围:
                            [6   7   8]
                                ↑
```

### 5.3 实现代码

```python
class SlidingWindowAttention(nn.Module):
    """
    Sliding Window Attention
    
    window_size: 滑动窗口大小（如 512、1024）
    """
    def __init__(self, d_model, num_heads, window_size=512):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.window_size = window_size
        
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
    
    def forward(self, x, mask=None):
        batch_size, seq_len = x.size(0), x.size(1)
        d_k = self.d_model // self.num_heads
        
        Q = self.W_q(x).view(batch_size, seq_len, self.num_heads, d_k).transpose(1, 2)
        K = self.W_k(x).view(batch_size, seq_len, self.num_heads, d_k).transpose(1, 2)
        V = self.W_v(x).view(batch_size, seq_len, self.num_heads, d_k).transpose(1, 2)
        
        # 创建滑动窗口 mask
        # 位置 i 只能关注 [i-window_size, i] 范围内的 token
        causal_mask = torch.tril(torch.ones(seq_len, seq_len, device=x.device))
        
        # 计算相对位置偏移
        # 远处位置的注意力分数设为 -inf
        positions = torch.arange(seq_len, device=x.device)
        relative_positions = positions.unsqueeze(0) - positions.unsqueeze(1)
        window_mask = (relative_positions.abs() <= self.window_size).float()
        
        # 组合 mask
        final_mask = causal_mask * window_mask
        final_mask = final_mask.masked_fill(final_mask == 0, float('-inf'))
        
        # 标准注意力计算
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)
        scores = scores + final_mask.unsqueeze(0).unsqueeze(0)
        
        attn_weights = F.softmax(scores, dim=-1)
        context = torch.matmul(attn_weights, V)
        
        output = context.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        return self.W_o(output)
```

### 5.4 扩展阅读：Longformer 和 BigBird

- **Longformer**：使用局部窗口 + 全局注意力的组合
- **BigBird**：使用窗口 + 随机 + 全局三种注意力的组合
- 两者都可以处理非常长的序列（16K+）

---

## 六、Sparse Attention（稀疏注意力）

### 6.1 核心思想

Sparse Attention 只计算部分 token 之间的注意力关系，通过预设的稀疏模式大幅减少计算量。

### 6.2 常见稀疏模式

```
全局注意力:     某些特殊 token（如 [CLS]）可以关注所有 token
    
局部块注意:    每个 token 只关注固定大小的块内
    
随机注意力:    随机选择一些 token 进行注意力计算
    
扩张注意力:    使用扩张卷积式的跳跃窗口
```

### 6.3 实现示例

```python
class SparseAttention(nn.Module):
    """
    稀疏注意力：结合局部和全局注意力
    """
    def __init__(self, d_model, num_heads, block_size=64, num_global=4):
        super().__init__()
        self.block_size = block_size
        self.num_global = num_global
        
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
    
    def forward(self, x):
        batch_size, seq_len = x.size(0), x.size(1)
        d_k = x.size(-1) // self.num_heads
        
        Q = self.W_q(x).view(batch_size, seq_len, self.num_heads, d_k).transpose(1, 2)
        K = self.W_k(x).view(batch_size, seq_len, self.num_heads, d_k).transpose(1, 2)
        V = self.W_v(x).view(batch_size, seq_len, self.num_heads, d_k).transpose(1, 2)
        
        output = torch.zeros_like(Q)
        
        # 1. 全局 token 的注意力（可以关注所有 token）
        global_q = Q[:, :, :self.num_global]
        global_scores = torch.matmul(global_q, K.transpose(-2, -1)) / math.sqrt(d_k)
        global_attn = F.softmax(global_scores, dim=-1)
        output[:, :, :self.num_global] = torch.matmul(global_attn, V)
        
        # 2. 局部块注意力
        for i in range(0, seq_len, self.block_size):
            end = min(i + self.block_size, seq_len)
            local_q = Q[:, :, i:end]
            local_k = K[:, :, max(0, i-self.block_size):end]
            local_v = V[:, :, max(0, i-self.block_size):end]
            
            local_scores = torch.matmul(local_q, local_k.transpose(-2, -1)) / math.sqrt(d_k)
            local_attn = F.softmax(local_scores, dim=-1)
            output[:, :, i:end] = torch.matmul(local_attn, local_v)
        
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        return self.W_o(output)
```

---

## 七、实战：注意力机制的选择

### 7.1 如何选择合适的注意力机制

| 场景 | 推荐方案 | 原因 |
|-----|---------|------|
| 短文本 (< 2K) | 标准 MHA | 简单高效 |
| 长文本 + 高质量 | Flash Attention + GQA | 平衡速度和质量 |
| 超长上下文 (32K+) | Flash Attention + Sliding Window | 显存可控 |
| 资源受限环境 | MQA | 最小的 KV 缓存 |
| 生产环境推理 | GQA | 最佳平衡 |

### 7.2 配置示例

```python
# LLaMA 模型中的注意力配置
config = {
    # 模型维度
    "hidden_size": 4096,
    "num_attention_heads": 32,
    "num_key_value_heads": 8,  # GQA: 32 Query 头，8 KV 头
    
    # 滑动窗口
    "sliding_window": 4096,  # LLaMA 2 使用 4096 窗口
    
    # Flash Attention
    "use_flash_attention": True,
}
```

### 7.3 性能测试

```python
import torch
import time
from xformers.ops import memory_efficient_attention

def benchmark_attention(seq_len, hidden_size, num_heads, num_iters=100):
    """测试不同注意力机制的性能"""
    d_k = hidden_size // num_heads
    batch_size = 1
    
    Q = torch.randn(batch_size, num_heads, seq_len, d_k).cuda()
    K = torch.randn(batch_size, num_heads, seq_len, d_k).cuda()
    V = torch.randn(batch_size, num_heads, seq_len, d_k).cuda()
    
    # Warm up
    for _ in range(10):
        _ = memory_efficient_attention(Q, K, V)
    
    # Benchmark
    start = time.time()
    for _ in range(num_iters):
        _ = memory_efficient_attention(Q, K, V)
    torch.cuda.synchronize()
    
    elapsed = time.time() - start
    print(f"Seq: {seq_len:5d} | Time: {elapsed/num_iters*1000:.2f}ms")

# 测试不同序列长度
for seq_len in [512, 2048, 8192, 32768]:
    benchmark_attention(seq_len, 4096, 32)
```

---

## 八、总结

| 技术 | 计算复杂度 | 显存占用 | 质量影响 | 适用场景 |
|-----|-----------|---------|---------|---------|
| 标准 Attention | O(n²) | O(n²) | - | 短序列 |
| Flash Attention | O(n) | O(n) | 无 | 通用 |
| MQA | O(n²) | O(n) | 轻微 | 推理为主 |
| GQA | O(n²) | O(n) | 轻微 | 平衡 |
| Sliding Window | O(n×w) | O(n×w) | 可控 | 长序列 |

**核心原则**：根据实际场景选择合适的注意力机制，在速度、显存和质量之间取得平衡。

---

## 参考资料

1. Flash Attention 论文: [FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness](https://arxiv.org/abs/2205.14135)
2. MQA 论文: [Fast Transformer Decoding: One Write-Head is All You Need](https://arxiv.org/abs/1911.02150)
3. GQA 论文: [GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints](https://arxiv.org/abs/2305.13245)
4. LLaMA 2 论文: [LLaMA 2: Open Foundation and Chat Models](https://arxiv.org/abs/2307.01394)
