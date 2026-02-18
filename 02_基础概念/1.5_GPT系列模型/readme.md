# 1.5 GPT 系列模型

GPT（Generative Pre-trained Transformer）系列是 OpenAI 开发的基于 Transformer 的生成式预训练模型。从 GPT-1 到 GPT-4，再到 Claude，展示了大型语言模型的演进历程。本章将详细解析各代模型的架构、训练方法和关键创新。

## 一、GPT 系列演进概览

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        GPT 系列模型演进                                   │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  2018        2019          2020          2022          2023      2024  │
│   │           │             │             │             │            │
│   ▼           ▼             ▼             ▼             ▼            │
│ ┌──────┐  ┌──────┐     ┌──────┐      ┌──────┐      ┌──────┐         │
│ │GPT-1 │  │GPT-2 │     │GPT-3 │      │GPT-3.5│      │GPT-4 │         │
│ │ 117M │  │1.5B  │     │ 175B │      │ChatGPT│ │   │ ???? │         │
│ └──────┘  └──────┘     └──────┘      └──────┘      └──────┘         │
│                                                                          │
│  + Claude (Anthropic)                                                   │
│       │       │           │             │             │              │
│       ▼       ▼           ▼             ▼             ▼              │
│    ┌────┐ ┌────┐     ┌────┐        ┌────┐       ┌────┐              │
│    │1.0 │ │2.0 │     │3.0 │        │3.5 │       │3.5 │              │
│    └────┘ └────┘     └────┘        └────┘       └────┘              │
│                                                                          │
│  趋势: 参数规模爆炸 ↑ | 能力涌现 ↑ | 应用范围扩展 ↑                      │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

## 二、GPT-1：开创性的预训练范式

### 2.1 核心创新

GPT-1 首次提出了"预训练 + 微调"的两阶段范式：

```
┌─────────────────────────────────────────────────────────────────┐
│                    GPT-1 训练范式                                 │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  阶段 1: 无监督预训练                                             │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  大规模文本语料库                                          │   │
│  │       │                                                    │   │
│  │       ▼                                                    │   │
│  │  GPT-1 (单向 Transformer Decoder)                        │   │
│  │       │                                                    │   │
│  │       ▼                                                    │   │
│  │  语言建模目标: 预测下一个词                                 │   │
│  └─────────────────────────────────────────────────────────┘   │
│                              │                                   │
│                              ▼                                   │
│  阶段 2: 监督微调                                                 │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  特定任务数据集                                            │   │
│  │       │                                                    │   │
│  │       ▼                                                    │   │
│  │  预训练 GPT-1 + 任务输出层                                 │   │
│  │       │                                                    │   │
│  │       ▼                                                    │   │
│  │  任务预测                                                  │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### 2.2 GPT-1 架构

```python
import torch
import torch.nn as nn

class GPT1Config:
    vocab_size = 50257
    n_positions = 1024
    n_ctx = 1024
    n_embd = 768
    n_layer = 12
    n_head = 12

class GPT1Model(nn.Module):
    """GPT-1 模型实现"""
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # 词嵌入和位置嵌入
        self.wte = nn.Embedding(config.vocab_size, config.n_embd)
        self.wpe = nn.Embedding(config.n_positions, config.n_embd)
        
        # Transformer Decoder 层
        self.blocks = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=config.n_embd,
                nhead=config.n_head,
                dim_feedforward=config.n_embd * 4,
                dropout=0.1,
                batch_first=True,
                norm_first=True
            ) for _ in range(config.n_layer)
        ])
        
        self.ln_f = nn.LayerNorm(config.n_embd)
        
        # 语言建模头
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        
        # 权重共享
        self.lm_head.weight = self.wte.weight
    
    def forward(self, input_ids, attention_mask=None):
        # 获取输入形状
        batch_size, seq_len = input_ids.shape
        
        # 创建位置编码
        position_ids = torch.arange(seq_len, dtype=torch.long, device=input_ids.device)
        position_ids = position_ids.unsqueeze(0).expand(batch_size, -1)
        
        # 计算嵌入
        hidden_states = self.wte(input_ids) + self.wpe(position_ids)
        
        # 通过 Transformer 层
        for block in self.blocks:
            hidden_states = block(hidden_states, src_key_padding_mask=attention_mask)
        
        hidden_states = self.ln_f(hidden_states)
        
        # 计算 logits
        lm_logits = self.lm_head(hidden_states)
        
        return lm_logits
```

### 2.3 GPT-1 关键特点

| 特性 | 说明 |
|------|------|
| 单向语言模型 | 只使用左侧上下文（从左到右） |
| 12 层 Transformer | 768 维隐藏层，12 个注意力头 |
| 1.17 亿参数 | 首次展示预训练-微调范式 |
| BookCorpus 数据 | 7,000 本书的文本 |

## 三、GPT-2：zero-shot 的突破

### 3.1 核心创新：Zero-shot Learning

GPT-2 提出了无需微调即可执行下游任务的能力：

```
┌─────────────────────────────────────────────────────────────────┐
│                    GPT-2 Zero-shot 能力                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  传统方法:                                                       │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  预训练模型 ──► Fine-tuning ──► 任务模型                   │   │
│  │           需要大量标注数据                                 │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                  │
│  GPT-2 方法:                                                     │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  预训练模型 ──► 直接利用                                   │   │
│  │                                                                │   │
│  │  示例提示:                                                  │   │
│  │  "Translate to French: The cat is sleeping."              │   │
│  │            ↓                                               │   │
│  │  模型直接输出翻译                                         │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### 3.2 GPT-2 模型系列

| 模型 | 层数 | 参数量 | 特点 |
|------|------|--------|------|
| GPT-2 Small | 12 | 117M | 基础版本 |
| GPT-2 Medium | 24 | 345M | 中等规模 |
| GPT-2 Large | 36 | 774M | 大规模 |
| GPT-2 XL | 48 | 1.5B | 最大版本 |

### 3.3 GPT-2 架构改进

```python
class GPT2Config:
    """GPT-2 配置"""
    vocab_size = 50257
    n_positions = 1024
    n_ctx = 1024
    n_embd = 768
    n_layer = 12
    n_head = 12
    n_inner = 3072  # FFN 中间层维度
    resid_dropout = 0.1
    embd_dropout = 0.1
    attn_dropout = 0.1

class GPT2Attention(nn.Module):
    """GPT-2 改进的注意力机制"""
    
    def __init__(self, config):
        super().__init__()
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.split_size = config.n_head * config.n_embd
        
        # 分离的 QKV 投影
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_head * config.n_embd)
        self.c_proj = nn.Linear(config.n_head * config.n_embd, config.n_embd)
        
        self.attn_dropout = nn.Dropout(config.attn_dropout)
        self.resid_dropout = nn.Dropout(config.resid_dropout)
    
    def forward(self, x, attention_mask=None):
        batch_size, seq_len, _ = x.shape
        
        # QKV 投影
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.split_size, dim=-1)
        
        # Reshape for multi-head
        q = q.view(batch_size, seq_len, self.n_head, -1).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.n_head, -1).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.n_head, -1).transpose(1, 2)
        
        # 计算注意力
        attn = torch.matmul(q, k.transpose(-2, -1)) / (self.n_head ** 0.5)
        
        if attention_mask is not None:
            attn = attn + attention_mask
        
        attn = torch.softmax(attn, dim=-1)
        attn = self.attn_dropout(attn)
        
        # 应用注意力到 V
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        
        # 输出投影
        out = self.c_proj(out)
        out = self.resid_dropout(out)
        
        return out
```

## 四、GPT-3：few-shot 的革命

### 4.1 核心创新：Few-shot Learning

GPT-3 展示了通过提示（Prompt）即可实现少样本学习的能力：

```
┌─────────────────────────────────────────────────────────────────┐
│                    GPT-3 In-context Learning                     │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Zero-shot (无示例):                                             │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  翻译成法语: The cat is sleeping                        │   │
│  │  输出: Le chat dort                                     │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                  │
│  One-shot (一个示例):                                            │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  英语: I love you → 法语: Je t'aime                     │   │
│  │  英语: The cat is sleeping →                            │   │
│  │  输出: Le chat dort                                     │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                  │
│  Few-shot (几个示例):                                            │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  英语: I love you → 法语: Je t'aime                     │   │
│  │  英语: He runs fast → 法语: Il court vite                │   │
│  │  英语: The cat is sleeping →                            │   │
│  │  输出: Le chat dort                                     │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### 4.2 GPT-3 规模

| 模型 | 层数 | 维度 | 头数 | 参数量 |
|------|------|------|------|--------|
| GPT-3 Small | 32 | 2,048 | 16 | 125M |
| GPT-3 Base | 32 | 2,048 | 32 | 350M |
| GPT-3 1.3B | 24 | 2,048 | 32 | 1.3B |
| GPT-3 6.7B | 32 | 4,096 | 32 | 6.7B |
| GPT-3 175B | 96 | 12,288 | 96 | 175B |

### 4.3 GPT-3 训练数据

```
┌─────────────────────────────────────────────────────────────────┐
│                    GPT-3 训练数据集                               │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  来源                              权重       数量(Tokens)        │
│  ─────────────────────────────────────────────────────────────   │
│  Common Crawl (过滤后)             60%        410B              │
│  WebText2                          22%        19B               │
│  Books1                            12%        67B               │
│  Books2                            8%         328B               │
│  Wikipedia                         3%         3B                │
│                                                                  │
│  总计: ~ 825B Tokens                                        │
│  训练计算: ~ 3.64 × 10^23 FLOPs                              │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### 4.4 GPT-3 能力涌现

随着模型规模增大，GPT-3 展现出"涌现能力"（Emergent Abilities）：

```
┌─────────────────────────────────────────────────────────────────┐
│                    GPT-3 涌现能力                                 │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  小模型不具备但大模型具备的能力:                                 │
│                                                                  │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  1. 算术运算 (3位数加减法)                               │   │
│  │  2. 单词拼写纠正                                         │   │
│  │  3. 上下文学习 (In-context Learning)                     │   │
│  │  4. 思维链推理 (Chain-of-Thought)                        │   │
│  │  5. 指令跟随 (Instruction Following)                     │   │
│  │  6. 代码生成                                             │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                  │
│  涌现的临界点通常在 50-100B 参数左右                            │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

## 五、GPT-3.5 和 ChatGPT

### 5.1 从 GPT-3 到 GPT-3.5

GPT-3.5 是基于 GPT-3 的改进版本，通过以下技术提升：

| 技术 | 说明 |
|------|------|
| RLHF | 人类反馈强化学习 |
| InstructGPT | 指令微调 |
| Codex | 代码训练数据 |
| WebGPT | 网页浏览能力 |

### 5.2 RLHF 流程

```
┌─────────────────────────────────────────────────────────────────┐
│                RLHF (人类反馈强化学习)                            │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  步骤 1: 监督微调 (SFT)                                        │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  人类编写的高质量对话                                    │   │
│  │       │                                                    │   │
│  │       ▼                                                    │   │
│  │  微调 GPT-3                                              │   │
│  └─────────────────────────────────────────────────────────┘   │
│                              │                                   │
│                              ▼                                   │
│  步骤 2: 奖励模型 (RM)                                         │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  对同一问题生成多个回复                                  │   │
│  │       │                                                    │   │
│  │       ▼                                                    │   │
│  │  人类排序评分                                            │   │
│  │       │                                                    │   │
│  │       ▼                                                    │   │
│  │  训练奖励模型                                            │   │
│  └─────────────────────────────────────────────────────────┘   │
│                              │                                   │
│                              ▼                                   │
│  步骤 3: PPO 强化学习                                           │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  使用奖励模型作为奖励信号                                 │   │
│  │       │                                                    │   │
│  │       ▼                                                    │   │
│  │  PPO 算法优化策略                                         │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### 5.3 代码实现 RLHF

```python
import torch
import torch.nn as nn
from transformers import GPT2Model, GPT2LMHeadModel

class RewardModel(nn.Module):
    """奖励模型，用于 RLHF"""
    
    def __init__(self, gpt2_model_name='gpt2'):
        super().__init__()
        self.gpt = GPT2Model.from_pretrained(gpt2_model_name)
        self.reward_head = nn.Linear(self.gpt.config.n_embd, 1)
    
    def forward(self, input_ids, attention_mask=None):
        outputs = self.gpt(input_ids, attention_mask=attention_mask)
        # 使用最后一个 token 的表示作为奖励
        last_hidden = outputs.last_hidden_state
        reward = self.reward_head(last_hidden[:, -1, :])
        return reward

class PPOAgent:
    """PPO 强化学习智能体"""
    
    def __init__(self, policy_model, reward_model, value_model, 
                 optimizer, clip_epsilon=0.2):
        self.policy = policy_model
        self.reward_model = reward_model
        self.value = value_model
        self.optimizer = optimizer
        self.clip_epsilon = clip_epsilon
    
    def compute_gae(self, rewards, values, next_values, gamma=0.99, 
                    lam=0.95):
        """计算 Generalized Advantage Estimation"""
        advantages = []
        gae = 0
        
        for t in reversed(range(len(rewards))):
            delta = rewards[t] + gamma * next_values[t] - values[t]
            gae = delta + gamma * lam * gae
            advantages.insert(0, gae)
        
        return torch.tensor(advantages)
    
    def ppo_loss(self, log_probs, old_log_probs, advantages):
        """计算 PPO 损失"""
        ratio = torch.exp(log_probs - old_log_probs)
        
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 
                           1 + self.clip_epsilon) * advantages
        
        return -torch.min(surr1, surr2).mean()
```

## 六、GPT-4：多模态的跨越

### 6.1 GPT-4 核心特性

| 特性 | 说明 |
|------|------|
| 多模态 | 支持图像和文本输入 |
| 长上下文 | 128K 上下文窗口 |
| 指令跟随 | 更强的指令理解能力 |
| 幻觉减少 | 显著减少虚假信息生成 |
| 安全对齐 | 更安全的输出 |

### 6.2 GPT-4 架构（推测）

```
┌─────────────────────────────────────────────────────────────────┐
│                    GPT-4 架构（推测）                             │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  专家混合 (MoE) 架构:                                            │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │                    Gating Network                        │   │
│  │                         │                                │   │
│  │       ┌────────────────┼────────────────┐               │   │
│  │       ▼                ▼                ▼               │   │
│  │  ┌─────────┐     ┌─────────┐     ┌─────────┐            │   │
│  │  │ Expert1 │     │ Expert2 │     │ ExpertN │   ...      │   │
│  │  │ (路由)  │     │ (路由)  │     │ (路由)  │            │   │
│  │  └─────────┘     └─────────┘     └─────────┘            │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                  │
│  特点:                                                          │
│  - 8 个 220B 专家模型（活跃约 280B）                             │
│  - 16 个 MoE 专家                                               │
│  - 约 1.8T 总参数量                                             │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### 6.3 GPT-4 使用示例

```python
from openai import OpenAI

client = OpenAI(api_key="your-api-key")

# 使用 GPT-4 Vision 分析图像
response = client.chat.completions.create(
    model="gpt-4-vision-preview",
    messages=[
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {
                        "url": "https://example.com/image.jpg"
                    }
                },
                {
                    "type": "text",
                    "text": "描述这张图片"
                }
            ]
        }
    ],
    max_tokens=300
)

print(response.choices[0].message.content)

# 使用 GPT-4 处理长文本
response = client.chat.completions.create(
    model="gpt-4-1106-preview",  # 支持 128K 上下文
    messages=[
        {
            "role": "system",
            "content": "你是一个专业的文档分析助手"
        },
        {
            "role": "user",
            "content": "请分析以下长文档..."
        }
    ],
    temperature=0.7,
    max_tokens=2000
)
```

## 七、Claude 系列模型

### 7.1 Claude 发展历程

| 模型 | 发布时间 | 特点 |
|------|----------|------|
| Claude 1 | 2023.3 | 首个 Claude 模型 |
| Claude 2 | 2023.7 | 更长上下文，更安全 |
| Claude 3 | 2024.3 | 多模态，Opus/Haiku/Sonnet |
| Claude 3.5 | 2024.6 | 性能提升，成本降低 |

### 7.2 Claude 3 系列

```
┌─────────────────────────────────────────────────────────────────┐
│                    Claude 3 模型系列                             │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Claude 3 Opus                                                 │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  • 最强性能                                              │   │
│  │  • 200K 上下文                                          │   │
│  │  • 适用于复杂推理、写作                                  │   │
│  └─────────────────────────────────────────────────────────┘   │
│                              │                                   │
│                              ▼                                   │
│  Claude 3 Sonnet                                                │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  • 平衡性能与成本                                        │   │
│  │  • 200K 上下文                                          │   │
│  │  • 适用于企业应用                                        │   │
│  └─────────────────────────────────────────────────────────┘   │
│                              │                                   │
│                              ▼                                   │
│  Claude 3 Haiku                                                 │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  • 快速响应                                             │   │
│  │  • 200K 上下文                                          │   │
│  │  • 适用于简单任务                                        │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### 7.3 Claude 3.5 Sonnet 特点

```python
# Anthropic Claude API 使用示例
from anthropic import Anthropic

client = Anthropic(api_key="your-api-key")

# 基础对话
message = client.messages.create(
    model="claude-3-5-sonnet-20241022",
    max_tokens=1024,
    messages=[
        {"role": "user", "content": "请解释一下什么是 Transformer"}
    ]
)

print(message.content[0].text)

# 使用系统提示
message = client.messages.create(
    model="claude-3-5-sonnet-20241022",
    system="你是一个 Python 编程助手",
    max_tokens=1024,
    messages=[
        {"role": "user", "content": "写一个快速排序函数"}
    ]
)
```

## 八、总结与对比

### 8.1 GPT 系列 vs Claude 对比

| 维度 | GPT 系列 | Claude |
|------|----------|--------|
| 开发者 | OpenAI | Anthropic |
| 训练数据 | 公开互联网 | 精选数据集 |
| 特色 | 代码能力强 | 安全对齐更好 |
| 上下文 | GPT-4: 128K | Claude: 200K |
| 多模态 | GPT-4V | Claude 3 |

### 8.2 模型选择建议

```
┌─────────────────────────────────────────────────────────────────┐
│                    模型选择指南                                   │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  场景                          推荐模型                          │
│  ─────────────────────────────────────────────────────────────   │
│  代码生成                  GPT-4 / Claude 3.5 Opus              │
│  复杂推理                   Claude 3 Opus / GPT-4                │
│  快速原型                   GPT-3.5 / Claude 3 Haiku             │
│  长文档处理                Claude 3 (200K) / GPT-4              │
│  多模态理解                GPT-4V / Claude 3                    │
│  低成本应用                GPT-3.5 Turbo / Claude 3 Haiku       │
│  安全敏感场景              Claude 3                             │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

## 参考资料

- Radford, A., et al. (2018). "Improving Language Understanding by Generative Pre-Training"
- Radford, A., et al. (2019). "Language Models are Unsupervised Multitask Learners"
- Brown, T., et al. (2020). "Language Models are Few-Shot Learners"
- OpenAI. (2023). "GPT-4 Technical Report"
- Anthropic. (2024). "Claude 3 Model Card"
