# 文本分割

## 1. 概述

文本分割 (Text Splitting) 是将长文档拆分成较小块的過程。这是 RAG 系统的关键步骤，直接影响检索效果。

### 1.1 为什么需要文本分割？

| 问题 | 说明 |
|------|------|
| **向量维度** | 嵌入模型有最大输入长度限制 |
| **语义完整** | 保持每个块的语义完整性 |
| **检索精度** | 较小的块可以提高检索精度 |
| **上下文** | 保留足够的上下文信息 |

### 1.2 分割策略分类

```
┌─────────────────────────────────────────────────────────────────┐
│                       文本分割策略                               │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌───────────────┐    ┌───────────────┐    ┌───────────────┐ │
│  │ 固定长度分割   │    │ 递归分割       │    │ 语义分割       │ │
│  │ (Character)   │    │ (Recursive)   │    │ (Semantic)    │ │
│  ├───────────────┤    ├───────────────┤    ├───────────────┤ │
│  │ 简单快速      │    │ 保持层次结构   │    │ 基于AI理解    │ │
│  │ 可能切断句子   │    │ 推荐使用      │    │ 计算成本高    │ │
│  └───────────────┘    └───────────────┘    └───────────────┘ │
│                                                                 │
│  ┌───────────────┐    ┌───────────────┐                       │
│  │ Markdown分割  │    │ 代码分割       │                       │
│  │ (Markdown)    │    │ (Code)        │                       │
│  ├───────────────┤    ├───────────────┤                       │
│  │ 保持格式      │    │ 保持函数/类    │                       │
│  │ 适合文档      │    │ 适合代码库     │                       │
│  └───────────────┘    └───────────────┘                       │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

## 2. 固定长度分割

### 2.1 字符级分割

```python
from langchain.text_splitter import CharacterTextSplitter

# 创建分割器
splitter = CharacterTextSplitter(
    chunk_size=100,        # 块大小
    chunk_overlap=20,      # 块重叠大小
    separator="\n",       # 分隔符
    length_function=len,  # 计算长度的方式
)

# 分割文本
text = """
机器学习是人工智能的一个分支，专注于开发能够从数据中学习的算法。
深度学习是机器学习的一个分支，使用多层神经网络来学习数据的表示。
自然语言处理研究如何让计算机理解和生成自然语言。
"""

chunks = splitter.split_text(text)

for i, chunk in enumerate(chunks):
    print(f"--- Chunk {i+1} ---")
    print(chunk)
    print()
```

### 2.2 自定义分割器

```python
class FixedSizeSplitter:
    """固定大小文本分割器"""
    
    def __init__(self, chunk_size: int, overlap: int = 0):
        self.chunk_size = chunk_size
        self.overlap = overlap
    
    def split(self, text: str) -> list:
        """分割文本"""
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + self.chunk_size
            chunk = text[start:end]
            chunks.append(chunk)
            
            start = end - self.overlap
        
        return chunks

# 使用示例
splitter = FixedSizeSplitter(chunk_size=100, overlap=20)
chunks = splitter.split(text)
```

## 3. 递归分割 (推荐)

### 3.1 基础使用

```python
from langchain.text_splitter import RecursiveCharacterTextSplitter

# 创建递归分割器
splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,        # 块大小
    chunk_overlap=50,      # 重叠大小
    # 按优先级尝试的分隔符
    separators=[
        "\n\n",           # 段落分隔
        "\n",             # 换行
        "。", "！", "？", # 中文标点
        ". ", "! ", "? ", # 英文标点
        " ",              # 空格
        ""                # 单字符
    ],
    length_function=len,
)

# 分割文本
text = """
机器学习是人工智能的核心技术。它使计算机能够从数据中学习，而无需明确编程。

深度学习是机器学习的一个分支。它使用具有多个隐藏层的神经网络。

自然语言处理（NLP）是人工智能的一个领域，专注于使计算机能够理解和生成人类语言。

计算机视觉是另一个重要领域。它使机器能够理解和处理图像和视频。
"""

chunks = splitter.split_text(text)

for i, chunk in enumerate(chunks):
    print(f"Chunk {i+1} (长度: {len(chunk)}):")
    print(chunk)
    print("-" * 50)
```

### 3.2 处理中文文本

```python
from langchain.text_splitter import RecursiveCharacterTextSplitter

class ChineseTextSplitter:
    """针对中文优化的递归分割器"""
    
    def __init__(self, chunk_size: int = 500, overlap: int = 50):
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=overlap,
            separators=[
                "\n\n",      # 段落
                "\n",        # 换行
                "。！？",     # 句子结束
                ".,!?;:",    # 英文标点
                " ",         # 空格
                ""           # 单字符
            ],
            length_function=self._chinese_length
        )
    
    def _chinese_length(self, text: str) -> int:
        """中英文混合文本长度计算"""
        # 中文每个字符算1，英文每个字符算0.5
        chinese_chars = sum(1 for c in text if '\u4e00' <= c <= '\u9fff')
        other_chars = len(text) - chinese_chars
        return chinese_chars + int(other_chars * 0.5)
    
    def split(self, text: str) -> list:
        return self.splitter.split_text(text)

# 使用示例
chinese_text = """
深度学习（Deep Learning）是机器学习的一个分支，它是一种以人工神经网络为架构，对数据进行表征学习的算法。

卷积神经网络（CNN）是深度学习的一种重要架构，特别适用于处理图像数据。循环神经网络（RNN）则适用于处理序列数据。

Transformer 架构近年来在自然语言处理领域取得了巨大成功。BERT、GPT 等模型都是基于 Transformer 设计的。
"""

splitter = ChineseTextSplitter(chunk_size=200, overlap=30)
chunks = splitter.split(chinese_text)

for i, chunk in enumerate(chunks):
    print(f"Chunk {i+1}: {chunk[:50]}...")
```

## 4. Markdown 分割

### 4.1 使用 LangChain

```python
from langchain.text_splitter import MarkdownTextSplitter

# 创建 Markdown 分割器
splitter = MarkdownTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)

# Markdown 内容
markdown_text = """
# 机器学习概述

机器学习是人工智能的核心技术。

## 监督学习

监督学习是最常见的机器学习范式。

### 分类

分类算法用于预测离散的类别标签。

### 回归

回归算法用于预测连续的数值。

## 无监督学习

无监督学习不需要标注数据。
"""

chunks = splitter.split_text(markdown_text)

for i, chunk in enumerate(chunks):
    print(f"--- Chunk {i+1} ---")
    print(chunk)
    print()
```

### 4.2 自定义 Markdown 分割

```python
import re

class MarkdownSplitter:
    """Markdown 文档分割器"""
    
    def __init__(self, chunk_size: int = 500, overlap: int = 50):
        self.chunk_size = chunk_size
        self.overlap = overlap
    
    def split(self, text: str) -> list:
        # 按标题分割
        sections = self._split_by_headers(text)
        
        # 对每个 section 进一步分割
        chunks = []
        for section in sections:
            if len(section) <= self.chunk_size:
                chunks.append(section)
            else:
                sub_chunks = self._split_recursive(section)
                chunks.extend(sub_chunks)
        
        return chunks
    
    def _split_by_headers(self, text: str) -> list:
        """按标题分割"""
        # 匹配 Markdown 标题
        pattern = r'^(#{1,6})\s+(.+)$'
        
        sections = []
        current_section = ""
        
        for line in text.split('\n'):
            if re.match(pattern, line):
                if current_section:
                    sections.append(current_section)
                current_section = line + "\n"
            else:
                current_section += line + "\n"
        
        if current_section:
            sections.append(current_section)
        
        return sections
    
    def _split_recursive(self, text: str) -> list:
        """递归分割"""
        separators = ["\n\n", "\n", "。", "！", "？", ".", "!", "?", ""]
        
        for sep in separators:
            if sep in text:
                parts = text.split(sep)
                
                current_chunk = ""
                chunks = []
                
                for part in parts:
                    if len(current_chunk) + len(part) <= self.chunk_size:
                        current_chunk += part + sep
                    else:
                        if current_chunk:
                            chunks.append(current_chunk.strip())
                        current_chunk = part + sep
                
                if current_chunk:
                    chunks.append(current_chunk.strip())
                
                if len(chunks) > 1:
                    return chunks
        
        return [text[:self.chunk_size]]

# 使用示例
splitter = MarkdownSplitter(chunk_size=200, overlap=30)
chunks = splitter.split(markdown_text)
```

## 5. 代码分割

### 5.1 通用代码分割

```python
from langchain.text_splitter import Language

# 支持的语言
print([lang.value for lang in Language])
# ['cpp', 'go', 'java', 'javascript', 'python', 'ruby', 'rust', 'scala', 'swift']

# Python 代码分割
from langchain.text_splitter import RecursiveCharacterTextSplitter

python_code = """
import numpy as np

class DataProcessor:
    def __init__(self):
        self.data = []
    
    def load_data(self, filepath):
        '''加载数据'''
        with open(filepath, 'r') as f:
            self.data = f.readlines()
    
    def process(self):
        '''处理数据'''
        return [line.strip() for line in self.data]

def main():
    processor = DataProcessor()
    processor.load_data('data.txt')
    result = processor.process()
    print(result)

if __name__ == '__main__':
    main()
"""

splitter = RecursiveCharacterTextSplitter.from_language(
    language=Language.PYTHON,
    chunk_size=200,
    chunk_overlap=30
)

chunks = splitter.split_text(python_code)

for i, chunk in enumerate(chunks):
    print(f"--- Chunk {i+1} ---")
    print(chunk)
    print()
```

### 5.2 自定义代码分割器

```python
import re

class CodeSplitter:
    """代码分割器 - 按函数/类分割"""
    
    def __init__(self, chunk_size: int = 500):
        self.chunk_size = chunk_size
    
    def split(self, code: str, language: str = "python") -> list:
        if language == "python":
            return self._split_python(code)
        elif language == "javascript":
            return self._split_javascript(code)
        else:
            return self._split_generic(code)
    
    def _split_python(self, code: str) -> list:
        # 匹配函数和类定义
        pattern = r'^(class|def)\s+\w+'
        
        sections = []
        current = ""
        
        for line in code.split('\n'):
            if re.match(pattern, line) and current:
                sections.append(current)
                current = ""
            current += line + "\n"
        
        if current:
            sections.append(current)
        
        # 对过长的 section 进一步分割
        chunks = []
        for section in sections:
            if len(section) <= self.chunk_size:
                chunks.append(section)
            else:
                chunks.extend(self._split_generic(section))
        
        return chunks
    
    def _split_javascript(self, code: str) -> list:
        # 匹配函数声明
        pattern = r'^(function|const|let|var)\s+\w+\s*=|\bclass\s+\w+'
        
        sections = []
        current = ""
        
        for line in code.split('\n'):
            if re.match(pattern, line) and current:
                sections.append(current)
                current = ""
            current += line + "\n"
        
        if current:
            sections.append(current)
        
        return [s for s in sections if len(s) <= self.chunk_size] if sections else [code[:self.chunk_size]]
    
    def _split_generic(self, code: str) -> list:
        # 按行数分割
        lines = code.split('\n')
        chunks = []
        current = []
        current_size = 0
        
        for line in lines:
            current.append(line)
            current_size += len(line)
            
            if current_size >= self.chunk_size:
                chunks.append('\n'.join(current))
                current = []
                current_size = 0
        
        if current:
            chunks.append('\n'.join(current))
        
        return chunks

# 使用示例
splitter = CodeSplitter(chunk_size=300)
chunks = splitter.split(python_code, language="python")
```

## 6. 语义分割 (高级)

### 6.1 基于句子嵌入的分割

```python
from sentence_transformers import SentenceTransformer
import numpy as np

class SemanticSplitter:
    """基于语义的文本分割器"""
    
    def __init__(self, model_name: str = "paraphrase-multilingual-MiniLM-L12-v1", 
                 threshold: float = 0.5):
        self.model = SentenceTransformer(model_name)
        self.threshold = threshold
    
    def split(self, text: str) -> list:
        # 分割成句子
        sentences = self._split_sentences(text)
        
        if len(sentences) <= 1:
            return [text]
        
        # 计算句子嵌入
        embeddings = self.model.encode(sentences)
        
        # 计算相邻句子的相似度
        chunks = []
        current_chunk = [sentences[0]]
        
        for i in range(1, len(sentences)):
            # 计算相似度
            similarity = np.dot(embeddings[i-1], embeddings[i])
            
            if similarity > self.threshold:
                current_chunk.append(sentences[i])
            else:
                chunks.append(" ".join(current_chunk))
                current_chunk = [sentences[i]]
        
        # 最后一个 chunk
        if current_chunk:
            chunks.append(" ".join(current_chunk))
        
        return chunks
    
    def _split_sentences(self, text: str) -> list:
        # 简单的句子分割
        import re
        sentences = re.split(r'[。！？.!?]+', text)
        return [s.strip() for s in sentences if s.strip()]

# 使用示例
splitter = SemanticSplitter(threshold=0.6)
text = "机器学习是人工智能的核心技术。深度学习是机器学习的一个分支。它使用神经网络模型。自然语言处理研究语言的理解和生成。"
chunks = splitter.split(text)
```

## 7. 分割策略选择

### 7.1 选择指南

| 场景 | 推荐策略 |
|------|----------|
| 通用文本 | RecursiveCharacterTextSplitter |
| Markdown 文档 | MarkdownTextSplitter |
| 代码文件 | Language-specific Splitter |
| 对话数据 | 按轮次分割 |
| 长篇小说 | 按章节 + 递归 |
| 中文文档 | ChineseTextSplitter |

### 7.2 chunk_size 选择

```python
def recommend_chunk_size(embedding_model: str, use_case: str) -> int:
    """推荐 chunk size"""
    
    # 根据嵌入模型推荐
    chunk_sizes = {
        "text-embedding-ada-002": 8000,   # 最大 token 数
        "text-embedding-3-small": 8000,
    }
    
    base_size = chunk_sizes.get(embedding_model, 5000)
    
    # 根据用途调整
    if use_case == "问答":
        # 较小的块，精确匹配
        return min(500, base_size // 8)
    elif use_case == "摘要":
        # 较大的块，保持上下文
        return min(2000, base_size // 4)
    else:
        # 默认
        return min(1000, base_size // 5)

# 使用示例
size = recommend_chunk_size("text-embedding-ada-002", "问答")
print(f"推荐 chunk_size: {size}")
```

## 8. 总结

文本分割是 RAG 系统的关键环节：

1. **固定长度**: 简单快速，可能切断语义
2. **递归分割**: 推荐使用，保持语义完整性
3. **Markdown 分割**: 适合文档
4. **代码分割**: 保持函数/类结构
5. **语义分割**: 最智能但计算成本高

选择合适的分割策略需要考虑文档类型、用途和性能要求。
