# 向量数据库

## 1. 什么是向量数据库？

向量数据库是专门用于存储和检索高维向量数据的数据库。在 RAG (Retrieval-Augmented Generation) 系统中，向量数据库用于存储文档的语义嵌入，实现高效的相似性搜索。

### 1.1 为什么需要向量数据库？

| 传统数据库 | 向量数据库 |
|------------|------------|
| 精确匹配 | 语义相似性搜索 |
| 字符串/数值 | 高维向量 |
| B-tree 索引 | ANN 近似最近邻 |
| SQL 查询 | 向量相似度查询 |

### 1.2 核心概念

```
┌─────────────────────────────────────────────────────────────────┐
│                       向量数据库工作流                           │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌─────────────┐      ┌─────────────┐      ┌─────────────┐     │
│  │   原始文档   │ ──▶ │  文本嵌入    │ ──▶ │   向量存储   │     │
│  │ "机器学习..."│      │ [0.1, 0.3...]│      │  (向量索引)  │     │
│  └─────────────┘      └─────────────┘      └─────────────┘     │
│                                                      │          │
│  ┌─────────────┐      ┌─────────────┐                │          │
│  │  查询向量   │ ──▶  │  相似度搜索  │ ◀─────────────┘          │
│  │ [0.2, 0.1..]│      │  Top-K 结果  │                           │
│  └─────────────┘      └─────────────┘                           │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

## 2. Chroma

### 2.1 简介

**Chroma** 是一个轻量级、嵌入优先的向量数据库，专为 AI 应用设计。它使用 Python 实现，易于使用，无需复杂的部署。

### 2.2 安装

```bash
pip install chromadb
```

### 2.3 基础使用

```python
import chromadb
from chromadb.config import Settings

# 初始化客户端
client = chromadb.Client(Settings(
    persist_directory="./chroma_db",  # 持久化目录
    anonymized_telemetry=False        # 禁用遥测
))

# 创建集合（类似于表）
collection = client.create_collection(
    name="my_documents",
    metadata={"description": "我的文档集合"}  # 可选元数据
)

# 添加文档
collection.add(
    documents=[
        "机器学习是人工智能的一个分支，专注于开发能够从数据中学习的算法。",
        "深度学习是机器学习的一个分支，使用多层神经网络来学习数据的表示。",
        "自然语言处理是计算机科学和人工智能的交叉领域，研究如何让计算机理解和生成自然语言。"
    ],
    ids=["doc1", "doc2", "doc3"],  # 文档 ID
    metadatas=[
        {"source": "AI教程", "category": "ML"},
        {"source": "AI教程", "category": "DL"},
        {"source": "AI教程", "category": "NLP"}
    ]
)

# 查询
results = collection.query(
    query_texts=["什么是深度学习？"],
    n_results=2  # 返回前 2 个结果
)

print(results)
```

### 2.4 使用自定义嵌入

```python
import chromadb
from chromadb.config import Settings

# 使用 Sentence Transformers 作为嵌入函数
from sentence_transformers import SentenceTransformer

# 初始化嵌入模型
embedding_model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

# 创建 Chroma 客户端
client = chromadb.Client(Settings(persist_directory="./custom_embeddings"))

# 创建集合
collection = client.create_collection(
    name="custom_docs",
    embedding_function=embedding_model  # 自定义嵌入函数
)

# 添加文档
collection.add(
    documents=[
        "Python 是一种高级编程语言。",
        "JavaScript 是 Web 开发的主要语言。",
        "Java 是一种面向对象的编程语言。"
    ],
    ids=["doc1", "doc2", "doc3"]
)

# 查询（使用相同的嵌入函数）
results = collection.query(
    query_texts=["请介绍一下 Python"],
    n_results=2
)

print(results)
```

## 3. FAISS

### 3.1 简介

**FAISS (Facebook AI Similarity Search)** 是 Facebook 开发的开源库，专注于高效的相似性搜索和密集向量的聚类。

### 3.2 安装

```bash
pip install faiss-cpu  # CPU 版本
# 或
pip install faiss-gpu  # GPU 版本
```

### 3.3 基础使用

```python
import numpy as np
import faiss

# 1. 准备数据
# 生成随机向量作为示例
np.random.seed(42)
dimension = 128  # 向量维度
n_vectors = 10000  # 向量数量

# 训练数据
train_data = np.random.random((n_vectors, dimension)).astype('float32')

# 查询数据
query_data = np.random.random((5, dimension)).astype('float32')

# 2. 创建索引
# 使用 IVF 索引（倒排文件）
n_clusters = 100  # 聚类数量
quantizer = faiss.IndexFlatL2(dimension)  # 使用 L2 距离的量化器
index = faiss.IndexIVFFlat(quantizer, dimension, n_clusters)

# 3. 训练索引
index.train(train_data)

# 4. 添加向量
index.add(train_data)

# 5. 设置搜索参数
index.nprobe = 10  # 搜索的聚类数量

# 6. 搜索
k = 5  # 返回前 5 个最近邻
distances, indices = index.search(query_data, k)

print("查询结果:")
print(f"距离: {distances}")
print(f"索引: {indices}")
```

### 3.4 使用预训练模型

```python
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

# 1. 加载嵌入模型
model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

# 2. 准备文档
documents = [
    "机器学习是人工智能的核心技术。",
    "深度学习使用神经网络模型。",
    "自然语言处理研究语言的计算表示。",
    "计算机视觉让机器理解图像和视频。",
    "强化学习通过试错学习最优策略。"
]

# 3. 生成向量
embeddings = model.encode(documents, convert_to_numpy=True)
dimension = embeddings.shape[1]

# 4. 创建索引
index = faiss.IndexFlatIP(dimension)  # 内积索引（用于余弦相似度）

# 归一化向量（用于余弦相似度）
faiss.normalize_L2(embeddings)
index.add(embeddings)

# 5. 查询
query = "什么是机器学习？"
query_embedding = model.encode([query], convert_to_numpy=True)
faiss.normalize_L2(query_embedding)

k = 3
distances, indices = index.search(query_embedding, k)

print(f"查询: {query}")
print("Top-K 结果:")
for i, (idx, dist) in enumerate(zip(indices[0], distances[0])):
    print(f"  {i+1}. {documents[idx]} (相似度: {dist:.4f})")
```

### 3.5 保存和加载索引

```python
import numpy as np
import faiss

# 创建索引
dimension = 128
index = faiss.IndexFlatL2(dimension)

# 添加数据
data = np.random.random((1000, dimension)).astype('float32')
index.add(data)

# 保存索引
faiss.write_index(index, "faiss_index.bin")

# 加载索引
loaded_index = faiss.read_index("faiss_index.bin")

# 使用加载的索引
query = np.random.random((1, dimension)).astype('float32')
distances, indices = loaded_index.search(query, k=5)
```

## 4. Milvus

### 4.1 简介

**Milvus** 是一个云原生、开源的向量数据库，专为大规模向量数据设计。它支持多种索引类型，具备高可扩展性和高性能。

### 4.2 安装

```bash
pip install pymilvus
```

### 4.3 基础使用

```python
from pymilvus import connections, Collection, CollectionSchema, FieldSchema, DataType, utility

# 1. 连接 Milvus
connections.connect(
    alias="default",
    host="localhost",
    port="19530"
)

# 2. 定义集合 schema
fields = [
    FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
    FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=128),
    FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=1000),
    FieldSchema(name="category", dtype=DataType.VARCHAR, max_length=100),
]
schema = CollectionSchema(fields, description="文档集合")

# 3. 创建集合
collection = Collection("my_documents", schema)

# 4. 创建索引
index_params = {
    "metric_type": "L2",
    "index_type": "IVF_FLAT",
    "params": {"nlist": 128}
}
collection.create_index("embedding", index_params)

# 5. 插入数据
import numpy as np

# 准备数据
entities = [
    [1, 2, 3],  # id (auto_id=True 时可省略)
    [[np.random.rand(128).tolist() for _ in range(3)]],  # embeddings
    ["文本1", "文本2", "文本3"],  # text
    ["cat1", "cat2", "cat3"]  # category
]

# 插入
collection.insert(entities)

# 加载集合到内存
collection.load()

# 6. 搜索
search_params = {"metric_type": "L2", "params": {"nprobe": 10}}

query_embedding = [np.random.rand(128).tolist()]
results = collection.search(
    query_embedding,
    "embedding",
    search_params,
    limit=3,
    output_fields=["text", "category"]
)

# 打印结果
for result in results:
    for hit in result:
        print(f"ID: {hit.id}, 距离: {hit.distance}, 文本: {hit.entity.get('text')}")

# 7. 查询
filtering = 'category == "cat1"'
query_results = collection.query(
    expr=filtering,
    output_fields=["id", "text", "category"],
    limit=10
)

# 8. 删除
utility.drop_collection("my_documents")
connections.disconnect("default")
```

### 4.4 使用 Embedding 函数

```python
from pymilvus import connections, Collection
from pymilvus.model.hybrid import BGEM3EmbeddingFunction
from pymilvus import DataType

# 使用 BGE-M3 嵌入函数
ef = BGEM3EmbeddingFunction(
    model_name="BAAI/bge-m3",
    device="cpu",
    use_fp16=False
)

# 连接
connections.connect(host="localhost", port="19530")

# 定义 schema
fields = [
    FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
    FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=1000),
]
schema = CollectionSchema(fields, description="测试集合")

collection = Collection("test_collection", schema)

# 准备文档
docs = [
    "机器学习是人工智能的核心技术。",
    "深度学习是机器学习的一个分支。",
    "自然语言处理研究如何让计算机理解语言。"
]

# 生成嵌入
embeddings = ef(docs)
print(f"生成 {len(embeddings['dense'])} 个向量")

# 插入数据
# 注意: BGE-M3 生成的是 dense 和 sparse 向量
# 这里简化处理，只使用 dense 向量
dense_embeddings = [e.tolist() for e in embeddings['dense']]

entities = [
    dense_embeddings,  # embedding
    docs  # text
]

collection.insert(entities)
collection.load()

# 搜索
query = "什么是机器学习？"
query_embedding = ef([query])
query_vector = [query_embedding['dense'][0].tolist()]

results = collection.search(
    query_vector,
    "embedding",
    {"metric_type": "IP", "params": {}},
    limit=2,
    output_fields=["text"]
)

for result in results:
    for hit in result:
        print(f"文本: {hit.entity.get('text')}, 分数: {hit.score}")
```

## 5. 向量数据库对比

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         向量数据库对比                                    │
├─────────────┬───────────┬───────────────┬───────────┬─────────────────┤
│    特性      │  Chroma   │    FAISS      │  Milvus   │     Pinecone    │
├─────────────┼───────────┼───────────────┼───────────┼─────────────────┤
│  类型        │ 轻量级    │ 库            │ 云原生    │  云服务         │
│ 部署         │ 本地      │ 本地          │ 本地/云   │  仅云           │
│ 扩展性       │ 低        │ 中            │ 高        │  高             │
│ 功能         │ 基础      │ 中等          │ 完整      │  完整           │
│ 易用性       │ 简单      │ 中等          │ 中等      │  简单           │
│ ANN 算法     │ HNSW     │ IVF/HNSW      │ 多种      │  HNSW           │
│ 推荐场景     │ 原型/小   │ 中等规模     │ 大规模生产│  云部署         │
│              │ 型项目    │               │ 环境      │                 │
└─────────────┴───────────┴───────────────┴───────────┴─────────────────┘
```

## 6. 最佳实践

### 6.1 选择向量数据库

```python
def choose_vector_db(use_case, scale):
    """
    选择合适的向量数据库
    
    Args:
        use_case: 使用场景
        scale: 数据规模
    """
    
    if scale < 100000:
        # 小规模，使用 Chroma
        return "chroma"
    elif scale < 10000000:
        # 中等规模，使用 FAISS
        return "faiss"
    else:
        # 大规模，使用 Milvus
        return "milvus"
```

### 6.2 索引选择

| 索引类型 | 适用场景 | 精度 | 速度 |
|----------|----------|------|------|
| Flat | 小数据集 | 最高 | 慢 |
| IVF_FLAT | 中等规模 | 高 | 中 |
| HNSW | 大规模 | 高 | 快 |
| PQ | 超大规模 | 中 | 最快 |

### 6.3 性能优化

```python
# 1. 批量处理
def batch_search(collection, queries, batch_size=100):
    """批量搜索"""
    results = []
    for i in range(0, len(queries), batch_size):
        batch = queries[i:i+batch_size]
        results.extend(collection.search(batch, ...))
    return results

# 2. 缓存嵌入
import hashlib

def get_embedding_cached(text, model):
    """带缓存的嵌入获取"""
    cache_key = hashlib.md5(text.encode()).hexdigest()
    
    if cache_key in embedding_cache:
        return embedding_cache[cache_key]
    
    embedding = model.encode(text)
    embedding_cache[cache_key] = embedding
    return embedding
```

## 7. 总结

向量数据库是 RAG 系统的核心组件：

1. **Chroma**: 轻量级，易于使用，适合原型开发
2. **FAISS**: 高性能，适合中等规模数据
3. **Milvus**: 功能完整，适合大规模生产环境

选择合适的向量数据库需要考虑数据规模、功能需求和部署环境。
