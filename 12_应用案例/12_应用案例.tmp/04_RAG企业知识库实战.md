# 实战案例：RAG 企业知识库实战

> 本文将详细介绍如何使用 RAG（检索增强生成）技术构建企业级知识库系统，实现智能问答。

---

## 📋 案例概述

### 场景
企业需要一个基于自有文档的智能问答系统，能够：
- 导入企业文档（PDF、Word、Markdown 等）
- 智能理解用户问题
- 从文档中检索相关内容
- 生成准确、可溯源的回答

### 技术栈
- **大模型**：GPT-4o / Claude 4 / 通义千问
- **向量数据库**：Chroma / FAISS / Milvus / Pinecone
- **文档处理**：LangChain / Unstructured / PyPDF2
- **嵌入模型**：OpenAI Embedding / BGE / Jina Embedding

---

## 🏗️ 系统架构

```
┌─────────────────────────────────────────────────────────────────┐
│                     RAG 系统架构                                 │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐      │
│  │   文档加载   │───▶│   文本分割   │───▶│  嵌入向量化 │      │
│  │  (Loaders)   │    │  (Splitters)│    │ (Embeddings)│      │
│  └──────────────┘    └──────────────┘    └──────────────┘      │
│                                                     │            │
│                                                     ▼            │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐      │
│  │   答案生成   │◀───│   相似度检索 │◀───│  向量存储    │      │
│  │    (LLM)     │    │  (Retriever) │    │ (Vector Store)│      │
│  └──────────────┘    └──────────────┘    └──────────────┘      │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## 🚀 第一步：环境准备

### 1.1 安装依赖

```bash
# 创建虚拟环境
python -m venv rag-env
source rag-env/bin/activate

# 安装核心依赖
pip install langchain langchain-community langchain-openai langchain-anthropic

# 安装向量数据库
pip install chromadb faiss-cpu  # 本地开发
# pip install pymilvus pinecone-client  # 云端服务

# 安装文档处理工具
pip install pypdf python-docx markdown

# 安装嵌入模型
pip install sentence-transformers

# 安装其他工具
pip install tiktoken beautifulsoup4
```

### 1.2 目录结构

```
rag_project/
├── data/
│   ├── docs/           # 原始文档
│   └── processed/     # 处理后的文本
├── storage/           # 向量数据库存储
├── src/
│   ├── __init__.py
│   ├── loader.py      # 文档加载
│   ├── splitter.py    # 文本分割
│   ├── vectorstore.py # 向量存储
│   ├── retriever.py   # 检索模块
│   └── rag_chain.py   # RAG 链
├── .env
└── main.py
```

---

## 📥 第二步：文档加载与处理

### 2.1 文档加载器

```python
# src/loader.py
from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    MarkdownLoader,
    DocxLoader,
    CSVLoader,
    WebLoader
)
from langchain_community.document_loaders.base import BaseLoader
from typing import List
from langchain_core.documents import Document
import os

class DocumentLoader:
    """文档加载器工厂"""
    
    @staticmethod
    def get_loader(file_path: str) -> BaseLoader:
        """根据文件类型返回对应的加载器"""
        
        ext = os.path.splitext(file_path)[1].lower()
        
        loaders = {
            '.pdf': PyPDFLoader,
            '.txt': TextLoader,
            '.md': MarkdownLoader,
            '.markdown': MarkdownLoader,
            '.docx': DocxLoader,
            '.doc': DocxLoader,
            '.csv': CSVLoader,
        }
        
        loader_class = loaders.get(ext, TextLoader)
        return loader_class(file_path)
    
    @staticmethod
    def load_directory(directory: str) -> List[Document]:
        """加载目录下所有支持的文档"""
        
        documents = []
        supported_extensions = ['.pdf', '.txt', '.md', '.docx', '.csv']
        
        for root, dirs, files in os.walk(directory):
            for file in files:
                ext = os.path.splitext(file)[1].lower()
                if ext in supported_extensions:
                    file_path = os.path.join(root, file)
                    try:
                        loader = DocumentLoader.get_loader(file_path)
                        docs = loader.load()
                        documents.extend(docs)
                        print(f"✓ 加载成功: {file}")
                    except Exception as e:
                        print(f"✗ 加载失败: {file} - {e}")
        
        return documents

# 使用示例
if __name__ == "__main__":
    loader = DocumentLoader()
    docs = loader.load_directory("./data/docs")
    print(f"共加载 {len(docs)} 个文档")
```

### 2.2 文档清洗与预处理

```python
# src/preprocessor.py
from langchain_core.documents import Document
from typing import List
import re

class DocumentPreprocessor:
    """文档预处理器"""
    
    @staticmethod
    def clean_text(text: str) -> str:
        """清洗文本"""
        
        # 移除多余空白
        text = re.sub(r'\s+', ' ', text)
        
        # 移除特殊字符（保留中文、英文、数字、常用标点）
        text = re.sub(r'[^\u4e00-\u9fa5a-zA-Z0-9\s.,!?;:\'\"-]', '', text)
        
        return text.strip()
    
    @staticmethod
    def add_metadata(doc: Document, source: str, page: int = None) -> Document:
        """添加元数据"""
        
        doc.metadata['source'] = source
        if page:
            doc.metadata['page'] = page
        doc.metadata['processed'] = True
        
        return doc
    
    @staticmethod
    def process(documents: List[Document]) -> List[Document]:
        """批量处理文档"""
        
        processed = []
        for doc in documents:
            # 清洗文本
            doc.page_content = DocumentPreprocessor.clean_text(doc.page_content)
            
            # 添加元数据
            source = doc.metadata.get('source', 'unknown')
            page = doc.metadata.get('page')
            doc = DocumentPreprocessor.add_metadata(doc, source, page)
            
            processed.append(doc)
        
        return processed
```

---

## ✂️ 第三步：文本分割

### 3.1 基础分割器

```python
# src/splitter.py
from langchain.text_splitter import (
    CharacterTextSplitter,
    RecursiveCharacterTextSplitter,
    MarkdownHeaderTextSplitter,
    PythonCodeTextSplitter
)
from langchain_core.documents import Document
from typing import List

class TextSplitter:
    """文本分割器"""
    
    @staticmethod
    def get_recursive_splitter(
        chunk_size: int = 500,
        chunk_overlap: int = 50
    ) -> RecursiveCharacterTextSplitter:
        """获取递归字符分割器（推荐）"""
        
        return RecursiveCharacterTextSplitter(
            separators=["\n\n", "\n", "。", "！", "？", ". ", " ", ""],
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
        )
    
    @staticmethod
    def get_markdown_splitter() -> MarkdownHeaderTextSplitter:
        """获取 Markdown 标题分割器"""
        
        return MarkdownHeaderTextSplitter(
            headers_to_split_on=[
                ("#", "H1"),
                ("##", "H2"),
                ("###", "H3"),
            ]
        )
    
    @staticmethod
    def split_documents(
        documents: List[Document],
        chunk_size: int = 500,
        chunk_overlap: int = 50
    ) -> List[Document]:
        """分割文档"""
        
        splitter = TextSplitter.get_recursive_splitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        
        return splitter.split_documents(documents)
```

### 3.2 智能分割策略

```python
# src/smart_splitter.py
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from typing import List, Callable
import re

class SmartTextSplitter:
    """智能文本分割器"""
    
    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 100,
        length_function: Callable[[str], int] = len
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.length_function = length_function
    
    def split_by_paragraph(self, text: str) -> List[str]:
        """按段落分割"""
        paragraphs = re.split(r'\n\s*\n', text)
        return [p.strip() for p in paragraphs if p.strip()]
    
    def split_by_sentence(self, text: str) -> List[str]:
        """按句子分割"""
        sentences = re.split(r'[。！？.!?]+', text)
        return [s.strip() for s in sentences if s.strip()]
    
    def create_chunks(self, documents: List[Document]) -> List[Document]:
        """创建文档块"""
        
        all_chunks = []
        
        for doc in documents:
            text = doc.page_content
            
            # 尝试按段落分割
            paragraphs = self.split_by_paragraph(text)
            
            current_chunk = ""
            for para in paragraphs:
                # 如果当前块加上新段落超过大小限制，保存当前块
                if self.length_function(current_chunk) + self.length_function(para) > self.chunk_size:
                    if current_chunk:
                        all_chunks.append(Document(
                            page_content=current_chunk,
                            metadata=doc.metadata.copy()
                        ))
                    
                    # 如果单个段落就超过大小，按句子分割
                    if self.length_function(para) > self.chunk_size:
                        sentences = self.split_by_sentence(para)
                        current_chunk = ""
                        for sent in sentences:
                            if self.length_function(current_chunk) + self.length_function(sent) > self.chunk_size:
                                if current_chunk:
                                    all_chunks.append(Document(
                                        page_content=current_chunk,
                                        metadata=doc.metadata.copy()
                                    ))
                                current_chunk = sent
                            else:
                                current_chunk += "。" + sent if current_chunk else sent
                    else:
                        current_chunk = para
                else:
                    current_chunk += "\n\n" + para if current_chunk else para
            
            # 保存最后一个块
            if current_chunk:
                all_chunks.append(Document(
                    page_content=current_chunk,
                    metadata=doc.metadata.copy()
                ))
        
        return all_chunks
```

---

## 💾 第四步：向量存储

### 4.1 使用 Chroma（本地开发）

```python
# src/vectorstore.py
from langchain.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_anthropic import AnthropicEmbeddings
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document
from typing import List, Optional
import os

class VectorStoreManager:
    """向量存储管理器"""
    
    def __init__(self, embedding_model: str = "openai"):
        self.embedding_model = embedding_model
        self.embeddings = self._get_embeddings(embedding_model)
        self.vectorstore = None
    
    def _get_embeddings(self, model: str):
        """获取嵌入模型"""
        
        if model == "openai":
            from langchain_openai import OpenAIEmbeddings
            return OpenAIEmbeddings(
                model="text-embedding-3-small",
                api_key=os.getenv("OPENAI_API_KEY")
            )
        
        elif model == "anthropic":
            return AnthropicEmbeddings(
                model="claude-embedding-3-sonnet-20240229",
                api_key=os.getenv("ANTHROPIC_API_KEY")
            )
        
        elif model == "bge-base":
            return HuggingFaceEmbeddings(
                model_name="BAAI/bge-base-zh-v1.5",
                model_kwargs={'device': 'cpu'}
            )
        
        elif model == "jina":
            return HuggingFaceEmbeddings(
                model_name="jinaai/jina-embeddings-v2-base-zh",
                model_kwargs={'device': 'cpu'}
            )
        
        else:
            raise ValueError(f"不支持的嵌入模型: {model}")
    
    def create_vectorstore(
        self,
        documents: List[Document],
        persist_directory: str = "./storage/chroma"
    ) -> Chroma:
        """创建向量存储"""
        
        self.vectorstore = Chroma.from_documents(
            documents=documents,
            embedding=self.embeddings,
            persist_directory=persist_directory
        )
        
        return self.vectorstore
    
    def load_vectorstore(
        self,
        persist_directory: str = "./storage/chroma"
    ) -> Chroma:
        """加载已存在的向量存储"""
        
        self.vectorstore = Chroma(
            persist_directory=persist_directory,
            embedding_function=self.embeddings
        )
        
        return self.vectorstore
    
    def add_documents(self, documents: List[Document]):
        """添加文档到向量存储"""
        
        if self.vectorstore is None:
            raise ValueError("请先创建或加载向量存储")
        
        self.vectorstore.add_documents(documents)
    
    def delete_collection(self):
        """删除向量集合"""
        
        if self.vectorstore:
            self.vectorstore.delete_collection()
```

### 4.2 使用 FAISS（大规模数据）

```python
# src/faiss_store.py
from langchain.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document
from typing import List

def create_faiss_index(
    documents: List[Document],
    embedding_model: str = "text-embedding-3-small"
) -> FAISS:
    """创建 FAISS 索引"""
    
    embeddings = OpenAIEmbeddings(model=embedding_model)
    
    vectorstore = FAISS.from_documents(
        documents=documents,
        embedding=embeddings
    )
    
    return vectorstore

def save_faiss_index(vectorstore: FAISS, path: str):
    """保存 FAISS 索引"""
    vectorstore.save_local(path)

def load_faiss_index(path: str) -> FAISS:
    """加载 FAISS 索引"""
    embeddings = OpenAIEmbeddings()
    return FAISS.load_local(path, embeddings, allow_dangerous_deserialization=True)
```

---

## 🔍 第五步：检索系统

### 5.1 基础检索

```python
# src/retriever.py
from langchain.vectorstores import VectorStoreRetriever
from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from typing import List, Optional
from langchain_core.documents import Document

class RetrieverManager:
    """检索管理器"""
    
    def __init__(self, vectorstore, search_type: str = "similarity"):
        self.vectorstore = vectorstore
        self.search_type = search_type
        self.retriever = None
    
    def create_retriever(
        self,
        search_type: str = "similarity",
        k: int = 4,
        score_threshold: float = 0.5,
        filter: dict = None
    ) -> VectorStoreRetriever:
        """创建检索器"""
        
        search_kwargs = {"k": k}
        
        # 添加过滤条件
        if filter:
            search_kwargs["filter"] = filter
        
        # 添加相似度阈值
        if score_threshold and search_type == "similarity_score_threshold":
            search_kwargs["score_threshold"] = score_threshold
        
        self.retriever = self.vectorstore.as_retriever(
            search_type=search_type,
            search_kwargs=search_kwargs
        )
        
        return self.retriever
    
    def get_relevant_documents(self, query: str) -> List[Document]:
        """获取相关文档"""
        
        if self.retriever is None:
            raise ValueError("请先创建检索器")
        
        return self.retriever.invoke(query)
    
    def get_relevant_documents_with_score(self, query: str) -> List[tuple]:
        """获取相关文档及分数"""
        
        if self.vectorstore is None:
            raise ValueError("请先创建向量存储")
        
        return self.vectorstore.similarity_search_with_score(query)
```

### 5.2 高级检索策略

```python
# src/advanced_retriever.py
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor
from langchain_openai import ChatOpenAI
from langchain.retrievers.self_query import SelfQueryRetriever
from langchain.schema import Document

class AdvancedRetriever:
    """高级检索器"""
    
    def __init__(self, vectorstore, llm):
        self.vectorstore = vectorstore
        self.llm = llm
    
    def get_compression_retriever(self, base_retriever):
        """获取压缩检索器 - 提取关键信息"""
        
        compressor = LLMChainExtractor.from_llm(self.llm)
        
        return ContextualCompressionRetriever(
            base_compressor=compressor,
            base_retriever=base_retriever
        )
    
    def get_multi_query_retriever(self, query: str) -> List[Document]:
        """多查询检索 - 生成多个查询变体"""
        
        from langchain.retrievers import MultiQueryRetriever
        
        multi_retriever = MultiQueryRetriever.from_llm(
            retriever=self.vectorstore.as_retriever(search_kwargs={"k": 4}),
            llm=self.llm
        )
        
        return multi_retriever.invoke(query)
    
    def get_ensemble_retriever(self, retrievers: list, weights: list = None):
        """集成检索 - 组合多个检索器"""
        
        from langchain.retrievers import EnsembleRetriever
        
        if weights is None:
            weights = [1/len(retrievers)] * len(retrievers)
        
        return EnsembleRetriever(
            retrievers=retrievers,
            weights=weights
        )
```

---

## 🔗 第六步：RAG 链构建

### 6.1 基础 RAG 链

```python
# src/rag_chain.py
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.prompts import PromptTemplate
from typing import List, Optional

class RAGChain:
    """RAG 链管理器"""
    
    def __init__(self, retriever, model: str = "openai", model_name: str = "gpt-4o"):
        self.retriever = retriever
        self.llm = self._get_llm(model, model_name)
        self.chain = None
    
    def _get_llm(self, model: str, model_name: str):
        """获取 LLM"""
        
        import os
        from dotenv import load_dotenv
        load_dotenv()
        
        if model == "openai":
            from langchain_openai import ChatOpenAI
            return ChatOpenAI(
                model=model_name,
                temperature=0
            )
        elif model == "anthropic":
            from langchain_anthropic import ChatAnthropic
            return ChatAnthropic(
                model=model_name,
                temperature=0
            )
    
    def create_chain(self, system_prompt: str = None):
        """创建 RAG 链"""
        
        # 默认系统提示词
        if system_prompt is None:
            system_prompt = """你是一个专业的问答助手。
            
            请根据以下参考文档回答用户的问题。
            
            参考文档：
            {context}
            
            要求：
            1. 只根据提供的文档内容回答，不要编造信息
            2. 如果文档中没有相关信息，请明确告知用户
            3. 回答要准确、简洁、有条理
            4. 在回答末尾标注信息来源"""
        
        # 创建文档处理链
        prompt = PromptTemplate.from_template(system_prompt)
        combine_docs_chain = create_stuff_documents_chain(
            self.llm,
            prompt,
            document_variable_name="context"
        )
        
        # 创建检索增强链
        self.chain = create_retrieval_chain(
            self.retriever,
            combine_docs_chain
        )
        
        return self.chain
    
    def invoke(self, query: str) -> dict:
        """执行问答"""
        
        if self.chain is None:
            raise ValueError("请先创建 RAG 链")
        
        return self.chain.invoke({"input": query})
    
    def get_answer(self, query: str) -> str:
        """获取答案（仅返回文本）"""
        
        result = self.invoke(query)
        return result["answer"]
    
    def get_sources(self, query: str) -> List[str]:
        """获取来源文档"""
        
        result = self.invoke(query)
        sources = []
        
        for doc in result.get("context", []):
            source = doc.metadata.get("source", "未知")
            page = doc.metadata.get("page", "")
            sources.append(f"{source} (页码: {page})" if page else source)
        
        return list(set(sources))
```

### 6.2 高级 RAG 链

```python
# src/advanced_rag_chain.py
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate

class AdvancedRAGChain:
    """高级 RAG 链"""
    
    def __init__(self, vectorstore, llm):
        self.vectorstore = vectorstore
        self.llm = llm
        self.memory = None
    
    def create_conversational_chain(self, system_prompt: str = None):
        """创建对话式 RAG 链"""
        
        # 对话记忆
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True,
            output_key="answer"
        )
        
        # 自定义提示词
        if system_prompt:
            prompt = PromptTemplate.from_template(system_prompt)
        else:
            prompt = None
        
        # 创建对话检索链
        chain = ConversationalRetrievalChain.from_llm(
            llm=self.llm,
            retriever=self.vectorstore.as_retriever(search_kwargs={"k": 4}),
            memory=self.memory,
            combine_docs_chain_kwargs={"prompt": prompt} if prompt else {},
            return_source_documents=True,
            verbose=True
        )
        
        return chain
    
    def create_hyde_chain(self):
        """创建 HyDE（假设性文档嵌入）链"""
        
        from langchain.chains import HypotheticalDocumentsEmbedder
        from langchain.chains import LLMChain
        
        # 使用 LLM 生成假设性回答作为查询
        hyde_prompt = PromptTemplate.from_template(
            "请用一段简洁的文字回答以下问题，这个问题可能涉及到公司内部文档：\n{question}"
        )
        
        hyde_chain = LLMChain(llm=self.llm, prompt=hyde_prompt)
        
        # 创建基于 HyDE 的检索器
        base_retriever = self.vectorstore.as_retriever(search_kwargs={"k": 4})
        
        # 这里简化处理，实际可以使用 HypotheticalDocumentsEmbedder
        return hyde_chain, base_retriever
```

---

## 🖥️ 第七步：构建完整知识库系统

### 7.1 主程序

```python
# main.py
import os
from dotenv import load_dotenv
from src.loader import DocumentLoader
from src.preprocessor import DocumentPreprocessor
from src.splitter import TextSplitter
from src.vectorstore import VectorStoreManager
from src.retriever import RetrieverManager
from src.rag_chain import RAGChain

load_dotenv()

class EnterpriseKnowledgeBase:
    """企业知识库系统"""
    
    def __init__(self, embedding_model: str = "openai"):
        self.embedding_model = embedding_model
        self.vectorstore_manager = VectorStoreManager(embedding_model)
        self.retriever_manager = None
        self.rag_chain = None
    
    def build_knowledge_base(
        self,
        docs_directory: str,
        chunk_size: int = 500,
        chunk_overlap: int = 50,
        persist_directory: str = "./storage/knowledge_base"
    ):
        """构建知识库"""
        
        print("📂 步骤1: 加载文档...")
        documents = DocumentLoader.load_directory(docs_directory)
        print(f"   加载了 {len(documents)} 个文档")
        
        print("🔧 步骤2: 预处理文档...")
        documents = DocumentPreprocessor.process(documents)
        
        print("✂️ 步骤3: 分割文档...")
        documents = TextSplitter.split_documents(
            documents,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        print(f"   分割成 {len(documents)} 个文本块")
        
        print("💾 步骤4: 创建向量存储...")
        vectorstore = self.vectorstore_manager.create_vectorstore(
            documents=documents,
            persist_directory=persist_directory
        )
        print(f"   向量存储已创建，共 {vectorstore._collection.count()} 个向量")
        
        print("🔍 步骤5: 创建检索器...")
        self.retriever_manager = RetrieverManager(vectorstore)
        retriever = self.retriever_manager.create_retriever(k=4)
        
        print("🔗 步骤6: 创建 RAG 链...")
        self.rag_chain = RAGChain(retriever)
        self.rag_chain.create_chain()
        
        print("✅ 知识库构建完成！")
        return vectorstore
    
    def load_knowledge_base(self, persist_directory: str = "./storage/knowledge_base"):
        """加载已有知识库"""
        
        print("📥 加载已有知识库...")
        vectorstore = self.vectorstore_manager.load_vectorstore(persist_directory)
        
        self.retriever_manager = RetrieverManager(vectorstore)
        retriever = self.retriever_manager.create_retriever(k=4)
        
        self.rag_chain = RAGChain(retriever)
        self.rag_chain.create_chain()
        
        print(f"   知识库加载完成，共 {vectorstore._collection.count()} 个向量")
        return vectorstore
    
    def query(self, question: str) -> dict:
        """查询"""
        
        if self.rag_chain is None:
            raise ValueError("请先构建或加载知识库")
        
        result = self.rag_chain.invoke(question)
        
        return {
            "answer": result["answer"],
            "sources": self.rag_chain.get_sources(question)
        }
    
    def query_with_context(self, question: str) -> str:
        """带上下文的查询（简化输出）"""
        
        result = self.query(question)
        
        response = f"📝 回答：\n{result['answer']}\n\n"
        response += f"📚 参考来源：\n"
        for i, source in enumerate(result['sources'], 1):
            response += f"   {i}. {source}\n"
        
        return response


# 使用示例
if __name__ == "__main__":
    # 初始化
    kb = EnterpriseKnowledgeBase(embedding_model="openai")
    
    # 构建知识库（首次运行）
    # kb.build_knowledge_base("./data/docs")
    
    # 加载已有知识库
    kb.load_knowledge_base()
    
    # 查询
    while True:
        question = input("\n❓ 请输入问题（输入 q 退出）：\n> ")
        if question.lower() == 'q':
            break
        
        result = kb.query_with_context(question)
        print(result)
```

### 7.2 API 服务

```python
# api.py
from flask import Flask, request, jsonify
from flask_cors import CORS
from main import EnterpriseKnowledgeBase
import os
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)
CORS(app)

# 初始化知识库
print("🚀 启动知识库系统...")
kb = EnterpriseKnowledgeBase(embedding_model="openai")
kb.load_knowledge_base()

@app.route("/query", methods=["POST"])
def query():
    """问答接口"""
    
    data = request.json
    question = data.get("question", "")
    
    if not question:
        return jsonify({"error": "问题不能为空"}), 400
    
    try:
        result = kb.query(question)
        return jsonify({
            "answer": result["answer"],
            "sources": result["sources"]
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/health", methods=["GET"])
def health():
    """健康检查"""
    return jsonify({"status": "ok"})

if __name__ == "__main__":
    app.run(debug=True, port=8000)
```

---

## 📦 第八步：部署与优化

### 8.1 Docker 部署

```dockerfile
# Dockerfile
FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

EXPOSE 8000

CMD ["python", "api.py"]
```

```yaml
# docker-compose.yml
version: '3.8'
services:
  rag-api:
    build: .
    ports:
      - "8000:8000"
    environment:
      - OPENAI_API_KEY=${OPENAI_API_KEY}
      - ANTHROPIC_API_KEY=${ANTHROPIC_API_KEY}
    volumes:
      - ./storage:/app/storage
      - ./data:/app/data
```

### 8.2 性能优化

| 优化方向 | 具体方法 |
|----------|----------|
| **检索质量** | 调整 chunk_size、尝试不同嵌入模型 |
| **回答准确率** | 添加 Query Rewriting、使用 rerank 模型 |
| **响应速度** | 添加缓存、使用异步处理 |
| **成本控制** | 使用本地嵌入模型、批量处理 |

---

## 📚 总结

本案例完整展示了企业级 RAG 知识库系统的构建过程：

1. **文档处理**：支持多种格式文档的加载与清洗
2. **文本分割**：多种分割策略满足不同场景
3. **向量存储**：Chroma/FAISS 等多种选择
4. **检索系统**：基础检索 + 高级检索策略
5. **RAG 链**：对话式 + 多种高级特性
6. **部署上线**：Docker 容器化部署

通过本案例的学习，您应该能够：
- 构建完整的企业知识库系统
- 实现基于自有文档的智能问答
- 根据实际需求优化系统性能

---

## 🔗 延伸阅读

- [LangChain RAG 教程](https://python.langchain.com/docs/tutorials/rag/)
- [Chroma 向量数据库文档](https://docs.trychroma.com/)
- [RAG 最佳实践指南](https://github.com/run-llama/llama_index/blob/main/docs/docs/optimizing/building_indices.md)

---

> 📝 **编写者**: GPT-Notes 团队  
> 📅 **更新日期**: 2026年2月  
> ⭐ **如果你觉得有帮助，欢迎提交改进建议！**
