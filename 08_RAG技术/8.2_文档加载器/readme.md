# 文档加载器

## 1. 概述

在 RAG 系统中，文档加载器负责将各种格式的文档（PDF、Word、Markdown 等）解析并转换为可处理的文本格式。本文将介绍常用的文档加载工具和实现方法。

### 1.1 文档加载流程

```
┌─────────────────────────────────────────────────────────────────┐
│                      文档加载流程                                │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌──────────┐    ┌──────────────┐    ┌─────────────┐          │
│  │ 原始文档 │ ──▶ │   文档解析    │ ──▶ │  文本提取    │          │
│  │ PDF/Docx │    │ (PyPDF2/Docx)│    │             │          │
│  └──────────┘    └──────────────┘    └──────┬──────┘          │
│                                                │                 │
│                                                ▼                 │
│  ┌──────────┐    ┌──────────────┐    ┌─────────────┐          │
│  │  文本存储 │ ◀── │  后处理       │ ◀── │  标准化处理  │          │
│  │ (List)   │    │ (分段/清洗)   │    │             │          │
│  └──────────┘    └──────────────┘    └─────────────┘          │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

## 2. PDF 加载

### 2.1 使用 PyPDF2

```python
from PyPDF2 import PdfReader

def load_pdf_pypdf2(file_path: str) -> str:
    """使用 PyPDF2 加载 PDF"""
    
    reader = PdfReader(file_path)
    text = ""
    
    for page in reader.pages:
        text += page.extract_text() + "\n"
    
    return text

# 使用示例
pdf_text = load_pdf_pypdf2("document.pdf")
print(f"提取文本长度: {len(pdf_text)}")
```

### 2.2 使用 pdfplumber (推荐)

```python
import pdfplumber

def load_pdf_pdfplumber(file_path: str) -> str:
    """使用 pdfplumber 加载 PDF（推荐）"""
    
    text = ""
    
    with pdfplumber.open(file_path) as pdf:
        for page in pdf.pages:
            # 提取文本
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
            
            # 提取表格
            tables = page.extract_tables()
            for table in tables:
                text += "\n[表格]\n"
                for row in table:
                    text += " | ".join([str(cell) for cell in row if cell]) + "\n"
    
    return text

# 使用示例
pdf_text = load_pdf_pdfplumber("document.pdf")
print(f"提取文本长度: {len(pdf_text)}")
```

### 2.3 使用 LangChain 的 PDF 加载器

```python
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders import PDFPlumberLoader

# 方式1: PyPDFLoader
loader = PyPDFLoader("document.pdf")
pages = loader.load()

print(f"页数: {len(pages)}")
for i, page in enumerate(pages[:3]):
    print(f"\n--- Page {i+1} ---")
    print(page.page_content[:200])  # 前200字符

# 方式2: PDFPlumberLoader (提取更完整)
loader = PDFPlumberLoader("document.pdf")
pages = loader.load()

# 处理每一页
for page in pages:
    print(f"元数据: {page.metadata}")
    print(f"内容: {page.page_content[:100]}")
```

### 2.4 扫描版 PDF (OCR)

```python
import pytesseract
from pdf2image import convert_from_path
from PIL import Image

def load_scanned_pdf(file_path: str) -> str:
    """处理扫描版 PDF（需要 OCR）"""
    
    # PDF 转图片
    images = convert_from_path(file_path)
    text = ""
    
    for i, image in enumerate(images):
        print(f"处理第 {i+1} 页...")
        
        # OCR 识别
        page_text = pytesseract.image_to_string(image, lang='chi_sim+eng')
        text += f"\n--- Page {i+1} ---\n" + page_text
    
    return text

# 安装依赖
# pip install pytesseract pdf2image poppler-utils
# sudo apt-get install poppler-utils tesseract-ocr
```

## 3. Word 文档加载

### 3.1 使用 python-docx

```python
from docx import Document

def load_word_docx(file_path: str) -> str:
    """加载 Word 文档 (.docx)"""
    
    doc = Document(file_path)
    text = ""
    
    # 提取段落
    for para in doc.paragraphs:
        if para.text.strip():
            text += para.text + "\n"
    
    # 提取表格
    for table in doc.tables:
        for row in table.rows:
            row_text = " | ".join([cell.text for cell in row.cells])
            text += row_text + "\n"
    
    return text

# 使用示例
doc_text = load_word_docx("document.docx")
print(doc_text[:500])
```

### 3.2 使用 LangChain 的 DocxLoader

```python
from langchain_community.document_loaders import Docx2txtLoader

loader = Docx2txtLoader("document.docx")
documents = loader.load()

for doc in documents:
    print(f"内容: {doc.page_content}")
    print(f"元数据: {doc.metadata}")
```

## 4. Markdown 加载

### 4.1 使用 LangChain 的 MarkdownLoader

```python
from langchain_community.document_loaders import MarkdownLoader

loader = MarkdownLoader("readme.md")
documents = loader.load()

for doc in documents:
    print(doc.page_content)
    print(doc.metadata)
```

### 4.2 自定义 Markdown 解析

```python
import re

def parse_markdown(file_path: str) -> list:
    """解析 Markdown 文件，返回文档块列表"""
    
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 按标题分割
    sections = []
    current_section = {"title": "Introduction", "content": "", "level": 1}
    
    lines = content.split('\n')
    for line in lines:
        # 检测标题
        match = re.match(r'^(#{1,6})\s+(.+)$', line)
        if match:
            # 保存前一个 section
            if current_section["content"].strip():
                sections.append(current_section)
            
            # 开始新的 section
            level = len(match.group(1))
            title = match.group(2)
            current_section = {"title": title, "content": "", "level": level}
        else:
            current_section["content"] += line + "\n"
    
    # 保存最后一个 section
    if current_section["content"].strip():
        sections.append(current_section)
    
    return sections

# 使用示例
sections = parse_markdown("readme.md")
for section in sections:
    print(f"\n{'#' * section['level']} {section['title']}")
    print(section['content'][:200])
```

## 5. 文本文件加载

### 5.1 纯文本

```python
def load_text_file(file_path: str) -> str:
    """加载纯文本文件"""
    
    with open(file_path, 'r', encoding='utf-8') as f:
        return f.read()

# 或使用 LangChain
from langchain_community.document_loaders import TextLoader

loader = TextLoader("text.txt", encoding="utf-8")
documents = loader.load()
```

### 5.2 CSV

```python
import csv

def load_csv(file_path: str) -> list:
    """加载 CSV 文件"""
    
    data = []
    
    with open(file_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            data.append(row)
    
    return data

# 使用 LangChain
from langchain_community.document_loaders import CSVLoader

loader = CSVLoader("data.csv", encoding="utf-8")
documents = loader.load()
```

## 6. HTML 加载

```python
from langchain_community.document_loaders import BSHTMLLoader

# 加载 HTML
loader = BSHTMLLoader("page.html")
documents = loader.load()

# 或者使用 UnstructuredHTMLLoader
from langchain_community.document_loaders import UnstructuredHTMLLoader

loader = UnstructuredHTMLLoader("page.html")
documents = loader.load()
```

## 7. 组合文档加载

### 7.1 目录批量加载

```python
from pathlib import Path
from langchain_community.document_loaders import (
    PyPDFLoader,
    Docx2txtLoader,
    TextLoader,
    MarkdownLoader,
)

def load_directory(directory_path: str) -> list:
    """加载目录下所有支持的文档"""
    
    directory = Path(directory_path)
    documents = []
    
    # 扩展名到加载器的映射
    loaders = {
        '.pdf': PyPDFLoader,
        '.docx': Docx2txtLoader,
        '.doc': Docx2txtLoader,
        '.txt': TextLoader,
        '.md': MarkdownLoader,
    }
    
    for file_path in directory.rglob('*'):
        if file_path.suffix in loaders:
            try:
                loader = loaders[file_path.suffix](str(file_path))
                docs = loader.load()
                documents.extend(docs)
                print(f"加载成功: {file_path.name}")
            except Exception as e:
                print(f"加载失败: {file_path.name}, 错误: {e}")
    
    return documents

# 使用示例
all_docs = load_directory("./documents")
print(f"共加载 {len(all_docs)} 个文档")
```

### 7.2 通用文档加载器

```python
from langchain_community.document_loaders import (
    DirectoryLoader,
    UnstructuredFileLoader,
)

# 方式1: 使用目录加载器
loader = DirectoryLoader(
    "./documents",
    glob="**/*.*",
    show_progress=True,
    loader_cls=UnstructuredFileLoader  # 自动检测文件类型
)
documents = loader.load()

# 方式2: 使用通用加载器（支持多种格式）
from langchain_community.document_loaders import UnstructuredFileLoader

loader = UnstructuredFileLoader(
    "document.pdf",
    mode="elements"  # "single" 或 "elements"
)
documents = loader.load()
```

## 8. 文档加载最佳实践

### 8.1 处理编码问题

```python
import chardet

def detect_encoding(file_path: str) -> str:
    """检测文件编码"""
    
    with open(file_path, 'rb') as f:
        raw_data = f.read()
        result = chardet.detect(raw_data)
        return result['encoding']

def load_text_with_encoding(file_path: str) -> str:
    """自动检测编码加载文本"""
    
    encoding = detect_encoding(file_path)
    
    # 常见编码尝试
    encodings = [encoding, 'utf-8', 'gbk', 'gb2312', 'latin-1']
    
    for enc in encodings:
        try:
            with open(file_path, 'r', encoding=enc) as f:
                return f.read()
        except (UnicodeDecodeError, LookupError):
            continue
    
    # 最后尝试忽略错误
    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
        return f.read()
```

### 8.2 大文档处理

```python
def load_large_pdf(file_path: str, chunk_size: int = 100) -> list:
    """分块加载大 PDF 文件"""
    
    from PyPDF2 import PdfReader
    
    reader = PdfReader(file_path)
    chunks = []
    current_chunk = []
    current_size = 0
    
    for page in reader.pages:
        text = page.extract_text()
        
        # 按页累积，达到阈值后保存
        current_chunk.append(text)
        current_size += len(text)
        
        if current_size >= chunk_size * 1000:  # 约 chunk_size KB
            chunks.append("\n".join(current_chunk))
            current_chunk = []
            current_size = 0
    
    # 保存最后一块
    if current_chunk:
        chunks.append("\n".join(current_chunk))
    
    return chunks
```

### 8.3 提取元数据

```python
from datetime import datetime
from pathlib import Path

def extract_metadata(file_path: str) -> dict:
    """提取文件元数据"""
    
    path = Path(file_path)
    stat = path.stat()
    
    return {
        "source": path.name,
        "file_type": path.suffix,
        "file_size": stat.st_size,
        "created": datetime.fromtimestamp(stat.st_ctime).isoformat(),
        "modified": datetime.fromtimestamp(stat.st_mtime).isoformat(),
    }

# 使用示例
metadata = extract_metadata("document.pdf")
print(metadata)
```

## 9. 总结

文档加载是 RAG 系统的第一步：

1. **PDF**: 使用 pdfplumber 或 PyPDF2
2. **Word**: 使用 python-docx
3. **Markdown**: 使用 LangChain 内置加载器
4. **批量处理**: 使用 DirectoryLoader
5. **编码问题**: 注意文件编码检测

选择合适的加载器可以提高文档解析的准确性和效率。
