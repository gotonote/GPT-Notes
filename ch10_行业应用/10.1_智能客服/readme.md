# 智能客服系统

## 概述

智能客服是 LLM 最广泛的应用场景之一。通过大语言模型，可以构建能够理解用户意图、进行多轮对话、提供个性化服务的智能客服机器人。本章将详细介绍智能客服系统的架构设计、核心功能实现和实际案例。

## 1. 智能客服系统架构

### 1.1 整体架构

```
┌─────────────────────────────────────────────────────────────────┐
│                    智能客服系统架构图                             │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │                      用户渠道层                          │    │
│  │  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐  │    │
│  │  │  Web    │ │  App    │ │  微信   │ │  WhatsApp│  │    │
│  │  └──────────┘ └──────────┘ └──────────┘ └──────────┘  │    │
│  └─────────────────────────────────────────────────────────┘    │
│                              │                                   │
│                              ▼                                   │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │                      网关层                              │    │
│  │  ┌──────────────┐ ┌──────────────┐ ┌──────────────┐  │    │
│  │  │  负载均衡    │ │  限流熔断    │ │  安全认证    │  │    │
│  │  └──────────────┘ └──────────────┘ └──────────────┘  │    │
│  └─────────────────────────────────────────────────────────┘    │
│                              │                                   │
│                              ▼                                   │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │                      业务逻辑层                          │    │
│  │                                                         │    │
│  │  ┌────────────┐ ┌────────────┐ ┌────────────┐        │    │
│  │  │  对话管理   │ │  意图识别   │ │  实体提取   │        │    │
│  │  └────────────┘ └────────────┘ └────────────┘        │    │
│  │                                                         │    │
│  │  ┌────────────┐ ┌────────────┐ ┌────────────┐        │    │
│  │  │  RAG检索   │ │  知识库   │ │  情绪识别   │        │    │
│  │  └────────────┘ └────────────┘ └────────────┘        │    │
│  │                                                         │    │
│  └─────────────────────────────────────────────────────────┘    │
│                              │                                   │
│                              ▼                                   │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │                      模型服务层                          │    │
│  │  ┌────────────┐ ┌────────────┐ ┌────────────┐        │    │
│  │  │ vLLM/LMDeploy│ │  Agent   │ │  Embedding │        │    │
│  │  └────────────┘ └────────────┘ └────────────┘        │    │
│  └─────────────────────────────────────────────────────────┘    │
│                              │                                   │
│                              ▼                                   │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │                      数据存储层                          │    │
│  │  ┌────────┐ ┌────────┐ ┌────────┐ ┌────────┐         │    │
│  │  │会话历史 │ │ 知识库  │ │用户画像 │ │日志分析 │         │    │
│  │  └────────┘ └────────┘ └────────┘ └────────┘         │    │
│  └─────────────────────────────────────────────────────────┘    │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 1.2 核心模块

| 模块 | 功能 | 技术选型 |
|------|------|----------|
| 对话管理 | 多轮对话状态跟踪 | LangChain Memory |
| 意图识别 | 理解用户意图 | 微调 LLM / 规则 |
| RAG 检索 | 产品/FAQ 检索 | 向量数据库 |
| 情绪识别 | 检测用户情绪 | 情感分析模型 |
| 话术生成 | 生成回复 | LLM + Prompt 工程 |

## 2. 快速实现

### 2.1 基础智能客服

```python
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.memory import ConversationBufferMemory

# 初始化 LLM
llm = ChatOpenAI(
    model="gpt-4",
    temperature=0.7,
)

# 定义系统提示词
system_template = """你是一家电商平台的智能客服助手。
你的职责是：
1. 热情、耐心地回答客户问题
2. 了解公司产品信息，帮助客户选购
3. 遇到无法解决的问题，引导客户联系人工客服
4. 保持专业、友好的语气

公司信息：
- 退货政策：7天无理由退货
- 客服时间：9:00-21:00
- 快递：顺丰包邮
"""

# 构建提示词模板
prompt = ChatPromptTemplate.from_messages([
    ("system", system_template),
    ("placeholder", "{chat_history}"),
    ("human", "{input}")
])

# 对话记忆
memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True
)

# 创建对话链
conversation_chain = prompt | llm

# 对话函数
def chat(user_input: str) -> str:
    # 获取历史消息
    chat_history = memory.load_memory_variables({})["chat_history"]
    
    # 调用模型
    response = conversation_chain.invoke({
        "input": user_input,
        "chat_history": chat_history
    })
    
    # 保存对话到记忆
    memory.save_context(
        {"input": user_input},
        {"output": response.content}
    )
    
    return response.content

# 测试
print(chat("你好，我想买一部手机"))
print("-" * 50)
print(chat("有什么推荐吗？"))
print("-" * 50)
print(chat("价格多少？"))
```

### 2.2 带 RAG 的智能客服

```python
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import TextLoader
from langchain.prompts import ChatPromptTemplate
from langchain.memory import ConversationBufferMemory

# 1. 加载知识库
documents = []
# 实际使用中可以加载 PDF、网页等
sample_docs = [
    """
    产品：iPhone 15 Pro
    价格：7999元
    配置：A17 Pro芯片，6.1英寸显示屏，4800万像素摄像头
    特点：钛金属设计，Action按钮，专业相机系统
    """,
    """
    产品：iPhone 15
    价格：5999元
    配置：A16芯片，6.1英寸显示屏，4800万像素摄像头
    特点：灵动岛设计，USB-C接口，升级的相机系统
    """,
    """
    退货政策：
    1. 7天内无理由退货（商品完好）
    2. 15天内质量问题换货
    3. 退货需保留包装和配件
    4. 退款将在收到商品后3-5个工作日内处理
    """
]

from langchain.schema import Document
documents = [Document(page_content=doc) for doc in sample_docs]

# 2. 向量化
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
splits = text_splitter.split_documents(documents)

embeddings = OpenAIEmbeddings()
vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings)
retriever = vectorstore.as_retriever(search_kwargs={"k": 2})

# 3. 构建 RAG 提示词
template = """你是一个专业的电商客服助手。请根据以下知识库信息回答用户问题。

知识库：
{context}

对话历史：
{chat_history}

用户问题：{question}

请给出专业、友好的回答。如果知识库中没有相关信息，请说明并建议用户联系人工客服。
"""

prompt = ChatPromptTemplate.from_template(template)

# 4. 对话链
llm = ChatOpenAI(model="gpt-4", temperature=0.7)
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

def rag_chat(user_input: str) -> str:
    # 检索相关知识
    docs = retriever.get_relevant_documents(user_input)
    context = "\n\n".join([doc.page_content for doc in docs])
    
    # 获取历史
    chat_history = memory.load_memory_variables({})["chat_history"]
    
    # 生成回答
    response = llm.invoke(
        prompt.format(
            context=context,
            chat_history=chat_history,
            question=user_input
        )
    )
    
    # 保存记忆
    memory.save_context({"input": user_input}, {"output": response.content})
    
    return response.content

# 测试
print(rag_chat("iPhone 15 Pro 价格是多少？"))
print("-" * 50)
print(rag_chat("可以退货吗？"))
```

## 3. 意图识别与路由

### 3.1 意图识别系统

```python
from enum import Enum
from typing import List
from pydantic import BaseModel
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate

class Intent(str, Enum):
    PRODUCT_QUERY = "产品查询"
    ORDER_STATUS = "订单状态"
    REFUND = "退款退货"
    COMPLAINT = "投诉建议"
    GREETING = "问候闲聊"
    OTHER = "其他"

# 意图识别
class IntentClassifier:
    def __init__(self):
        self.llm = ChatOpenAI(model="gpt-4", temperature=0)
        
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", """请分析用户消息的意图，只能选择以下一种：
- 产品查询：询问产品价格、配置、功能等
- 订单状态：查询订单物流、发货时间等
- 退款退货：申请退货、退款等
- 投诉建议：对产品或服务不满，提出投诉
- 问候闲聊：打招呼、闲聊等
- 其他：不属于以上类别

直接输出意图类别，不要有其他内容。"""),
            ("human", "{message}")
        ])
        
        self.chain = self.prompt | self.llm
    
    def classify(self, message: str) -> Intent:
        result = self.chain.invoke({"message": message})
        
        # 解析意图
        for intent in Intent:
            if intent.value in result.content:
                return intent
        return Intent.OTHER

# 使用
classifier = IntentClassifier()
print(classifier.classify("iPhone 15 多少钱？"))  # 产品查询
print(classifier.classify("我的订单到哪了？"))    # 订单状态
print(classifier.classify("我要退货"))            # 退款退货
```

### 3.2 智能路由

```python
from typing import Callable, Dict

class Router:
    def __init__(self):
        self.routes: Dict[Intent, Callable] = {}
    
    def register(self, intent: Intent):
        def decorator(func: Callable):
            self.routes[intent] = func
            return func
        return decorator
    
    def route(self, message: str) -> str:
        intent = classifier.classify(message)
        
        handler = self.routes.get(intent)
        if handler:
            return handler(message)
        else:
            return self.routes[Intent.OTHER](message)

# 创建路由
router = Router()

@router.register(Intent.PRODUCT_QUERY)
def handle_product_query(message: str) -> str:
    return rag_chat(message)

@router.register(Intent.ORDER_STATUS)
def handle_order_status(message: str) -> str:
    return "抱歉，订单状态查询功能正在升级中，请稍后再试或联系人工客服。"

@router.register(Intent.REFUND)
def handle_refund(message: str) -> str:
    return "您可以进入'我的订单'页面申请退货，或者告诉我您的订单号，我来帮您处理。"

@router.register(Intent.COMPLAINT)
def handle_complaint(message: str) -> str:
    return "非常抱歉给您带来不好的体验。请告诉我具体情况，我们会尽快为您解决。"

@router.register(Intent.GREETING)
def handle_greeting(message: str) -> str:
    return "您好！很高兴为您服务。请问有什么可以帮您？"

@router.register(Intent.OTHER)
def handle_other(message: str) -> str:
    return "我理解您的需求，让我换个方式帮您..."

# 使用
response = router.route("iPhone 15 Pro 有什么优惠？")
print(response)
```

## 4. 多轮对话管理

### 4.1 带状态机的对话管理

```python
from enum import Enum
from typing import Optional
import json

class DialogState(str, Enum):
    INIT = "初始化"
    GREETING = "问候"
    PRODUCT_QUERY = "产品查询中"
    COLLECTING_INFO = "收集信息"
    RECOMMENDING = "推荐中"
    ORDERING = "下单中"
    RESOLVED = "已解决"

class DialogSession:
    def __init__(self, session_id: str):
        self.session_id = session_id
        self.state = DialogState.INIT
        self.context = {}
        self.history = []
    
    def update_state(self, new_state: DialogState):
        self.state = new_state
    
    def add_context(self, key: str, value: any):
        self.context[key] = value
    
    def add_history(self, role: str, content: str):
        self.history.append({"role": role, "content": content})
    
    def to_dict(self):
        return {
            "session_id": self.session_id,
            "state": self.state.value,
            "context": self.context,
            "history": self.history
        }

# 对话管理器
class DialogManager:
    def __init__(self):
        self.sessions: Dict[str, DialogSession] = {}
        self.classifier = IntentClassifier()
    
    def get_session(self, session_id: str) -> DialogSession:
        if session_id not in self.sessions:
            self.sessions[session_id] = DialogSession(session_id)
        return self.sessions[session_id]
    
    def process_message(self, session_id: str, message: str) -> str:
        session = self.get_session(session_id)
        
        # 意图识别
        intent = self.classifier.classify(message)
        
        # 状态转移逻辑
        if intent == Intent.GREETING:
            session.update_state(DialogState.GREETING)
            return "您好！欢迎光临，请问有什么可以帮您？"
        
        elif intent == Intent.PRODUCT_QUERY:
            session.update_state(DialogState.PRODUCT_QUERY)
            session.add_context("pending_query", message)
            return rag_chat(message)
        
        elif intent == Intent.REFUND:
            session.update_state(DialogState.COLLECTING_INFO)
            session.add_context("intent", "refund")
            return "好的，我来帮您处理退货。请问您的订单号是多少？"
        
        elif session.state == DialogState.COLLECTING_INFO and "intent" in session.context:
            # 收集信息状态
            if "order_id" not in session.context:
                session.add_context("order_id", message)
                return "好的，订单号已记录。请问退货原因是什么？"
            else:
                session.update_state(DialogState.RESOLVED)
                return f"好的，您的退货申请已提交。订单号：{session.context['order_id']}，退货原因：{message}。请保持手机畅通，客服会尽快联系您。"
        
        else:
            return rag_chat(message)

# 使用
manager = DialogManager()
session_id = "user_123"

print(manager.process_message(session_id, "你好"))
print(manager.process_message(session_id, "我想退货"))
print(manager.process_message(session_id, "ORDER123456"))
print(manager.process_message(session_id, "不想要了"))
```

### 4.2 带槽位填充的对话

```python
from typing import Optional, Dict, List
from pydantic import BaseModel

# 槽位定义
class OrderSlot(BaseModel):
    product_name: Optional[str] = None
    quantity: Optional[int] = None
    address: Optional[str] = None
    phone: Optional[str] = None

class SlotFillingDialog:
    required_slots = ["product_name", "quantity", "address", "phone"]
    
    def __init__(self):
        self.slots: Dict[str, OrderSlot] = {}
    
    def process(self, session_id: str, message: str) -> str:
        if session_id not in self.slots:
            self.slots[session_id] = OrderSlot()
        
        slot = self.slots[session_id]
        
        # 检测槽位
        # 实际应用中可以用 NER 或 LLM 提取
        if not slot.product_name and any(word in message for word in ["手机", "电脑", "平板"]):
            slot.product_name = message
            return "好的，请问需要购买多少台？"
        
        if slot.product_name and not slot.quantity:
            try:
                slot.quantity = int(message)
                return "好的，请提供收货地址。"
            except:
                pass
        
        if slot.product_name and slot.quantity and not slot.address:
            slot.address = message
            return "好的，请提供联系电话。"
        
        if slot.address and not slot.phone:
            slot.phone = message
            return self._confirm_order(slot)
        
        return "好的，我来帮您下单。请告诉我想要购买的产品。"
    
    def _confirm_order(self, slot: OrderSlot) -> str:
        return f"""请确认订单信息：
产品：{slot.product_name}
数量：{slot.quantity}
收货地址：{slot.address}
联系电话：{slot.phone}

确认请回复"确认"，修改请回复具体信息。
"""

# 使用
dialog = SlotFillingDialog()
print(dialog.process("user_001", "我想买一部手机"))
print(dialog.process("user_001", "1台"))
print(dialog.process("user_001", "北京市朝阳区xxx"))
print(dialog.process("user_001", "13800138000"))
```

## 5. 情绪识别与安抚

### 5.1 情绪分析

```python
from enum import Enum

class Emotion(str, Enum):
    HAPPY = "开心"
    NEUTRAL = "中性"
    ANGRY = "愤怒"
    SAD = "悲伤"
    ANXIOUS = "焦虑"

class EmotionDetector:
    def __init__(self):
        self.llm = ChatOpenAI(model="gpt-4", temperature=0)
    
    def detect(self, message: str) -> Emotion:
        prompt = f"""分析用户消息的情绪类别，只能选择以下一种：
- 开心：用户表达满意、高兴
- 中性：用户普通询问，无明显情绪
- 愤怒：用户表达不满、生气
- 悲伤：用户表达难过、失望
- 焦虑：用户表达担忧、急切

用户消息：{message}

直接输出情绪类别。
"""
        result = self.llm.invoke(prompt)
        
        for emotion in Emotion:
            if emotion.value in result.content:
                return emotion
        return Emotion.NEUTRAL

# 使用
detector = EmotionDetector()
print(detector.detect("太棒了，这正是我想要的！"))  # 开心
print(detect("我已经等了三天了，怎么还没发货！"))   # 愤怒
print(detector.detect("请问这个产品怎么样？"))      # 中性
```

### 5.2 情绪安抚策略

```python
class EmpatheticResponse:
    def __init__(self):
        self.llm = ChatOpenAI(model="gpt-4", temperature=0.7)
    
    def generate_response(self, message: str, emotion: Emotion) -> str:
        empathy_prompts = {
            Emotion.HAPPY: "用户表达开心，请给出积极热情的回应。",
            Emotion.NEUTRAL: "请给出专业、友好的回应。",
            Emotion.ANGRY: "用户很生气，请先表达歉意和理解，然后积极解决问题。",
            Emotion.SAD: "用户很难过，请表达关心和安慰。",
            Emotion.ANXIOUS: "用户很焦虑，请给出明确的时间节点和解决方案。",
        }
        
        prompt = f"""{empathy_prompts[emotion]}

用户消息：{message}

请根据情绪生成合适的回复。
"""
        
        return self.llm.invoke(prompt).content

# 使用
emotion_detector = EmotionDetector()
emotion_response = EmpatheticResponse()

message = "我已经等了三天了，怎么还没发货！"
emotion = emotion_detector.detect(message)
response = emotion_response.generate_response(message, emotion)

print(f"用户情绪: {emotion.value}")
print(f"回复: {response}")
```

## 6. 完整案例：电商客服机器人

### 6.1 项目结构

```
customer_service/
├── app/
│   ├── __init__.py
│   ├── main.py              # FastAPI 主入口
│   ├── config.py           # 配置
│   ├── chains/
│   │   ├── __init__.py
│   │   ├── chat_chain.py    # 对话链
│   │   └── rag_chain.py     # RAG 链
│   ├── services/
│   │   ├── __init__.py
│   │   ├── intent.py        # 意图识别
│   │   ├── emotion.py       # 情绪识别
│   │   └── dialog.py        # 对话管理
│   └── models/
│       └── schemas.py       # 数据模型
├── knowledge/
│   └── product_kb.txt      # 产品知识库
├── requirements.txt
└── run.sh
```

### 6.2 核心代码

```python
# app/main.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional
import uvicorn

from app.chains.chat_chain import CustomerServiceChain

app = FastAPI(title="智能客服系统")
cs_chain = CustomerServiceChain()

class ChatRequest(BaseModel):
    session_id: str
    message: str

class ChatResponse(BaseModel):
    response: str
    intent: str
    emotion: str

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    result = cs_chain.process(
        session_id=request.session_id,
        message=request.message
    )
    return ChatResponse(**result)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

```python
# app/chains/chat_chain.py
from typing import Dict
from app.services.intent import IntentClassifier
from app.services.emotion import EmotionDetector
from app.services.dialog import DialogManager
from app.chains.rag_chain import RAGChain

class CustomerServiceChain:
    def __init__(self):
        self.intent_classifier = IntentClassifier()
        self.emotion_detector = EmotionDetector()
        self.dialog_manager = DialogManager()
        self.rag_chain = RAGChain()
    
    def process(self, session_id: str, message: str) -> Dict:
        # 1. 情绪识别
        emotion = self.emotion_detector.detect(message)
        
        # 2. 意图识别
        intent = self.intent_classifier.classify(message)
        
        # 3. 对话管理
        response = self.dialog_manager.process(session_id, message, intent, self.rag_chain)
        
        # 4. 情绪适配
        if emotion.value in ["愤怒", "悲伤", "焦虑"]:
            response = self._empathetic_response(response, emotion)
        
        return {
            "response": response,
            "intent": intent.value,
            "emotion": emotion.value
        }
    
    def _empathetic_response(self, response: str, emotion) -> str:
        # 对负面情绪添加额外安抚
        if emotion.value == "愤怒":
            return f"非常抱歉给您带来困扰。{response}"
        elif emotion.value == "悲伤":
            return f"我理解您的心情。{response}"
        elif emotion.value == "焦虑":
            return f"请放心，我会尽快帮您处理。{response}"
        return response
```

### 6.3 部署配置

```yaml
# docker-compose.yml
version: '3.8'

services:
  customer-service:
    build: .
    ports:
      - "8000:8000"
    environment:
      - OPENAI_API_KEY=${OPENAI_API_KEY}
      - OPENAI_API_BASE=${OPENAI_API_BASE}
    volumes:
      - ./knowledge:/app/knowledge
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]

  # 向量数据库
  chroma:
    image: chromadb/chroma
    ports:
      - "8001:8000"
    volumes:
      - chroma_data:/chroma/chroma

volumes:
  chroma_data:
```

## 7. 性能优化与监控

### 7.1 缓存策略

```python
from functools import lru_cache
import hashlib
import json

class ResponseCache:
    def __init__(self, max_size=1000):
        self.cache = {}
        self.max_size = max_size
    
    def _hash(self, message: str) -> str:
        return hashlib.md5(message.encode()).hexdigest()
    
    def get(self, message: str) -> Optional[str]:
        key = self._hash(message)
        return self.cache.get(key)
    
    def set(self, message: str, response: str):
        if len(self.cache) >= self.max_size:
            # 简单的 LRU 实现：删除第一个
            self.cache.pop(next(iter(self.cache)))
        
        key = self._hash(message)
        self.cache[key] = response

# 使用
cache = ResponseCache()

def chat_with_cache(session_id: str, message: str) -> str:
    # 先检查缓存
    cached = cache.get(message)
    if cached:
        return cached
    
    # 正常处理
    response = cs_chain.process(session_id, message)["response"]
    
    # 存入缓存
    cache.set(message, response)
    
    return response
```

### 7.2 监控指标

```python
# 监控指标
METRICS = {
    "total_requests": 0,
    "successful_requests": 0,
    "failed_requests": 0,
    "avg_response_time": 0,
    "intent_distribution": {},
    "emotion_distribution": {},
}

def record_metrics(result: dict, response_time: float):
    METRICS["total_requests"] += 1
    METRICS["successful_requests"] += 1
    
    # 意图分布
    intent = result.get("intent", "unknown")
    METRICS["intent_distribution"][intent] = METRICS["intent_distribution"].get(intent, 0) + 1
    
    # 情绪分布
    emotion = result.get("emotion", "unknown")
    METRICS["emotion_distribution"][emotion] = METRICS["emotion_distribution"].get(emotion, 0) + 1
    
    # 平均响应时间
    METRICS["avg_response_time"] = (
        METRICS["avg_response_time"] * (METRICS["total_requests"] - 1) + response_time
    ) / METRICS["total_requests"]
```

## 8. 总结

本章我们详细介绍了智能客服系统的构建：

1. **系统架构**：分层设计，包括用户渠道、网关、业务逻辑、模型服务等层次
2. **基础实现**：基于 LangChain 的对话系统
3. **RAG 集成**：结合知识库的问答系统
4. **意图识别**：分类用户意图并智能路由
5. **对话管理**：多轮对话状态机、槽位填充
6. **情绪识别**：检测用户情绪并做出同理心回应
7. **完整案例**：电商客服机器人的实现
8. **性能优化**：缓存策略、监控指标

智能客服是企业级 LLM 应用最成熟的场景之一，通过合理设计可以显著提升客户满意度和服务效率。
