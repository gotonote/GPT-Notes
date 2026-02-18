# 实战案例：AI Agent 智能体开发实战

> 本文将详细介绍如何使用 LangChain 和最新的大模型技术，从零构建一个智能 AI Agent（智能体）。

---

## 📋 案例概述

### 场景
企业需要一个能够自动完成复杂任务的 AI Agent，具备：
- 多轮对话能力
- 工具调用能力（调用外部 API、搜索、计算等）
- 任务规划与分解能力
- 记忆与上下文保持能力

### 技术栈
- **大模型**：Claude 4 / GPT-4o / 通义千问
- **开发语言**：Python 3.10+
- **框架**：LangChain + LangGraph
- **工具**：Tavily 搜索、Python REPL、文件操作

---

## 🏗️ 系统架构

```
┌─────────────────────────────────────────────────────────┐
│                      AI Agent                           │
├─────────────────────────────────────────────────────────┤
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐ │
│  │   规划器    │───▶│   执行器    │───▶│   工具集    │ │
│  │  (Planner)  │    │  (Executor) │    │  (Tools)    │ │
│  └─────────────┘    └─────────────┘    └─────────────┘ │
│         │                  │                  │        │
│         └──────────────────┼──────────────────┘        │
│                            ▼                            │
│                   ┌─────────────┐                       │
│                   │   记忆模块   │                       │
│                   │  (Memory)   │                       │
│                   └─────────────┘                       │
└─────────────────────────────────────────────────────────┘
```

---

## 🚀 第一步：环境准备

### 1.1 安装依赖

```bash
# 创建虚拟环境
python -m venv agent-env
source agent-env/bin/activate  # Linux/Mac
# agent-env\Scripts\activate   # Windows

# 安装 LangChain 核心包
pip install langchain langchain-core langchain-anthropic langchain-openai

# 安装工具依赖
pip install tavily-python langchain-community

# 安装 LangGraph（新一代 Agent 框架）
pip install langgraph
```

### 1.2 配置 API 密钥

创建 `.env` 文件：

```env
# Anthropic Claude
ANTHROPIC_API_KEY=your_anthropic_key

# OpenAI
OPENAI_API_KEY=your_openai_key

# 搜索工具
TAVILY_API_KEY=your_tavily_key
```

---

## 📝 第二步：理解 LangChain Agent 核心概念

### 2.1 什么是 Agent？

Agent（智能体）是能够自主决策、执行复杂任务的人工智能系统。与简单的 LLM 调用不同，Agent 具有：

| 特性 | 描述 |
|------|------|
| **自主性** | 能够自主决定下一步行动 |
| **工具使用** | 可以调用外部工具完成任务 |
| **规划能力** | 将复杂任务分解为步骤 |
| **反思能力** | 能够评估和修正自己的行为 |

### 2.2 Agent 的工作流程

```
用户输入 → 理解意图 → 规划步骤 → 执行工具 → 评估结果 → 反馈输出
```

---

## 🔧 第三步：构建基础 Agent

### 3.1 简单对话 Agent

```python
# basic_agent.py
import os
from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain.agents import create_agent
from langchain.tools import Tool

load_dotenv()

# 初始化模型
llm = ChatAnthropic(
    model="claude-sonnet-4-20250514",
    anthropic_api_key=os.getenv("ANTHROPIC_API_KEY")
)

# 定义简单工具
def get_current_time():
    """获取当前时间"""
    from datetime import datetime
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

# 创建工具列表
tools = [
    Tool(
        name="get_time",
        func=get_current_time,
        description="获取当前日期和时间"
    )
]

# 创建 Agent
agent = create_agent(
    llm,
    tools,
    system_prompt="你是一个有用的助手，可以帮助用户完成各种任务。"
)

# 运行 Agent
result = agent.invoke({
    "messages": [("user", "现在几点了？")]
})

print(result["messages"][-1].content)
```

### 3.2 带搜索功能的 Agent

```python
# search_agent.py
import os
from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain.agents import create_agent
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain.tools import Tool

load_dotenv()

# 初始化模型
llm = ChatAnthropic(
    model="claude-sonnet-4-20250514",
    anthropic_api_key=os.getenv("ANTHROPIC_API_KEY")
)

# 创建搜索工具
search = TavilySearchResults(max_results=3)
search_tool = Tool(
    name="web_search",
    func=search.invoke,
    description="搜索最新的信息，用于回答实时问题"
)

# 创建 Agent
agent = create_agent(
    llm,
    [search_tool],
    system_prompt="""你是一个研究助手，擅长查找和分析信息。
    
    当用户询问实时信息或你不确定的问题时，请使用搜索工具查找最新信息。
    提供准确、全面的回答。"""
)

# 运行 Agent
result = agent.invoke({
    "messages": [("user", "2025年AI领域最重要的技术突破是什么？")]
})

print(result["messages"][-1].content)
```

---

## 🧠 第四步：使用 LangGraph 构建高级 Agent

### 4.1 LangGraph 核心概念

LangGraph 是 LangChain 的新一代框架，专门用于构建有状态、多步骤的 Agent 应用。

```python
# langgraph_agent.py
from langgraph.graph import StateGraph, END
from langchain_anthropic import ChatAnthropic
from langchain_community.tools.tavily_search import TavilySearchResults
from typing import TypedDict, Annotated
import operator

# 定义状态
class AgentState(TypedDict):
    messages: list
    next_action: str
    search_results: list

# 初始化组件
llm = ChatAnthropic(model="claude-sonnet-4-20250514")
search = TavilySearchResults(max_results=3)

# 定义节点
def should_search(state: AgentState) -> str:
    """决定是否需要搜索"""
    last_message = state["messages"][-1]
    if "?" in last_message.content or "最新" in last_message.content:
        return "search"
    return "respond"

def search_node(state: AgentState):
    """执行搜索"""
    query = state["messages"][-1].content
    results = search.invoke(query)
    return {"search_results": [results]}

def respond_node(state: AgentState):
    """生成回复"""
    response = llm.invoke(state["messages"])
    return {"messages": [response]}

# 构建图
workflow = StateGraph(AgentState)

# 添加节点
workflow.add_node("search", search_node)
workflow.add_node("respond", respond_node)

# 添加边
workflow.set_entry_point("respond")
workflow.add_conditional_edges(
    "respond",
    should_search,
    {
        "search": "search",
        "respond": END
    }
)
workflow.add_edge("search", "respond")

# 编译图
graph = workflow.compile()
```

### 4.2 带记忆的 Agent

```python
# memory_agent.py
from langgraph.graph import StateGraph, END
from langchain_anthropic import ChatAnthropic
from langchain.memory import ConversationBufferMemory
from typing import TypedDict

class AgentState(TypedDict):
    messages: list
    memory: dict

llm = ChatAnthropic(model="claude-sonnet-4-20250514")

# 简单记忆实现
memory_store = {}

def chat_node(state: AgentState):
    """对话节点"""
    # 获取历史消息
    history = memory_store.get("history", [])
    full_messages = history + state["messages"]
    
    # 调用 LLM
    response = llm.invoke(full_messages)
    
    # 更新记忆
    memory_store["history"] = full_messages + [response]
    
    return {"messages": [response]}

# 使用 LangChain 的记忆组件
from langgraph.checkpoint.memory import MemorySaver

# 创建带持久化的图
checkpointer = MemorySaver()
workflow = StateGraph(AgentState)
workflow.add_node("chat", chat_node)
workflow.set_entry_point("chat")
workflow.add_edge("chat", END)

graph = workflow.compile(checkpointer=checkpointer)

# 运行（支持多轮对话）
config = {"configurable": {"thread_id": "user_123"}}
result = graph.invoke({"messages": [("user", "我叫张三")]}, config)
result = graph.invoke({"messages": [("user", "我叫什么名字？")]}, config)
```

---

## 🔨 第五步：构建多功能工具 Agent

### 5.1 工具定义

```python
# tools.py
from langchain.tools import tool
from datetime import datetime
import math

@tool
def calculate(expression: str) -> str:
    """执行数学计算
    
    Args:
        expression: 数学表达式，如 "2+3*5"
    Returns:
        计算结果
    """
    try:
        result = eval(expression, {"__builtins__": {}}, {"math": math})
        return f"计算结果: {result}"
    except Exception as e:
        return f"计算错误: {str(e)}"

@tool
def get_weather(city: str) -> str:
    """获取城市天气
    
    Args:
        city: 城市名称，如 "北京"、"上海"
    Returns:
        天气信息
    """
    # 实际项目中可以调用天气 API
    weather_data = {
        "北京": "晴，15-25°C",
        "上海": "多云，18-27°C",
        "广州": "雷阵雨，24-32°C"
    }
    return weather_data.get(city, f"未找到{city}的天气信息")

@tool
def send_email(to: str, subject: str, body: str) -> str:
    """发送邮件
    
    Args:
        to: 收件人邮箱
        subject: 邮件主题
        body: 邮件正文
    Returns:
        发送结果
    """
    # 实际项目中需要集成邮件服务
    print(f"发送邮件到 {to}")
    print(f"主题: {subject}")
    print(f"内容: {body}")
    return f"邮件已发送至 {to}"

@tool
def read_file(filename: str) -> str:
    """读取文件内容
    
    Args:
        filename: 文件路径
    Returns:
        文件内容
    """
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            return f.read()
    except Exception as e:
        return f"读取失败: {str(e)}"
```

### 5.2 整合所有工具

```python
# tool_agent.py
from langchain_anthropic import ChatAnthropic
from langchain.agents import create_agent
from tools import calculate, get_weather, send_email, read_file
import os
from dotenv import load_dotenv

load_dotenv()

llm = ChatAnthropic(model="claude-sonnet-4-20250514")

# 收集所有工具
tools = [calculate, get_weather, send_email, read_file]

# 创建 Agent
agent = create_agent(
    llm,
    tools,
    system_prompt="""你是一个多功能助手，可以使用各种工具帮助用户。
    
    可用工具：
    - calculate: 数学计算
    - get_weather: 查询天气
    - send_email: 发送邮件
    - read_file: 读取文件
    
    根据用户需求选择合适的工具完成任务。"""
)

# 测试各种工具
test_queries = [
    "计算 123 * 456 的结果",
    "北京今天天气怎么样？",
    "帮我读取 test.txt 文件",
    "给 test@example.com 发送一封主题为'测试'的邮件"
]

for query in test_queries:
    print(f"\n用户: {query}")
    result = agent.invoke({"messages": [("user", query)]})
    print(f"助手: {result['messages'][-1].content}")
```

---

## 🖥️ 第六步：构建交互式 Web Agent

### 6.1 Flask API 服务

```python
# app.py
from flask import Flask, request, jsonify
from flask_cors import CORS
from langchain_anthropic import ChatAnthropic
from langchain.agents import create_agent
from langchain.memory import ConversationBufferMemory
import os
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)
CORS(app)

# 初始化
llm = ChatAnthropic(model="claude-sonnet-4-20250514")
memory = ConversationBufferMemory(return_messages=True)

# Agent 创建函数
def create_session_agent():
    return create_agent(
        llm,
        [calculate, get_weather],  # 工具列表
        memory=memory,
        system_prompt="你是一个友好的AI助手。"
    )

# 会话存储
sessions = {}

@app.route("/chat", methods=["POST"])
def chat():
    data = request.json
    session_id = data.get("session_id", "default")
    message = data.get("message", "")
    
    # 获取或创建会话
    if session_id not in sessions:
        sessions[session_id] = {
            "memory": ConversationBufferMemory(return_messages=True),
            "agent": None
        }
    
    session = sessions[session_id]
    
    # 创建 Agent（带记忆）
    if session["agent"] is None:
        from langchain.agents import create_agent
        session["agent"] = create_agent(
            llm,
            [calculate, get_weather],
            memory=session["memory"],
            system_prompt="你是一个友好的AI助手，擅长帮助用户解决问题。"
        )
    
    # 调用 Agent
    result = session["agent"].invoke({
        "input": message
    })
    
    return jsonify({
        "response": result["output"],
        "session_id": session_id
    })

@app.route("/clear", methods=["POST"])
def clear_session():
    session_id = request.json.get("session_id", "default")
    if session_id in sessions:
        del sessions[session_id]
    return jsonify({"status": "cleared"})

if __name__ == "__main__":
    app.run(debug=True, port=5000)
```

### 6.2 前端示例

```html
<!DOCTYPE html>
<html>
<head>
    <title>AI Agent 聊天</title>
    <style>
        body { font-family: Arial, sans-serif; max-width: 800px; margin: 50px auto; }
        #chat-box { height: 400px; border: 1px solid #ccc; overflow-y: auto; padding: 20px; }
        #input-area { display: flex; gap: 10px; margin-top: 20px; }
        input { flex: 1; padding: 10px; }
        button { padding: 10px 20px; background: #007bff; color: white; border: none; cursor: pointer; }
        .message { margin: 10px 0; }
        .user { color: #007bff; }
        .assistant { color: #28a745; }
    </style>
</head>
<body>
    <h1>🤖 AI Agent 助手</h1>
    <div id="chat-box"></div>
    <div id="input-area">
        <input type="text" id="message" placeholder="输入消息..." onkeypress="handleKeyPress(event)">
        <button onclick="sendMessage()">发送</button>
    </div>

    <script>
        let sessionId = 'session_' + Date.now();
        
        async function sendMessage() {
            const input = document.getElementById('message');
            const message = input.value;
            if (!message) return;
            
            addMessage('user', message);
            input.value = '';
            
            const response = await fetch('/chat', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({ message, session_id: sessionId })
            });
            
            const data = await response.json();
            addMessage('assistant', data.response);
        }
        
        function addMessage(role, content) {
            const chatBox = document.getElementById('chat-box');
            chatBox.innerHTML += `<div class="message ${role}"><strong>${role === 'user' ? '你' : '助手'}:</strong> ${content}</div>`;
            chatBox.scrollTop = chatBox.scrollHeight;
        }
        
        function handleKeyPress(event) {
            if (event.key === 'Enter') sendMessage();
        }
    </script>
</body>
</html>
```

---

## 📦 第七步：部署与优化

### 7.1 Docker 部署

```dockerfile
# Dockerfile
FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

EXPOSE 5000

CMD ["python", "app.py"]
```

```yaml
# docker-compose.yml
version: '3.8'
services:
  agent-api:
    build: .
    ports:
      - "5000:5000"
    environment:
      - ANTHROPIC_API_KEY=${ANTHROPIC_API_KEY}
      - TAVILY_API_KEY=${TAVILY_API_KEY}
    volumes:
      - ./data:/app/data
```

### 7.2 性能优化技巧

| 优化方向 | 具体方法 |
|----------|----------|
| **响应速度** | 使用流式输出 (stream=True)、添加缓存 |
| **成本控制** | 合理设置 max_tokens、使用更小的模型处理简单任务 |
| **稳定性** | 添加重试机制、错误处理、限流保护 |
| **准确性** | 优化提示词、添加Few-shot示例 |

```python
# 优化示例：流式输出
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

agent = create_agent(
    llm,
    tools,
    streaming=True,
    callbacks=[StreamingStdOutCallbackHandler()]
)

# 优化示例：重试机制
from tenacity import retry, stop_after_attempt, wait_exponential

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
def call_agent_with_retry(agent, message):
    return agent.invoke(message)
```

---

## 📚 总结

本案例展示了如何使用 LangChain 和 LangGraph 构建功能强大的 AI Agent：

1. **基础概念**：理解 Agent 的核心特性
2. **工具集成**：为 Agent 配备各种能力
3. **记忆系统**：实现多轮对话
4. **高级架构**：使用 LangGraph 构建复杂工作流
5. **部署上线**：Docker 容器化部署

通过本案例的学习，您应该能够：
- 使用 LangChain 创建基础 Agent
- 为 Agent 添加自定义工具
- 实现对话记忆功能
- 构建完整的 Web 服务

---

## 🔗 延伸阅读

- [LangChain 官方文档](https://python.langchain.com/docs/introduction/)
- [LangGraph 官方文档](https://langchain-ai.github.io/langgraph/)
- [Anthropic Claude API 文档](https://docs.anthropic.com/)

---

> 📝 **编写者**: GPT-Notes 团队  
> 📅 **更新日期**: 2026年2月  
> ⭐ **如果你觉得有帮助，欢迎提交改进建议！**
