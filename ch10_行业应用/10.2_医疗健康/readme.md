# 医疗健康 AI 应用

## 概述

医疗健康是 LLM 应用的重要领域。本章将介绍如何构建医疗健康 AI 系统，包括健康咨询、症状分析、药物助手、医学知识问答等应用场景，同时强调医疗 AI 的合规性和安全性要求。

## 1. 医疗 AI 系统架构

### 1.1 系统架构图

```
┌─────────────────────────────────────────────────────────────────┐
│                    医疗健康 AI 系统架构                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │                      用户交互层                          │    │
│  │  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐  │    │
│  │  │  健康App  │ │  微信小程序│ │  Web端   │ │  语音助手 │  │    │
│  │  └──────────┘ └──────────┘ └──────────┘ └──────────┘  │    │
│  └─────────────────────────────────────────────────────────┘    │
│                              │                                   │
│                              ▼                                   │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │                      安全层                              │    │
│  │  ┌──────────────┐ ┌──────────────┐ ┌──────────────┐  │    │
│  │  │  身份认证    │ │  隐私保护    │ │  风险提示    │  │    │
│  │  └──────────────┘ └──────────────┘ └──────────────┘  │    │
│  └─────────────────────────────────────────────────────────┘    │
│                              │                                   │
│                              ▼                                   │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │                      业务层                              │    │
│  │                                                         │    │
│  │  ┌────────────┐ ┌────────────┐ ┌────────────┐        │    │
│  │  │ 健康咨询   │ │ 症状分析   │ │ 药物助手   │        │    │
│  │  └────────────┘ └────────────┘ └────────────┘        │    │
│  │                                                         │    │
│  │  ┌────────────┐ ┌────────────┐ ┌────────────┐        │    │
│  │  │ 医学知识库 │ │ 用药提醒   │ │ 预约导诊   │        │    │
│  │  └────────────┘ └────────────┘ └────────────┘        │    │
│  │                                                         │    │
│  └─────────────────────────────────────────────────────────┘    │
│                              │                                   │
│                              ▼                                   │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │                      模型层                              │    │
│  │  ┌────────────┐ ┌────────────┐ ┌────────────┐        │    │
│  │  │  医疗 LLM  │ │  Embedding │ │  RAG 检索  │        │    │
│  │  └────────────┘ └────────────┘ └────────────┘        │    │
│  └─────────────────────────────────────────────────────────┘    │
│                              │                                   │
│                              ▼                                   │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │                      数据层                              │    │
│  │  ┌────────┐ ┌────────┐ ┌────────┐ ┌────────┐         │    │
│  │  │医学文献 │ │ 药品库  │ │ 病历库  │ │健康档案 │         │    │
│  │  └────────┘ └────────┘ └────────┘ └────────┘         │    │
│  └─────────────────────────────────────────────────────────┘    │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 1.2 合规性要求

```
┌─────────────────────────────────────────────────────────────────┐
│                    医疗 AI 合规性要求                            │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ⚠️ 重要声明                                                    │
│  ─────────────────────────────────────────────────────────────  │
│  医疗 AI 系统必须遵守以下原则：                                   │
│                                                                 │
│  1. 辅助决策而非诊断                                             │
│     AI 提供的是健康信息参考，不能替代医生诊断                     │
│                                                                 │
│  2. 明确免责声明                                                │
│     始终告知用户 AI 不能替代专业医疗建议                          │
│                                                                 │
│  3. 数据隐私保护                                                │
│     符合 HIPAA、GDPR 等医疗数据保护法规                          │
│                                                                 │
│  4. 安全性优先                                                  │
│     对涉及生命安全的建议，设置人工复核环节                        │
│                                                                 │
│  5. 可追溯性                                                    │
│     记录所有建议的生成过程，便于审计                             │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

## 2. 健康咨询助手

### 2.1 系统设计

```python
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.memory import ConversationBufferMemory
from pydantic import BaseModel
from typing import Optional, List
from enum import Enum

class HealthDomain(str, Enum):
    GENERAL = " general_health"
    NUTRITION = "nutrition"
    EXERCISE = "exercise"
    MENTAL = "mental_health"
    SLEEP = "sleep"
    CHRONIC = "chronic_disease"

class HealthConsultant:
    def __init__(self):
        self.llm = ChatOpenAI(model="gpt-4", temperature=0.7)
        
        # 系统提示词
        self.system_prompt = """你是一位专业的健康顾问助手。你的职责是：
        
1. 提供基于证据的健康信息和建议
2. 帮助用户了解健康生活方式
3. 解释医学术语和概念
4. 提醒用户何时需要寻求专业医疗帮助

⚠️ 重要声明：
- 你提供的仅是健康信息参考，不能替代医生诊断
- 如果用户描述的症状可能严重，请立即建议就医
- 不要为急诊情况提供建议，立刻让用户拨打急救电话

请用友好、专业的方式回复。"""

        self.prompt = ChatPromptTemplate.from_messages([
            ("system", self.system_prompt),
            ("placeholder", "{chat_history}"),
            ("human", "{input}")
        ])
        
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True
        )
        
        self.chain = self.prompt | self.llm
    
    def chat(self, user_input: str) -> str:
        # 获取历史
        chat_history = self.memory.load_memory_variables({})["chat_history"]
        
        # 生成回复
        response = self.chain.invoke({
            "input": user_input,
            "chat_history": chat_history
        })
        
        # 保存记忆
        self.memory.save_context(
            {"input": user_input},
            {"output": response.content}
        )
        
        return response.content
    
    def reset(self):
        """重置对话"""
        self.memory.clear()

# 使用
consultant = HealthConsultant()
print(consultant.chat("什么是健康的饮食方式？"))
print("-" * 50)
print(consultant.chat("我应该如何开始运动？"))
```

### 2.2 带安全检查的健康咨询

```python
from typing import List, Tuple
import re

class SafeHealthConsultant:
    # 危险信号关键词
    DANGER_KEYWORDS = [
        "胸痛", "呼吸困难", "大出血", "意识不清", 
        "中毒", "自杀", "严重过敏", "休克"
    ]
    
    # 需要就医的关键词
    SEE_DOCTOR_KEYWORDS = [
        "发烧", "持续疼痛", "体重骤降", "异常出血",
        "长期咳嗽", "头痛", "眩晕", "皮疹"
    ]
    
    def __init__(self):
        self.consultant = HealthConsultant()
    
    def check_danger(self, message: str) -> Tuple[bool, str]:
        """检查是否涉及紧急情况"""
        message = message.lower()
        
        for keyword in self.DANGER_KEYWORDS:
            if keyword in message:
                return True, self._get_emergency_response(keyword)
        
        return False, ""
    
    def check_see_doctor(self, message: str) -> str:
        """检查是否建议就医"""
        for keyword in self.SEE_DOCTOR_KEYWORDS:
            if keyword in message:
                return self._get_see_doctor_suggestion(keyword)
        return ""
    
    def _get_emergency_response(self, keyword: str) -> str:
        return f"""
⚠️ 紧急提示

您描述的情况可能涉及紧急医疗状况，请立即：
1. 拨打 120 急救电话
2. 或前往最近医院的急诊科

{keyword} 可能预示严重健康问题，请不要延迟，立即寻求医疗帮助！
"""
    
    def _get_see_doctor_suggestion(self, keyword: str) -> str:
        return f"""
💡 建议：您描述的情况涉及 {keyword}，建议您：
- 尽快预约医生进行面诊
- 如症状持续或加重，立即就医
- 记录症状出现的时间、频率和伴随因素，供医生参考

本建议仅供参考，不能替代专业医疗诊断。"""

    def chat(self, user_input: str) -> str:
        # 1. 检查紧急情况
        is_danger, danger_response = self.check_danger(user_input)
        if is_danger:
            return danger_response
        
        # 2. 正常健康咨询
        response = self.consultant.chat(user_input)
        
        # 3. 检查是否需要添加就医建议
        see_doctor = self.check_see_doctor(user_input)
        if see_doctor:
            response += see_doctor
        
        # 4. 添加通用免责声明
        response += "\n\n⚠️ 本回答仅供参考，不能替代专业医疗建议。如有疑虑，请咨询医生。"
        
        return response

# 使用
safe_consultant = SafeHealthConsultant()

# 紧急情况
print(safe_consultant.chat("我突然胸口很痛，呼吸困难"))
print("=" * 50)

# 普通咨询
print(safe_consultant.chat("每天走10000步对身体好吗？"))
```

## 3. 症状分析助手

### 3.1 症状分析 Chain

```python
from typing import Optional, List, Dict
from pydantic import BaseModel
from enum import Enum

class UrgencyLevel(str, Enum):
    EMERGENCY = "emergency"      # 立即就医/拨打120
    URGENT = "urgent"           # 尽快就医（24小时内）
    ROUTINE = "routine"         # 常规就医
    SELF_CARE = "self_care"    # 可以在家观察

class SymptomAnalysis(BaseModel):
    possible_conditions: List[Dict]
    urgency: UrgencyLevel
    recommendations: List[str]
    see_doctor_urgency: str

class SymptomAnalyzer:
    def __init__(self):
        self.llm = ChatOpenAI(model="gpt-4", temperature=0)
        
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", """你是一个症状分析助手。请根据用户描述的症状进行分析。

重要规则：
1. 只提供可能性分析，不能诊断
2. 严重症状必须提示立即就医
3. 提供时给出可信度评估
4. 始终建议咨询专业医生

请以JSON格式输出：
{
    "possible_conditions": [{"condition": "疾病名", "likelihood": "high/medium/low", "reason": "原因"}],
    "urgency": "emergency/urgent/routine/self_care",
    "recommendations": ["建议1", "建议2"],
    "see_doctor_urgency": "就医紧急程度说明"
}"""),
            ("human", "症状描述：{symptoms}\n持续时间：{duration}\n伴随症状：{accompanying}")
        ])
        
        self.chain = self.prompt | self.llm
    
    def analyze(self, symptoms: str, duration: str = "未知", 
                accompanying: str = "无") -> SymptomAnalysis:
        result = self.chain.invoke({
            "symptoms": symptoms,
            "duration": duration,
            "accompanying": accompanying
        })
        
        # 解析 JSON（实际使用中需要更健壮的解析）
        import json
        try:
            data = json.loads(result.content)
            return SymptomAnalysis(**data)
        except:
            return SymptomAnalysis(
                possible_conditions=[],
                urgency=UrgencyLevel.ROUTINE,
                recommendations=["请咨询专业医生"],
                see_doctor_urgency="请咨询医生获取准确诊断"
            )

# 使用
analyzer = SymptomAnalyzer()
result = analyzer.analyze(
    symptoms="头痛",
    duration="2天",
    accompanying="轻微发热，乏力"
)

print(f"紧急程度: {result.urgency.value}")
print("可能的状况:")
for condition in result.possible_conditions:
    print(f"  - {condition['condition']} (可信度: {condition['likelihood']})")
print("建议:", result.recommendations)
print("就医建议:", result.see_doctor_urgency)
```

### 3.2 症状追踪

```python
from datetime import datetime, timedelta
from typing import Dict, List
import json

class SymptomTracker:
    def __init__(self, user_id: str):
        self.user_id = user_id
        self.symptoms: List[Dict] = []
    
    def log_symptom(self, symptom: str, severity: int, 
                    notes: str = "", medication: str = ""):
        """记录症状"""
        self.symptoms.append({
            "timestamp": datetime.now().isoformat(),
            "symptom": symptom,
            "severity": severity,  # 1-10
            "notes": notes,
            "medication": medication
        })
    
    def get_symptom_summary(self) -> str:
        """生成症状摘要"""
        if not self.symptoms:
            return "暂无症状记录"
        
        summary = "📊 症状摘要\n\n"
        
        # 按症状类型分组
        symptom_types = {}
        for record in self.symptoms:
            s = record["symptom"]
            if s not in symptom_types:
                symptom_types[s] = []
            symptom_types[s].append(record)
        
        for symptom, records in symptom_types.items():
            avg_severity = sum(r["severity"] for r in records) / len(records)
            first_date = records[0]["timestamp"][:10]
            last_date = records[-1]["timestamp"][:10]
            
            summary += f"• {symptom}\n"
            summary += f"  记录次数: {len(records)}\n"
            summary += f"  平均严重程度: {avg_severity:.1f}/10\n"
            summary += f"  持续时间: {first_date} ~ {last_date}\n\n"
        
        return summary
    
    def export_for_doctor(self) -> str:
        """导出给医生的报告"""
        if not self.symptoms:
            return "暂无症状记录可导出"
        
        report = "=" * 50 + "\n"
        report += "患者症状记录报告\n"
        report += f"患者ID: {self.user_id}\n"
        report += f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n"
        report += "=" * 50 + "\n\n"
        
        report += self.get_symptom_summary()
        
        report += "\n详细记录:\n"
        for i, record in enumerate(self.symptoms, 1):
            report += f"\n{i}. {record['timestamp'][:19]}\n"
            report += f"   症状: {record['symptom']}\n"
            report += f"   严重程度: {record['severity']}/10\n"
            if record['notes']:
                report += f"   备注: {record['notes']}\n"
            if record['medication']:
                report += f"   用药: {record['medication']}\n"
        
        return report

# 使用
tracker = SymptomTracker("user_001")

# 记录症状
tracker.log_symptom("头痛", 6, notes="可能是睡眠不足", medication="布洛芬")
tracker.log_symptom("头痛", 5, notes="轻微缓解", medication="布洛芬")
tracker.log_symptom("疲劳", 4, notes="最近工作压力大")

print(tracker.get_symptom_summary())
print("=" * 50)
print(tracker.export_for_doctor())
```

## 4. 药物助手

### 4.1 用药咨询

```python
class MedicationAssistant:
    def __init__(self):
        self.llm = ChatOpenAI(model="gpt-4", temperature=0)
        
        # 药物信息模板
        self.medication_prompt = ChatPromptTemplate.from_messages([
            ("system", """你是一位药物咨询助手。请提供以下信息：
            
1. 药物适应症
2. 用法用量
3. 注意事项
4. 禁忌
5. 不良反应

⚠️ 重要声明：
- 这只是药物信息参考，不能替代医生处方
- 务必遵医嘱用药
- 如有疑问，请咨询医生或药师"""),
            ("human", "请介绍以下药物：{drug_name}")
        ])
        
        # 用药提醒模板
        self.reminder_prompt = ChatPromptTemplate.from_messages([
            ("system", "你是一个用药提醒助手。根据用户的用药计划，生成提醒内容。"),
            ("human", "药物：{drug_name}\n剂量：{dosage}\n频率：{frequency}\n开始日期：{start_date}\n持续天数：{days}")
        ])
        
        self.medication_chain = self.medication_prompt | self.llm
        self.reminder_chain = self.reminder_prompt | self.llm
    
    def get_drug_info(self, drug_name: str) -> str:
        """获取药物信息"""
        return self.medication_chain.invoke({"drug_name": drug_name}).content
    
    def check_interactions(self, drugs: List[str]) -> str:
        """检查药物相互作用"""
        prompt = f"""请分析以下药物之间可能存在的相互作用：
        
药物列表：{', '.join(drugs)}

请列出已知的相互作用和注意事项。"""
        
        return self.llm.invoke(prompt).content
    
    def generate_reminder(self, drug_name: str, dosage: str, 
                          frequency: str, start_date: str, days: int) -> str:
        """生成用药提醒"""
        return self.reminder_chain.invoke({
            "drug_name": drug_name,
            "dosage": dosage,
            "frequency": frequency,
            "start_date": start_date,
            "days": days
        }).content

# 使用
med_assistant = MedicationAssistant()

# 药物信息查询
print(med_assistant.get_drug_info("布洛芬"))
print("=" * 50)

# 药物相互作用检查
print(med_assistant.check_interactions(["阿司匹林", "布洛芬", "华法林"]))
```

### 4.2 智能用药提醒

```python
from datetime import datetime, timedelta
from typing import List

class MedicationReminder:
    def __init__(self):
        self.reminders: List[dict] = []
    
    def add_reminder(self, drug_name: str, dosage: str, 
                     times: List[str], duration_days: int = 30):
        """添加用药提醒"""
        reminder = {
            "drug_name": drug_name,
            "dosage": dosage,
            "times": times,
            "start_date": datetime.now(),
            "duration_days": duration_days,
            "enabled": True
        }
        self.reminders.append(reminder)
    
    def get_today_reminders(self) -> str:
        """获取今日提醒"""
        today = datetime.now().strftime("%Y-%m-%d")
        message = f"📅 {today} 用药提醒\n\n"
        
        for i, rem in enumerate(self.reminders, 1):
            if not rem["enabled"]:
                continue
            
            message += f"{i}. 💊 {rem['drug_name']}\n"
            message += f"   剂量: {rem['dosage']}\n"
            message += f"   时间: {', '.join(rem['times'])}\n\n"
        
        if message == f"📅 {today} 用药提醒\n\n":
            return "今天没有用药提醒"
        
        message += "\n⚠️ 请遵医嘱用药"
        return message
    
    def generate_schedule(self, days: int = 7) -> str:
        """生成_schedule表"""
        schedule = "📅 用药计划\n\n"
        
        for day in range(days):
            date = datetime.now() + timedelta(days=day)
            date_str = date.strftime("%Y-%m-%d (%A)")
            
            schedule += f"{date_str}\n"
            
            for rem in self.reminders:
                if not rem["enabled"]:
                    continue
                
                # 检查是否在持续时间内
                days_diff = (date.date() - rem["start_date"].date()).days
                if 0 <= days_diff < rem["duration_days"]:
                    schedule += f"  💊 {rem['drug_name']} {rem['dosage']}"
                    schedule += f" - {', '.join(rem['times'])}\n"
            
            schedule += "\n"
        
        return schedule

# 使用
reminder = MedicationReminder()

reminder.add_reminder(
    drug_name="阿司匹林",
    dosage="1片(100mg)",
    times=["08:00", "20:00"],
    duration_days=30
)

reminder.add_reminder(
    drug_name="维生素D",
    dosage="1粒(400IU)",
    times=["08:00"],
    duration_days=90
)

print(reminder.get_today_reminders())
print("=" * 50)
print(reminder.generate_schedule(3))
```

## 5. 医学知识库 RAG

### 5.1 构建医学知识库

```python
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders import TextLoader, PyPDFLoader

class MedicalKnowledgeBase:
    def __init__(self, persist_directory: str = "./medical_kb"):
        self.persist_directory = persist_directory
        self.embeddings = OpenAIEmbeddings()
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=100
        )
    
    def load_documents(self, file_paths: List[str]):
        """加载文档"""
        documents = []
        
        for path in file_paths:
            if path.endswith('.txt'):
                loader = TextLoader(path)
            elif path.endswith('.pdf'):
                loader = PyPDFLoader(path)
            else:
                continue
            
            documents.extend(loader.load())
        
        return documents
    
    def build_knowledge_base(self, documents: List):
        """构建知识库"""
        # 分割文档
        splits = self.text_splitter.split_documents(documents)
        
        # 构建向量数据库
        vectorstore = Chroma.from_documents(
            documents=splits,
            embedding=self.embeddings,
            persist_directory=self.persist_directory
        )
        
        vectorstore.persist()
        
        return vectorstore.as_retriever(search_kwargs={"k": 3})
    
    def query(self, question: str, retriever) -> str:
        """查询知识库"""
        # 检索相关文档
        docs = retriever.get_relevant_documents(question)
        
        context = "\n\n".join([doc.page_content for doc in docs])
        
        # 生成回答
        prompt = f"""基于以下医学知识库内容回答问题。如果知识库中没有相关信息，请说明。

知识库内容：
{context}

问题：{question}

回答："""
        
        from langchain_openai import ChatOpenAI
        llm = ChatOpenAI(model="gpt-4", temperature=0)
        
        return llm.invoke(prompt).content

# 使用
# kb = MedicalKnowledgeBase()
# documents = kb.load_documents(["./medical_kb/diseases.txt", "./medical_kb/drugs.txt"])
# retriever = kb.build_knowledge_base(documents)
# answer = kb.query("糖尿病的症状有哪些？", retriever)
```

### 5.2 医疗问答 Chain

```python
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate

class MedicalQASystem:
    def __init__(self, retriever):
        self.llm = ChatOpenAI(model="gpt-4", temperature=0)
        self.retriever = retriever
        
        self.qa_prompt = ChatPromptTemplate.from_messages([
            ("system", """你是一个专业的医学知识问答助手。

要求：
1. 基于提供的医学知识库内容回答问题
2. 使用通俗易懂的语言解释专业医学术语
3. 回答要准确、客观
4. 注明信息来源
5. 提醒用户咨询专业医生

⚠️ 免责声明：此回答仅供参考，不能替代医生诊断。"""),
            ("human", "知识库内容：{context}\n\n问题：{question}")
        ])
        
        self.chain = self.qa_prompt | self.llm
    
    def answer(self, question: str) -> str:
        # 检索相关知识
        docs = self.retriever.get_relevant_documents(question)
        context = "\n\n".join([doc.page_content for doc in docs])
        
        if not context:
            return "抱歉，知识库中没有找到相关信息。建议您咨询专业医生。"
        
        # 生成回答
        response = self.chain.invoke({
            "context": context,
            "question": question
        })
        
        # 添加免责声明
        response.content += "\n\n⚠️ 本回答基于医学知识库，仅供参考，不能替代专业医疗诊断。"
        
        return response.content
```

## 6. 健康报告生成

### 6.1 健康摘要报告

```python
from datetime import datetime
from typing import Dict, List

class HealthReportGenerator:
    def __init__(self):
        self.llm = ChatOpenAI(model="gpt-4", temperature=0.5)
    
    def generate_summary_report(self, 
                                symptoms: List[Dict],
                                medications: List[Dict],
                                vitals: Dict) -> str:
        """生成健康摘要报告"""
        
        # 构建数据摘要
        symptom_summary = self._summarize_symptoms(symptoms)
        medication_summary = self._summarize_medications(medications)
        vitals_summary = self._summarize_vitals(vitals)
        
        # 生成报告
        prompt = f"""请根据以下健康数据生成一份个人健康摘要报告：

{symptom_summary}

{medication_summary}

{vitals_summary}

要求：
1. 总结整体健康状况
2. 提醒需要注意的问题
3. 给出健康建议
4. 语气专业、温和

⚠️ 此报告仅供参考，不能替代医生诊断。"""
        
        report = self.llm.invoke(prompt).content
        
        # 添加报告头部
        header = f"""
╔══════════════════════════════════════════════════════════════╗
║                    个人健康摘要报告                          ║
║                      {datetime.now().strftime('%Y-%m-%d')}                       ║
╚══════════════════════════════════════════════════════════════╝

"""
        
        return header + report
    
    def _summarize_symptoms(self, symptoms: List[Dict]) -> str:
        if not symptoms:
            return "📝 症状记录：无"
        
        summary = "📝 症状记录：\n"
        for s in symptoms[-7:]:  # 最近7条
            summary += f"  - {s.get('symptom', '未知')} (严重程度: {s.get('severity', 'N/A')}/10)\n"
        return summary
    
    def _summarize_medications(self, medications: List[Dict]) -> str:
        if not medications:
            return "💊 用药记录：无"
        
        summary = "💊 当前用药：\n"
        for m in medications:
            summary += f"  - {m.get('drug', '未知')}: {m.get('dosage', 'N/A')}\n"
        return summary
    
    def _summarize_vitals(self, vitals: Dict) -> str:
        if not vitals:
            return "📊 生命体征：无记录"
        
        summary = "📊 生命体征：\n"
        for key, value in vitals.items():
            summary += f"  - {key}: {value}\n"
        return summary

# 使用
generator = HealthReportGenerator()

symptoms = [
    {"symptom": "头痛", "severity": 5, "date": "2024-01-15"},
    {"symptom": "疲劳", "severity": 4, "date": "2024-01-14"},
    {"symptom": "失眠", "severity": 6, "date": "2024-01-13"},
]

medications = [
    {"drug": "维生素B", "dosage": "1片/天"},
    {"drug": "褪黑素", "dosage": "1片/睡前"},
]

vitals = {
    "血压": "120/80 mmHg",
    "心率": "72 bpm",
    "体重": "70 kg",
    "睡眠时长": "6.5 小时",
}

report = generator.generate_summary_report(symptoms, medications, vitals)
print(report)
```

## 7. 隐私与安全

### 7.1 数据脱敏

```python
import re

class DataAnonymizer:
    """医疗数据脱敏"""
    
    # 脱敏规则
    PATTERNS = {
        "phone": r"1[3-9]\d{9}",
        "id_card": r"\d{17}[\dXx]",
        "bank_card": r"\d{16,19}",
        "name": r"([张李王刘陈杨黄赵周吴徐孙马朱胡郭何高林罗郑梁谢宋唐许韩邓冯曹彭曾肖田董袁潘于蒋蔡余杜叶程魏苏吕丁任沈",
    }
    
    @classmethod
    def anonymize(cls, text: str) -> str:
        """脱敏处理"""
        # 手机号
        text = re.sub(cls.PATTERNS["phone"], "1**********", text)
        
        # 身份证号
        text = re.sub(cls.PATTERNS["id_card"], "*******************", text)
        
        # 银行卡
        text = re.sub(cls.PATTERNS["bank_card"], "****", text)
        
        return text
    
    @classmethod
    def extract_medical_info(cls, text: str) -> dict:
        """提取需要关注的医疗信息"""
        info = {
            "symptoms": [],
            "medications": [],
            "diseases": []
        }
        
        # 症状关键词
        symptom_keywords = ["疼", "痛", "晕", "吐", "泻", "发烧", "咳嗽"]
        for kw in symptom_keywords:
            if kw in text:
                info["symptoms"].append(kw)
        
        return info
```

## 8. 总结

本章我们介绍了医疗健康 AI 系统的构建：

1. **系统架构**：分层设计，包含用户交互、安全、业务、模型、数据层
2. **合规性要求**：辅助决策、免责声明、隐私保护
3. **健康咨询**：带安全检查的对话系统
4. **症状分析**：可能性评估、紧急程度判断
5. **药物助手**：用药咨询、相互作用检查、用药提醒
6. **医学知识库**：基于 RAG 的知识问答
7. **健康报告**：个人健康摘要生成
8. **隐私安全**：数据脱敏、处理

⚠️ **重要提醒**：医疗 AI 应用必须严格遵守法规，明确 AI 的辅助定位，不能替代专业医疗诊断。
