import json
import os
import re
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
PERSIST_DIRECTORY = BASE_DIR / "chroma_db"

from langchain_chroma import Chroma
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI


@tool
def query_campus_contact(department_name: str) -> str:
    """查询中山大学职能部门的办公地点、楼层与公开电话。"""
    contacts = {
        "教务部": {"location": "南校园行政楼 101 室", "phone": "020-84112345", "desc": "负责本科生学籍、选课、考试安排。"},
        "学生处": {"location": "南校园行政楼 203 室", "phone": "020-84111234", "desc": "负责奖学金、评优、宿舍管理与学生资助。"},
        "研究生院": {"location": "南校园行政楼 305 室", "phone": "020-84111688", "desc": "负责研究生培养与学位管理。"},
        "校医院": {"location": "南校园东区路口", "phone": "020-84111200", "desc": "提供日常门诊与医疗服务。"},
        "保卫处": {"location": "南校园西区", "phone": "020-84110110", "desc": "负责校园治安与户籍管理。"},
    }
    for key, info in contacts.items():
        if key in department_name or department_name in key:
            return (
                f"【官方部门名片】\n* **部门名称**：{key}\n"
                f"* **办公地点**：{info['location']}\n"
                f"* **联系电话**：`{info['phone']}`\n"
                f"* **核心职责**：{info['desc']}"
            )
    return f"【提示】未找到“{department_name}”的直接部门信息，建议通过学校主页或信息门户查询。"


@tool
def get_document_download_link(form_name: str) -> str:
    """查询常用校园事务表格的下载链接。"""
    forms = {
        "选课": {"name": "中山大学本科生选课/退课申请表.docx", "url": "http://jwb.sysu.edu.cn/file/xuanoke.docx"},
        "休学": {"name": "中山大学学生休学/复学申请审批表.pdf", "url": "http://jwb.sysu.edu.cn/file/xiuxue.pdf"},
        "免修": {"name": "中山大学学生课程免修申请表.docx", "url": "http://jwb.sysu.edu.cn/file/mianxiu.docx"},
        "转专业": {"name": "中山大学本科生转专业申请审批表.pdf", "url": "http://jwb.sysu.edu.cn/file/zhuanye.pdf"},
    }
    for key, info in forms.items():
        if key in form_name or form_name in key:
            return (
                f"【官方表格下载】\n* **表格名称**：《{info['name']}》\n"
                f"* **操作指引**：[点击下载官方文档]({info['url']})"
            )
    return "【提示】该表格暂未收录至快捷直链库，请前往教务部官网下载专区获取。"


TOOLS = [query_campus_contact, get_document_download_link]
TOOL_MAP = {tool.name: tool for tool in TOOLS}

REFUSAL_OUT_OF_SCOPE = """
不好意思，我的知识库仅限中大校内规章与办事文件，这个问题超出了我的服务范围。
"""
REFUSAL_NEED_MORE_INFO = """
为了给您最准确的指引，我还需要了解一些关键信息。请问您具体要申请什么事项（如奖助、学籍、场地等）？另外请告知您所在的年级和院系，我好为您匹配对应的文件。
"""

SYSTEM_PROMPT = """你是一个中山大学校园政策咨询与办事 Agent。你可以通过查阅官方政策文档，或者调用本地办事实用工具来服务学生。

【回答规则】
1. 只能使用参考资料中的事实回答，不得补充资料未提及的流程、条件、数字或背景。
2. 先直接回答用户所问内容，只保留必要信息；通常控制在1至3句话，不重复、不延伸讨论无关事项。
3. 每项政策事实都要注明对应的来源文件名；不同文件的规定要分别说明，不能混为一谈。
4. 如果召回内容不足以回答，明确说明“现有资料未说明”，不要猜测。
5. 咨询地点、电话或表格时，按办事工具结果回答，不得自行编造或扩展。
6. 使用专业、清晰、口语友好的表达。
"""

PROMPT = ChatPromptTemplate.from_template(
    """你是一个中山大学校园政策咨询与办事 Agent。你可以通过查阅官方政策文档，或者调用本地办事实用工具来服务学生。

【前文对话历史】
{chat_history}

【参考官方文档与工具结果】
{context}

【回答规则】
1. 只能使用上方召回片段和工具结果中的事实回答，不得补充片段未提及的流程、条件、数字或背景。
2. 先直接回答用户所问内容，只保留必要信息；通常控制在1至3句话，不重复、不延伸讨论无关事项。
3. 每项政策事实都要注明对应的来源文件名；不同文件的规定要分别说明，不能混为一谈。
4. 如果召回内容不足以回答，明确说明“现有资料未说明”，不要猜测。
5. 咨询地点、电话或表格时，按办事工具结果回答，不得自行编造或扩展。
6. 使用专业、清晰、口语友好的表达。

用户当前问题: {question}
"""
)


def create_llm() -> ChatOpenAI | None:
    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        return None
    return ChatOpenAI(
        model="deepseek-chat",
        api_key=api_key,
        base_url="https://api.deepseek.com/v1",
        temperature=0.1,
    ).bind_tools(TOOLS)


def route_intent(query: str, llm: ChatOpenAI | None) -> dict[str, str | float]:
    if llm is None:
        return {"intent": "unknown", "confidence": 1.0, "reason": "意图识别模型不可用"}

    routing_prompt = """你是中山大学校园问答系统的意图识别器。请只输出一个 JSON 对象，不要输出 Markdown 或解释。

用户问题：{query}

意图类别：
- policy_qa：查询中大校内规章、政策条件、流程或结论
- contact：咨询部门、地点、办公时间或电话
- download：索要表格、申请表或下载链接
- out_of_scope：与中大校内规章、办事文件无关的问题
- unknown：无法可靠判断意图

输出格式：{{"intent":"类别","confidence":0到1之间的小数,"reason":"简短理由"}}
"""

    try:
        response = llm.invoke([HumanMessage(content=routing_prompt.format(query=query))])
        content = response.content
        if not isinstance(content, str):
            content = str(content)
        parsed = json.loads(content)
        if not isinstance(parsed, dict):
            raise ValueError("意图识别结果不是对象")

        intent = parsed.get("intent")
        valid_intents = {"policy_qa", "contact", "download", "out_of_scope", "unknown"}
        if intent not in valid_intents:
            raise ValueError("意图类别无效")

        confidence = float(parsed["confidence"])
        confidence = max(0.0, min(1.0, confidence))
        reason = parsed.get("reason", "")
        if not isinstance(reason, str):
            reason = str(reason)
        return {"intent": intent, "confidence": confidence, "reason": reason}
    except json.JSONDecodeError:
        return {"intent": "unknown", "confidence": 0.0, "reason": "意图识别结果不是合法 JSON"}
    except (KeyError, TypeError, ValueError):
        return {"intent": "unknown", "confidence": 0.0, "reason": "意图识别结果格式无效"}
    except Exception:
        return {"intent": "unknown", "confidence": 1.0, "reason": "意图识别调用失败，降级为正常检索生成"}


def tokenize_chinese_text(text: str) -> list[str]:
    tokens = []
    for segment in re.findall(r"[\u3400-\u9fff]+|[a-zA-Z0-9]+", text.lower()):
        if "\u3400" <= segment[0] <= "\u9fff":
            if len(segment) == 1:
                tokens.append(segment)
            else:
                tokens.extend(segment[index : index + 2] for index in range(len(segment) - 1))
        else:
            tokens.append(segment)
    return tokens


def initialize_rag_components() -> BM25Retriever:
    if not PERSIST_DIRECTORY.exists():
        raise FileNotFoundError(f"找不到 Chroma 知识库：{PERSIST_DIRECTORY}")

    vectorstore = Chroma(
        persist_directory=str(PERSIST_DIRECTORY),
        embedding_function=None,
    )
    all_data = vectorstore.get()
    texts = all_data["documents"] or []
    metadatas = all_data["metadatas"] or []
    documents = [
        Document(page_content=text, metadata=metadata or {})
        for text, metadata in zip(texts, metadatas)
    ]
    if not documents:
        raise ValueError("Chroma 知识库为空，请先运行 ingest.py。")

    return BM25Retriever.from_documents(
        documents,
        preprocess_func=tokenize_chinese_text,
        k=8,
    )


def retrieve_policy_documents(query: str, bm25_retriever: BM25Retriever) -> list[Document]:
    return bm25_retriever.invoke(query)


def format_local_fallback(source_docs: list[Document], tool_output: str) -> str:
    sections = ["当前无法使用 DeepSeek，以下为本地知识库检索结果："]
    for document in source_docs:
        source = Path(str(document.metadata.get("source", "未知文件"))).name
        sections.append(f"【来源：{source}】\n{document.page_content}")
    if tool_output:
        sections.append(tool_output.strip())
    if not source_docs and not tool_output:
        sections.append("本地知识库未检索到相关内容，请稍后重试或联系相关部门。")
    return "\n\n".join(sections)


def answer_with_tools(
    query: str,
    bm25_retriever: BM25Retriever,
    llm: ChatOpenAI | None,
    chat_history_text: str = "",
) -> tuple[str, list[Document], str]:
    intent_route = route_intent(query, llm)
    if intent_route["intent"] == "out_of_scope":
        return (REFUSAL_OUT_OF_SCOPE, [], "")
    if intent_route["intent"] == "unknown" and intent_route["confidence"] < 0.5:
        return (REFUSAL_NEED_MORE_INFO, [], "")

    source_docs = retrieve_policy_documents(query, bm25_retriever)
    context_sections = [
        f"【来源文件：{Path(str(document.metadata.get('source', '未知文件'))).name}】\n"
        f"{document.page_content}"
        for document in source_docs
    ]
    context = "\n\n".join(context_sections)

    if llm is None:
        return format_local_fallback(source_docs, ""), source_docs, ""

    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(
            content=(
                f"[参考资料]\n{context}\n\n"
                f"[前文对话历史]\n{chat_history_text or '无历史对话'}\n\n"
                f"[用户问题]\n{query}"
            )
        ),
    ]
    tool_output = ""

    try:
        response = llm.invoke(messages)
        messages.append(response)
        for _ in range(2):
            tool_calls = getattr(response, "tool_calls", None) or []
            if not tool_calls:
                break

            round_outputs = []
            for call in tool_calls:
                try:
                    selected_tool = TOOL_MAP[call["name"]]
                    result = selected_tool.invoke(call["args"])
                    output = str(result)
                except Exception as exc:
                    output = f"工具调用失败：{exc}"

                round_outputs.append(output)
                tool_output = "\n\n" + "\n\n".join(round_outputs)
                messages.append(
                    ToolMessage(content=output, tool_call_id=call["id"])
                )

            response = llm.invoke(messages)
            messages.append(response)
    except Exception:
        return format_local_fallback(source_docs, tool_output), source_docs, tool_output

    content = response.content
    if not isinstance(content, str):
        content = str(content)
    return content, source_docs, tool_output


def answer_question(
    query: str,
    bm25_retriever: BM25Retriever,
    llm: ChatOpenAI | None,
    chat_history: str = "无历史对话",
) -> tuple[str, list[Document], str]:
    return answer_with_tools(query, bm25_retriever, llm, chat_history)


# answer_question_v1_backup（原函数完整备份，不再运行）
# def answer_question_v1_backup(
#     query: str,
#     bm25_retriever: BM25Retriever,
#     llm: ChatOpenAI | None,
#     chat_history: str = "无历史对话",
# ) -> tuple[str, list[Document], str]:
#     tool_output = ""
#     if any(term in query for term in ("电话", "地点", "教务部", "学生处", "校医院")):
#         department = (
#             "教务部" if "教务" in query else
#             "学生处" if "学生处" in query else
#             "校医院" if "医院" in query else "教务部"
#         )
#         tool_output = "\n\n" + query_campus_contact.invoke({"department_name": department})
#     elif any(term in query for term in ("表", "下载", "免修", "休学表")):
#         form_name = "免修" if "免修" in query else "休学" if "休学" in query else "选课"
#         tool_output = "\n\n" + get_document_download_link.invoke({"form_name": form_name})
#
#     source_docs = retrieve_policy_documents(query, bm25_retriever)
#     if llm is None:
#         return format_local_fallback(source_docs, tool_output), source_docs, tool_output
#
#     context_sections = [
#         f"【来源文件：{Path(str(document.metadata.get('source', '未知文件'))).name}】\n"
#         f"{document.page_content}"
#         for document in source_docs
#     ]
#     if tool_output:
#         context_sections.append(tool_output.strip())
#     context = "\n\n".join(context_sections)
#     try:
#         response = llm.invoke(
#             PROMPT.format(
#                 chat_history=chat_history,
#                 context=context,
#                 question=query,
#             )
#         ).content
#     except Exception:
#         return format_local_fallback(source_docs, tool_output), source_docs, tool_output
#
#     if not isinstance(response, str):
#         response = str(response)
#     return response, source_docs, tool_output