import os
import re
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
PERSIST_DIRECTORY = BASE_DIR / "chroma_db"

from langchain_chroma import Chroma
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from langchain_core.messages import ToolMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI

SEMANTIC_MODEL = "shibing624/text2vec-base-chinese"
_semantic_store: Chroma | None = None
_intent_llm: ChatOpenAI | None = None

# 固定话术：让模型自由发挥拒绝的方式，每次说法都不一样，用户会以为系统坏了。
# 拒答也保持三段式结构，保证前端渲染一致。
REJECT_OUT_OF_SCOPE = (
    "【结论】\n这个问题超出了我的服务范围：我的知识库仅覆盖中山大学校内规章制度与办事文件。\n\n"
    "【依据】\n无（该问题不涉及知识库收录的校内规章）。\n\n"
    "【下一步】\n建议前往中山大学官方网站、信息门户或咨询所在院系教务老师获取相关信息。"
)

CLARIFY_LOW_CONFIDENCE = (
    "【结论】\n为了给您最准确的指引，我还需要了解一些关键信息。\n\n"
    "【依据】\n无（当前提问信息不足，暂未检索知识库）。\n\n"
    "【下一步】\n请问您具体要办理什么事项（如奖助、学籍、场地等）？另外请告知您所在的年级和院系，以便我为您匹配对应的文件。"
)

INTENT_PROMPT = ChatPromptTemplate.from_template(
    """你是中山大学校园政策问答助手的意图分类器。请把用户问题归入以下五类之一，并给出置信度（高/中/低）：

1. 政策问答：询问校内规章、政策的条件、流程、金额、期限等（如保研条件、补助金额、请假手续）。
2. 部门咨询：询问职能部门的位置、联系电话、职责（如"教务部在哪"）。
3. 表格下载：询问申请表、证明文件在哪里下载（如"休学申请表去哪下"）。
4. 超纲：与中大校内事务无关的问题（如写代码、聊心情、问天气、其他学校的政策、时事新闻）。
5. 判断不了：问题过于含糊、缺少关键信息，无法确定用户想要什么（如只说"怎么办""帮我查一下"）。

判断规则：
- 只要问题涉及中大校内场景（课室、自习室、督导、听课、社团、宿舍、奖助、学籍、选课等），即使没有出现"政策""规定"等字眼，也不能判为超纲。
- 问题中提到了具体事项时，优先归入所涉事项的类别，而不是"判断不了"。

只按格式输出一行，不要输出任何解释：
类别|置信度

用户问题：{question}
"""
)


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

PROMPT = ChatPromptTemplate.from_template(
    """你是一个中山大学校园政策咨询与办事 Agent。你可以通过查阅官方政策文档，或者调用本地办事实用工具来服务学生。

【前文对话历史】
{chat_history}

【参考官方文档与工具结果】
{context}

【回答规则】
1. 只能使用上方召回片段和工具结果中的事实回答，不得补充片段未提及的流程、条件、数字或背景。
2. 回答必须严格采用以下三段结构，三段缺一不可，不得增删段落或修改段落标题：
【结论】
（1至3句，直接回答用户所问；不同文件的规定分开说明，不能混为一谈）
【依据】
（1至3条；每条格式为《文件名》：原文关键摘录——文件名必须原样照抄参考资料开头的【来源文件】或【来源文件名】，不得简写、不得省略.pdf；没有可引用的原文时，本段写"无"）
【下一步】
（1句，给出明确的建议动作，如前往哪个部门、下载哪份表格；无明确建议时写"如有疑问请咨询所在院系教务老师"）
3. 如果召回内容不足以回答，【结论】必须写"现有资料未说明"，不要猜测，不要用常识补充。
4. 咨询地点、电话或表格时，按办事工具结果回答，不得自行编造或扩展。
5. 使用专业、清晰、口语友好的表达。

用户当前问题：{question}
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


def classify_intent(query: str) -> tuple[str, str]:
    """意图分类 + 置信度分级。返回 (类别, 置信度)。

    分类失败或没有 API Key 时默认放行到检索流程——宁可多答，不误杀。
    """
    global _intent_llm
    try:
        api_key = os.getenv("DEEPSEEK_API_KEY")
        if not api_key:
            return "政策问答", "高"
        if _intent_llm is None:
            _intent_llm = ChatOpenAI(
                model="deepseek-chat",
                api_key=api_key,
                base_url="https://api.deepseek.com/v1",
                temperature=0,
                max_tokens=10,
            )
        raw = _intent_llm.invoke(INTENT_PROMPT.format(question=query)).content
        if not isinstance(raw, str):
            raw = str(raw)
        parts = [part.strip() for part in raw.strip().split("|")]
        category = parts[0] if parts else "政策问答"
        confidence = parts[1] if len(parts) > 1 else "高"
        if category not in ("政策问答", "部门咨询", "表格下载", "超纲", "判断不了"):
            category = "政策问答"
        if confidence not in ("高", "中", "低"):
            confidence = "高"
        print(f"[intent] {category}|{confidence} :: {query[:30]}")
        return category, confidence
    except Exception as exc:
        print(f"[intent] 分类失败，默认放行：{exc}")
        return "政策问答", "高"


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
    global _semantic_store
    if not PERSIST_DIRECTORY.exists():
        raise FileNotFoundError(f"找不到 Chroma 知识库：{PERSIST_DIRECTORY}")

    # 优先带 embedding 打开（语义路可用）；模型缺失或网络不通时退回纯 BM25，不阻断启动
    if os.getenv("RAG_HYBRID", "1") == "1":
        try:
            embeddings = HuggingFaceEmbeddings(model_name=SEMANTIC_MODEL)
            _semantic_store = Chroma(
                persist_directory=str(PERSIST_DIRECTORY),
                embedding_function=embeddings,
            )
        except Exception as exc:
            print(f"[rag_core] 语义检索不可用，退回 BM25 单路：{exc}")
            _semantic_store = None
    else:
        _semantic_store = None

    vectorstore = _semantic_store or Chroma(
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


def _rrf_fuse(ranked_lists: list[list[Document]], top_k: int = 8, k: int = 60) -> list[Document]:
    """Reciprocal Rank Fusion：每路按 1/(k+rank+1) 计分求和后重排，k=60 为原论文推荐常数。"""
    scores: dict[str, float] = {}
    docs: dict[str, Document] = {}
    for ranked in ranked_lists:
        for rank, document in enumerate(ranked):
            key = document.page_content
            scores[key] = scores.get(key, 0.0) + 1.0 / (k + rank + 1)
            docs.setdefault(key, document)
    ordered = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    return [docs[key] for key, _ in ordered[:top_k]]


def retrieve_policy_documents(
    query: str,
    bm25_retriever: BM25Retriever,
    top_k: int = 8,
) -> list[Document]:
    bm25_results = bm25_retriever.invoke(query)[:top_k]
    if _semantic_store is None:
        return bm25_results
    try:
        vector_results = _semantic_store.similarity_search(query, k=top_k)
    except Exception:
        return bm25_results
    return _rrf_fuse([bm25_results, vector_results], top_k=top_k)


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


def answer_question(
    query: str,
    bm25_retriever: BM25Retriever,
    llm: ChatOpenAI | None,
    chat_history: str = "无历史对话",
) -> tuple[str, list[Document], str]:
    # 1) 意图分类 + 置信度分级：先判该不该答，再判怎么答。
    #    顺序很重要——让模型拿不该答的问题去检索，越检索越错。
    #    超纲且高置信才拒答，置信度不足时放行检索——宁可多答，不误杀。
    category, confidence = classify_intent(query)
    if category == "超纲" and confidence == "高":
        return REJECT_OUT_OF_SCOPE, [], ""
    if category == "判断不了":
        return CLARIFY_LOW_CONFIDENCE, [], ""

    # 2) 部门 / 表格类意图直接路由到确定性工具；关键词预触发作为双保险
    tool_output = ""
    if category == "部门咨询" or any(term in query for term in ("电话", "地点", "教务部", "学生处", "校医院")):
        department = (
            "教务部" if "教务" in query else
            "学生处" if "学生处" in query else
            "校医院" if "医院" in query else "教务部"
        )
        tool_output = "\n\n" + query_campus_contact.invoke({"department_name": department})
    elif category == "表格下载" or any(term in query for term in ("表", "下载", "免修", "休学表")):
        form_name = "免修" if "免修" in query else "休学" if "休学" in query else "选课"
        tool_output = "\n\n" + get_document_download_link.invoke({"form_name": form_name})

    # 3) 政策问答走混合检索 + 三段式生成
    source_docs = retrieve_policy_documents(query, bm25_retriever)
    if llm is None:
        return format_local_fallback(source_docs, tool_output), source_docs, tool_output

    context_sections = [
        f"【来源文件：{Path(str(document.metadata.get('source', '未知文件'))).name}】\n"
        f"{document.page_content}"
        for document in source_docs
    ]
    if tool_output:
        context_sections.append(tool_output.strip())
    context = "\n\n".join(context_sections)
    try:
        prompt_messages = PROMPT.format_messages(
            chat_history=chat_history,
            context=context,
            question=query,
        )
        ai_message = llm.invoke(prompt_messages)

        # 模型自主发起的工具调用：执行工具、回传结果、再生成最终回答。
        # 最多循环 2 轮，防止调用失控；没有可执行的工具时直接使用当前回答。
        rounds = 0
        while getattr(ai_message, "tool_calls", None) and rounds < 2:
            tool_map = {t.name: t for t in TOOLS}
            follow_up = prompt_messages + [ai_message]
            executed = False
            for call in ai_message.tool_calls:
                tool = tool_map.get(call.get("name", ""))
                if tool is None:
                    continue
                observation = tool.invoke(call.get("args") or {})
                follow_up.append(ToolMessage(str(observation), tool_call_id=call.get("id", "")))
                executed = True
            if not executed:
                break
            ai_message = llm.invoke(follow_up)
            rounds += 1

        response = ai_message.content
    except Exception:
        return format_local_fallback(source_docs, tool_output), source_docs, tool_output

    if not isinstance(response, str):
        response = str(response)
    return response, source_docs, tool_output
