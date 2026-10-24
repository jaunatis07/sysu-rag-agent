import streamlit as st
from rag_core import answer_question, create_llm, initialize_rag_components

# 1. 页面基本配置
st.set_page_config(
    page_title="SYSU Policy & Service Agent",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="auto",
)

# 2. 注入现代 AI App 风格 CSS
st.markdown("""
<style>
#MainMenu, footer {visibility: hidden !important;}
header[data-testid="stHeader"] {height: 0 !important; background: transparent !important;}
[data-testid="stToolbar"] {visibility: hidden !important;}
[data-testid="stDecoration"] {display: none !important;}

.stApp {
    background:
        radial-gradient(circle at 12% 10%, rgba(197, 184, 239, 0.22), transparent 25%),
        radial-gradient(circle at 88% 12%, rgba(244, 194, 215, 0.18), transparent 22%),
        linear-gradient(135deg, #f7f8fc 0%, #f3f4fa 52%, #fbf7fa 100%);
    font-family: "PingFang SC", "Microsoft YaHei", sans-serif;
}
.block-container {max-width: 980px; padding-top: 0.7rem; padding-bottom: 6rem;}

section[data-testid="stSidebar"] {
    background: rgba(255, 255, 255, 0.72);
    border-right: 1px solid rgba(109, 94, 145, 0.08);
    backdrop-filter: blur(18px);
}
section[data-testid="stSidebar"] > div {padding-top: 1.5rem;}

.app-header {display: flex; justify-content: center; align-items: center; margin: 0.2rem 0 1rem;}
.app-brand {display: flex; align-items: center; gap: 0.65rem; color: #29273a; font-size: 1rem; font-weight: 750;}
.brand-mark {
    width: 38px; height: 38px; border-radius: 12px; display: grid; place-items: center;
    background: linear-gradient(135deg, #7268a6, #a68bbf);
    box-shadow: 0 8px 22px rgba(114, 104, 166, 0.18);
}
.brand-dot {width: 7px; height: 7px; border-radius: 50%; background: #9ee2b7; box-shadow: 0 0 0 3px rgba(158, 226, 183, 0.18);}

.welcome {text-align: center; margin: 1rem 0 1.45rem;}
.welcome h1 {margin: 0; color: #29273a; font-size: clamp(1.55rem, 3vw, 2.1rem); line-height: 1.2; letter-spacing: 0;}
.welcome p {margin: 0.65rem 0 0; color: #85819a; font-size: 0.92rem;}

.shortcut-card {
    height: 148px; padding: 1.15rem; border-radius: 20px;
    border: 1px solid rgba(255, 255, 255, 0.86);
    background: rgba(255, 255, 255, 0.54);
    backdrop-filter: blur(16px); -webkit-backdrop-filter: blur(16px);
    box-shadow: 0 12px 32px rgba(87, 78, 116, 0.06); color: #2f2c42;
}
.shortcut-icon {
    width: 42px; height: 42px; margin-bottom: 0.9rem; border-radius: 13px;
    display: grid; place-items: center; font-size: 1.25rem;
    background: rgba(239, 235, 252, 0.88);
}
.shortcut-card strong {display: block; font-size: 0.98rem; margin-bottom: 0.35rem;}
.shortcut-card span {color: #85819a; font-size: 0.78rem; line-height: 1.5;}

[data-testid="stChatMessage"] {border: none !important; background: transparent !important; max-width: 82%; width: fit-content;}
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]),
[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) {margin-left: auto;}
[data-testid="stChatMessage"] > div:last-child {padding: 0 !important;}
[data-testid="stChatMessageContent"] {padding: 0.85rem 1rem; border-radius: 20px;}
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) [data-testid="stChatMessageContent"] {
    background: linear-gradient(135deg, rgba(230, 224, 246, 0.96), rgba(242, 232, 245, 0.96));
    border-bottom-right-radius: 7px; color: #45415a;
}
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarAssistant"]) [data-testid="stChatMessageContent"] {
    background: rgba(255, 255, 255, 0.96); border-bottom-left-radius: 7px;
    box-shadow: 0 10px 28px rgba(61, 54, 91, 0.08); color: #353348;
}

[data-testid="stChatInput"] {width: min(920px, calc(100% - 2rem)); margin: 0 auto; bottom: 1rem;}
[data-testid="stChatInput"] > div {
    border-radius: 24px !important; border: 2px solid rgba(153, 145, 183, 0.26) !important;
    background: rgba(255, 255, 255, 0.90) !important;
    box-shadow: 0 14px 38px rgba(78, 69, 113, 0.12); backdrop-filter: blur(18px);
}
[data-testid="stChatInput"] textarea {font-size: 0.92rem !important;}

.stButton > button {
    width: 100%; min-height: 42px; border: 1px solid rgba(112, 105, 145, 0.10);
    border-radius: 14px; background: rgba(255, 255, 255, 0.58); color: #575169;
    font-weight: 650; box-shadow: none;
}
.stButton > button:hover {border-color: rgba(126, 112, 177, 0.22); background: rgba(255, 255, 255, 0.92); color: #51486f;}
[data-testid="stExpander"] {border: 1px solid rgba(112, 105, 145, 0.09); border-radius: 14px; background: rgba(255, 255, 255, 0.58);}
hr {border-color: rgba(112, 105, 145, 0.08);}

@media (max-width: 700px) {
    .block-container {padding-left: 0.8rem; padding-right: 0.8rem;}
    [data-testid="stChatMessage"] {max-width: 92%;}
    .shortcut-card {height: 132px;}
}
</style>
""", unsafe_allow_html=True)

# 3. 加载本地向量数据库与检索组件
@st.cache_resource
def load_rag_components():
    return initialize_rag_components()

with st.spinner("正在加载校园政策知识库与办事工具箱..."):
    bm25_retriever = load_rag_components()

llm = create_llm()

# ==================== 5. 现代化侧边栏 ====================
with st.sidebar:
    st.markdown("### ✦ SYSU Agent")
    st.caption("校园政策智能助手")
    st.divider()

    st.markdown("#### 工作台状态")
    st.markdown("**知识库** · 83 份官方 PDF")
    st.markdown("**检索引擎** · 本地 BM25")
    st.markdown("**生成模型** · DeepSeek-V3")

    st.divider()
    st.markdown("#### 常用入口")
    if st.button("📋 课程免修申请"):
        st.session_state.quick_query = "我想申请免修，请问哪里可以下载免修申请表？"
    if st.button("☎️ 联系教务部"):
        st.session_state.quick_query = "教务部的办公地点和电话是多少？"
    if st.button("📝 补考与成绩政策"):
        st.session_state.quick_query = "考试不及格以后可以参加补考吗？"

    st.divider()
    if st.button("清空当前对话", type="secondary"):
        st.session_state.messages = []
        st.rerun()

# ==================== 6. 主界面 ====================
st.markdown("""
<div class="app-header">
    <div class="app-brand">
        <span class="brand-mark">🎓</span>
        <span>SYSU Campus AI</span>
        <span class="brand-dot"></span>
    </div>
</div>
""", unsafe_allow_html=True)

if "messages" not in st.session_state:
    st.session_state.messages = []

# 初始欢迎区与快捷入口卡片
if not st.session_state.messages:
    st.markdown("""
    <div class="welcome">
        <h1>有什么校园问题，我来帮你查</h1>
        <p>教务、学籍、宿舍、奖助与办事流程，直接问我。</p>
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3, col4 = st.columns(4, gap="medium")
    with col1:
        st.markdown("""
        <div class="shortcut-card"><div class="shortcut-icon">📚</div><strong>评奖评优</strong><span>奖学金、评优与申请资格</span></div>
        """, unsafe_allow_html=True)
        if st.button("咨询评奖评优", key="card_award"):
            st.session_state.quick_query = "本科生奖学金申请有哪些条件？"
    with col2:
        st.markdown("""
        <div class="shortcut-card"><div class="shortcut-icon">📄</div><strong>办事流程</strong><span>申请表、提交入口与材料</span></div>
        """, unsafe_allow_html=True)
        if st.button("咨询办事流程", key="card_process"):
            st.session_state.quick_query = "学生证丢了怎么办，补办流程是什么？"
    with col3:
        st.markdown("""
        <div class="shortcut-card"><div class="shortcut-icon">🏠</div><strong>宿舍生活</strong><span>入住、调整与退宿政策</span></div>
        """, unsafe_allow_html=True)
        if st.button("咨询宿舍政策", key="card_dorm"):
            st.session_state.quick_query = "学生宿舍退宿需要办理什么手续？"
    with col4:
        st.markdown("""
        <div class="shortcut-card"><div class="shortcut-icon">📝</div><strong>考试成绩</strong><span>补考、成绩与课程考核</span></div>
        """, unsafe_allow_html=True)
        if st.button("咨询考试政策", key="card_exam"):
            st.session_state.quick_query = "课程考试不及格可以参加补考吗？"

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

query = st.chat_input("在此输入您关于中大教务、学籍、宿舍的疑问...")
if "quick_query" in st.session_state and st.session_state.quick_query:
    query = st.session_state.quick_query
    st.session_state.quick_query = None

if query:
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    with st.chat_message("assistant"):
        with st.spinner("AI 正在联动知识库与办事工具..."):
            
            chat_history_text = ""
            recent_messages = st.session_state.messages[-6:-1] 
            for msg in recent_messages:
                role = "用户" if msg["role"] == "user" else "助手"
                chat_history_text += f"{role}: {msg['content']}\n"
            
            if not chat_history_text:
                chat_history_text = "无历史对话"

            response, source_docs, tool_output = answer_question(
                query,
                bm25_retriever,
                llm,
                chat_history_text,
            )

            st.markdown(response)

            with st.expander("📌 点击查看底层 RAG 检索与办事 Tool 调用详情"):
                if tool_output:
                    st.markdown(f"**调用的办事工具输出**: `{tool_output}`")
                for i, doc in enumerate(source_docs):
                    source_name = doc.metadata.get("source", "未知文件")
                    st.markdown(f"**来源文档 {i+1}**: `{source_name}`")
                    st.text(doc.page_content[:300] + "...")
                    st.divider()

    st.session_state.messages.append({"role": "assistant", "content": response})
