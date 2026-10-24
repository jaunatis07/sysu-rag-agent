import streamlit as st
from rag_core import answer_question, create_llm, initialize_rag_components

# 1. 页面基本配置
st.set_page_config(
    page_title="SYSU Policy & Service Agent",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="collapsed",
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

section[data-testid="stSidebar"] {display: none !important;}
[data-testid="collapsedControl"] {display: none !important;}

.welcome {text-align: center; margin: 1rem 0 1.45rem;}
.welcome h1 {margin: 0; color: #29273a; font-size: clamp(1.55rem, 3vw, 2.1rem); line-height: 1.2; letter-spacing: 0;}

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

[data-testid="stChatInput"] {width: 100%; margin: 0; bottom: 0;}
[data-testid="stChatInput"] > div {
    border-radius: 24px !important; border: 2px solid rgba(153, 145, 183, 0.26) !important;
    background: rgba(255, 255, 255, 0.90) !important;
    box-shadow: 0 14px 38px rgba(78, 69, 113, 0.12); backdrop-filter: blur(18px);
}
[data-testid="stChatInput"] textarea {font-size: 0.92rem !important;}
.bottom-composer {padding-bottom: 0.6rem;}
.bottom-composer button[kind="secondary"] {
    width: 48px !important; height: 48px; min-height: 48px; padding: 0 !important;
    border-radius: 50%; border: 1px solid rgba(153, 145, 183, 0.20);
    background: rgba(255, 255, 255, 0.82); color: #756f88;
    box-shadow: 0 8px 20px rgba(78, 69, 113, 0.08);
}
.bottom-composer [data-testid="column"]:first-child {display:flex; align-items:flex-end; justify-content:center;}
.bottom-composer [data-testid="column"]:last-child {padding-left: 0.25rem;}

.stButton > button:hover {border-color: rgba(126, 112, 177, 0.22); background: rgba(255, 255, 255, 0.92); color: #51486f;}
[data-testid="stExpander"] {border: 1px solid rgba(112, 105, 145, 0.09); border-radius: 14px; background: rgba(255, 255, 255, 0.58);}
hr {border-color: rgba(112, 105, 145, 0.08);}

@media (max-width: 700px) {
    .block-container {padding-left: 0.7rem; padding-right: 0.7rem;}
    [data-testid="stChatMessage"] {max-width: 92%;}
    .bottom-composer button[kind="secondary"] {width: 44px !important; height: 44px; min-height: 44px;}
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

# ==================== 5. 主界面 ====================

if "messages" not in st.session_state:
    st.session_state.messages = []

# 顶部居中标题
st.markdown("""
<div class="welcome">
    <h1>有什么校园问题，我来帮你查</h1>
</div>
""", unsafe_allow_html=True)

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

st.markdown('<div class="bottom-composer">', unsafe_allow_html=True)
clear_col, input_col = st.columns([0.06, 0.94], gap="small", vertical_alignment="bottom")
with clear_col:
    if st.button("🗑️", key="clear_chat", help="清空对话", type="secondary"):
        st.session_state.messages = []
        st.rerun()
with input_col:
    query = st.chat_input("在此输入您关于中大教务、学籍、宿舍的疑问...")
st.markdown('</div>', unsafe_allow_html=True)

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
