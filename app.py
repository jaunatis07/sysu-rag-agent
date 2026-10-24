import streamlit as st
from rag_core import answer_question, create_llm, initialize_rag_components

# 1. 页面基本配置
st.set_page_config(
    page_title="SYSU Policy & Service Agent",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 2. 注入高级感定制 CSS（参考高端杂志与毛玻璃风格）
st.markdown("""
    <style>
    /* 隐藏默认元素 */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* 全局背景与字体基调 */
    .stApp {
        background: linear-gradient(135deg, #f6f8fb 0%, #edf1f7 100%);
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    }
    
    /* 侧边栏精美化 */
    section[data-testid="stSidebar"] {
        background-color: #ffffff;
        border-right: 1px solid rgba(0, 45, 98, 0.06);
    }
    
    /* 顶栏主标题样式（具有 Editorial 杂志感） */
    .hero-container {
        padding: 2rem 1rem 1rem 1rem;
        background: rgba(255, 255, 255, 0.7);
        backdrop-filter: blur(20px);
        border-radius: 20px;
        border: 1px solid rgba(255, 255, 255, 0.8);
        box-shadow: 0 10px 30px rgba(0, 45, 98, 0.03);
        margin-bottom: 2rem;
    }
    .main-title {
        font-size: 28px;
        font-weight: 800;
        letter-spacing: -0.5px;
        color: #002D62; /* 中大蓝 */
        margin-bottom: 4px;
    }
    .sub-title {
        font-size: 14px;
        color: #64748b;
        font-weight: 400;
    }
    
    /* 快捷按钮卡片美化 */
    .stButton>button {
        border-radius: 12px;
        border: 1px solid rgba(0, 45, 98, 0.1);
        background-color: #ffffff;
        color: #002D62;
        font-weight: 500;
        transition: all 0.3s ease;
        box-shadow: 0 2px 5px rgba(0,0,0,0.02);
    }
    .stButton>button:hover {
        background-color: #002D62;
        color: #ffffff;
        border-color: #002D62;
        box-shadow: 0 5px 15px rgba(0, 45, 98, 0.15);
    }
    
    /* 聊天气泡与输入框微调 */
    .stChatInputContainer {
        padding-bottom: 1rem;
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

# ==================== 5. 侧边栏设计 (Sidebar) ====================
with st.sidebar:
    st.markdown("### ✦ SYSU AGENT")
    st.caption("Editorial & Intelligent Assistant")
    st.divider()
    
    st.markdown("#### 📊 系统运行状态")
    st.markdown("- **知识库**: 83 份官方 PDF")
    st.markdown("- **引擎**: 本地 BM25 文本检索")
    st.markdown("- **模型**: DeepSeek-V3")
    
    st.divider()
    st.markdown("#### ⚡ 快捷场景体验")
    if st.button("📋 办理课程免修申请"):
        st.session_state.quick_query = "我想申请免修，请问哪里可以下载免修申请表？"
    if st.button("📞 查询教务部联系方式"):
        st.session_state.quick_query = "教务部的办公地点和电话是多少？"
    if st.button("🔄 期末考试不及格补考政策"):
        st.session_state.quick_query = "大一学生如果考试不及格，可以参加补考吗？"
        
    st.divider()
    if st.button("✨ 清空当前对话", type="secondary"):
        st.session_state.messages = []
        st.rerun()

# ==================== 6. 主界面排版 ====================
st.markdown("""
    <div class="hero-container">
        <div class="main-title">中山大学校园政策咨询与办事助理</div>
        <div class="sub-title">精准溯源、多轮对话与智能办事工具的无缝融合。</div>
    </div>
""", unsafe_allow_html=True)

if "messages" not in st.session_state:
    st.session_state.messages = []

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
