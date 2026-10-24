import os
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

# 1. 检查 data 目录
data_dir = "./data"
if not os.path.exists(data_dir):
    print("错误：找不到 data 文件夹，请确认 83 个 PDF 是否在 data 目录下！")
else:
    print("正在加载 data 目录下的所有 PDF 文件...")

    # 2. 批量加载 PDF
    loader = DirectoryLoader(
        data_dir,
        glob="**/*.pdf",
        loader_cls=PyPDFLoader
    )
    docs = loader.load()
    print(f"成功加载了 {len(docs)} 页原始 PDF 内容。")

    # 3. 文本分块（Chunking）- 针对规章制度优化
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,     # 每个 chunk 大约 500 个字符
        chunk_overlap=100   # 相邻 chunk 重叠 100 字符，保证上下文连贯
    )
    splits = text_splitter.split_documents(docs)
    print(f"文本切分完成，共生成了 {len(splits)} 个文本块（Chunks）。")

    # 4. 初始化中文 Embedding 模型（用于将文字转为向量）
    print("正在加载中文向量化模型（Embedding）...")
    embeddings = HuggingFaceEmbeddings(model_name="shibing624/text2vec-base-chinese")

    # 5. 存入本地 Chroma 向量数据库
    persist_directory = "./chroma_db"
    print(f"正在将向量写入本地数据库（保存在 {persist_directory} 目录下）...")
    
    vectorstore = Chroma.from_documents(
        documents=splits,
        embedding=embeddings,
        persist_directory=persist_directory
    )
    print("🎉 恭喜！所有 PDF 已成功向量化并存入向量数据库！")
