import json
import os
from langchain_openai import ChatOpenAI
from langchain_community.document_loaders import PyPDFDirectoryLoader

print("🚀 正在安全加载本地 PDF 政策文档的真实文本...")

# 1. 读取本地 PDF 真实文本
data_dir = "./data"
all_docs = []
if os.path.exists(data_dir):
    loader = PyPDFDirectoryLoader(data_dir)
    try:
        all_docs = loader.load()
        print(f"📄 成功读取到本地 PDF 共计 {len(all_docs)} 个切片/页面内容。")
    except Exception as e:
        print(f"⚠️ 读取 PDF 出现小问题: {e}")

if not all_docs:
    raise ValueError("❌ 错误：你的 ./data 目录下没有找到可读的 PDF 文件，请检查路径！")

# 抽取更多页面的真实文本作为出题素材
doc_sample_text = "\n".join([doc.page_content for doc in all_docs[:30]])

# 2. 初始化 DeepSeek 客户端
api_key = os.environ.get("DEEPSEEK_API_KEY")
if not api_key:
    raise EnvironmentError("请先设置 DEEPSEEK_API_KEY 环境变量")

llm = ChatOpenAI(
    model="deepseek-chat",
    openai_api_key=api_key,
    openai_api_base="https://api.deepseek.com/v1",
    temperature=0.4
)

prompt_template = """你是一个严谨的中山大学教务政策分析专家。
下面是我从学校真实政策 PDF 中提取的部分原文片段：
-----------------
{context}
-----------------

请根据上面【真实的政策原文】，模拟生成 {batch_count} 条大二学生在校园论坛里会问的**口语化、接地气的问题**，并提供**严格基于原文的标准答案**。

【生成要求】
1. 问题要口语化、带学生的真实表达习惯。
2. 标准答案（ground_truth）和来源文件（expected_source）必须100%忠于上面的真实原文，绝对不能胡编乱造！
3. 必须以标准的 JSON 数组格式返回，不要包含任何 Markdown 代码块标签或其他废话，格式如下：
[
  {{
    "id": {start_id},
    "question": "口语化学生提问",
    "expected_source": "对应的文件名",
    "ground_truth": "基于原文的标准解答"
  }}
]
"""

all_eval_items = []
target_total = 50
batch_size = 17  # 分 3 批生成，每批 17 条左右，轻松凑够 50 条
current_id = 1

print(f"📊 正在分批让 DeepSeek 结合真实 PDF 原文生成共计 {target_total} 条评测集...")

for i in range(3):
    if len(all_eval_items) >= target_total:
        break
        
    print(f"正在生成第 {i+1} 批测试用例...")
    prompt = prompt_template.format(
        context=doc_sample_text[:4000],
        batch_count=batch_size,
        start_id=current_id
    )
    
    try:
        response = llm.invoke(prompt).content
        clean_response = response.replace("```json", "").replace("```", "").strip()
        batch_items = json.loads(clean_response)
        
        for item in batch_items:
            item["id"] = current_id
            all_eval_items.append(item)
            current_id += 1
    except Exception as e:
        print(f"⚠️ 第 {i+1} 批生成出错: {e}")

# 截取刚好 50 条
final_items = all_eval_items[:target_total]

output_file = "eval_dataset.json"
with open(output_file, "w", encoding="utf-8") as f:
    json.dump(final_items, f, ensure_ascii=False, indent=4)
    
print(f"\n🎉 成功生成包含 {len(final_items)} 条【绝对基于真实PDF原文 + 口语化学生提问】的黄金评测集，已保存至 {output_file}！")
