import json
import os
import re
import time
from pathlib import Path

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from pypdf import PdfReader


BASE_DIR = Path(__file__).resolve().parent
DATASET_PATH = BASE_DIR / "eval_dataset.json"
DISTRIBUTION_PATH = BASE_DIR / "eval_distribution.csv"
DATA_DIR = BASE_DIR / "data"
OUTPUT_PATH = BASE_DIR / "eval_dataset_v2.json"
REQUEST_INTERVAL_SECONDS = 1.0
MAX_RETRIES = 3

load_dotenv(BASE_DIR / ".env")

GENERATION_PROMPT = """你是一名中山大学校园政策评测集编写专家。请严格依据下面提供的真实政策原文，只生成一条高质量评测数据。

【来源文件】
{source_name}

【政策原文】
{document_text}

【任务要求】
1. question：学生可能在实际办理或咨询时提出的口语化问题，只能由原文支持。
2. ground_truth：用简洁、准确的中文回答，包含原文明确给出的关键条件、流程或数值；不得补充原文外的知识。
3. expected_source：原样使用上面的【来源文件】。
4. 只输出一个 JSON 对象，不要输出 Markdown、代码块或其他说明。
5. JSON 格式：{{"question":"...","ground_truth":"...","expected_source":"..."}}
"""


def read_uncovered_source_names() -> list[str]:
    if not DISTRIBUTION_PATH.exists():
        raise FileNotFoundError(f"找不到统计文件：{DISTRIBUTION_PATH}")

    source_names = []
    with DISTRIBUTION_PATH.open(encoding="utf-8-sig", newline="") as csv_file:
        import csv

        for row in csv.DictReader(csv_file):
            if row.get("record_type") != "uncovered_pdf":
                continue
            source_name = (row.get("source_file") or "").strip()
            if source_name:
                source_names.append(source_name)
    return source_names


def extract_pdf_text(pdf_path: Path, max_chars: int = 12000) -> str:
    reader = PdfReader(str(pdf_path))
    text_parts = []
    for page in reader.pages:
        page_text = page.extract_text() or ""
        text_parts.append(page_text)
        if sum(len(part) for part in text_parts) >= max_chars:
            break
    return "\n".join(text_parts).strip()[:max_chars]


def parse_json_object(content: str) -> dict:
    content = content.strip()
    content = re.sub(r"^```(?:json)?\s*", "", content, flags=re.I)
    content = re.sub(r"\s*```$", "", content)

    try:
        parsed = json.loads(content)
    except json.JSONDecodeError as exc:
        raise ValueError(f"模型返回的不是合法 JSON：{exc}\n原文：{content}") from exc

    if not isinstance(parsed, dict):
        raise ValueError("模型返回的 JSON 顶层不是对象")

    required = ("question", "ground_truth", "expected_source")
    for field in required:
        if not isinstance(parsed.get(field), str) or not parsed[field].strip():
            raise ValueError(
                f"模型返回缺少非空字段：{field}\n"
                f"【大模型原始返回文本】\n{content}"
            )
    return parsed


def invoke_with_retry(llm: ChatOpenAI, prompt: str) -> str:
    last_error = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = llm.invoke(prompt)
            content = response.content
            if not isinstance(content, str):
                content = str(content)
            return content
        except Exception as exc:
            last_error = exc
            print(f"  第 {attempt}/{MAX_RETRIES} 次调用失败：{exc}")
            if attempt < MAX_RETRIES:
                time.sleep(REQUEST_INTERVAL_SECONDS * attempt)
    raise RuntimeError(f"调用 DeepSeek 重试 {MAX_RETRIES} 次后仍失败") from last_error


def main() -> None:
    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        raise EnvironmentError("请先设置 DEEPSEEK_API_KEY 环境变量")

    if not DATASET_PATH.exists():
        raise FileNotFoundError(f"找不到评测集：{DATASET_PATH}")

    existing_items = json.loads(DATASET_PATH.read_text(encoding="utf-8"))
    if not isinstance(existing_items, list):
        raise ValueError("eval_dataset.json 顶层必须是数组")

    uncovered_names = read_uncovered_source_names()
    if not uncovered_names:
        raise ValueError("eval_distribution.csv 中没有 uncovered_pdf 记录")

    llm = ChatOpenAI(
        model="deepseek-chat",
        api_key=api_key,
        base_url="https://api.deepseek.com/v1",
        temperature=0.2,
    )

    generated_items = []
    next_id = max((item.get("id", 0) for item in existing_items if isinstance(item, dict)), default=0) + 1
    total = len(uncovered_names)

    for index, source_name in enumerate(uncovered_names, start=1):
        pdf_path = DATA_DIR / source_name
        print(f"[{index}/{total}] 处理中：{source_name}")

        if not pdf_path.is_file():
            print("  跳过：知识库中找不到该 PDF")
            continue

        try:
            document_text = extract_pdf_text(pdf_path)
            if not document_text:
                print("  跳过：PDF 未提取到可用文本")
                continue

            prompt = GENERATION_PROMPT.format(
                source_name=source_name,
                document_text=document_text,
            )
            item = parse_json_object(invoke_with_retry(llm, prompt))
            if item["expected_source"] != source_name:
                raise ValueError(
                    f"模型返回的来源与目标 PDF 不一致："
                    f"{item['expected_source']!r} != {source_name!r}"
                )

            item = {
                "id": next_id,
                "question": item["question"].strip(),
                "expected_source": source_name,
                "ground_truth": item["ground_truth"].strip(),
            }
            generated_items.append(item)
            next_id += 1
            print("  已生成一条")
        except Exception:
            import traceback

            traceback.print_exc()

        if index < total:
            time.sleep(REQUEST_INTERVAL_SECONDS)

    augmented_items = [*existing_items, *generated_items]
    OUTPUT_PATH.write_text(
        json.dumps(augmented_items, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print(f"\n原评测集：{len(existing_items)} 条")
    print(f"成功生成：{len(generated_items)} 条")
    print(f"最终评测集：{len(augmented_items)} 条")
    print(f"已保存到：{OUTPUT_PATH}")


if __name__ == "__main__":
    main()
