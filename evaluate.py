import json
import os
import re
from pathlib import Path
from typing import Any

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from rag_core import answer_question, create_llm, initialize_rag_components


BASE_DIR = Path(__file__).resolve().parent
DATASET_PATH = BASE_DIR / "eval_dataset.json"
RESULTS_PATH = BASE_DIR / "eval_results.json"
PASS_THRESHOLD = 0.8

JUDGE_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "你是中山大学校园政策问答系统的严格评测裁判。"
            "判断模型回答是否正确回应问题，并与标准答案在关键事实上相符。"
            "意思等价、措辞不同可以通过；关键政策条件遗漏、事实错误、与标准答案矛盾，"
            "或回答没有解决问题则不通过。额外的无依据具体信息也判 FAIL。"
            "只输出 PASS 或 FAIL，不要输出解释。",
        ),
        (
            "human",
            "问题：{question}\n"
            "参考来源：{expected_source}\n"
            "标准答案：{ground_truth}\n"
            "待评测回答：{answer}\n\n"
            "裁判结果（仅 PASS 或 FAIL）：",
        ),
    ]
)


def load_dataset() -> list[dict[str, Any]]:
    if not DATASET_PATH.exists():
        raise FileNotFoundError(f"找不到评测集：{DATASET_PATH}")

    dataset = json.loads(DATASET_PATH.read_text(encoding="utf-8"))
    if not isinstance(dataset, list) or not dataset:
        raise ValueError("eval_dataset.json 必须是非空 JSON 数组。")
    required_fields = ("question", "ground_truth")
    for index, item in enumerate(dataset, start=1):
        if not isinstance(item, dict) or any(
            not isinstance(item.get(field), str) or not item[field].strip()
            for field in required_fields
        ):
            raise ValueError(f"第 {index} 条数据必须包含非空 question 和 ground_truth。")
    return dataset


def judge_answer(
    judge: ChatOpenAI,
    question: str,
    expected_source: str,
    ground_truth: str,
    answer: str,
) -> str:
    response = judge.invoke(
        JUDGE_PROMPT.format(
            question=question,
            expected_source=expected_source or "未指定",
            ground_truth=ground_truth,
            answer=answer,
        )
    ).content
    if not isinstance(response, str):
        response = str(response)

    match = re.match(r"^\s*(PASS|FAIL)\b", response.strip(), re.IGNORECASE)
    if not match:
        raise ValueError(f"裁判返回了无法识别的结果：{response!r}")
    return match.group(1).upper()


def main() -> None:
    if not os.getenv("DEEPSEEK_API_KEY"):
        raise EnvironmentError("请先设置 DEEPSEEK_API_KEY 环境变量。")

    dataset = load_dataset()
    bm25_retriever = initialize_rag_components()
    rag_llm = create_llm()
    judge = ChatOpenAI(
        model="deepseek-chat",
        api_key=os.environ["DEEPSEEK_API_KEY"],
        base_url="https://api.deepseek.com/v1",
        temperature=0,
    )

    results = []
    passed = 0
    for index, item in enumerate(dataset, start=1):
        question = item["question"].strip()
        answer, _, _ = answer_question(
            question,
            bm25_retriever,
            rag_llm,
        )
        verdict = judge_answer(
            judge,
            question,
            str(item.get("expected_source", "")),
            item["ground_truth"].strip(),
            answer,
        )
        passed += verdict == "PASS"
        results.append(
            {
                "id": item.get("id", index),
                "question": question,
                "expected_source": item.get("expected_source", ""),
                "ground_truth": item["ground_truth"].strip(),
                "answer": answer,
                "verdict": verdict,
            }
        )
        print(f"[{index}/{len(dataset)}] {verdict}: {question}")

    accuracy = passed / len(dataset)
    report = {
        "total": len(dataset),
        "passed": passed,
        "failed": len(dataset) - passed,
        "accuracy": accuracy,
        "threshold": PASS_THRESHOLD,
        "meets_threshold": accuracy >= PASS_THRESHOLD,
        "results": results,
    }
    RESULTS_PATH.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print(f"\n准确率：{passed}/{len(dataset)} = {accuracy:.1%}")
    print(f"达标线：{PASS_THRESHOLD:.0%}，{'达标' if accuracy >= PASS_THRESHOLD else '未达标'}")
    print(f"详细结果：{RESULTS_PATH}")


if __name__ == "__main__":
    main()