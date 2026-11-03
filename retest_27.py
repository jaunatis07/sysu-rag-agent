import os
import re
from pathlib import Path

from langchain_openai import ChatOpenAI

from rag_core import answer_with_tools, create_llm, initialize_rag_components


BASE_DIR = Path(__file__).resolve().parent
HISTORICAL_FAILURES_PATH = BASE_DIR / "failures_report.txt"
EXPECTED_FAILURE_COUNT = 27

STRICT_JUDGE_SYSTEM_PROMPT = (
    "你是中山大学校园政策问答系统的严格评测裁判。"
    "判断模型回答是否正确回应问题，并与标准答案在关键事实上相符。"
    "意思等价、措辞不同可以通过；关键政策条件遗漏、事实错误、与标准答案矛盾，"
    "或回答没有解决问题则不通过。额外的无依据具体信息也判 FAIL。"
    "只输出 PASS 或 FAIL，不要输出解释。"
)

STRICT_JUDGE_USER_PROMPT = (
    "问题：{question}\n"
    "参考来源：{expected_source}\n"
    "标准答案：{ground_truth}\n"
    "待评测回答：{answer}\n\n"
    "裁判结果（仅 PASS 或 FAIL）："
)


def load_historical_failures() -> list[dict[str, str]]:
    if not HISTORICAL_FAILURES_PATH.exists():
        raise FileNotFoundError(
            f"找不到历史错题清单：{HISTORICAL_FAILURES_PATH}"
        )

    text = HISTORICAL_FAILURES_PATH.read_text(encoding="utf-8")
    lines = text.splitlines()
    if not lines or lines[0].strip() != f"失败题目：{EXPECTED_FAILURE_COUNT} 条":
        raise ValueError(
            f"历史错题清单首行必须标明 {EXPECTED_FAILURE_COUNT} 条失败记录。"
        )

    failures: list[dict[str, str]] = []
    current: dict[str, str] = {}
    field_pattern = re.compile(r"^  (expected_source|question|ground_truth): (.*)$")

    for line in lines[1:]:
        match = field_pattern.match(line)
        if not match:
            continue

        field, value = match.groups()
        if field == "expected_source":
            if current:
                failures.append(current)
            current = {}
        current[field] = value

    if current:
        failures.append(current)

    required_fields = ("expected_source", "question", "ground_truth")
    for index, failure in enumerate(failures, start=1):
        if any(not failure.get(field, "").strip() for field in required_fields):
            raise ValueError(f"历史错题清单第 {index} 条缺少必需字段。")
    if len(failures) != EXPECTED_FAILURE_COUNT:
        raise ValueError(
            f"历史错题清单应包含 {EXPECTED_FAILURE_COUNT} 条记录，"
            f"实际解析到 {len(failures)} 条。"
        )
    return failures


def judge_answer(
    judge: ChatOpenAI,
    question: str,
    expected_source: str,
    ground_truth: str,
    answer: str,
) -> str:
    prompt = STRICT_JUDGE_USER_PROMPT.format(
        question=question,
        expected_source=expected_source or "未指定",
        ground_truth=ground_truth,
        answer=answer,
    )
    response = judge.invoke(
        [
            ("system", STRICT_JUDGE_SYSTEM_PROMPT),
            ("human", prompt),
        ]
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

    failures = load_historical_failures()
    bm25_retriever = initialize_rag_components()
    rag_llm = create_llm()
    judge = ChatOpenAI(
        model="deepseek-chat",
        api_key=os.environ["DEEPSEEK_API_KEY"],
        base_url="https://api.deepseek.com/v1",
        temperature=0,
    )

    rescued = 0
    still_failed: list[dict[str, str]] = []
    for index, item in enumerate(failures, start=1):
        answer, _, _ = answer_with_tools(
            item["question"],
            bm25_retriever,
            rag_llm,
        )
        verdict = judge_answer(
            judge,
            item["question"],
            item["expected_source"],
            item["ground_truth"],
            answer,
        )
        print(f"[{index}/{len(failures)}] {verdict}: {item['question']}")
        if verdict == "PASS":
            rescued += 1
        else:
            still_failed.append({**item, "answer": answer})

    print(f"\n救回：{rescued} 题")
    print(f"依然失败：{len(still_failed)} 题")

    if still_failed:
        print("\n===== 依然失败的题目 =====")
        for index, item in enumerate(still_failed, start=1):
            print(f"\n【{index}】问题：{item['question']}")
            print(f"标准答案：{item['ground_truth']}")
            print(f"Agent 现在的回答：{item['answer']}")


if __name__ == "__main__":
    main()
