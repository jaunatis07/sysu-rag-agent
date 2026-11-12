import json
import os
import re
import time
from pathlib import Path
from typing import Any, Callable

from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from rag_core import answer_question, create_llm, initialize_rag_components

# 自动加载项目根目录 .env 里的 DEEPSEEK_API_KEY，无需每次手动 export
load_dotenv(Path(__file__).resolve().parent / ".env")


BASE_DIR = Path(__file__).resolve().parent
DATASET_PATH = BASE_DIR / "eval_dataset_v2.json"
RESULTS_PATH = BASE_DIR / "eval_results_v2.json"
CHECKPOINT_PATH = BASE_DIR / "eval_checkpoint.json"  # 断点续跑：每判完一条立即落盘
PASS_THRESHOLD = 0.8
MAX_RETRIES = 3          # API 瞬时故障（限流/超时）自动重试次数
RETRY_WAIT_SECONDS = 10  # 重试等待基数，按次数递增

JUDGE_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "你是中山大学校园政策问答系统的严格评测裁判。"
            "判断模型回答是否正确回应问题，并与标准答案在关键事实上相符。"
            "意思等价、措辞不同可以通过；关键政策条件遗漏、事实错误、与标准答案矛盾，"
            "或回答没有解决问题则不通过。额外的无依据具体信息也判 FAIL。"
            "如果回答准确提取了标准答案中的核心事实（如数字、条款），即使句式和标点不同也必须判 PASS；"
            "不要仅因少了不影响核心含义的修饰词而判 FAIL。"
            "来源文件名判断应宽容：若回答引用的文件名包含参考来源 expected_source 的核心关键词，即视为来源匹配；"
            "书名号、括号补充说明及是否带“.pdf”等格式差异，不得单独作为判 FAIL 的理由。"
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
        raise ValueError("eval_dataset_v2.json 必须是非空 JSON 数组。")
    required_fields = ("question", "ground_truth")
    for index, item in enumerate(dataset, start=1):
        if not isinstance(item, dict) or any(
            not isinstance(item.get(field), str) or not item[field].strip()
            for field in required_fields
        ):
            raise ValueError(f"第 {index} 条数据必须包含非空 question 和 ground_truth。")
    return dataset


def invoke_with_retry(call: Callable[[], Any], label: str) -> Any:
    """API 瞬时故障重试：最多 MAX_RETRIES 次，等待时间按次数递增。"""
    last_exc: Exception | None = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            return call()
        except Exception as exc:  # noqa: BLE001 网络层异常类型较多，统一兜住重试
            last_exc = exc
            wait = RETRY_WAIT_SECONDS * attempt
            print(f"\n[{label}] 第 {attempt}/{MAX_RETRIES} 次调用失败（{type(exc).__name__}），{wait} 秒后重试…")
            time.sleep(wait)
    raise RuntimeError(f"[{label}] 重试 {MAX_RETRIES} 次仍失败") from last_exc


def get_answer(question: str, bm25_retriever, rag_llm) -> str:
    """生成回答。生成层异常会内部降级为本地兜底文案，视为本次调用失败并重试。"""

    def call() -> str:
        answer, _, _ = answer_question(question, bm25_retriever, rag_llm)
        if answer.startswith("当前无法使用 DeepSeek"):
            raise RuntimeError("生成降级（API 异常）")
        return answer

    return invoke_with_retry(call, f"回答 id={question[:12]}…")


def judge_answer(
    judge: ChatOpenAI,
    question: str,
    expected_source: str,
    ground_truth: str,
    answer: str,
) -> str:
    response = invoke_with_retry(
        lambda: judge.invoke(
            JUDGE_PROMPT.format(
                question=question,
                expected_source=expected_source or "未指定",
                ground_truth=ground_truth,
                answer=answer,
            )
        ).content,
        f"判卷 id={question[:12]}…",
    )
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

    # 断点续跑：读取已有进度，跳过已判定的题目
    done: dict[str, dict[str, Any]] = {}
    if CHECKPOINT_PATH.exists():
        done = json.loads(CHECKPOINT_PATH.read_text(encoding="utf-8"))
        print(f"检测到断点文件，已完成 {len(done)} 条，自动跳过。")

    results = []
    for index, item in enumerate(dataset, start=1):
        key = str(item.get("id", index))
        if key in done:
            results.append(done[key])
            continue

        question = item["question"].strip()
        answer = get_answer(question, bm25_retriever, rag_llm)
        verdict = judge_answer(
            judge,
            question,
            str(item.get("expected_source", "")),
            item["ground_truth"].strip(),
            answer,
        )
        record = {
            "id": item.get("id", index),
            "question": question,
            "expected_source": item.get("expected_source", ""),
            "ground_truth": item["ground_truth"].strip(),
            "answer": answer,
            "verdict": verdict,
        }
        results.append(record)

        # 每判完一条立即落盘，中途崩溃也不丢进度
        done[key] = record
        CHECKPOINT_PATH.write_text(
            json.dumps(done, ensure_ascii=False), encoding="utf-8"
        )
        print(f"[{index}/{len(dataset)}] {verdict}: {question}")

    passed = sum(result["verdict"] == "PASS" for result in results)
    core_results = results[:50]
    tail_results = results[50:130]
    core_passed = sum(result["verdict"] == "PASS" for result in core_results)
    tail_passed = sum(result["verdict"] == "PASS" for result in tail_results)
    core_accuracy = core_passed / len(core_results) if core_results else 0
    tail_accuracy = tail_passed / len(tail_results) if tail_results else 0

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
    CHECKPOINT_PATH.unlink(missing_ok=True)  # 全量跑完后清除断点文件

    print(f"\n准确率：{passed}/{len(dataset)} = {accuracy:.1%}")
    print(f"前 50 条（原有核心文件）准确率：{core_passed}/{len(core_results)} = {core_accuracy:.1%}")
    print(f"后 80 条（新补充长尾文件）准确率：{tail_passed}/{len(tail_results)} = {tail_accuracy:.1%}")
    print(f"达标线：{PASS_THRESHOLD:.0%}，{'达标' if accuracy >= PASS_THRESHOLD else '未达标'}")
    print(f"详细结果：{RESULTS_PATH}")


if __name__ == "__main__":
    main()
