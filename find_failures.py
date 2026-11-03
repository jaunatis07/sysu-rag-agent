import json
from collections import defaultdict
from pathlib import Path
from typing import Any


RESULTS_PATH = Path(__file__).resolve().parent / "eval_results_v2.json"
FAIL_MARKERS = (
    "FAIL",
    "FAILED",
    "NOT PASS",
    "NOT_PASS",
    "NOT-PASS",
    "不通过",
    "未通过",
    "错误",
    "WRONG",
    "INCORRECT",
)


def is_failure(verdict: Any) -> bool:
    if not isinstance(verdict, str):
        return False
    normalized = verdict.strip().upper()
    return any(marker in normalized for marker in FAIL_MARKERS)


def main() -> None:
    with RESULTS_PATH.open(encoding="utf-8") as results_file:
        report = json.load(results_file)

    if not isinstance(report, dict) or not isinstance(report.get("results"), list):
        raise ValueError("eval_results_v2.json 必须包含 results 数组。")

    failures: list[dict[str, Any]] = []
    for index, result in enumerate(report["results"], start=1):
        if not isinstance(result, dict):
            raise ValueError(f"results 第 {index} 条必须是 JSON 对象。")
        if is_failure(result.get("verdict")):
            failures.append(result)

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for result in failures:
        source = result.get("expected_source")
        source_name = source.strip() if isinstance(source, str) and source.strip() else "未记录"
        grouped[source_name].append(result)

    print(f"失败题目：{len(failures)} 条\n")
    for source_name in sorted(grouped, key=lambda name: (-len(grouped[name]), name)):
        items = grouped[source_name]
        print(f"【{source_name}】（{len(items)} 条）")
        for result in items:
            question = result.get("question", "未记录")
            ground_truth = result.get("ground_truth", "未记录")
            print(f"  expected_source: {source_name}")
            print(f"  question: {question}")
            print(f"  ground_truth: {ground_truth}\n")


if __name__ == "__main__":
    main()
