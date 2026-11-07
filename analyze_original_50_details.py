import csv
import json
import re
from collections import Counter
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
DATASET_PATH = BASE_DIR / "eval_dataset.json"
DATA_DIR = BASE_DIR / "data"
OUTPUT_PATH = BASE_DIR / "original_50_details.csv"
BASELINE_SIZE = 50
TRAILING_QUALIFIER = re.compile(r"[\(（][^()（）]*[\)）]\s*$")


def normalize_source_name(value: str) -> str:
    name = value.strip()
    while True:
        normalized = TRAILING_QUALIFIER.sub("", name).strip()
        if normalized == name:
            return name
        name = normalized


def find_pdf(source: str, pdf_names: list[str]) -> str | None:
    source_name = normalize_source_name(source)
    if not source_name:
        return None

    matches = [
        pdf_name
        for pdf_name in pdf_names
        if source_name in normalize_source_name(Path(pdf_name).stem)
        or normalize_source_name(Path(pdf_name).stem) in source_name
    ]
    return matches[0] if len(matches) == 1 else None


def main() -> None:
    with DATASET_PATH.open(encoding="utf-8") as dataset_file:
        samples = json.load(dataset_file)

    if not isinstance(samples, list):
        raise ValueError("eval_dataset.json 顶层必须是数组")

    baseline = samples[:BASELINE_SIZE]
    pdf_names = sorted(path.name for path in DATA_DIR.rglob("*.pdf"))
    rows = []
    for index, sample in enumerate(baseline, start=1):
        if not isinstance(sample, dict):
            raise ValueError(f"第 {index} 条数据必须是 JSON 对象")

        raw_source = sample.get("expected_source", "")
        raw_source = raw_source if isinstance(raw_source, str) else ""
        matched_pdf = find_pdf(raw_source, pdf_names)
        rows.append({
            "题号": sample.get("id", index),
            "问题": sample.get("question", ""),
            "expected_source原值": raw_source,
            "归属PDF文件名": matched_pdf or "未匹配",
            "是否匹配成功": "是" if matched_pdf else "否",
        })

    fieldnames = [
        "题号",
        "问题",
        "expected_source原值",
        "归属PDF文件名",
        "是否匹配成功",
    ]
    with OUTPUT_PATH.open("w", encoding="utf-8-sig", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    counts = Counter(row["归属PDF文件名"] for row in rows)
    sorted_counts = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    print(f"前 {len(rows)} 条按归属后的文件名统计：")
    for pdf_name, count in sorted_counts:
        print(f"《{pdf_name}》 ： {count} 条")

    top_three_total = sum(count for _, count in sorted_counts[:3])
    remaining_total = sum(count for _, count in sorted_counts[3:])
    print(f"\n前 3 名合计：{top_three_total} 条")
    print(f"剩余文件合计：{remaining_total} 条")
    print(f"\n明细 CSV 已保存到：{OUTPUT_PATH}")


if __name__ == "__main__":
    main()
