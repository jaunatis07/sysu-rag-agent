import csv
import json
from collections import Counter
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
DATASET_PATH = BASE_DIR / "eval_dataset.json"
DATA_DIR = BASE_DIR / "data"
OUTPUT_PATH = BASE_DIR / "eval_distribution.csv"


def get_source_name(sample: dict) -> str:
    for key in ("expected_source", "source", "source_file"):
        value = sample.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()

    metadata = sample.get("metadata")
    if isinstance(metadata, dict):
        for key in ("source", "source_file", "expected_source"):
            value = metadata.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
    return ""


def main() -> None:
    with DATASET_PATH.open(encoding="utf-8") as dataset_file:
        samples = json.load(dataset_file)

    if not isinstance(samples, list):
        raise ValueError("eval_dataset.json 顶层必须是数组")

    source_names = [get_source_name(sample) for sample in samples]
    if not source_names or not any(source_names):
        raise SystemExit("取不到")

    pdf_paths = sorted(DATA_DIR.glob("**/*.pdf"), key=lambda path: path.name)
    pdf_names = [path.stem for path in pdf_paths]
    coverage = Counter(name for name in source_names if name)

    pdf_coverage = Counter()
    matched_source_names = set()
    for source_name in coverage:
        matched_pdfs = [
            pdf_name
            for pdf_name in pdf_names
            if source_name in pdf_name or pdf_name in source_name
        ]
        if matched_pdfs:
            matched_source_names.add(source_name)
            pdf_coverage.update({pdf_name: coverage[source_name] for pdf_name in matched_pdfs})

    covered_pdfs = set(pdf_coverage)
    uncovered_pdfs = sorted(set(pdf_names) - covered_pdfs)
    unmatched_sources = sorted(set(source_names) - matched_source_names)
    sorted_coverage = sorted(pdf_coverage.items(), key=lambda item: (-item[1], item[0]))

    print("每条样本的来源文件名：")
    for index, (sample, source_name) in enumerate(zip(samples, source_names), start=1):
        print(f"  {index}. [{sample.get('id', index)}] {source_name or '未记录'}")

    print("\n每个知识库 PDF 的模糊匹配覆盖数（按条数降序）：")
    for source_name, count in sorted_coverage:
        print(f"  {source_name}: {count}")

    print("\n未被样本覆盖的知识库 PDF：")
    for pdf_name in uncovered_pdfs:
        print(f"  {pdf_name}.pdf")
    print(f"\n知识库 PDF 总数：{len(pdf_paths)}")
    print(f"已覆盖 PDF 数（模糊包含匹配）：{len(covered_pdfs)}")
    print(f"未覆盖 PDF 数：{len(uncovered_pdfs)}")

    if unmatched_sources:
        print("\n未能模糊匹配到知识库 PDF 的来源记录（前 10 条）：")
        for source_name in unmatched_sources[:10]:
            print(f"  {source_name}")
        if len(unmatched_sources) > 10:
            print(f"  … 另有 {len(unmatched_sources) - 10} 条未显示")

    fieldnames = [
        "record_type",
        "sample_id",
        "question",
        "source_file",
        "sample_count",
        "pdf_count",
        "uncovered_count",
    ]
    with OUTPUT_PATH.open("w", encoding="utf-8-sig", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=fieldnames)
        writer.writeheader()

        for sample, source_name in zip(samples, source_names):
            writer.writerow({
                "record_type": "sample",
                "sample_id": sample.get("id", ""),
                "question": sample.get("question", ""),
                "source_file": source_name,
            })

        for pdf_name, count in sorted_coverage:
            writer.writerow({
                "record_type": "coverage",
                "source_file": f"{pdf_name}.pdf",
                "sample_count": count,
            })

        for pdf_name in uncovered_pdfs:
            writer.writerow({
                "record_type": "uncovered_pdf",
                "source_file": f"{pdf_name}.pdf",
            })

        writer.writerow({
            "record_type": "summary",
            "pdf_count": len(pdf_paths),
            "uncovered_count": len(uncovered_pdfs),
        })

    print(f"\nCSV 已保存到：{OUTPUT_PATH}")


if __name__ == "__main__":
    main()
