import json
from collections import Counter
from pathlib import Path


DATASET_PATH = Path(__file__).resolve().parent / "eval_dataset.json"
BASELINE_SIZE = 50


def main() -> None:
    with DATASET_PATH.open(encoding="utf-8") as dataset_file:
        samples = json.load(dataset_file)

    if not isinstance(samples, list):
        raise ValueError("eval_dataset.json 顶层必须是数组")

    baseline = samples[:BASELINE_SIZE]
    source_names = []
    for index, sample in enumerate(baseline, start=1):
        source = sample.get("expected_source") if isinstance(sample, dict) else None
        if not isinstance(source, str) or not source.strip():
            raise ValueError(f"第 {index} 条数据缺少有效的 expected_source")
        source_names.append(source.strip())

    source_counts = Counter(source_names)
    sorted_sources = sorted(source_counts.items(), key=lambda item: (-item[1], item[0]))

    print(f"最初评测集前 {len(baseline)} 条的 expected_source 分布：")
    for source, count in sorted_sources:
        print(f"《{source}》 ： {count} 条")

    top_three = sorted_sources[:3]
    remaining = sorted_sources[3:]
    print(f"\n排名前 3 的文件共 {sum(count for _, count in top_three)} 条。")
    print(f"其余文件共 {sum(count for _, count in remaining)} 条：")
    for source, count in remaining:
        print(f"《{source}》 ： {count} 条")


if __name__ == "__main__":
    main()
