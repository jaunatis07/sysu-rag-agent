import json
from pathlib import Path
from typing import Any


BASE_DIR = Path(__file__).resolve().parent
INPUT_PATH = BASE_DIR / "eval_dataset_v2.json"
OUTPUT_PATH = BASE_DIR / "eval_dataset_v2_cleaned.json"

BRACKET_PAIRS = {
    ")": "(",
    "）": "（",
    "]": "[",
    "］": "［",
    "】": "【",
    "〕": "〔",
}


def remove_trailing_parenthetical(text: str) -> str:
    text = text.rstrip()
    while text and text[-1] in BRACKET_PAIRS:
        expected_open = BRACKET_PAIRS[text[-1]]
        stack = [expected_open]
        start = len(text) - 2

        while start >= 0 and stack:
            char = text[start]
            if char in BRACKET_PAIRS:
                stack.append(BRACKET_PAIRS[char])
            elif char == stack[-1]:
                stack.pop()
            start -= 1

        if stack:
            break
        text = text[: start + 1].rstrip()
    return text


def clean_source(value: Any) -> Any:
    if not isinstance(value, str):
        return value

    source = value.replace("《", "").replace("》", "").strip()
    if source.lower().endswith(".pdf"):
        source = source[:-4].rstrip()
    return remove_trailing_parenthetical(source)


def main() -> None:
    with INPUT_PATH.open(encoding="utf-8") as input_file:
        dataset = json.load(input_file)

    if not isinstance(dataset, list):
        raise ValueError("eval_dataset_v2.json 顶层必须是数组。")

    for index, item in enumerate(dataset, start=1):
        if not isinstance(item, dict):
            raise ValueError(f"第 {index} 条数据必须是 JSON 对象。")
        if "expected_source" in item:
            item["expected_source"] = clean_source(item["expected_source"])

    with OUTPUT_PATH.open("w", encoding="utf-8") as output_file:
        json.dump(dataset, output_file, ensure_ascii=False, indent=2)
        output_file.write("\n")

    print(f"清洗完成，结果已保存到：{OUTPUT_PATH}")


if __name__ == "__main__":
    main()
