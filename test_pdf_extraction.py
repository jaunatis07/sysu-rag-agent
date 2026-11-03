from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"


def main() -> None:
    pdf_paths = sorted(DATA_DIR.glob("**/*申请表*.pdf"))
    if not pdf_paths:
        raise FileNotFoundError(f"在 {DATA_DIR} 下没有找到文件名包含“申请表”的 PDF。")

    for pdf_path in pdf_paths:
        documents = PyPDFLoader(str(pdf_path)).load()
        extracted_text = "\n".join(document.page_content for document in documents)
        print(f"===== {pdf_path.name} =====")
        print(extracted_text[:500])
        print()


if __name__ == "__main__":
    main()
