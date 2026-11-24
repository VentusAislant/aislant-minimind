"""流式查找训练数据中带有个人、作者或项目品牌标识的样本。"""

import argparse
import json
from collections import Counter
from pathlib import Path


DEFAULT_TERMS = [
    "minimind",
    "jingyaogong",
    "景尧",
    "景曜",
]
SUPPORTED_SUFFIXES = {".jsonl", ".json", ".txt"}


def parse_args():
    parser = argparse.ArgumentParser(description="查找数据集中的个人或品牌标识样本")
    parser.add_argument("--data_dir", type=Path, default=Path("data"), help="需要递归扫描的数据目录")
    parser.add_argument(
        "--output_path",
        type=Path,
        default=Path("outputs/identity_samples.jsonl"),
        help="命中样本的输出报告",
    )
    parser.add_argument(
        "--terms",
        nargs="+",
        default=DEFAULT_TERMS,
        help="需要查找的关键词，不区分英文大小写",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    terms = list(dict.fromkeys(term.strip() for term in args.terms if term.strip()))
    normalized_terms = {term: term.casefold() for term in terms}
    data_files = sorted(
        path for path in args.data_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in SUPPORTED_SUFFIXES
    )

    if not data_files:
        raise FileNotFoundError(f"在 {args.data_dir} 下没有找到可扫描的数据文件")

    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    file_counts = Counter()
    term_counts = Counter()
    scanned_lines = 0
    matched_lines = 0

    with args.output_path.open("w", encoding="utf-8") as output_file:
        for data_file in data_files:
            print(f"正在扫描: {data_file}", flush=True)
            with data_file.open("r", encoding="utf-8", errors="replace") as source_file:
                for line_number, raw_line in enumerate(source_file, start=1):
                    scanned_lines += 1
                    normalized_line = raw_line.casefold()
                    matched_terms = [
                        term for term, normalized_term in normalized_terms.items()
                        if normalized_term in normalized_line
                    ]
                    if not matched_terms:
                        continue

                    matched_lines += 1
                    relative_path = str(data_file.relative_to(args.data_dir))
                    file_counts[relative_path] += 1
                    term_counts.update(matched_terms)

                    try:
                        sample = json.loads(raw_line)
                    except json.JSONDecodeError:
                        sample = raw_line.rstrip("\n")

                    result = {
                        "source_file": relative_path,
                        "line_number": line_number,
                        "matched_terms": matched_terms,
                        "sample": sample,
                    }
                    output_file.write(json.dumps(result, ensure_ascii=False) + "\n")

    print("\n扫描完成")
    print(f"扫描文件数: {len(data_files)}")
    print(f"扫描行数: {scanned_lines}")
    print(f"命中样本数: {matched_lines}")
    print(f"报告位置: {args.output_path}")

    if file_counts:
        print("\n按文件统计:")
        for file_name, count in file_counts.most_common():
            print(f"  {file_name}: {count}")

        print("\n按关键词统计:")
        for term, count in term_counts.most_common():
            print(f"  {term}: {count}")


if __name__ == "__main__":
    main()
