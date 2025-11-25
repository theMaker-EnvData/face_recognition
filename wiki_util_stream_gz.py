#!/usr/bin/env python
"""
Wikidata latest-all.json.gz 스트리밍 뷰어

- 실제 gz 파일을 gzip 스트리밍으로 열어서 앞부분 N줄을 그대로 보여줌
- 줄 번호와 repr() 형태로 찍어서, 공백/슬래시/따옴표까지 확인 가능
- 이걸로 실제 포맷이 "엔티티 1줄"인지 "멀티라인"인지 바로 확인 가능함

사용 예:
    python show_wikidata_gz_head.py          # 기본 50줄
    python show_wikidata_gz_head.py --max-lines 200
"""

import argparse
import gzip
from pathlib import Path

DUMP_GZ = Path(
    r"C:\Users\iamth\Downloads\dumps.wikimedia.org_wikidatawiki_entities_latest-all.json.gz"
)


def stream_head(max_lines: int, skip_empty: bool = False) -> None:
    if not DUMP_GZ.exists():
        raise SystemExit(f"[error] dump not found: {DUMP_GZ}")

    print(f"[info] streaming from: {DUMP_GZ}")
    print(f"[info] max_lines = {max_lines}, skip_empty = {skip_empty}")
    print("-" * 80)

    count = 0
    with gzip.open(DUMP_GZ, "rt", encoding="utf-8", errors="replace") as f:
        for line_no, raw_line in enumerate(f, start=1):
        
            line = raw_line.rstrip("\n")

            if skip_empty and not line.strip():
                continue

            count += 1
            print(f"{count:6d} (file line {line_no:6d}): {repr(line)}")

            if count >= max_lines:
                break

    print("-" * 80)
    print(f"[done] printed {count} lines")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--max-lines",
        type=int,
        default=50,
        help="출력할 최대 줄 수 (기본 50)",
    )
    parser.add_argument(
        "--skip-empty",
        action="store_true",
        help="내용 없는 줄은 건너뛰기",
    )
    args = parser.parse_args()
    stream_head(max_lines=args.max_lines, skip_empty=args.skip_empty)


if __name__ == "__main__":
    main()
