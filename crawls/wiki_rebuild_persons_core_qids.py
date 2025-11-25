#!/usr/bin/env python
import polars as pl
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

P18_PATH = ROOT / "crawls" / "wiki_derived_persons_core_P18.parquet"
NOP18_PATH = ROOT / "crawls" / "wiki_derived_persons_core_without_P18.parquet"
OUT_PATH = ROOT / "crawls" / "wiki_persons_core_qids.csv"


def extract_qids_from_series(s: pl.Series) -> pl.Series:
    """
    s: 어떤 타입이든 받을 수 있음 (Utf8, List, Int, Null)
    return: QID 문자열만 들어있는 단일 Series
    """

    # List 타입인 경우 → explode 해서 string으로 변환
    if s.dtype == pl.List:
        # explode 후 각 원소를 문자열로 변환
        s = s.explode().cast(pl.Utf8, strict=False)

    # 문자열이 아닌 경우 → 문자열로 강제 변환
    elif s.dtype != pl.Utf8:
        s = s.cast(pl.Utf8, strict=False)

    # 이제 s는 Utf8 또는 Null
    # 셀마다 QID 패턴 모아 리스트로 추출 후 explode
    return s.str.extract_all(r"Q\d+").explode()


def main():
    print("[info] loading parquet files...")
    df_p18 = pl.read_parquet(P18_PATH)
    df_no = pl.read_parquet(NOP18_PATH)
    df = pl.concat([df_p18, df_no], how="vertical_relaxed")

    print("[info] collecting QIDs (excluding first col qid)...")
    cols = [c for c in df.columns if c != "qid"]

    qid_series_list = []

    for col in cols:
        print(f"[info] processing column: {col}")
        qid_s = extract_qids_from_series(df[col])
        qid_series_list.append(qid_s.alias("q"))

    print("[info] concatenating...")
    all_q = pl.concat(qid_series_list)

    print("[info] cleaning...")
    all_q = all_q.drop_nulls().unique()

    out_df = all_q.to_frame("q")

    OUT_PATH.unlink(missing_ok=True)
    out_df.write_csv(OUT_PATH)

    print(f"[done] wrote {out_df.height:,} QIDs → {OUT_PATH}")


if __name__ == "__main__":
    main()
