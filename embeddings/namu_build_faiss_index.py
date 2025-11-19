#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
build_faiss_index.py
--------------------
namu_face 테이블의 face_vec (512D float32, L2-normalized) 를 읽어서
Faiss IndexFlatIP + IDMap 인덱스를 만든다.

출력:
- data/namu_facevec_ip.faiss
"""

from __future__ import annotations
import logging
import sqlite3
from pathlib import Path

import numpy as np
import faiss  # pip install faiss-cpu  (또는 faiss-gpu)

BASE_DIR = Path(__file__).resolve().parents[1]
DB_PATH = BASE_DIR / "data" / "namu_wiki_facevec.sqlite"
OUT_INDEX = BASE_DIR / "data" / "namu_facevec_ip.faiss"

FACE_DIM = 512

LOGGER = logging.getLogger("build_faiss")


def setup_logging():
    h = logging.StreamHandler()
    f = logging.Formatter("[%(levelname)s] %(message)s")
    h.setFormatter(f)
    LOGGER.addHandler(h)
    LOGGER.setLevel(logging.INFO)


def load_vectors():
    if not DB_PATH.exists():
        raise SystemExit(f"DB 없음: {DB_PATH}")

    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row

    LOGGER.info("face_vec 로딩 중...")

    cur = conn.execute(
        """
        SELECT id, face_vec
        FROM namu_face
        WHERE face_status = 1
          AND face_vec IS NOT NULL
        ORDER BY id
        """
    )

    ids = []
    vecs = []

    for r in cur:
        pid = int(r["id"])
        blob = r["face_vec"]
        if blob is None:
            continue
        vec = np.frombuffer(blob, dtype="float32")
        if vec.shape[0] != FACE_DIM:
            continue
        vecs.append(vec)
        ids.append(pid)

    conn.close()

    if not vecs:
        raise SystemExit("유효한 face_vec 이 없음")

    x = np.vstack(vecs).astype("float32")
    ids_arr = np.asarray(ids, dtype="int64")

    LOGGER.info("총 벡터 수: %d", x.shape[0])
    return ids_arr, x


def build_index():
    setup_logging()

    ids, x = load_vectors()

    LOGGER.info("Faiss IndexFlatIP + IDMap 생성...")
    index = faiss.IndexFlatIP(FACE_DIM)
    index = faiss.IndexIDMap(index)

    # 안전하게 한 번 더 정규화 (혹시 모를 경우 대비)
    faiss.normalize_L2(x)

    index.add_with_ids(x, ids)

    faiss.write_index(index, str(OUT_INDEX))
    LOGGER.info("인덱스 저장 완료: %s", OUT_INDEX)


def main():
    build_index()


if __name__ == "__main__":
    main()
