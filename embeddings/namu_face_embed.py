#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
namu_face_embed.py
------------------
- SRC_DB: namu_wiki_persons.sqlite (읽기 전용)
- DST_DB: data/namu_wiki_facevec.sqlite
- 테이블: namu_face
    id, page_id, page_title, page_url, image_path, face_vec, face_status

GPU 사용 + 샤딩 방식 병렬 처리
----------------------------
- 기본은 단일 프로세스로 실행 (shard=0, num_shards=1).
- 여러 프로세스로 병렬 실행하고 싶으면:

  예) num_shards=4 로 나누고 4개 터미널에서 동시에 실행:
    python -m embeddings.namu_face_embed --shard 0 --num-shards 4
    python -m embeddings.namu_face_embed --shard 1 --num-shards 4
    python -m embeddings.namu_face_embed --shard 2 --num-shards 4
    python -m embeddings.namu_face_embed --shard 3 --num-shards 4

- 각 프로세스는  id % num_shards == shard 인 row만 처리하므로
  DB lock 충돌이 거의 없고, 재실행/중단도 독립적.

L4 VM 권장값 (대략)
-------------------
- CTX_ID = 0        (GPU 사용)
- BATCH_COMMIT = 256
- num_shards: 2 ~ 4 (GPU/CPU 사용률 보고 조절)
"""

from __future__ import annotations
import argparse
import logging
import sqlite3
import time
from pathlib import Path
from typing import List, Tuple, Optional

import cv2
import numpy as np
import insightface

# ---------------- CONFIG ----------------

BASE_DIR = Path(__file__).resolve().parents[1]

SRC_DB = BASE_DIR / "crawls" / "namu_wiki_persons.sqlite"
DST_DB = BASE_DIR / "data" / "namu_wiki_facevec.sqlite"

IMAGES_ROOT = BASE_DIR  # image_path: 'images\\namu_wiki\\1.webp' 같은 상대 경로라고 가정

FACE_DIM = 512
BATCH_COMMIT = 256      # L4 + 32GB RAM 기준: 256 정도면 무난
CTX_ID = -1              # GPU 사용 (로컬 CPU 테스트하려면 -1 로 변경)

LOGGER = logging.getLogger("namu_face_embed")


# ---------------- logging ----------------

def setup_logging():
    h = logging.StreamHandler()
    f = logging.Formatter("[%(levelname)s] %(message)s")
    h.setFormatter(f)
    LOGGER.addHandler(h)
    LOGGER.setLevel(logging.INFO)


# ---------------- DB helpers ----------------

def connect_db(path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(str(path))
    conn.row_factory = sqlite3.Row
    return conn


def ensure_dst_schema(conn: sqlite3.Connection):
    cur = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='namu_face'"
    )
    if cur.fetchone():
        return

    LOGGER.info("새 DB에 namu_face 테이블 생성")
    conn.execute(
        """
        CREATE TABLE namu_face (
          id          INTEGER PRIMARY KEY,
          page_id     INTEGER NOT NULL,
          page_title  TEXT    NOT NULL,
          page_url    TEXT    NOT NULL,
          image_path  TEXT,
          face_vec    BLOB,
          face_status INTEGER NOT NULL DEFAULT 0
        )
        """
    )
    conn.execute(
        "CREATE UNIQUE INDEX idx_namu_face_page_id ON namu_face(page_id)"
    )
    conn.commit()


def sync_metadata(src: sqlite3.Connection, dst: sqlite3.Connection):
    """
    원본 namu_person 의 메타데이터를 namu_face 로 복사.
    image_path 가 NULL 이 아닌 row만.
    이미 있는 id/page_id 는 INSERT OR IGNORE 로 스킵.
    """
    LOGGER.info("원본 DB에서 메타데이터 동기화 시작")

    cur = src.execute(
        """
        SELECT id, page_id, page_title, page_url, image_path
        FROM namu_person
        WHERE image_path IS NOT NULL
        """
    )
    rows = cur.fetchall()
    LOGGER.info("원본에서 image_path 있는 row: %d", len(rows))

    dst.executemany(
        """
        INSERT OR IGNORE INTO namu_face
          (id, page_id, page_title, page_url, image_path)
        VALUES (?, ?, ?, ?, ?)
        """,
        [(r["id"], r["page_id"], r["page_title"], r["page_url"], r["image_path"])
         for r in rows],
    )
    dst.commit()
    LOGGER.info("메타데이터 동기화 완료")


def count_targets(dst: sqlite3.Connection, shard: int, num_shards: int) -> int:
    cur = dst.execute(
        """
        SELECT COUNT(*)
        FROM namu_face
        WHERE face_vec IS NULL
          AND face_status = 0
          AND (id % ? = ?)
        """,
        (num_shards, shard),
    )
    (n,) = cur.fetchone()
    return int(n)


def iter_targets(dst: sqlite3.Connection, shard: int, num_shards: int):
    cur = dst.execute(
        """
        SELECT id, image_path
        FROM namu_face
        WHERE face_vec IS NULL
          AND face_status = 0
          AND (id % ? = ?)
        ORDER BY id
        """,
        (num_shards, shard),
    )
    for r in cur:
        yield int(r["id"]), r["image_path"]


# ---------------- InsightFace ----------------

def init_insightface():
    LOGGER.info("InsightFace buffalo_l 로드 (ctx_id=%s)...", CTX_ID)
    app = insightface.app.FaceAnalysis(name="buffalo_l")
    # det_size 는 워크로드 보고 640~1024 사이에서 조정 가능
    app.prepare(ctx_id=CTX_ID, det_size=(640, 640))
    LOGGER.info("모델 준비 완료")
    return app


def select_best_face(faces) -> Optional[object]:
    if not faces:
        return None

    females: List[Tuple[int, object]] = []
    males: List[Tuple[int, object]] = []

    for f in faces:
        x1, y1, x2, y2 = f.bbox.astype(int)
        area = (x2 - x1) * (y2 - y1)
        if getattr(f, "gender", 0) == 1:  # 1=female
            females.append((area, f))
        else:
            males.append((area, f))

    if females:
        females.sort(key=lambda x: x[0], reverse=True)
        return females[0][1]
    if males:
        males.sort(key=lambda x: x[0], reverse=True)
        return males[0][1]
    return None


# ---------------- misc helpers ----------------

def resolve_image_path(image_path: str) -> Path:
    normalized = image_path.replace("\\", "/")
    rel = Path(normalized)
    if rel.is_absolute():
        return rel
    return IMAGES_ROOT / rel


def safe_commit(
    conn: sqlite3.Connection,
    pending_vec: List[Tuple[bytes, int, int]],
    pending_status: List[Tuple[int, int]],
):
    if not pending_vec and not pending_status:
        return

    for attempt in range(10):
        try:
            if pending_vec:
                conn.executemany(
                    "UPDATE namu_face SET face_vec = ?, face_status = ? WHERE id = ?",
                    pending_vec,
                )
            if pending_status:
                conn.executemany(
                    "UPDATE namu_face SET face_status = ? WHERE id = ?",
                    pending_status,
                )
            conn.commit()
            return
        except sqlite3.OperationalError as e:
            if "locked" in str(e).lower():
                time.sleep(0.15 * (attempt + 1))
                continue
            raise
    LOGGER.error("commit 실패: database locked")


# ---------------- main pipeline ----------------

def process_shard(shard: int, num_shards: int):
    setup_logging()

    if not SRC_DB.exists():
        raise SystemExit(f"원본 DB 없음: {SRC_DB}")

    DST_DB.parent.mkdir(parents=True, exist_ok=True)

    LOGGER.info("프로젝트 루트: %s", BASE_DIR)
    LOGGER.info("SRC_DB: %s", SRC_DB)
    LOGGER.info("DST_DB: %s", DST_DB)
    LOGGER.info("shard=%d num_shards=%d", shard, num_shards)

    src_conn = connect_db(SRC_DB)
    dst_conn = connect_db(DST_DB)

    try:
        dst_conn.execute("PRAGMA journal_mode=WAL;")
        dst_conn.execute("PRAGMA synchronous=NORMAL;")
        ensure_dst_schema(dst_conn)
        sync_metadata(src_conn, dst_conn)

        total = count_targets(dst_conn, shard, num_shards)
        LOGGER.info("face_vec 생성 대상 (이 shard): %d", total)

        if total == 0:
            LOGGER.info("이 shard에서 처리할 대상 없음")
            return

        app = init_insightface()

        processed = 0
        encoded = 0
        missing = 0
        no_face = 0

        pending_vec: List[Tuple[bytes, int, int]] = []
        pending_status: List[Tuple[int, int]] = []

        for pid, image_path in iter_targets(dst_conn, shard, num_shards):
            processed += 1

            img_path = resolve_image_path(image_path)
            if not img_path.exists():
                missing += 1
                pending_status.append((-2, pid))
            else:
                img = cv2.imread(str(img_path))
                if img is None:
                    missing += 1
                    pending_status.append((-2, pid))
                else:
                    faces = app.get(img)
                    if not faces:
                        no_face += 1
                        pending_status.append((-1, pid))
                    else:
                        best = select_best_face(faces)
                        vec = getattr(best, "embedding", None) if best is not None else None
                        if vec is None or vec.shape[0] != FACE_DIM:
                            no_face += 1
                            pending_status.append((-1, pid))
                        else:
                            pending_vec.append(
                                (vec.astype("float32").tobytes(), 1, pid)
                            )
                            encoded += 1

            if len(pending_vec) + len(pending_status) >= BATCH_COMMIT:
                safe_commit(dst_conn, pending_vec, pending_status)
                pending_vec.clear()
                pending_status.clear()
                LOGGER.info(
                    "[중간 commit] shard=%d processed=%d/%d encoded=%d missing=%d no_face=%d",
                    shard, processed, total, encoded, missing, no_face,
                )

            if processed % 500 == 0:
                LOGGER.info(
                    "[진행] shard=%d processed=%d/%d encoded=%d missing=%d no_face=%d",
                    shard, processed, total, encoded, missing, no_face,
                )

        if pending_vec or pending_status:
            safe_commit(dst_conn, pending_vec, pending_status)

        LOGGER.info(
            "shard=%d 완료: processed=%d encoded=%d missing=%d no_face=%d",
            shard, processed, encoded, missing, no_face,
        )

    finally:
        src_conn.close()
        dst_conn.close()


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--shard", type=int, default=0,
                   help="처리할 shard 번호 (0 <= shard < num_shards)")
    p.add_argument("--num-shards", type=int, default=1,
                   help="총 shard 개수 (병렬 실행 시 사용)")
    return p.parse_args()


def main():
    args = parse_args()
    if not (0 <= args.shard < args.num_shards):
        raise SystemExit("shard 는 0 <= shard < num_shards 여야 함")
    process_shard(args.shard, args.num_shards)


if __name__ == "__main__":
    main()
