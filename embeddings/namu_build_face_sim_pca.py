#!/usr/bin/env python
"""
face_vec(InsightFace 임베딩) 전체에 PCA를 걸어서
저차원 face_sim_vec을 만들고, 그걸 이용한 FAISS index를 생성한다.

- 입력 DB : data/namu_wiki_facevec.sqlite   (테이블 namu_face)
  - 컬럼: id INTEGER PRIMARY KEY
          face_vec BLOB (float32, D차원)

- 출력:
  - DB 컬럼 추가: face_sim_vec BLOB
  - FAISS index : data/namu_facevec_sim_ip.faiss
"""

import os
import sqlite3
from pathlib import Path

import numpy as np
import faiss

# ---------------- 경로 설정 ----------------

BASE_DIR = Path(__file__).resolve().parent.parent  # 프로젝트 루트
DATA_DB_PATH = BASE_DIR / "data" / "namu_wiki_facevec.sqlite"
FAISS_OUT_PATH = BASE_DIR / "data" / "namu_facevec_sim_ip.faiss"
PCA_PARAM_PATH = BASE_DIR / "data" / "namu_face_pca_params.npz"

SIM_DIM = 128  # face_sim_vec 차원 (원본 512D → 128D 추천)


# ---------------- 유틸 ----------------


def get_conn(db_path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    return conn


def ensure_face_sim_column(conn: sqlite3.Connection):
    cur = conn.execute("PRAGMA table_info(namu_face)")
    cols = [row["name"] for row in cur.fetchall()]
    if "face_sim_vec" not in cols:
        print("[INFO] namu_face 테이블에 face_sim_vec 컬럼 추가")
        conn.execute("ALTER TABLE namu_face ADD COLUMN face_sim_vec BLOB")
        conn.commit()
    else:
        print("[INFO] face_sim_vec 컬럼 이미 존재")


def load_all_face_vecs(conn: sqlite3.Connection):
    print("[INFO] face_vec 전체 로딩 중...")
    cur = conn.execute("SELECT id, face_vec FROM namu_face WHERE face_vec IS NOT NULL")
    ids = []
    vecs = []

    for row in cur:
        blob = row["face_vec"]
        if blob is None:
            continue
        v = np.frombuffer(blob, dtype="float32")
        ids.append(row["id"])
        vecs.append(v)

    ids = np.asarray(ids, dtype="int64")
    X = np.stack(vecs, axis=0).astype("float32")
    print(f"[INFO] 로딩 완료: N={X.shape[0]}, D={X.shape[1]}")
    return ids, X


def compute_pca_projection(X: np.ndarray, out_dim: int):
    """
    간단한 PCA: X_center = X - mean
    SVD(X_center) = U S Vt  → 상위 out_dim개의 V 사용
    """
    N, D = X.shape
    if out_dim > D:
        out_dim = D
    print(f"[INFO] PCA 계산 (N={N}, D={D}, out_dim={out_dim})")

    mean = X.mean(axis=0, keepdims=True)
    Xc = X - mean

    # full_matrices=False 로 메모리 절약
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    components = Vt[:out_dim].T  # (D, out_dim)

    # 투영
    Y = Xc @ components  # (N, out_dim)

    # L2 정규화
    norms = np.linalg.norm(Y, axis=1, keepdims=True) + 1e-12
    Y_norm = (Y / norms).astype("float32")

    print("[INFO] PCA 변환 완료")
    return mean.astype("float32"), components.astype("float32"), Y_norm


def update_face_sim_vecs(conn: sqlite3.Connection, ids: np.ndarray, Y: np.ndarray):
    print("[INFO] DB에 face_sim_vec 업데이트 중...")
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute("BEGIN")

    try:
        for i, vid in enumerate(ids):
            blob = Y[i].tobytes()
            conn.execute(
                "UPDATE namu_face SET face_sim_vec=? WHERE id=?",
                (blob, int(vid)),
            )
        conn.commit()
        print("[INFO] face_sim_vec 업데이트 완료")
    except Exception:
        conn.rollback()
        raise


def build_faiss_index(ids: np.ndarray, Y: np.ndarray, out_path: Path):
    print("[INFO] FAISS index 생성 중...")
    N, D = Y.shape
    # Inner Product index + IDMap
    index = faiss.IndexFlatIP(D)
    index = faiss.IndexIDMap2(index)
    index.add_with_ids(Y, ids)
    faiss.write_index(index, str(out_path))
    print(f"[INFO] FAISS index 저장 완료: {out_path} (N={N}, D={D})")


# ---------------- 메인 ----------------


def main():
    if not DATA_DB_PATH.exists():
        raise FileNotFoundError(f"DB not found: {DATA_DB_PATH}")

    print(f"[INFO] 사용 DB: {DATA_DB_PATH}")
    conn = get_conn(DATA_DB_PATH)

    # 1) 컬럼 확인/추가
    ensure_face_sim_column(conn)

    # 2) face_vec 전체 로딩
    ids, X = load_all_face_vecs(conn)

    # 3) PCA 기반 face_sim_vec 계산
    mean, components, Y = compute_pca_projection(X, SIM_DIM)
    np.savez(PCA_PARAM_PATH, mean=mean.squeeze(), components=components)

    # 4) DB에 face_sim_vec 저장
    update_face_sim_vecs(conn, ids, Y)
    conn.close()

    # 5) face_sim_vec 기반 FAISS index 생성
    build_faiss_index(ids, Y, FAISS_OUT_PATH)


if __name__ == "__main__":
    main()
