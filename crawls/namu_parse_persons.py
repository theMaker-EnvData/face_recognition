# crawls/parse_namu_persons.py
#
# namu_wiki_pages.sqlite 에서 is_person_candidate = 1 인 페이지를 읽어와
# 사람(또는 가상 인물) 후보의 프로필을 파싱해서
# namu_wiki_persons.sqlite 에 저장하는 스크립트.
#
# - 기존 페이지 DB는 절대 건드리지 않음 (read-only 용도).
# - persons DB 에는 page_id 기준 UNIQUE 로 기록.
# - 이미 기록된 page_id 는 재시작 시 자동으로 건너뜀.
# - MAX_PAGES 로 테스트 배치만 돌려볼 수 있고, None 이면 전체.

from __future__ import annotations

import json
import os
import sqlite3
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional
from urllib.parse import urljoin, urlparse

from io import BytesIO
from PIL import Image

import requests
from bs4 import BeautifulSoup

# ───────────────── 설정 ─────────────────

THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = THIS_DIR.parent

PAGES_DB_PATH = THIS_DIR / "namu_wiki_pages.sqlite"      # 읽기 전용
PERSONS_DB_PATH = THIS_DIR / "namu_wiki_persons.sqlite"  # 새로 만드는 DB
IMG_DIR = PROJECT_ROOT / "images" / "namu_wiki"

BASE_URL = "https://namu.wiki"

# 테스트용: 5개만 파싱하고 싶으면 5, 전부 돌리고 싶으면 None
MAX_PAGES: Optional[int] = None

# discover_namu_tree 와 비슷한 톤
MAX_WORKERS = 2
BATCH_SIZE = 30
REQUEST_TIMEOUT = 15


class TooManyRequestsError(Exception):
    pass


# ───────────────── DB 유틸 (pages) ─────────────────

def get_pages_conn() -> sqlite3.Connection:
    if not PAGES_DB_PATH.exists():
        raise FileNotFoundError(f"pages DB not found: {PAGES_DB_PATH}")
    conn = sqlite3.connect(PAGES_DB_PATH, timeout=30)
    conn.row_factory = sqlite3.Row
    return conn


# ───────────────── DB 유틸 (persons) ─────────────────

CREATE_PERSONS_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS namu_person (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    page_id       INTEGER NOT NULL UNIQUE,
    page_title    TEXT NOT NULL,
    page_url      TEXT NOT NULL,
    name          TEXT NOT NULL,
    birth         TEXT,
    death         TEXT,
    infobox_json  TEXT,
    image_url     TEXT,
    image_path    TEXT,
    parsed_at     TEXT NOT NULL
);
"""


def get_persons_conn() -> sqlite3.Connection:
    PERSONS_DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(PERSONS_DB_PATH, timeout=30)
    conn.row_factory = sqlite3.Row

    cur = conn.cursor()
    cur.execute("PRAGMA journal_mode=WAL;")
    cur.execute("PRAGMA synchronous=NORMAL;")
    cur.execute("PRAGMA temp_store=MEMORY;")
    cur.execute("PRAGMA mmap_size=134217728;")   # 128MB
    cur.execute("PRAGMA busy_timeout=30000;")    # 30초
    conn.commit()
    cur.close()

    return conn


def init_persons_db() -> None:
    conn = get_persons_conn()
    cur = conn.cursor()
    cur.execute(CREATE_PERSONS_TABLE_SQL)
    conn.commit()
    cur.close()
    conn.close()


def get_already_parsed_ids() -> set[int]:
    if not PERSONS_DB_PATH.exists():
        return set()
    conn = get_persons_conn()
    cur = conn.cursor()
    cur.execute("SELECT page_id FROM namu_person;")
    ids = {row[0] for row in cur.fetchall()}
    cur.close()
    conn.close()
    return ids


# ───────────────── 후보 목록 뽑기 ─────────────────

def load_person_candidates() -> List[Dict[str, Any]]:
    pages_conn = get_pages_conn()
    persons_done = get_already_parsed_ids()

    # 이미 파싱된 것들 중 가장 큰 page_id
    max_done = max(persons_done) if persons_done else 0

    cur = pages_conn.cursor()
    if max_done > 0:
        cur.execute(
            """
            SELECT id, page_title, page_url
            FROM wiki_page
            WHERE is_person_candidate = 1
              AND id > ?
            ORDER BY id ASC;
            """,
            (max_done,),
        )
    else:
        # 아직 아무 것도 파싱 안 했으면 전체
        cur.execute(
            """
            SELECT id, page_title, page_url
            FROM wiki_page
            WHERE is_person_candidate = 1
            ORDER BY id ASC;
            """
        )

    rows = cur.fetchall()
    cur.close()
    pages_conn.close()

    candidates: List[Dict[str, Any]] = []
    for r in rows:
        pid = r["id"]
        # 논리상 여기서는 pid > max_done 이라 persons_done 에 없지만,
        # 혹시 모를 중복을 막기 위해 한 번 더 방어적으로 체크.
        if pid in persons_done:
            continue
        candidates.append(
            {
                "page_id": pid,
                "page_title": r["page_title"],
                "page_url": r["page_url"],
            }
        )

    if MAX_PAGES is not None:
        candidates = candidates[:MAX_PAGES]

    return candidates



# ───────────────── HTTP & 파싱 유틸 ─────────────────

def fetch_html(url: str) -> Optional[str]:
    headers = {
        "User-Agent": "face-recognition-person-parser/0.1 (personal project)",
    }
    try:
        resp = requests.get(url, headers=headers, timeout=REQUEST_TIMEOUT)
    except Exception as e:
        print(f"  [WARN] request error: {url} ({e})")
        return None

    status = resp.status_code
    if 200 <= status < 300:
        return resp.text

    if status == 429:
        raise TooManyRequestsError(f"429 Too Many Requests: {url}")

    if 500 <= status < 600:
        print(f"  [WARN] {status} server error: {url}")
        return None

    print(f"  [WARN] HTTP {status}: {url}")
    return None


def find_infobox(soup):
    # 인물 인포박스에 거의 항상 들어가는 라벨들
    label_keywords = ["성명", "본명", "출생", "사망", "국적", "직업"]

    for table in soup.find_all("table"):
        text = table.get_text(" ", strip=True)
        if any(k in text for k in label_keywords):
            return table
    return None


def parse_infobox(infobox):
    info = {}

    for row in infobox.find_all("tr"):
        # 이미지 행(한 칸만 있는 행) / 빈 행은 스킵
        cells = row.find_all("td", recursive=False)
        if len(cells) != 2:
            continue

        key = cells[0].get_text(" ", strip=True)
        val = cells[1].get_text(" ", strip=True)

        if not key or not val:
            continue

        info[key] = val

    # 인물 기본 필드 정리
    name = info.get("성명") or info.get("본명")
    birth = info.get("출생")
    death = info.get("사망")

    return name, birth, death, json.dumps(info, ensure_ascii=False)


def find_main_image_url(soup: BeautifulSoup, page_url: str) -> Optional[str]:
    """
    인포박스 안에 있는 '얼굴 사진'만 대표사진으로 사용한다.

    - 인포박스에서 <td>가 1개뿐인 행(tr)을 찾고,
      그 셀 안에 img가 있을 때만 대표사진으로 인정.
    - 이런 행이 하나도 없으면 사진이 없는 문서로 보고 None을 리턴.
    - 로고/아이콘(정당 로고, 나무위키 트리 등)은
      보통 2칸짜리 행의 값 부분에 들어가거나,
      alt/src에 '로고', 'logo' 등이 들어가므로 추가로 걸러준다.
    """
    infobox = find_infobox(soup)
    if not infobox:
        return None

    portrait_src = None

    for tr in infobox.find_all("tr"):
        cells = tr.find_all("td", recursive=False)
        # 사진 행: <tr><td colspan="2"><img ...></td></tr>
        if len(cells) != 1:
            continue

        cell = cells[0]
        img = cell.find("img")
        if not img:
            continue

        src = img.get("src")
        if not src:
            continue

        alt = (img.get("alt") or "").lower()
        src_lower = src.lower()

        # 나무위키 로고 / CC 로고 / 기타 명백한 아이콘 필터
        if any(k in alt for k in ["나무위키", "namuwiki", "cc by", "creativecommons", "로고", "logo"]):
            continue
        if any(k in src_lower for k in ["namu_logo", "cc-by", "creativecommons", "/logo"]):
            continue

        portrait_src = src
        break

    if not portrait_src:
        return None

    return urljoin(page_url, portrait_src)


def guess_image_filename(page_id: int, image_url: str) -> str:
    ext = os.path.splitext(urlparse(image_url).path)[1]
    if not ext:
        ext = ".jpg"
    return f"{page_id}{ext}"


def download_image(image_url: str, dest_path: Path) -> None:
    dest_path.parent.mkdir(parents=True, exist_ok=True)

    headers = {
        "User-Agent": "face-recognition-person-parser/0.1 (personal project)",
    }
    resp = requests.get(image_url, headers=headers, timeout=30)
    resp.raise_for_status()

    # SVG 같은 벡터 이미지는 건너뛴다.
    ctype = resp.headers.get("Content-Type", "").lower()
    if "svg" in ctype:
        raise ValueError(f"skip svg image: {image_url}")

    img = Image.open(BytesIO(resp.content))
    # 채널 맞추기 (팔레트/투명 PNG 등)
    if img.mode not in ("RGB", "L"):
        img = img.convert("RGB")

    w, h = img.size
    max_side = max(w, h)

    # 긴 변이 384보다 클 때만 다운스케일
    if max_side > 384:
        scale = 384.0 / max_side
        new_size = (int(w * scale), int(h * scale))
        img = img.resize(new_size, Image.LANCZOS)

    # 원래 확장자 그대로 저장 (dest_path 에 이미 확장자 들어있음)
    img.save(dest_path)


# ───────────────── 워커 ─────────────────

def process_page(row: Dict[str, Any]) -> None:
    page_id = row["page_id"]
    page_title = row["page_title"]
    page_url = row["page_url"]

    print(f"[WORKER] page_id={page_id} title={page_title!r}")

    html = fetch_html(page_url)
    if html is None:
        print("  [WARN] fetch failed, will retry next run.")
        return

    soup = BeautifulSoup(html, "html.parser")

    # 이름은 일단 페이지 타이틀
    name = page_title
    birth = None
    death = None
    infobox_json = None

    # 인포박스
    infobox_table = find_infobox(soup)
    if infobox_table is not None:
        parsed_name, parsed_birth, parsed_death, parsed_json = parse_infobox(infobox_table)

        if parsed_name:   # 인포박스에 '성명'이나 '본명'이 있으면 그걸 이름으로 사용
            name = parsed_name
        if parsed_birth:
            birth = parsed_birth
        if parsed_death:
            death = parsed_death
        infobox_json = parsed_json

    # 대표 이미지 (인포박스 안에 없으면 이 페이지는 스킵)
    image_url = find_main_image_url(soup, page_url)
    if not image_url:
        print("  [INFO] no infobox image found – skip this page.")
        return  # DB에 아무 것도 쓰지 않고 종료

    image_path_str: Optional[str] = None
    filename = guess_image_filename(page_id, image_url)
    dest_path = IMG_DIR / filename
    try:
        download_image(image_url, dest_path)
        image_path_str = str(dest_path.relative_to(PROJECT_ROOT))
    except Exception as e:
        print(f"  [WARN] image download failed: {e}")
        return  # 이미지까지 실패하면 이 페이지는 건너뜀

    parsed_at = datetime.utcnow().isoformat(timespec="seconds") + "Z"

    # persons DB 에 기록
    conn = get_persons_conn()
    cur = conn.cursor()
    try:
        cur.execute(
            """
            INSERT OR IGNORE INTO namu_person (
                page_id, page_title, page_url,
                name, birth, death,
                infobox_json, image_url, image_path, parsed_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
            """,
            (
                page_id,
                page_title,
                page_url,
                name,
                birth,
                death,
                infobox_json,   # 이미 JSON 문자열 상태
                image_url,
                image_path_str,
                parsed_at,
            ),
        )
        conn.commit()
    finally:
        cur.close()
        conn.close()


# ───────────────── 메인 ─────────────────

def main() -> None:
    print(f"[INFO] pages DB  : {PAGES_DB_PATH}")
    print(f"[INFO] persons DB: {PERSONS_DB_PATH}")
    print(f"[INFO] images dir: {IMG_DIR}")
    print(f"[INFO] MAX_PAGES = {MAX_PAGES}")

    init_persons_db()
    candidates = load_person_candidates()

    print(f"[INFO] candidates to process (after skipping already parsed): {len(candidates)}")
    if not candidates:
        return

    try:
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            # 배치 단위로 나눠서 조금씩 보내기
            for start in range(0, len(candidates), BATCH_SIZE):
                batch = candidates[start:start + BATCH_SIZE]
                print(
                    f"[MAIN] dispatch batch {start // BATCH_SIZE + 1} "
                    f"({len(batch)} pages, workers={MAX_WORKERS})"
                )

                futures = [executor.submit(process_page, row) for row in batch]

                for fut in as_completed(futures):
                    try:
                        fut.result()
                    except TooManyRequestsError as e:
                        print(f"[FATAL] {e} → stop and retry later.")
                        return
                    except Exception as e:
                        print(f"[WARN] worker error: {e}")
                        # 개별 페이지 에러는 무시, 나중에 다시 시도할 수 있음
                        continue

    except KeyboardInterrupt:
        print("\n[MAIN] KeyboardInterrupt detected. Stopping gracefully.")
        # 이미 커밋된 row 들은 그대로 남고,
        # 처리 안 된 page_id 들은 다음 실행에서 다시 candidates 로 잡힌다.


if __name__ == "__main__":
    main()
