# crawls/discover_namu_tree.py
from __future__ import annotations

import sqlite3
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Tuple, Dict, Any, List
from urllib.parse import urljoin, urlparse, unquote

import requests
from bs4 import BeautifulSoup

# ─────────────────────────────────────
# 설정
# ─────────────────────────────────────
BASE_URL = "https://namu.wiki"
ROOT_CATEGORY_URL = (
    "https://namu.wiki/w/%EB%B6%84%EB%A5%98:%EB%82%98%EB%9D%BC%EB%B3%84%20%EC%A7%81%EC%97%85%EB%B3%84%20%EC%9D%B8%EB%AC%BC"
)

THIS_DIR = Path(__file__).resolve().parent
DB_PATH = THIS_DIR / "namu_wiki_pages.sqlite"

MAX_WORKERS = 2          # 동시에 처리할 카테고리 수
BATCH_SIZE = 30          # 한 번에 잡아올 pending 카테고리 수
REQUEST_TIMEOUT = 15


class TooManyRequestsError(Exception):
    """HTTP 429가 발생했을 때 던지는 예외."""
    pass


# ─────────────────────────────────────
# DB 유틸
# ─────────────────────────────────────
def get_conn() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH, timeout=30)
    conn.row_factory = sqlite3.Row

    cur = conn.cursor()
    cur.execute("PRAGMA journal_mode=WAL;")
    cur.execute("PRAGMA synchronous=NORMAL;")
    cur.execute("PRAGMA temp_store=MEMORY;")
    cur.execute("PRAGMA mmap_size=134217728;")   # 128MB
    cur.execute("PRAGMA busy_timeout=30000;")    # 30초
    cur.close()

    return conn


def init_db() -> None:
    conn = get_conn()
    cur = conn.cursor()

    # status = pending / in_progress / done
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS category_page (
            id                    INTEGER PRIMARY KEY AUTOINCREMENT,
            category_title        TEXT NOT NULL,
            page_url              TEXT NOT NULL UNIQUE,
            parent_category_title TEXT,
            depth                 INTEGER NOT NULL DEFAULT 0,
            status                TEXT NOT NULL DEFAULT 'pending',
            last_visited_at       TEXT
        );
        """
    )

    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS wiki_page (
            id                    INTEGER PRIMARY KEY AUTOINCREMENT,
            page_title            TEXT NOT NULL,
            page_url              TEXT NOT NULL UNIQUE,
            first_category_title  TEXT,
            discovered_at         TEXT NOT NULL
        );
        """
    )

    # 루트 분류
    cur.execute(
        """
        INSERT OR IGNORE INTO category_page
            (category_title, page_url, parent_category_title, depth, status)
        VALUES (?, ?, NULL, 0, 'pending');
        """,
        ("분류:나라별 직업별 인물", ROOT_CATEGORY_URL),
    )

    conn.commit()
    cur.close()
    conn.close()


def reset_in_progress() -> None:
    """이전 실행 도중 죽어서 in_progress로 남은 애들 다시 pending으로."""
    conn = get_conn()
    cur = conn.cursor()
    cur.execute(
        """
        UPDATE category_page
        SET status = 'pending',
            last_visited_at = NULL
        WHERE status = 'in_progress';
        """
    )
    affected = cur.rowcount
    conn.commit()
    cur.close()
    conn.close()
    if affected:
        print(f"[INIT] reset {affected} in_progress rows back to pending")


# ─────────────────────────────────────
# HTTP 요청 (429 → 예외 던지고 전체 중단)
# ─────────────────────────────────────
def fetch_once(url: str) -> str | None:
    session = requests.Session()
    session.headers.update({
        "User-Agent": "face-recognition-crawler/0.1 (research)",
    })

    try:
        resp = session.get(url, timeout=REQUEST_TIMEOUT)
    except Exception as e:
        print(f"[WARN] request error: {url} ({e})")
        return None

    status = resp.status_code

    if 200 <= status < 300:
        return resp.text

    if status == 429:
        msg = f"429 Too Many Requests: {url}"
        print(f"[WARN] {msg}")
        raise TooManyRequestsError(msg)

    if 500 <= status < 600:
        print(f"[WARN] {status} server error: {url}")
        return None

    print(f"[WARN] HTTP {status}: {url}")
    return None


# ─────────────────────────────────────
# 링크 유틸
# ─────────────────────────────────────
def extract_title_from_url(url: str) -> str:
    parsed = urlparse(url)
    path = parsed.path
    if "/w/" in path:
        title_part = path.split("/w/", 1)[1]
    else:
        title_part = path.lstrip("/")
    return unquote(title_part)


def classify_link(href: str) -> Tuple[str | None, str | None]:
    """
    /w/ 로 시작하는 링크만 카테고리/문서로 분류.
    그 외(나무위키:, 파일:, 외부 링크 등)는 모두 무시.
    """
    if not href.startswith("/w/"):
        return None, None

    abs_url = urljoin(BASE_URL, href)
    title = extract_title_from_url(abs_url)

    if title.startswith("나무위키:") or title.startswith("파일:"):
        return None, None

    if title.startswith("분류:"):
        return "category", abs_url

    return "page", abs_url


def iter_section_links(
    soup: BeautifulSoup,
    keywords: List[str],
) -> List[Any]:
    """
    주어진 키워드를 포함하는 제목(하위 분류, \"…분류에 속하는 문서\" 등)을 찾아서
    그 섹션(다음 heading 전까지)에 포함된 <a>들을 모두 반환.
    - 상단 breadcrumb, 오른쪽 실시간 검색어/최근 변경은 건드리지 않음.
    """
    links: List[Any] = []

    # h2 / h3 / h4 정도만 보면 충분하다고 가정
    for heading in soup.find_all(["h2", "h3", "h4"]):
        text = heading.get_text(strip=True)
        if not any(k in text for k in keywords):
            continue

        # 이 heading 이후부터, 다음 heading 전까지가 섹션
        for sib in heading.next_siblings:
            if getattr(sib, "name", None) in {"h2", "h3", "h4"}:
                break
            if hasattr(sib, "find_all"):
                for a in sib.find_all("a", href=True):
                    links.append(a)

    return links


# ─────────────────────────────────────
# 워커
# ─────────────────────────────────────
def mark_category_done(category_id: int) -> None:
    conn = get_conn()
    cur = conn.cursor()
    now = datetime.utcnow().isoformat(timespec="seconds") + "Z"
    cur.execute(
        """
        UPDATE category_page
        SET status = 'done',
            last_visited_at = ?
        WHERE id = ?;
        """,
        (now, category_id),
    )
    conn.commit()
    cur.close()
    conn.close()


def process_category_page(row: Dict[str, Any]) -> Tuple[int, int]:
    """
    하나의 category_page row 처리:
      - "하위 분류" 섹션의 링크만 카테고리로 따라감
      - "\"…분류에 속하는 문서\"" 섹션의 링크만 wiki_page로 저장
      - 429가 나면 TooManyRequestsError 예외를 그대로 던진다 (done 표시 안 함).
    """
    cat_id = row["id"]
    cat_title = row["category_title"]
    page_url = row["page_url"]
    depth = row["depth"]

    print(f"[WORKER] depth={depth} {cat_title} ({page_url})")

    html = fetch_once(page_url)
    if html is None:
        print(f"[WORKER]   fetch failed (non-429), leave status=in_progress for retry later")
        return (0, 0)

    soup = BeautifulSoup(html, "html.parser")
    conn = get_conn()
    cur = conn.cursor()

    new_cat = 0
    new_page = 0
    now = datetime.utcnow().isoformat(timespec="seconds") + "Z"

    # 1) 하위 분류 섹션: 여기의 링크만 카테고리로 인식
    subcat_links = iter_section_links(soup, ["하위 분류"])

    # 2) "분류에 속하는 문서" 섹션: 여기의 링크만 문서로 인식
    page_links = iter_section_links(
        soup,
        ["분류에 속하는 문서", "분류에 속한 문서"],
    )

    # 카테고리 먼저 처리
    for a in subcat_links:
        href = a["href"]

        # "?cfrom=..." 같은 페이지네이션은 현재 경로와 합쳐준다.
        if href.startswith("?"):
            base_path = urlparse(page_url).path
            href = base_path + href

        kind, abs_url = classify_link(href)
        if kind != "category":
            continue

        title = extract_title_from_url(abs_url)
        cur.execute(
            """
            INSERT OR IGNORE INTO category_page
                (category_title, page_url, parent_category_title, depth, status)
            VALUES (?, ?, ?, ?, 'pending');
            """,
            (title, abs_url, cat_title, depth + 1),
        )
        if cur.rowcount > 0:
            new_cat += 1

    # 분류에 속하는 문서들 처리
    for a in page_links:
        href = a["href"]

        if href.startswith("?"):
            # 이 구간에는 거의 없겠지만, 혹시 모를 상대 링크 처리
            base_path = urlparse(page_url).path
            href = base_path + href

        kind, abs_url = classify_link(href)
        if kind != "page":
            continue

        title = extract_title_from_url(abs_url)
        cur.execute(
            """
            INSERT OR IGNORE INTO wiki_page
                (page_title, page_url, first_category_title, discovered_at)
            VALUES (?, ?, ?, ?);
            """,
            (title, abs_url, cat_title, now),
        )
        if cur.rowcount > 0:
            new_page += 1

    conn.commit()
    cur.close()
    conn.close()

    mark_category_done(cat_id)

    print(
        f"[WORKER] done depth={depth} {cat_title}: "
        f"+{new_cat} categories, +{new_page} pages"
    )
    return (new_cat, new_page)


# ─────────────────────────────────────
# 메인 루프
# ─────────────────────────────────────
def fetch_pending_batch(conn: sqlite3.Connection, limit: int) -> List[Dict[str, Any]]:
    cur = conn.cursor()
    cur.execute(
        """
        SELECT id, category_title, page_url, depth
        FROM category_page
        WHERE status = 'pending'
        ORDER BY depth, id
        LIMIT ?;
        """,
        (limit,),
    )
    rows = cur.fetchall()

    if not rows:
        cur.close()
        return []

    ids = [r["id"] for r in rows]
    now = datetime.utcnow().isoformat(timespec="seconds") + "Z"
    cur.execute(
        f"""
        UPDATE category_page
        SET status = 'in_progress',
            last_visited_at = ?
        WHERE id IN ({",".join(["?"] * len(ids))});
        """,
        (now, *ids),
    )
    conn.commit()
    cur.close()

    return [
        {
            "id": r["id"],
            "category_title": r["category_title"],
            "page_url": r["page_url"],
            "depth": r["depth"],
        }
        for r in rows
    ]


def main():
    init_db()
    reset_in_progress()

    conn = get_conn()
    total_cat = 0
    total_page = 0
    iteration = 0

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        while True:
            batch = fetch_pending_batch(conn, BATCH_SIZE)
            if not batch:
                print("[MAIN] No more pending categories. Done.")
                break

            iteration += 1
            print(
                f"[MAIN] Iteration {iteration}: dispatch {len(batch)} category pages "
                f"(workers={MAX_WORKERS})"
            )

            futures = [executor.submit(process_category_page, row) for row in batch]

            stop_due_to_429 = False

            for fut in as_completed(futures):
                try:
                    new_cat, new_page = fut.result()
                    total_cat += new_cat
                    total_page += new_page
                except TooManyRequestsError as e:
                    print(f"[FATAL] {e} → stopping crawler so you can 재시도 later.")
                    stop_due_to_429 = True
                    break
                except Exception as e:
                    print(f"[WARN] worker raised exception: {e}")
                    continue

            if stop_due_to_429:
                break

            print(
                f"[MAIN]   cumulative new categories: {total_cat}, "
                f"new pages: {total_page}"
            )

    conn.close()
    print(f"[MAIN] Finished. DB at {DB_PATH}")


if __name__ == "__main__":
    main()
