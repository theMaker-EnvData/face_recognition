# filter_person_candidates_namu.py
#
# namu_wiki_pages.sqlite 안의 wiki_page 테이블에
# is_person_candidate 플래그를 추가/갱신해서
# "사람(또는 가상 인물) 후보"만 남기기 위한 스크립트.
#
# 규칙 요약:
#   1) 기본값은 모두 1 (후보).
#   2) 아래 조건에 걸리면 0으로 내림.
#      - page_title 에 '/' 포함  (서브 문서, 목록, 활동/논란 등)
#      - page_title 에 '사건', '사고', '논란' 포함
#      - page_title 에 '(음반)', '(앨범)', '(싱글)', '(EP)', '(곡)', '(노래)', '(음악)', '(OST)'
#      - first_category_title 에
#        '음반', '앨범', '싱글', 'EP', '곡', '노래', '음악', '디스코그래피',
#        '콘텐츠', '프로듀싱',
#        '사건', '사고', '일지',
#        '투어', '콘서트', '공연',
#        '음악 방송 직캠', '방송 직캠'
#
# 필요하면 아래 토큰 리스트만 수정해서 튜닝하면 됨.

from pathlib import Path
import sqlite3


# 프로젝트 루트에서 실행한다고 가정
DB_PATH = Path(__file__).resolve().parent / "namu_wiki_pages.sqlite"


CATEGORY_BLACKLIST_SUBSTRINGS = [
    "음반",
    "앨범",
    "싱글",
    "EP",
    "곡",
    "노래",
    "음악",
    "디스코그래피",
    "콘텐츠",
    "프로듀싱",
    "사건",
    "사고",
    "일지",
    "투어",
    "콘서트",
    "공연",
    "음악 방송 직캠",
    "방송 직캠",
]

# page_title 전체에 들어가면 사람 아님으로 보는 토큰
PAGE_TITLE_CONTAINS_BLACKLIST = [
    "사건",
    "사고",
    "논란",
]

# page_title 안에서 괄호 타입으로 붙는 토큰 (예: Foo(음반))
PAGE_TITLE_PAREN_BLACKLIST = [
    "음반",
    "앨범",
    "싱글",
    "EP",
    "곡",
    "노래",
    "음악",
    "OST",
]


def ensure_is_person_candidate_column(conn: sqlite3.Connection) -> None:
    cur = conn.cursor()
    cur.execute("PRAGMA table_info('wiki_page');")
    cols = [row[1] for row in cur.fetchall()]

    if "is_person_candidate" not in cols:
        print("[INFO] Adding is_person_candidate column to wiki_page...")
        cur.execute(
            "ALTER TABLE wiki_page "
            "ADD COLUMN is_person_candidate INTEGER NOT NULL DEFAULT 1"
        )
        conn.commit()
    else:
        print("[INFO] is_person_candidate column already exists.")


def reset_flags(conn: sqlite3.Connection) -> None:
    """모든 행을 일단 1로 초기화."""
    cur = conn.cursor()
    print("[INFO] Resetting is_person_candidate to 1 for all rows...")
    cur.execute("UPDATE wiki_page SET is_person_candidate = 1")
    conn.commit()


def apply_blacklist_rules(conn: sqlite3.Connection) -> None:
    cur = conn.cursor()

    # 1) page_title 에 '/'가 들어가면 서브 문서/목록/활동/논란 등으로 보고 제외
    print("[INFO] Applying rule: page_title contains '/' -> 0")
    cur.execute(
        "UPDATE wiki_page "
        "SET is_person_candidate = 0 "
        "WHERE page_title LIKE '%/%'"
    )

    # 2) page_title 에 사건/사고/논란 이 들어가면 사건 문서로 보고 제외
    print("[INFO] Applying rule: page_title contains 사건/사고/논란 -> 0")
    for token in PAGE_TITLE_CONTAINS_BLACKLIST:
        cur.execute(
            "UPDATE wiki_page "
            "SET is_person_candidate = 0 "
            "WHERE page_title LIKE ?",
            (f"%{token}%",),
        )

    # 3) page_title 에 '(음반)' 같은 타입 표시가 붙으면 제외
    print("[INFO] Applying rule: page_title has (음반/앨범/...) -> 0")
    for token in PAGE_TITLE_PAREN_BLACKLIST:
        cur.execute(
            "UPDATE wiki_page "
            "SET is_person_candidate = 0 "
            "WHERE page_title LIKE ?",
            (f"%({token})%",),
        )

    # 4) first_category_title 기반 블랙리스트
    print("[INFO] Applying rule: first_category_title contains media/event tokens -> 0")
    for token in CATEGORY_BLACKLIST_SUBSTRINGS:
        cur.execute(
            "UPDATE wiki_page "
            "SET is_person_candidate = 0 "
            "WHERE first_category_title LIKE ?",
            (f"%{token}%",),
        )

    conn.commit()


def print_summary(conn: sqlite3.Connection) -> None:
    cur = conn.cursor()
    total = cur.execute("SELECT COUNT(*) FROM wiki_page").fetchone()[0]
    candidates = cur.execute(
        "SELECT COUNT(*) FROM wiki_page WHERE is_person_candidate = 1"
    ).fetchone()[0]
    rejected = total - candidates

    print("----- SUMMARY -----")
    print(f"Total rows        : {total}")
    print(f"Person candidates : {candidates}")
    print(f"Rejected (0)      : {rejected}")
    if total:
        ratio = candidates / total * 100
        print(f"Candidate ratio   : {ratio:.2f}%")
    print("-------------------")


def main() -> None:
    if not DB_PATH.exists():
        raise FileNotFoundError(f"DB not found at {DB_PATH}")

    print(f"[INFO] Using DB: {DB_PATH}")

    conn = sqlite3.connect(DB_PATH)

    try:
        ensure_is_person_candidate_column(conn)
        reset_flags(conn)
        apply_blacklist_rules(conn)
        print_summary(conn)
    finally:
        conn.close()


if __name__ == "__main__":
    main()
