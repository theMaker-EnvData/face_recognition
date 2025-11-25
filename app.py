import os
import io
import base64
import json
import sqlite3

import numpy as np
import cv2
import faiss
from flask import Flask, request, render_template_string, send_from_directory

from insightface.app import FaceAnalysis

# ---------------- 기본 설정 ----------------

BASE_DIR = os.path.abspath(os.path.dirname(__file__))

DATA_DB_PATH = os.path.join(BASE_DIR, "data", "namu_wiki_facevec.sqlite")
PERSON_DB_PATH = os.path.join(BASE_DIR, "crawls", "namu_wiki_persons.sqlite")

FAISS_ID_PATH = os.path.join(BASE_DIR, "data", "namu_facevec_ip.faiss")        # 동일인
FAISS_SIM_PATH = os.path.join(BASE_DIR, "data", "namu_facevec_sim_ip.faiss")   # 닮은꼴 (PCA)
PCA_PARAM_PATH = os.path.join(BASE_DIR, "data", "namu_face_pca_params.npz")

IMAGES_ROOT = os.path.join(BASE_DIR, "images")  # images/namu_wiki/...
TOP_K = 5


# ---------------- Flask 앱 ----------------

app = Flask(__name__)

# ---------------- 유틸 함수 ----------------


def get_sqlite_conn(db_path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn


def load_faiss_index(path: str):
    if not os.path.exists(path):
        raise FileNotFoundError(f"FAISS index not found: {path}")
    index = faiss.read_index(path)
    return index


def l2_normalize(x: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(x, axis=1, keepdims=True) + 1e-12
    return x / norm


def select_main_face(faces):
    """여러 얼굴 중 bbox 면적이 가장 큰 얼굴 선택"""
    if not faces:
        return None
    areas = []
    for f in faces:
        x1, y1, x2, y2 = f.bbox.astype(float)
        areas.append((x2 - x1) * (y2 - y1))
    idx = int(np.argmax(areas))
    return faces[idx]


def clean_image_rel_path(image_path: str) -> str:
    # 예: 'images\\namu_wiki\\1.webp' -> 'namu_wiki/1.webp'
    rel = image_path.replace("\\", "/")
    if rel.startswith("images/"):
        rel = rel[len("images/") :]
    return rel

def infobox_to_rows(infobox_json: str):
    """infobox_json을 테이블로 표시할 수 있게 (key, value) 리스트로 변환."""
    if not infobox_json:
        return []

    try:
        obj = json.loads(infobox_json)
    except Exception:
        # 파싱 안 되면 그냥 한 줄짜리로라도 뿌리기
        return [("", infobox_json)]

    rows = []

    if isinstance(obj, dict):
        for k, v in obj.items():
            if isinstance(v, (list, tuple)):
                v = ", ".join(str(x) for x in v)
            rows.append((str(k), str(v)))

    elif isinstance(obj, list):
        # [ ["키", "170cm"], ["몸무게", "52kg"] ] 이런 형태도 대충 맞춰줌
        for item in obj:
            if isinstance(item, (list, tuple)) and len(item) == 2:
                k, v = item
            elif isinstance(item, dict) and "key" in item and "value" in item:
                k, v = item["key"], item["value"]
            else:
                k, v = "", item
            rows.append((str(k), str(v)))

    else:
        rows.append(("", str(obj)))

    return rows

def project_to_sim(vec: np.ndarray) -> np.ndarray:
    """
    vec: (1, D) face_vec
    PCA 서브스페이스로 투영해서 L2-normalized vector 반환.
    """
    if _pca_mean is None or _pca_components is None:
        return None
    # (1, D) - (D,) -> (1, D)
    xc = vec - _pca_mean.reshape(1, -1)
    y = xc @ _pca_components   # (1, sim_dim)
    y = y.astype("float32")
    return l2_normalize(y)

def build_results_from_ids(scores, ids):
    """
    FAISS 결과 (scores, ids)로부터 카드 리스트를 만들어 반환.
    """
    results = []

    conn_face = get_sqlite_conn(DATA_DB_PATH)
    conn_person = get_sqlite_conn(PERSON_DB_PATH)

    try:
        for score, face_id in zip(scores, ids):
            if int(face_id) < 0:
                continue

            row_face = conn_face.execute(
                "SELECT id, page_id, page_title, page_url, image_path "
                "FROM namu_face WHERE id = ?",
                (int(face_id),),
            ).fetchone()
            if not row_face:
                continue

            page_id = row_face["page_id"]
            page_title = row_face["page_title"]
            page_url = row_face["page_url"]
            image_path = row_face["image_path"]

            row_person = conn_person.execute(
                "SELECT infobox_json, image_path FROM namu_person WHERE page_id = ?",
                (page_id,),
            ).fetchone()

            infobox_json = None
            person_image_path = None
            if row_person:
                infobox_json = row_person["infobox_json"]
                person_image_path = row_person["image_path"] or image_path
            else:
                person_image_path = image_path

            rel_path = clean_image_rel_path(person_image_path)
            image_url = None
            if rel_path:
                image_url = f"/images/{rel_path}"

            infobox_rows = infobox_to_rows(infobox_json) if infobox_json else []

            results.append(
                dict(
                    page_title=page_title,
                    page_url=page_url,
                    image_url=image_url,
                    score=float(score),
                    infobox_rows=infobox_rows,
                )
            )
    finally:
        conn_face.close()
        conn_person.close()

    return results


# ---------------- 전역 리소스 로드 ----------------

print("[INFO] InsightFace buffalo_l 로드 중...")
try:
    _face_engine = FaceAnalysis(name="buffalo_l")
    # ctx_id=0 : GPU / 실패하면 CPU(-1) 로 폴백
    try:
        _face_engine.prepare(ctx_id=0, det_size=(640, 640))
        print("[INFO] InsightFace 준비 완료 (GPU)")
    except Exception:
        _face_engine.prepare(ctx_id=-1, det_size=(640, 640))
        print("[INFO] InsightFace 준비 완료 (CPU)")
except Exception as e:
    raise RuntimeError(f"Failed to initialize FaceAnalysis: {e}")

print("[INFO] FAISS 인덱스 로드 중...")
_index_id = load_faiss_index(FAISS_ID_PATH)   # face_vec용 (동일인)
_index_sim = load_faiss_index(FAISS_SIM_PATH) # face_sim_vec용 (닮은꼴)
print(f"[INFO] ID index:   dim={_index_id.d}, ntotal={_index_id.ntotal}")
print(f"[INFO] SIM index:  dim={_index_sim.d}, ntotal={_index_sim.ntotal}")

print("[INFO] PCA 파라미터 로드 중...")
_pca_mean = None
_pca_components = None
if os.path.exists(PCA_PARAM_PATH):
    params = np.load(PCA_PARAM_PATH)
    _pca_mean = params["mean"].astype("float32")            # (D,)
    _pca_components = params["components"].astype("float32") # (D, sim_dim)
    print(f"[INFO] PCA loaded: D={_pca_components.shape[0]}, sim_dim={_pca_components.shape[1]}")
else:
    print("[WARN] PCA 파라미터 파일이 없습니다. 닮은꼴 모드는 동작하지 않습니다.")


# ---------------- 정적 이미지 서빙 ----------------


@app.route("/images/<path:filename>")
def serve_image(filename):
    # images/ 밑을 그대로 노출
    return send_from_directory(IMAGES_ROOT, filename)


# ---------------- 메인 페이지 & 검색 ----------------

HTML_TEMPLATE = """
<!doctype html>
<html lang="ko">
<head>
  <meta charset="utf-8">
  <title>Namu Face Search</title>
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <style>
    body { font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
           margin: 0; padding: 1rem; background:#111; color:#eee; }
    h1 { font-size: 1.4rem; margin-bottom: 0.5rem; }
    .container { max-width: 640px; margin: 0 auto; }
    .upload-box { border: 1px dashed #555; padding: 1rem; border-radius: 0.75rem;
                  text-align: center; margin-bottom: 1rem; background:#1a1a1a; }
    input[type=file] { width: 100%; margin-bottom: 0.5rem; }
    button { padding: 0.5rem 1rem; border-radius: 999px; border:none;
             background:#4f46e5; color:white; font-weight:600; }
    button:active { transform: scale(0.97); }
    .thumb { max-width: 100%; border-radius: 0.75rem; margin-bottom: 0.75rem; }
    .result { background:#1a1a1a; border-radius: 0.75rem; padding:0.75rem; margin-bottom:0.75rem; display:flex; gap:0.75rem; }
    .result img { width: 96px; height: 96px; object-fit: cover; border-radius:0.5rem; flex-shrink:0; }
    .result-title { font-weight:600; margin-bottom:0.25rem; }
    .result-meta { font-size:0.8rem; color:#aaa; margin-bottom:0.25rem; }
    pre { white-space: pre-wrap; word-break: break-word; font-size:0.75rem; background:#000; padding:0.5rem; border-radius:0.5rem; }
    .msg { margin-top:0.5rem; font-size:0.9rem; color:#f97373; }
    table.ibox { width:100%; border-collapse:collapse; font-size:0.8rem; }
    table.ibox th, table.ibox td { padding:0.25rem 0.35rem; vertical-align:top; }
    table.ibox th { width:30%; color:#a5b4fc; text-align:left; white-space:nowrap; }
    table.ibox tr:nth-child(odd) { background:#0b0b0b; }
    table.ibox tr:nth-child(even) { background:#050505; }
    .tabs { display:flex; gap:0.5rem; margin:0.5rem 0; }
    .tab-btn {
      flex:1;
      padding:0.4rem 0.5rem;
      border-radius:999px;
      border:none;
      background:#27272f;
      color:#e5e7eb;
      font-weight:600;
      font-size:0.9rem;
    }
    .tab-btn.active {
      background:#4f46e5;
      color:white;
    }
    .result-section { margin-top:0.25rem; }
  </style>
</head>
<body>
<div class="container">
  <h1>나무위키 얼굴 검색 (테스트)</h1>

  <form class="upload-box" method="POST" enctype="multipart/form-data">
    <div style="margin-bottom:0.5rem;">사진 한 장을 업로드해서 비슷한 얼굴을 찾아봅니다.</div>
    <label style="display:block; margin-bottom:0.5rem;">
      <input type="radio" name="source" value="gallery" checked style="width:auto; margin-right:0.25rem;">
      갤러리에서 선택
    </label>
    <label style="display:block; margin-bottom:0.5rem;">
      <input type="radio" name="source" value="camera" style="width:auto; margin-right:0.25rem;">
      카메라로 촬영
    </label>
    <input type="file" id="imageInput" name="image" accept="image/*" required>
    <button type="submit">검색</button>
    {% if message %}
      <div class="msg">{{ message }}</div>
    {% endif %}
  </form>

  {% if uploaded_image %}
    <div style="margin-bottom:0.75rem;">
      <div style="font-size:0.9rem; margin-bottom:0.25rem;">입력 이미지</div>
      <img class="thumb" src="data:image/jpeg;base64,{{ uploaded_image }}" alt="uploaded">
    </div>
  {% endif %}

  {% if identity_results or sim_results %}
    <div class="tabs">
      <button type="button"
              class="tab-btn active"
              data-mode="identity"
              onclick="showMode('identity')">
        동일인 Top {{ identity_results|length or 0 }}
      </button>
      <button type="button"
              class="tab-btn"
              data-mode="lookalike"
              onclick="showMode('lookalike')">
        닮은꼴 Top {{ sim_results|length or 0 }}
      </button>
    </div>

    <div id="section-identity" class="result-section">
      {% for r in identity_results %}
        <div class="result">
          {% if r.image_url %}
            <img src="{{ r.image_url }}" alt="face">
          {% endif %}
          <div>
            <div class="result-title">{{ r.page_title }}</div>
            <div class="result-meta">
              score: {{ "%.3f"|format(r.score) }}
              {% if r.page_url %}
                · <a href="{{ r.page_url }}" target="_blank" style="color:#a5b4fc;">나무위키</a>
              {% endif %}
            </div>
            {% if r.infobox_rows %}
              <table class="ibox">
                {% for k, v in r.infobox_rows %}
                  <tr>
                    <th>{{ k }}</th>
                    <td>{{ v }}</td>
                  </tr>
                {% endfor %}
              </table>
            {% endif %}
          </div>
        </div>
      {% endfor %}
    </div>

    <div id="section-lookalike" class="result-section" style="display:none;">
      {% for r in sim_results %}
        <div class="result">
          {% if r.image_url %}
            <img src="{{ r.image_url }}" alt="face">
          {% endif %}
          <div>
            <div class="result-title">{{ r.page_title }}</div>
            <div class="result-meta">
              score: {{ "%.3f"|format(r.score) }}
              {% if r.page_url %}
                · <a href="{{ r.page_url }}" target="_blank" style="color:#a5b4fc;">나무위키</a>
              {% endif %}
            </div>
            {% if r.infobox_rows %}
              <table class="ibox">
                {% for k, v in r.infobox_rows %}
                  <tr>
                    <th>{{ k }}</th>
                    <td>{{ v }}</td>
                  </tr>
                {% endfor %}
              </table>
            {% endif %}
          </div>
        </div>
      {% endfor %}
    </div>
  {% endif %}

</div>
<script>
function showMode(mode) {
  const idSec = document.getElementById('section-identity');
  const simSec = document.getElementById('section-lookalike');
  const btns = document.querySelectorAll('.tab-btn');
  if (!idSec || !simSec) return;

  if (mode === 'identity') {
    idSec.style.display = 'block';
    simSec.style.display = 'none';
  } else {
    idSec.style.display = 'none';
    simSec.style.display = 'block';
  }

  btns.forEach(btn => {
    if (btn.dataset.mode === mode) {
      btn.classList.add('active');
    } else {
      btn.classList.remove('active');
    }
  });
}

// 파일 입력 소스 전환 (갤러리 vs 카메라)
document.addEventListener('DOMContentLoaded', function() {
  const radios = document.querySelectorAll('input[name="source"]');
  const fileInput = document.getElementById('imageInput');
  
  if (radios && fileInput) {
    radios.forEach(radio => {
      radio.addEventListener('change', function() {
        if (this.value === 'camera') {
          fileInput.setAttribute('capture', 'environment');
        } else {
          fileInput.removeAttribute('capture');
        }
      });
    });
  }
});
</script>
</body>
</html>
"""


@app.route("/", methods=["GET", "POST"])
def index():
    uploaded_image_b64 = None
    identity_results = []
    sim_results = []
    message = None

    if request.method == "POST":
        file = request.files.get("image")
        if not file or file.filename == "":
            message = "이미지 파일을 선택해 주세요."
        else:
            data = file.read()
            uploaded_image_b64 = base64.b64encode(data).decode("ascii")

            img_array = np.frombuffer(data, dtype=np.uint8)
            img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
            if img is None:
                message = "이미지를 읽을 수 없습니다."
            else:
                faces = _face_engine.get(img)
                if not faces:
                    message = "얼굴을 찾지 못했습니다."
                else:
                    face = select_main_face(faces)
                    emb = face["embedding"].astype("float32")[np.newaxis, :]
                    emb_norm = l2_normalize(emb)

                    # 1) 동일인 검색 (face_vec 기반)
                    scores_id, ids_id = _index_id.search(emb_norm, TOP_K)
                    identity_results = build_results_from_ids(scores_id[0], ids_id[0])

                    # 2) 닮은꼴 검색 (PCA + face_sim_vec 기반)
                    emb_sim = project_to_sim(emb)
                    if emb_sim is None:
                        message = (message or "") + " (PCA 파라미터가 없어 닮은꼴 모드는 비활성화됩니다.)"
                        sim_results = []
                    else:
                        scores_sim, ids_sim = _index_sim.search(emb_sim, TOP_K)
                        sim_results = build_results_from_ids(scores_sim[0], ids_sim[0])

    return render_template_string(
        HTML_TEMPLATE,
        uploaded_image=uploaded_image_b64,
        identity_results=identity_results,
        sim_results=sim_results,
        message=message,
    )


if __name__ == "__main__":
    # VM에서 외부 접속하려면 host="0.0.0.0" 로
    app.run(host="0.0.0.0", port=8000, debug=False)
