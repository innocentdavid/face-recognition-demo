import os
import math
import json
import sqlite3
import subprocess
from dataclasses import dataclass
from typing import List, Tuple, Dict

import cv2
import numpy as np
import faiss
from tqdm import tqdm

from insightface.app import FaceAnalysis


# -----------------------------
# CONFIG (edit if needed)
# -----------------------------
VIDEOS_DIR = "/data/videos"
QUERY_IMAGE = "/data/query/selfie.jpg"
WORK_DIR = "/data/output"

FRAMES_DIR = os.path.join(WORK_DIR, "frames")
DB_PATH = os.path.join(WORK_DIR, "faces.db")
FAISS_INDEX_PATH = os.path.join(WORK_DIR, "faces.index")

SAMPLE_EVERY_SECONDS = 2          # try 1 for more accuracy, 3-5 for faster
MIN_FACE_DET_SCORE = 0.60         # face detection confidence
TOP_K = 50                        # how many nearest neighbors to retrieve
SIMILARITY_THRESHOLD = 0.35       # cosine distance threshold (lower is stricter-ish after normalization)
MERGE_WITHIN_SECONDS = 4          # merge duplicate near timestamps


# -----------------------------
# Helpers
# -----------------------------
def ensure_dirs():
    os.makedirs(WORK_DIR, exist_ok=True)
    os.makedirs(FRAMES_DIR, exist_ok=True)

def seconds_to_hhmmss(s: int) -> str:
    h = s // 3600
    m = (s % 3600) // 60
    sec = s % 60
    return f"{h:02d}:{m:02d}:{sec:02d}"

def compact_timestamps(timestamps: List[int], merge_within_seconds: int) -> List[int]:
    if not timestamps:
        return []
    timestamps = sorted(set(timestamps))
    out = [timestamps[0]]
    for t in timestamps[1:]:
        if t - out[-1] > merge_within_seconds:
            out.append(t)
    return out

def init_db():
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("""
      CREATE TABLE IF NOT EXISTS faces (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        video_name TEXT NOT NULL,
        ts_seconds INTEGER NOT NULL,
        frame_path TEXT NOT NULL,
        det_score REAL NOT NULL
      )
    """)
    cur.execute("CREATE INDEX IF NOT EXISTS idx_faces_video ON faces(video_name)")
    conn.commit()
    conn.close()

def db_insert_face(video_name: str, ts_seconds: int, frame_path: str, det_score: float) -> int:
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute(
        "INSERT INTO faces(video_name, ts_seconds, frame_path, det_score) VALUES (?,?,?,?)",
        (video_name, ts_seconds, frame_path, det_score)
    )
    face_id = cur.lastrowid
    conn.commit()
    conn.close()
    return face_id

def db_get_face_rows_by_ids(ids: List[int]) -> List[Tuple[int, str, int, str, float]]:
    if not ids:
        return []
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    qmarks = ",".join("?" for _ in ids)
    cur.execute(f"SELECT id, video_name, ts_seconds, frame_path, det_score FROM faces WHERE id IN ({qmarks})", ids)
    rows = cur.fetchall()
    conn.close()
    # keep same order as ids
    rows_map = {r[0]: r for r in rows}
    return [rows_map[i] for i in ids if i in rows_map]

def ffmpeg_extract_frames(video_path: str, out_dir: str, every_s: int) -> List[Tuple[str, int]]:
    """
    Extract 1 frame every `every_s` seconds.
    Frame index i corresponds to ts = (i-1)*every_s
    """
    os.makedirs(out_dir, exist_ok=True)
    out_pattern = os.path.join(out_dir, "%06d.jpg")

    cmd = [
        "ffmpeg", "-hide_banner", "-loglevel", "error",
        "-i", video_path,
        "-vf", f"fps=1/{every_s}",
        "-q:v", "3",
        out_pattern
    ]
    subprocess.run(cmd, check=True)

    frames = sorted([f for f in os.listdir(out_dir) if f.lower().endswith(".jpg")])
    results = []
    for idx, fname in enumerate(frames, start=1):
        ts = (idx - 1) * every_s
        results.append((os.path.join(out_dir, fname), ts))
    return results

def read_image_rgb(path: str) -> np.ndarray:
    bgr = cv2.imread(path)
    if bgr is None:
        raise RuntimeError(f"Failed to read image: {path}")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return rgb

def l2_normalize(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v, axis=1, keepdims=True) + 1e-12
    return v / n


# -----------------------------
# Pipeline
# -----------------------------
def build_index(face_app: FaceAnalysis) -> Tuple[faiss.IndexFlatIP, List[int]]:
    """
    Returns a FAISS cosine-sim index (using inner product on normalized embeddings)
    and the mapping list: faiss_row_index -> db_face_id
    """
    video_files = sorted([f for f in os.listdir(VIDEOS_DIR) if f.lower().endswith((".mp4", ".mov", ".mkv"))])
    if not video_files:
        raise SystemExit("No videos found in /data/videos")

    all_embs = []
    all_db_ids = []

    print(f"\n📼 Indexing {len(video_files)} video(s) — sampling every {SAMPLE_EVERY_SECONDS}s")
    for vf in video_files:
        video_path = os.path.join(VIDEOS_DIR, vf)
        per_video_dir = os.path.join(FRAMES_DIR, os.path.splitext(vf)[0])

        frames = ffmpeg_extract_frames(video_path, per_video_dir, SAMPLE_EVERY_SECONDS)
        for frame_path, ts in tqdm(frames, desc=f"Frames {vf}", unit="frame"):
            img = read_image_rgb(frame_path)
            faces = face_app.get(img)

            if not faces:
                continue

            # store ALL faces found in frame (works for crowds)
            for face in faces:
                det_score = float(getattr(face, "det_score", 0.0))
                if det_score < MIN_FACE_DET_SCORE:
                    continue

                emb = face.embedding.astype("float32")[None, :]  # (1, 512)
                all_embs.append(emb)
                db_id = db_insert_face(vf, ts, frame_path, det_score)
                all_db_ids.append(db_id)

    if not all_embs:
        raise SystemExit("No faces indexed. Try lowering MIN_FACE_DET_SCORE or sampling more frequently.")

    embs = np.vstack(all_embs).astype("float32")
    embs = l2_normalize(embs)

    # Cosine similarity = inner product if vectors are normalized
    index = faiss.IndexFlatIP(embs.shape[1])
    index.add(embs)

    # Persist index + ids mapping
    faiss.write_index(index, FAISS_INDEX_PATH)
    with open(os.path.join(WORK_DIR, "id_map.json"), "w") as f:
        json.dump(all_db_ids, f)

    print(f"\n✅ Indexed {len(all_db_ids)} face embeddings.")
    return index, all_db_ids

def load_index() -> Tuple[faiss.IndexFlatIP, List[int]]:
    if not os.path.exists(FAISS_INDEX_PATH):
        raise SystemExit("Index not found. Run indexing first.")
    index = faiss.read_index(FAISS_INDEX_PATH)
    with open(os.path.join(WORK_DIR, "id_map.json"), "r") as f:
        id_map = json.load(f)
    return index, id_map

def query_selfie(face_app: FaceAnalysis, index: faiss.IndexFlatIP, id_map: List[int]) -> Dict[str, List[int]]:
    print("\n🧑‍🦱 Querying selfie...")
    img = read_image_rgb(QUERY_IMAGE)
    faces = face_app.get(img)
    if not faces:
        raise SystemExit("No face detected in selfie. Use a clearer front-facing photo.")

    # pick the biggest face
    faces = sorted(faces, key=lambda f: (f.bbox[2]-f.bbox[0])*(f.bbox[3]-f.bbox[1]), reverse=True)
    q = faces[0].embedding.astype("float32")[None, :]
    q = l2_normalize(q)

    sims, idxs = index.search(q, TOP_K)  # sims are cosine similarities (higher is better)
    sims = sims[0]
    idxs = idxs[0]

    # Filter by similarity threshold (cosine sim)
    # Typical good matches might be >= 0.35–0.55 depending on footage quality.
    hits = []
    for sim, faiss_i in zip(sims, idxs):
        if faiss_i < 0:
            continue
        if float(sim) < SIMILARITY_THRESHOLD:
            continue
        hits.append((int(faiss_i), float(sim)))

    if not hits:
        return {}

    # Map FAISS ids -> db ids
    db_ids = [id_map[h[0]] for h in hits]
    rows = db_get_face_rows_by_ids(db_ids)

    # Aggregate timestamps per video
    out: Dict[str, List[int]] = {}
    for (rid, video_name, ts_seconds, frame_path, det_score) in rows:
        out.setdefault(video_name, []).append(int(ts_seconds))

    # Compact timestamps
    for v in list(out.keys()):
        out[v] = compact_timestamps(out[v], MERGE_WITHIN_SECONDS)

    return out


def main():
    ensure_dirs()
    init_db()

    # InsightFace model setup (CPU)
    face_app = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
    face_app.prepare(ctx_id=-1, det_size=(640, 640))

    # Build (or rebuild) index each run — simplest for a demo with 5 videos
    index, id_map = build_index(face_app)

    results = query_selfie(face_app, index, id_map)

    if not results:
        print("\n❌ No matches found.")
        print("Try:")
        print(" - set SAMPLE_EVERY_SECONDS=1")
        print(" - lower SIMILARITY_THRESHOLD a bit (e.g., 0.30)")
        print(" - use a clearer selfie / better-lit footage")
        return

    print("\n✅ Matches (video → timestamps):")
    for video_name, ts_list in sorted(results.items()):
        pretty = ", ".join(seconds_to_hhmmss(t) for t in ts_list[:40])
        more = "" if len(ts_list) <= 40 else f" (+{len(ts_list)-40} more)"
        print(f" - {video_name}: {pretty}{more}")


if __name__ == "__main__":
    main()

