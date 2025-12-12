import os
import json
import sqlite3
import subprocess
from typing import List, Tuple, Dict, Optional, Callable

import cv2
import numpy as np
import faiss
from tqdm import tqdm

from insightface.app import FaceAnalysis


class FaceRecognitionEngine:
    def __init__(self, videos_dir: str, query_image: str, work_dir: str,
                 sample_every_seconds: int = 2,
                 min_face_det_score: float = 0.40,
                 top_k: int = 50,
                 similarity_threshold: float = 0.25,
                 merge_within_seconds: int = 4,
                 progress_callback: Optional[Callable] = None):
        self.videos_dir = videos_dir
        self.query_image = query_image
        self.work_dir = work_dir
        self.sample_every_seconds = sample_every_seconds
        self.min_face_det_score = min_face_det_score
        self.top_k = top_k
        self.similarity_threshold = similarity_threshold
        self.merge_within_seconds = merge_within_seconds
        self.progress_callback = progress_callback or (lambda x: None)
        
        self.frames_dir = os.path.join(work_dir, "frames")
        self.db_path = os.path.join(work_dir, "faces.db")
        self.faiss_index_path = os.path.join(work_dir, "faces.index")
        
        self.ensure_dirs()
        self.init_db()
        
        # Initialize face analysis
        self.face_app = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
        self.face_app.prepare(ctx_id=-1, det_size=(640, 640))
    
    def ensure_dirs(self):
        os.makedirs(self.work_dir, exist_ok=True)
        os.makedirs(self.frames_dir, exist_ok=True)
    
    def seconds_to_hhmmss(self, s: int) -> str:
        h = s // 3600
        m = (s % 3600) // 60
        sec = s % 60
        return f"{h:02d}:{m:02d}:{sec:02d}"
    
    def compact_timestamps(self, timestamps: List[int]) -> List[int]:
        if not timestamps:
            return []
        timestamps = sorted(set(timestamps))
        out = [timestamps[0]]
        for t in timestamps[1:]:
            if t - out[-1] > self.merge_within_seconds:
                out.append(t)
        return out
    
    def init_db(self):
        conn = sqlite3.connect(self.db_path)
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
    
    def db_insert_face(self, video_name: str, ts_seconds: int, frame_path: str, det_score: float) -> int:
        conn = sqlite3.connect(self.db_path)
        cur = conn.cursor()
        cur.execute(
            "INSERT INTO faces(video_name, ts_seconds, frame_path, det_score) VALUES (?,?,?,?)",
            (video_name, ts_seconds, frame_path, det_score)
        )
        face_id = cur.lastrowid
        conn.commit()
        conn.close()
        return face_id
    
    def db_get_face_rows_by_ids(self, ids: List[int]) -> List[Tuple[int, str, int, str, float]]:
        if not ids:
            return []
        conn = sqlite3.connect(self.db_path)
        cur = conn.cursor()
        qmarks = ",".join("?" for _ in ids)
        cur.execute(f"SELECT id, video_name, ts_seconds, frame_path, det_score FROM faces WHERE id IN ({qmarks})", ids)
        rows = cur.fetchall()
        conn.close()
        rows_map = {r[0]: r for r in rows}
        return [rows_map[i] for i in ids if i in rows_map]
    
    def ffmpeg_extract_frames(self, video_path: str, out_dir: str, every_s: int) -> List[Tuple[str, int]]:
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
    
    def read_image_rgb(self, path: str) -> np.ndarray:
        bgr = cv2.imread(path)
        if bgr is None:
            raise RuntimeError(f"Failed to read image: {path}")
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        return rgb
    
    def l2_normalize(self, v: np.ndarray) -> np.ndarray:
        n = np.linalg.norm(v, axis=1, keepdims=True) + 1e-12
        return v / n
    
    def build_index(self) -> Tuple[faiss.IndexFlatIP, List[int]]:
        video_files = sorted([f for f in os.listdir(self.videos_dir) 
                            if f.lower().endswith((".mp4", ".mov", ".mkv"))])
        if not video_files:
            raise ValueError("No videos found")
        
        all_embs = []
        all_db_ids = []
        
        self.progress_callback(f"📼 Indexing {len(video_files)} video(s) — sampling every {self.sample_every_seconds}s")
        
        for vf in video_files:
            video_path = os.path.join(self.videos_dir, vf)
            per_video_dir = os.path.join(self.frames_dir, os.path.splitext(vf)[0])
            
            frames = self.ffmpeg_extract_frames(video_path, per_video_dir, self.sample_every_seconds)
            
            for frame_path, ts in frames:
                img = self.read_image_rgb(frame_path)
                faces = self.face_app.get(img)
                
                if not faces:
                    continue
                
                for face in faces:
                    det_score = float(getattr(face, "det_score", 0.0))
                    if det_score < self.min_face_det_score:
                        continue
                    
                    emb = face.embedding.astype("float32")[None, :]
                    all_embs.append(emb)
                    db_id = self.db_insert_face(vf, ts, frame_path, det_score)
                    all_db_ids.append(db_id)
        
        if not all_embs:
            raise ValueError("No faces indexed. Try lowering MIN_FACE_DET_SCORE or sampling more frequently.")
        
        embs = np.vstack(all_embs).astype("float32")
        embs = self.l2_normalize(embs)
        
        index = faiss.IndexFlatIP(embs.shape[1])
        index.add(embs)
        
        faiss.write_index(index, self.faiss_index_path)
        with open(os.path.join(self.work_dir, "id_map.json"), "w") as f:
            json.dump(all_db_ids, f)
        
        self.progress_callback(f"✅ Indexed {len(all_db_ids)} face embeddings.")
        return index, all_db_ids
    
    def load_index(self) -> Tuple[faiss.IndexFlatIP, List[int]]:
        if not os.path.exists(self.faiss_index_path):
            raise ValueError("Index not found. Run indexing first.")
        index = faiss.read_index(self.faiss_index_path)
        with open(os.path.join(self.work_dir, "id_map.json"), "r") as f:
            id_map = json.load(f)
        return index, id_map
    
    def query_selfie(self, index: faiss.IndexFlatIP, id_map: List[int]) -> Dict[str, List[int]]:
        self.progress_callback("🧑‍🦱 Querying selfie...")
        img = self.read_image_rgb(self.query_image)
        faces = self.face_app.get(img)
        if not faces:
            raise ValueError("No face detected in selfie. Use a clearer front-facing photo.")
        
        faces = sorted(faces, key=lambda f: (f.bbox[2]-f.bbox[0])*(f.bbox[3]-f.bbox[1]), reverse=True)
        q = faces[0].embedding.astype("float32")[None, :]
        q = self.l2_normalize(q)
        
        sims, idxs = index.search(q, self.top_k)
        sims = sims[0]
        idxs = idxs[0]
        
        hits = []
        for sim, faiss_i in zip(sims, idxs):
            if faiss_i < 0:
                continue
            if float(sim) < self.similarity_threshold:
                continue
            hits.append((int(faiss_i), float(sim)))
        
        if not hits:
            return {}
        
        db_ids = [id_map[h[0]] for h in hits]
        rows = self.db_get_face_rows_by_ids(db_ids)
        
        out: Dict[str, List[int]] = {}
        for (rid, video_name, ts_seconds, frame_path, det_score) in rows:
            out.setdefault(video_name, []).append(int(ts_seconds))
        
        for v in list(out.keys()):
            out[v] = self.compact_timestamps(out[v])
        
        return out

