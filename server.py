#!/usr/bin/env python3
"""
ASPIRE face server
==================
Detect  : YuNet (OpenCV FaceDetectorYN, ~230 KB)
Embed   : ArcFace via onnxruntime. EMBED_MODEL=r50 (ResNet-50, most accurate, default) or mbf (MobileFaceNet, small, for Render free)
          Faces are FOUND on a small copy but aligned + embedded from the full-resolution photo, flip-averaged.
Storage : Firebase Realtime Database over plain REST (falls back to a local JSON file when FIREBASE_URL is unset)
API     : Flask, protected by an API key (header  X-API-Key  or  ?key=)

Endpoints
  GET  /                    test page (no auth, but every call it makes needs the key)
  GET  /health              liveness + number of people (no auth)
  POST /enroll              image + name [+ relation, persona, notes]  -> remember a person
  POST /identify            image                                      -> who is this?
  GET  /people              list people (no embeddings)
  POST /people/<id>         update name / relation / persona / notes / append_note
  DELETE /people/<id>       forget a person
  POST /reload              re-read everyone from Firebase

Image input (both work, so the ESP32 can just POST raw JPEG bytes later):
  * multipart/form-data with a file field called "image"
  * raw body with  Content-Type: image/jpeg   (parameters then go in the query string)

Common parameters (form fields or query string)
  rotate   auto (default) | 0 | 90 | 180 | 270   degrees CLOCKWISE applied before detection
  flip     0/1   mirror horizontally BEFORE rotating (undoes the camera's hmirror)
  annotate 0/1   also return the image with boxes drawn, as base64 JPEG
  notify   0/1   send the annotated photo + details to your Telegram chat (needs TELEGRAM_BOT_TOKEN + TELEGRAM_CHAT_ID)
                 /enroll : green box + name         /identify : green box (known) or red box (unknown) + name + match score
"""
import base64
import hmac
import json
import os
import re
import secrets
import shutil
import sys
import threading
import time
import uuid
import zipfile

import cv2
import numpy as np
import requests

# ----------------------------------------------------------------------------------------------
# Config (all overridable with environment variables)
# ----------------------------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.environ.get("MODEL_DIR", os.path.join(BASE_DIR, "models"))
DATA_DIR = os.environ.get("DATA_DIR", os.path.join(BASE_DIR, "data"))

YUNET_PATH = os.path.join(MODEL_DIR, "face_detection_yunet_2023mar.onnx")
YUNET_URL = ("https://github.com/opencv/opencv_zoo/raw/main/models/"
             "face_detection_yunet/face_detection_yunet_2023mar.onnx")

# Embedding model.  r50 = ResNet-50 ArcFace (w600k_r50, ~167 MB): clearly more accurate, needs a normal PC/laptop.
#                   mbf = MobileFaceNet (w600k_mbf, ~13 MB): small enough for Render's free 512 MB tier.
EMBED_MODEL = os.environ.get("EMBED_MODEL", "r50").strip().lower()
if EMBED_MODEL not in ("r50", "mbf"):
    sys.exit("EMBED_MODEL must be 'r50' or 'mbf'")
EMBED_FILE = {"r50": "w600k_r50.onnx", "mbf": "w600k_mbf.onnx"}[EMBED_MODEL]
EMBED_PATH = os.path.join(MODEL_DIR, EMBED_FILE)
EMBED_ZIP_URL = ("https://github.com/deepinsight/insightface/releases/download/v0.7/"
                 + {"r50": "buffalo_l.zip", "mbf": "buffalo_sc.zip"}[EMBED_MODEL])
# Stored embeddings are only valid for the exact model + pipeline that made them. Bump PIPELINE_VERSION whenever
# the alignment/embedding maths changes; anyone stored under a different tag has to be re-enrolled.
PIPELINE_VERSION = 2
EMBED_TAG = f"{EMBED_MODEL}-v{PIPELINE_VERSION}"

FIREBASE_URL = os.environ.get("FIREBASE_URL", "").rstrip("/")   # e.g. https://xxxx-default-rtdb.firebaseio.com
FIREBASE_AUTH = os.environ.get("FIREBASE_AUTH", "")             # optional database secret / ID token
FIREBASE_NODE = os.environ.get("FIREBASE_NODE", "people")

API_KEY = os.environ.get("API_KEY", "")
API_KEY_GENERATED = False
if not API_KEY:                       # fail closed: never run open. The generated key is printed at start-up.
    API_KEY = secrets.token_urlsafe(18)
    API_KEY_GENERATED = True
NO_AUTH = os.environ.get("NO_AUTH", "0") == "1"                  # LOCAL TESTING ONLY: turns the API key check off

MATCH_THRESHOLD = float(os.environ.get("MATCH_THRESHOLD", "0.40"))   # cosine; tune on your own people
DET_SCORE = float(os.environ.get("DET_SCORE", "0.6"))                # YuNet confidence for identify
ENROLL_MIN_SCORE = float(os.environ.get("ENROLL_MIN_SCORE", "0.75"))
ENROLL_MIN_FACE = int(os.environ.get("ENROLL_MIN_FACE", "36"))       # px (in the processed image)
IDENTIFY_MIN_FACE = int(os.environ.get("IDENTIFY_MIN_FACE", "24"))
FULL_SIDE = int(os.environ.get("FULL_SIDE", "2048"))                 # long side kept for alignment (faces are embedded from this full-res copy)
DET_SIDE = int(os.environ.get("DET_SIDE", "800"))                    # long side of the small copy used to FIND faces (speed)
REFINE_BELOW = int(os.environ.get("REFINE_BELOW", "140"))            # faces narrower than this (px) get a zoomed re-detect for better landmarks
USE_TTA = os.environ.get("TTA", "1") == "1"                          # flip averaging: embed the face and its mirror image, then average
ENROLL_MIN_SHARP = float(os.environ.get("ENROLL_MIN_SHARP", "15"))   # Laplacian variance of the aligned crop; 0 turns the blur check off
ENROLL_MAX_YAW = float(os.environ.get("ENROLL_MAX_YAW", "0.30"))     # head-turn limit for enrollment photos (0 = facing the camera)
# Wide search for small / distant faces. Only runs when the quick pass finds nothing, so normal photos are unaffected.
FAR_SEARCH = os.environ.get("FAR_SEARCH", "1") == "1"
FAR_DET_SCORE = float(os.environ.get("FAR_DET_SCORE", "0.45"))   # looser detector threshold for the wide search (matching still decides who it is)
FAR_TILE = int(os.environ.get("FAR_TILE", "384"))                # tile size (px of the full-res photo)
FAR_ZOOM = float(os.environ.get("FAR_ZOOM", "2.0"))              # each tile is blown up by this before detection
FAR_BUDGET_S = float(os.environ.get("FAR_BUDGET_S", "6"))        # total time allowed for tiling, split across the rotations tried
FAR_MAX_FACES = int(os.environ.get("FAR_MAX_FACES", "3"))
MAX_FACES = int(os.environ.get("MAX_FACES", "5"))
MAX_EMB_PER_PERSON = int(os.environ.get("MAX_EMB_PER_PERSON", "8"))
DEFAULT_ROTATE = int(os.environ.get("DEFAULT_ROTATE", "0"))          # first rotation tried in "auto" mode
DEFAULT_FLIP = os.environ.get("DEFAULT_FLIP", "0") == "1"
ORT_THREADS = int(os.environ.get("ORT_THREADS", "1"))
MAX_UPLOAD_MB = int(os.environ.get("MAX_UPLOAD_MB", "8"))

# Telegram (optional): the server itself sends the annotated photo, so the ESP32 never has to forward it.
TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID", "")
NOTIFY_DEFAULT = os.environ.get("NOTIFY_DEFAULT", "0") == "1"      # value of notify= when a request doesn't say (the test page doesn't, so it never spams)
NOTIFY_UNKNOWN = os.environ.get("NOTIFY_UNKNOWN", "1") == "1"      # also send the photo when /identify does NOT recognise the face
TELEGRAM_MAX_SIDE = int(os.environ.get("TELEGRAM_MAX_SIDE", "1024"))

# 5-point ArcFace template for a 112x112 crop (left eye, right eye, nose, left mouth, right mouth)
ARC_TEMPLATE = np.array([[38.2946, 51.6963],
                         [73.5318, 51.5014],
                         [56.0252, 71.7366],
                         [41.5493, 92.3655],
                         [70.7299, 92.2041]], dtype=np.float32)

ROT_CODES = {90: cv2.ROTATE_90_CLOCKWISE, 180: cv2.ROTATE_180, 270: cv2.ROTATE_90_COUNTERCLOCKWISE}
_SEARCH = threading.local()          # per-request note on how the faces were found (read by /identify)


# ----------------------------------------------------------------------------------------------
# Model download
# ----------------------------------------------------------------------------------------------
def _download(url, dest):
    print(f"[models] downloading {url}", flush=True)
    r = requests.get(url, stream=True, timeout=120, allow_redirects=True)
    r.raise_for_status()
    tmp = dest + ".part"
    total = int(r.headers.get("content-length") or 0)
    done, next_mark = 0, 25 << 20
    with open(tmp, "wb") as f:
        for chunk in r.iter_content(1 << 20):
            f.write(chunk)
            done += len(chunk)
            if total and done >= next_mark:
                print(f"[models]   {done >> 20} / {total >> 20} MB", flush=True)
                next_mark += 25 << 20
    os.replace(tmp, dest)


def ensure_models():
    os.makedirs(MODEL_DIR, exist_ok=True)
    if not os.path.exists(YUNET_PATH):
        _download(YUNET_URL, YUNET_PATH)
    if not os.path.exists(EMBED_PATH):
        zpath = os.path.join(MODEL_DIR, os.path.basename(EMBED_ZIP_URL))
        _download(EMBED_ZIP_URL, zpath)
        with zipfile.ZipFile(zpath) as z:
            member = next(n for n in z.namelist() if n.endswith(EMBED_FILE))
            with z.open(member) as src, open(EMBED_PATH + ".part", "wb") as dst:
                shutil.copyfileobj(src, dst)
        os.replace(EMBED_PATH + ".part", EMBED_PATH)
        os.remove(zpath)
    print(f"[models] ready (embedder: {EMBED_MODEL})", flush=True)


if __name__ == "__main__" and "--download-models" in sys.argv:   # used as the Render build step
    ensure_models()
    sys.exit(0)

import onnxruntime as ort                                   # noqa: E402  (after the download-only exit, so builds stay light)
from flask import Flask, jsonify, request, send_from_directory   # noqa: E402


# ----------------------------------------------------------------------------------------------
# Face engine: detect -> align -> embed
# ----------------------------------------------------------------------------------------------
def similarity_lsq(src, dst):
    """Least-squares 2D similarity transform (Umeyama): the 2x3 matrix that maps src points onto dst points.
    This is the maths InsightFace uses to align faces. It uses all 5 landmarks; cv2's RANSAC/LMEDS variants
    can throw landmarks away, which hurts when there are only 5."""
    src = np.asarray(src, dtype=np.float64)
    dst = np.asarray(dst, dtype=np.float64)
    mu_s, mu_d = src.mean(0), dst.mean(0)
    s, d = src - mu_s, dst - mu_d
    U, S, Vt = np.linalg.svd(d.T @ s / len(src))
    sign = np.array([1.0, np.sign(np.linalg.det(U) * np.linalg.det(Vt)) or 1.0])
    R = U @ np.diag(sign) @ Vt
    scale = float((S * sign).sum() / s.var(0).sum())
    t = mu_d - scale * (R @ mu_s)
    return np.hstack([scale * R, t[:, None]])


class FaceEngine:
    def __init__(self):
        self.det = cv2.FaceDetectorYN.create(YUNET_PATH, "", (320, 320), DET_SCORE, 0.3, 20)
        so = ort.SessionOptions()
        so.intra_op_num_threads = ORT_THREADS
        so.inter_op_num_threads = 1
        so.enable_cpu_mem_arena = False      # keep RSS low on a 512 MB box
        so.enable_mem_pattern = False
        self.sess = ort.InferenceSession(EMBED_PATH, so, providers=["CPUExecutionProvider"])
        self.inp = self.sess.get_inputs()[0].name
        self.det_lock = threading.Lock()     # FaceDetectorYN is stateful (setInputSize), so serialise it

    def detect(self, img):
        h, w = img.shape[:2]
        with self.det_lock:
            self.det.setInputSize((w, h))
            _, faces = self.det.detect(img)
        if faces is None:
            return []
        faces = [f for f in faces if f[2] >= 8 and f[3] >= 8]
        faces.sort(key=lambda f: f[2] * f[3], reverse=True)       # biggest face first
        return faces[:MAX_FACES]

    def _detect_raw(self, img, thr):
        """One detector pass at a custom score threshold; faces come back as float32 rows in img coordinates."""
        h, w = img.shape[:2]
        with self.det_lock:
            self.det.setScoreThreshold(thr)
            try:
                self.det.setInputSize((w, h))
                _, faces = self.det.detect(img)
            finally:
                self.det.setScoreThreshold(DET_SCORE)
        return [] if faces is None else [np.array(f, dtype=np.float32) for f in faces]

    @staticmethod
    def _iou(a, b):
        iw = min(a[0] + a[2], b[0] + b[2]) - max(a[0], b[0])
        ih = min(a[1] + a[3], b[1] + b[3]) - max(a[1], b[1])
        if iw <= 0 or ih <= 0:
            return 0.0
        inter = float(iw * ih)
        return inter / float(a[2] * a[3] + b[2] * b[3] - inter)

    def detect_far(self, img, deadline):
        """Slower search for small / distant faces, used only when the quick pass found nothing:
        (1) the whole full-resolution image (far more detail than the shrunken copy), then
        (2) overlapping tiles of it, each blown up FAR_ZOOM times, until `deadline`.
        Highest-scoring faces first (a big box isn't more likely to be a face here)."""
        H, W = img.shape[:2]
        cands = []
        if max(H, W) > DET_SIDE:
            cands = self._detect_raw(img, FAR_DET_SCORE)
        if not cands:
            T, step = FAR_TILE, max(1, int(FAR_TILE * 0.6))

            def starts(L):
                return [0] if L <= T else list(range(0, L - T, step)) + [L - T]

            done = False
            for ty in starts(H):
                for tx in starts(W):
                    if time.time() > deadline:
                        done = True
                        break
                    tile = img[ty:ty + T, tx:tx + T]
                    big = cv2.resize(tile, None, fx=FAR_ZOOM, fy=FAR_ZOOM, interpolation=cv2.INTER_CUBIC)
                    for f in self._detect_raw(big, FAR_DET_SCORE):
                        f[0:14] /= FAR_ZOOM
                        f[0] += tx
                        f[1] += ty
                        f[4:14:2] += tx                     # landmark x
                        f[5:14:2] += ty                     # landmark y
                        cands.append(f)
                    if cands and max(float(f[14]) for f in cands) >= 0.85:
                        done = True                         # a confident face: no need to scan the rest
                        break
                if done:
                    break
        cands.sort(key=lambda f: float(f[14]), reverse=True)
        keep = []
        for f in cands:                                      # greedy non-max suppression across overlapping tiles
            if f[2] >= 8 and f[3] >= 8 and all(self._iou(f, k) < 0.3 for k in keep):
                keep.append(f)
        return keep[:FAR_MAX_FACES]

    def refine(self, img, face):
        """Re-detect zoomed in on a small face. YuNet's landmarks are noticeably more accurate when the face is
        big in its input, and the alignment (so the embedding) depends on them. Falls back to the original
        detection if the zoomed pass doesn't find the same face."""
        x, y, w, h = [float(v) for v in face[0:4]]
        zoom = min(4.0, 200.0 / max(w, 1.0))
        if w >= REFINE_BELOW or zoom <= 1.05:
            return face
        H, W = img.shape[:2]
        cx, cy, side = x + w / 2, y + h / 2, max(w, h) * 2.4
        x0, y0 = int(max(0, cx - side / 2)), int(max(0, cy - side / 2))
        x1, y1 = int(min(W, cx + side / 2)), int(min(H, cy + side / 2))
        crop = img[y0:y1, x0:x1]
        if crop.size == 0:
            return face
        big = cv2.resize(crop, None, fx=zoom, fy=zoom, interpolation=cv2.INTER_CUBIC)
        with self.det_lock:
            self.det.setInputSize((big.shape[1], big.shape[0]))
            _, found = self.det.detect(big)
        if found is None:
            return face
        ocx, ocy = (cx - x0) * zoom, (cy - y0) * zoom
        best = min(found, key=lambda f: (f[0] + f[2] / 2 - ocx) ** 2 + (f[1] + f[3] / 2 - ocy) ** 2)
        far = (best[0] + best[2] / 2 - ocx) ** 2 + (best[1] + best[3] / 2 - ocy) ** 2 > (0.4 * w * zoom) ** 2
        if far or not (0.6 * w * zoom <= best[2] <= 1.6 * w * zoom):
            return face                                           # a different face, or a nonsense box
        out = np.array(best, dtype=np.float32)
        out[0:14] /= zoom
        out[0:2] += (x0, y0)                                      # box origin
        out[4:14:2] += x0                                         # landmark x
        out[5:14:2] += y0                                         # landmark y
        out[14] = max(float(out[14]), float(face[14]))
        return out

    def align(self, img, face):
        """Warp the face to the 112x112 ArcFace template, from the FULL-resolution image.
        YuNet landmarks: right eye, left eye, nose, right mouth, left mouth (as seen in the image, i.e.
        image-left first), the same order as the template."""
        lm = np.asarray(face[4:14], dtype=np.float64).reshape(5, 2)
        M = similarity_lsq(lm, ARC_TEMPLATE)
        scale = float(np.sqrt(abs(np.linalg.det(M[:, :2]))))
        if scale <= 1e-6:
            return None
        if scale < 0.7:                  # face is much bigger than 112 px: area-average first so the warp doesn't alias
            r = scale / 0.8
            img = cv2.resize(img, None, fx=r, fy=r, interpolation=cv2.INTER_AREA)
            M = similarity_lsq(lm * r, ARC_TEMPLATE)
        return cv2.warpAffine(img, M, (112, 112), flags=cv2.INTER_LINEAR, borderValue=0)

    def _run(self, crop):
        blob = cv2.dnn.blobFromImage(crop, 1.0 / 127.5, (112, 112), (127.5, 127.5, 127.5), swapRB=True)
        v = self.sess.run(None, {self.inp: blob})[0][0].astype(np.float64)
        n = float(np.linalg.norm(v))
        return v / n if n > 0 else None

    def embed(self, img, face):
        """Unit-length embedding of one detected face (or None). With TTA the face and its mirror image are
        both embedded and averaged, which is standard for ArcFace models and steadies the vector."""
        crop = self.align(img, face)
        if crop is None:
            return None
        v = self._run(crop)
        if v is None:
            return None
        if USE_TTA:
            v2 = self._run(cv2.flip(crop, 1))
            if v2 is not None:
                v = v + v2
                n = float(np.linalg.norm(v))
                if n > 0:
                    v = v / n
        return v.astype(np.float32)

    @staticmethod
    def yaw(face):
        """Rough head-turn measure: how far the nose sits from the eye midpoint, in eye-distances
        (0 = facing the camera, about 0.3 = turned roughly 35 degrees)."""
        lm = np.asarray(face[4:14], dtype=np.float64).reshape(5, 2)
        ev = lm[1] - lm[0]
        d2 = float(ev @ ev)
        if d2 < 1e-6:
            return 9.9
        return float(((lm[2] - (lm[0] + lm[1]) / 2) @ ev) / d2)

    def sharpness(self, img, face):
        """Laplacian variance of the aligned crop: low = blurry / no real detail."""
        crop = self.align(img, face)
        if crop is None:
            return 0.0
        return float(cv2.Laplacian(cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var())


# ----------------------------------------------------------------------------------------------
# People store (Firebase REST, or a local JSON file) + in-memory match index
# ----------------------------------------------------------------------------------------------
def slugify(name):
    s = re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")
    return (s or "person")[:24]


class Store:
    def __init__(self):
        self.people = {}                 # id -> dict (embeddings kept as list of np.float32 arrays)
        self.lock = threading.RLock()
        self.mode = "firebase" if FIREBASE_URL else "local"
        self.matrix = np.zeros((0, 512), dtype=np.float32)
        self.owner = []
        os.makedirs(DATA_DIR, exist_ok=True)
        self.local_file = os.path.join(DATA_DIR, "people.json")

    # ---- raw IO ----
    def _fb_url(self, path=""):
        return f"{FIREBASE_URL}/{FIREBASE_NODE}{('/' + path) if path else ''}.json"

    def _fb_params(self):
        return {"auth": FIREBASE_AUTH} if FIREBASE_AUTH else {}

    @staticmethod
    def _to_json(p):
        d = dict(p)
        d.pop("stale", None)
        d["model"] = p.get("model", EMBED_TAG)
        d["embeddings"] = [[round(float(x), 5) for x in e] for e in p["embeddings"]]
        return d

    @staticmethod
    def _from_json(pid, d):
        model = d.get("model") or "mbf-v1"       # saved before versioning existed = the first pipeline
        stale = model != EMBED_TAG               # made by another model/pipeline: its vectors can't be compared with ours
        embs = []
        for e in ([] if stale else (d.get("embeddings") or [])):
            a = np.asarray(e, dtype=np.float32)
            if a.shape == (512,):
                embs.append(a)
        return {"id": pid, "name": d.get("name", ""), "relation": d.get("relation", ""),
                "persona": d.get("persona", ""), "notes": d.get("notes", ""),
                "created": d.get("created", 0), "updated": d.get("updated", 0), "embeddings": embs,
                "model": model, "stale": stale}

    def load(self):
        raw = {}
        if self.mode == "firebase":
            r = requests.get(self._fb_url(), params=self._fb_params(), timeout=20)
            r.raise_for_status()
            raw = r.json() or {}
        elif os.path.exists(self.local_file):
            with open(self.local_file, "r", encoding="utf-8") as f:
                raw = json.load(f)
        with self.lock:
            self.people = {pid: self._from_json(pid, d) for pid, d in raw.items() if isinstance(d, dict)}
            self.rebuild()
        print(f"[store] loaded {len(self.people)} people from {self.mode}", flush=True)
        stale = [p["name"] for p in self.people.values() if p["stale"]]
        if stale:
            print(f"[store] RE-ENROLL NEEDED (made with another model/pipeline, current is {EMBED_TAG}): "
                  + ", ".join(stale), flush=True)

    def _save_local(self):
        tmp = self.local_file + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump({pid: self._to_json(p) for pid, p in self.people.items()}, f)
        os.replace(tmp, self.local_file)

    def save(self, pid):
        """Persist one person. Returns True when it reached storage."""
        with self.lock:
            p = self.people[pid]
            p["updated"] = int(time.time())
            if self.mode == "firebase":
                try:
                    r = requests.put(self._fb_url(pid), params=self._fb_params(),
                                     data=json.dumps(self._to_json(p)), timeout=10)
                    r.raise_for_status()
                    return True
                except Exception as e:
                    print(f"[store] Firebase write failed: {e}", flush=True)
                    return False
            self._save_local()
            return True

    def delete(self, pid):
        with self.lock:
            if pid not in self.people:
                return False
            del self.people[pid]
            self.rebuild()
            if self.mode == "firebase":
                try:
                    requests.delete(self._fb_url(pid), params=self._fb_params(), timeout=10).raise_for_status()
                except Exception as e:
                    print(f"[store] Firebase delete failed: {e}", flush=True)
                    return False
            else:
                self._save_local()
            return True

    # ---- matching ----
    def rebuild(self):
        rows, owner = [], []
        for pid, p in self.people.items():
            for e in p["embeddings"]:
                rows.append(e)
                owner.append(pid)
        self.matrix = np.stack(rows).astype(np.float32) if rows else np.zeros((0, 512), dtype=np.float32)
        self.owner = owner

    def find_by_name(self, name):
        n = name.strip().casefold()
        with self.lock:
            for pid, p in self.people.items():
                if p["name"].strip().casefold() == n:
                    return pid
        return None

    def match(self, q, exclude=None):
        """Return [(pid, score)] best first.
        Score = mean of the person's two best cosine similarities (just the best if they have one
        sample). Averaging two keeps one lucky sample from a stranger's face from passing."""
        with self.lock:
            if self.matrix.shape[0] == 0:
                return []
            sims = self.matrix @ q
            per = {}
            for s, pid in zip(sims.tolist(), self.owner):
                if pid != exclude:
                    per.setdefault(pid, []).append(s)
        out = []
        for pid, ss in per.items():
            ss.sort(reverse=True)
            out.append((pid, float(np.mean(ss[:2]))))
        out.sort(key=lambda t: t[1], reverse=True)
        return out


# ----------------------------------------------------------------------------------------------
# Image pipeline
# ----------------------------------------------------------------------------------------------
class ApiError(Exception):
    def __init__(self, msg, status=400, extra=None):
        super().__init__(msg)
        self.msg, self.status, self.extra = msg, status, extra or {}


def read_image_bytes():
    if "image" in request.files:
        return request.files["image"].read()
    data = request.get_data()
    if data:
        return data
    raise ApiError("No image received. Send a multipart field called 'image' or a raw JPEG body.", 400)


def truthy(v, default=False):
    if v is None or v == "":
        return default
    return str(v).strip().lower() in ("1", "true", "yes", "on")


def rotate_img(img, deg):
    return img if deg == 0 else cv2.rotate(img, ROT_CODES[deg])


def prepare(raw):
    arr = np.frombuffer(raw, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ApiError("That isn't a readable image.", 400)
    h, w = img.shape[:2]
    scale = FULL_SIDE / float(max(h, w))                   # (imdecode already applies the photo's EXIF rotation)
    if scale < 1.0:
        img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    return img


def find_faces(img, rotate_param, flip):
    """Returns (rotated_full_res_img, faces, rotation_used). Faces are FOUND on a small copy (fast) but every
    coordinate is mapped back to the full-resolution image, which is what gets aligned and embedded.
    Small faces (a half-body shot) are then re-detected zoomed in for better landmarks.
    In auto mode the other rotations are tried too (a glasses frame is often sideways)."""
    if flip:
        img = cv2.flip(img, 1)
    rotate_param = (rotate_param or "auto").strip().lower()
    if rotate_param == "auto":
        order = [DEFAULT_ROTATE] + [d for d in (0, 90, 270, 180) if d != DEFAULT_ROTATE]
    else:
        try:
            deg = int(rotate_param) % 360
        except ValueError:
            raise ApiError("rotate must be auto, 0, 90, 180 or 270", 400)
        if deg not in (0, 90, 180, 270):
            raise ApiError("rotate must be auto, 0, 90, 180 or 270", 400)
        order = [deg]
    best = None                                   # (strength, image, faces, rotation)
    h, w = img.shape[:2]
    s = min(1.0, DET_SIDE / float(max(h, w)))
    small = cv2.resize(img, (max(1, int(w * s)), max(1, int(h * s))), interpolation=cv2.INTER_AREA) if s < 1.0 else img
    for deg in order:
        rimg = rotate_img(small, deg)                 # rimg is the SMALL rotated copy inside this loop
        faces = engine.detect(rimg)
        if not faces:
            continue
        f = faces[0]
        strength = float(f[14]) - 0.08 * order.index(deg)     # detector score first, biased toward the rotation tried first (upright)
        if best is None or strength > best[0]:
            best = (strength, rimg, faces, deg)
        if float(f[14]) >= 0.85:   # confident, usable face: stop looking
            break
    _SEARCH.far = False
    if not best:
        if FAR_SEARCH:                                 # nothing on the quick pass: hunt for small / distant faces
            per = FAR_BUDGET_S / max(1, len(order))
            for deg in order:
                fullr = rotate_img(img, deg)
                far = engine.detect_far(fullr, time.time() + per)
                if far:
                    _SEARCH.far = True
                    return fullr, [engine.refine(fullr, f) for f in far], deg
        return rotate_img(img, order[0]), [], order[0]
    _, small_rot, sfaces, deg = best
    full = rotate_img(img, deg)
    fx, fy = full.shape[1] / small_rot.shape[1], full.shape[0] / small_rot.shape[0]
    mapped = []
    for f in sfaces:
        g = np.array(f, dtype=np.float32)
        g[0:14:2] *= fx                               # x, width and the landmark x's
        g[1:14:2] *= fy                               # y, height and the landmark y's
        mapped.append(engine.refine(full, g))
    mapped.sort(key=lambda f: f[2] * f[3], reverse=True)
    return full, mapped, deg


GREEN = (60, 200, 60)       # BGR: known / remembered
RED = (60, 60, 230)         # BGR: unknown


def render_annotated(img, items, max_side=900, show_score=True):
    """Draw a box + label for every face and return JPEG bytes (or None).
    items: [{"box": [x, y, w, h] in img coordinates, "known": bool, "name": str|None, "score": float}]
    Known = green box, unknown = red box. The label has a filled background so it stays readable on any photo."""
    sc = min(1.0, max_side / float(max(img.shape[:2])))                # img is full-res: shrink it for the reply
    out = cv2.resize(img, None, fx=sc, fy=sc, interpolation=cv2.INTER_AREA) if sc < 1.0 else img.copy()
    H, W = out.shape[:2]
    thick = max(2, int(round(max(H, W) / 380)))
    fscale = max(0.5, max(H, W) / 1400.0)
    for it in items:
        x, y, w, h = [int(v * sc) for v in it["box"]]
        col = GREEN if it["known"] else RED
        cv2.rectangle(out, (x, y), (x + w, y + h), col, thick)
        name = it.get("name") or "unknown"
        label = f'{name} {it["score"]:.2f}' if show_score else str(name)
        label = label.encode("ascii", "replace").decode()                 # cv2 fonts can't draw non-ASCII
        (tw, th), base = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, fscale, max(1, thick - 1))
        pad = 4
        top = y - th - base - 2 * pad
        if top < 0:                                                        # no room above the box: put the label inside its top edge
            top = y + thick
        left = min(max(0, x), max(0, W - tw - 2 * pad))
        cv2.rectangle(out, (left, top), (left + tw + 2 * pad, top + th + base + 2 * pad), col, -1)
        cv2.putText(out, label, (left + pad, top + th + pad), cv2.FONT_HERSHEY_SIMPLEX, fscale,
                    (255, 255, 255), max(1, thick - 1), cv2.LINE_AA)
    ok, buf = cv2.imencode(".jpg", out, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return buf.tobytes() if ok else None


def draw_annotated(img, items):
    """Base64 version for the JSON reply (used by the test page)."""
    jpg = render_annotated(img, items, 900, True)
    return base64.b64encode(jpg).decode() if jpg else None


# ----------------------------------------------------------------------------------------------
# Telegram (fire-and-forget, so the glasses never wait for it)
# ----------------------------------------------------------------------------------------------
def telegram_ready():
    return bool(TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID)


def _telegram_send_photo(jpeg, caption):
    try:
        r = requests.post(f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendPhoto",
                          data={"chat_id": TELEGRAM_CHAT_ID, "caption": caption[:1000]},
                          files={"photo": ("aspire.jpg", jpeg, "image/jpeg")}, timeout=25)
        if not r.ok:
            print(f"[telegram] sendPhoto failed: HTTP {r.status_code} {r.text[:200]}", flush=True)
    except Exception as e:                       # never print `e` itself: requests puts the bot token (it is in the URL) in its messages
        print(f"[telegram] sendPhoto failed: {type(e).__name__}", flush=True)


def notify_telegram(jpeg, caption):
    """Queue a photo + caption. Returns True if it was handed to the sender thread."""
    if not telegram_ready() or not jpeg:
        return False
    threading.Thread(target=_telegram_send_photo, args=(jpeg, caption), daemon=True).start()
    return True


def enroll_caption(p, created, added, face_px):
    lines = [("\u2705 Remembered: " if created else "\u2795 New look saved for: ") + p["name"]
             if added else "\u2139\ufe0f Already knew this look of: " + p["name"]]
    if p.get("relation"):
        lines.append("Relation: " + p["relation"])
    if p.get("persona"):
        lines.append("Persona: " + p["persona"])
    if p.get("notes"):
        lines.append("Notes: " + p["notes"])
    lines.append(f"Looks stored: {len(p['embeddings'])}  |  face {face_px}px  |  model {EMBED_TAG}")
    lines.append(time.strftime("%Y-%m-%d %H:%M UTC", time.gmtime()))
    return "\n".join(lines)


def identify_caption(results, primary, p, maybe):
    if p:
        head = f"\U0001F7E2 Identified: {p['name']}"
        if p.get("relation"):
            head += f" ({p['relation']})"
        lines = [head, f"Match score: {primary['score']:.2f}  (needs {MATCH_THRESHOLD:.2f})"]
        if p.get("persona"):
            lines.append("Persona: " + p["persona"])
        if p.get("notes"):
            lines.append("Notes: " + p["notes"])
    else:
        lines = ["\U0001F534 Unknown person", f"Best match score: {primary['score']:.2f}  (needs {MATCH_THRESHOLD:.2f})"]
        if maybe:
            lines.append(f"Closest: {maybe}")
    others = [f"{r['name']} ({r['score']:.2f})" for r in results[1:] if r["known"]]
    if others:
        lines.append("Also in frame: " + ", ".join(others))
    if len(results) > 1:
        lines.append(f"Faces in frame: {len(results)}")
    lines.append(time.strftime("%Y-%m-%d %H:%M UTC", time.gmtime()))
    return "\n".join(lines)


def persona_text(p):
    parts = [p["name"]]
    if p.get("relation"):
        parts.append(f'({p["relation"]})')
    txt = " ".join(parts) + "."
    if p.get("persona"):
        txt += " Persona: " + p["persona"].strip()
        if not txt.endswith((".", "!", "?")):
            txt += "."
    if p.get("notes"):
        txt += " Notes: " + p["notes"].strip()
    return txt


def public_person(p):
    return {"id": p["id"], "name": p["name"], "relation": p["relation"], "persona": p["persona"],
            "notes": p["notes"], "samples": len(p["embeddings"]), "stale": p["stale"],
            "created": p["created"], "updated": p["updated"]}


# ----------------------------------------------------------------------------------------------
# App
# ----------------------------------------------------------------------------------------------
ensure_models()
engine = FaceEngine()
store = Store()
try:
    store.load()
except Exception as e:                       # keep the server up even if Firebase is down at boot
    print(f"[store] initial load failed: {e}", flush=True)

app = Flask(__name__, static_folder=None)
app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_MB * 1024 * 1024
OPEN_PATHS = {"/", "/health", "/favicon.ico"}


@app.before_request
def check_key():
    if NO_AUTH or request.path in OPEN_PATHS:
        return None
    given = request.headers.get("X-API-Key") or request.args.get("key") or ""
    if not hmac.compare_digest(given.encode(), API_KEY.encode()):
        return jsonify(ok=False, error="Bad or missing API key."), 401
    return None


@app.errorhandler(ApiError)
def on_api_error(e):
    return jsonify(ok=False, error=e.msg, **e.extra), e.status


@app.errorhandler(413)
def too_big(_):
    return jsonify(ok=False, error=f"Image too large (max {MAX_UPLOAD_MB} MB)."), 413


@app.errorhandler(Exception)
def on_error(e):
    code = getattr(e, "code", 500)
    if code == 500:
        print(f"[error] {type(e).__name__}: {e}", flush=True)
    return jsonify(ok=False, error=str(e) if code != 500 else "Server error."), code


@app.get("/")
def index():
    return send_from_directory(BASE_DIR, "index.html")


@app.get("/health")
def health():
    return jsonify(ok=True, people=len(store.people), storage=store.mode, threshold=MATCH_THRESHOLD,
                   model=EMBED_TAG, re_enroll_needed=sum(1 for p in store.people.values() if p["stale"]),
                   telegram=telegram_ready())


@app.post("/enroll")
def enroll():
    t0 = time.time()
    v = request.values
    name = (v.get("name") or "").strip()
    if not name:
        raise ApiError("Give me a name to remember this person by.", 400)
    raw = read_image_bytes()
    img = prepare(raw)
    rimg, faces, rot = find_faces(img, v.get("rotate"), truthy(v.get("flip"), DEFAULT_FLIP))

    if not faces:
        raise ApiError("I couldn't find a face in that photo. Try better light and look at the camera.", 422)
    face = faces[0]
    if len(faces) > 1 and (faces[1][2] * faces[1][3]) > 0.4 * (face[2] * face[3]):
        raise ApiError("There's more than one clear face in that photo. Enroll one person at a time.", 422)
    if float(face[14]) < ENROLL_MIN_SCORE:
        raise ApiError(f"The face isn't clear enough (confidence {float(face[14]):.2f}, need {ENROLL_MIN_SCORE}; "
                       f"rotation {rot}). Face the camera in good light.", 422)
    if face[2] < ENROLL_MIN_FACE:
        ih, iw = rimg.shape[:2]
        raise ApiError(f"The face is too small ({int(face[2])}px wide in a {iw}x{ih} image, rotation {rot}; "
                       f"need {ENROLL_MIN_FACE}px). Move closer to the camera.", 422)

    yaw = abs(engine.yaw(face))
    if ENROLL_MAX_YAW > 0 and yaw > ENROLL_MAX_YAW:
        raise ApiError(f"The head is turned too far to the side (turn {yaw:.2f}, max {ENROLL_MAX_YAW}). "
                       "Use a photo where the face points at the camera.", 422)
    sharp = engine.sharpness(rimg, face)
    if face[2] >= 80 and sharp < ENROLL_MIN_SHARP:       # tiny faces are soft by nature (upscaled): the size note covers those
        raise ApiError(f"The face looks blurry (sharpness {sharp:.0f}, need {ENROLL_MIN_SHARP:.0f}). "
                       "Use a sharper photo in better light.", 422)
    emb = engine.embed(rimg, face)
    if emb is None:
        raise ApiError("Couldn't align that face. Try a straighter photo.", 422)

    pid = store.find_by_name(name)
    resp_extra = {}
    if face[2] < 80:
        resp_extra["note"] = (f"Small face ({int(face[2])}px): matching will be less certain. "
                              "If you can, also add a closer photo of this person.")
    # does this face look like someone ELSE already enrolled? (wrong name / same person twice)
    others = store.match(emb, exclude=pid)
    if others and others[0][1] >= 0.55:
        resp_extra["warning"] = (f'This face looks a lot like {store.people[others[0][0]]["name"]} '
                                 f'(score {others[0][1]:.2f}). Is that the same person?')

    with store.lock:
        created = pid is None
        if created:
            pid = f"{slugify(name)}-{uuid.uuid4().hex[:5]}"
            store.people[pid] = {"id": pid, "name": name, "relation": "", "persona": "", "notes": "",
                                 "created": int(time.time()), "updated": 0, "embeddings": [],
                                 "model": EMBED_TAG, "stale": False}
        p = store.people[pid]
        if p["stale"]:                                       # old vectors came from another model/pipeline: start fresh
            p["embeddings"], p["model"], p["stale"] = [], EMBED_TAG, False
        for field in ("relation", "persona", "notes"):
            val = (v.get(field) or "").strip()
            if val:
                p[field] = val
        added = True
        if p["embeddings"] and max(float(e @ emb) for e in p["embeddings"]) > 0.97:
            added = False                                    # basically the same photo again
        if added:
            p["embeddings"].append(emb)
            if len(p["embeddings"]) > MAX_EMB_PER_PERSON:
                p["embeddings"] = p["embeddings"][-MAX_EMB_PER_PERSON:]
        store.rebuild()
        synced = store.save(pid)
        samples = len(p["embeddings"])

    say = (f"Okay, I'll remember {name}." if created else
           (f"Got it, I've added another look at {name}." if added else f"I already have a very similar look of {name}."))
    result = {"ok": True, "id": pid, "name": name, "created": created, "sample_added": added, "samples": samples,
              "face_score": round(float(face[14]), 3), "rotation_used": rot, "synced": synced,
              "quality": {"face_px": int(face[2]), "turn": round(yaw, 3), "sharpness": round(sharp, 1)},
              "say": say, "ms": int((time.time() - t0) * 1000), **resp_extra}
    if not synced:
        result["warning"] = "Saved in memory but the database write failed. It will be lost on restart."
    want_b64 = truthy(v.get("annotate"))
    want_tg = truthy(v.get("notify"), NOTIFY_DEFAULT)
    if want_b64 or want_tg:
        item = [{"box": face[0:4], "known": True, "name": name, "score": 1.0}]
        if want_tg:                                          # green box + name only (no score: it isn't a match)
            jpg = render_annotated(rimg, item, TELEGRAM_MAX_SIDE, show_score=False)
            result["telegram"] = notify_telegram(jpg, enroll_caption(p, created, added, int(face[2])))
        if want_b64:
            result["annotated"] = draw_annotated(rimg, item)
    return jsonify(result)


@app.post("/identify")
def identify():
    t0 = time.time()
    v = request.values
    raw = read_image_bytes()
    img = prepare(raw)
    rimg, faces, rot = find_faces(img, v.get("rotate"), truthy(v.get("flip"), DEFAULT_FLIP))

    faces = [f for f in faces if f[2] >= IDENTIFY_MIN_FACE]
    if not faces:
        return jsonify(ok=True, faces=0, known=False, name=None, score=0.0, rotation_used=rot,
                       say="I can't see a clear face. Get a bit closer.", persona_text="",
                       image=f"{rimg.shape[1]}x{rimg.shape[0]}", min_face_px=IDENTIFY_MIN_FACE,
                       searched="quick + wide + tiles" if FAR_SEARCH else "quick only",
                       ms=int((time.time() - t0) * 1000))

    results = []
    for f in faces:
        emb = engine.embed(rimg, f)
        ranked = store.match(emb) if emb is not None else []
        best_pid, best = (ranked[0] if ranked else (None, 0.0))
        known = best_pid is not None and best >= MATCH_THRESHOLD
        p = store.people.get(best_pid) if known else None
        results.append({
            "box": [round(float(x), 1) for x in f[0:4]],
            "det_score": round(float(f[14]), 3),
            "known": known,
            "person_id": best_pid if known else None,
            "name": p["name"] if p else None,
            "score": round(best, 3),
            "margin": round(best - ranked[1][1], 3) if len(ranked) > 1 else None,
            "candidates": [{"name": store.people[pid]["name"], "score": round(s, 3)} for pid, s in ranked[:3]],
        })

    primary = results[0]                                         # largest face
    p = store.people.get(primary["person_id"]) if primary["known"] else None
    maybe = None
    if p:
        say = f"That's {p['name']}."
        if p.get("relation"):
            say = f"That's {p['name']}, your {p['relation']}." if not p["relation"].lower().startswith(("your", "my")) else f"That's {p['name']}."
    else:
        say = "I don't recognise this person."
        c0 = primary["candidates"][0] if primary["candidates"] else None
        # a near miss that clearly stands out from everyone else: say "might be", but never claim it as known
        if c0 and primary["score"] >= MATCH_THRESHOLD - 0.10 and (primary["margin"] is None or primary["margin"] >= 0.08):
            maybe = c0["name"]
            say = f"I'm not sure, but that might be {maybe}."
    extra_known = [r["name"] for r in results[1:] if r["known"]]
    if extra_known:
        say += " Also here: " + ", ".join(extra_known) + "."

    out = {"ok": True, "faces": len(results), "known": primary["known"], "name": primary["name"],
           "person_id": primary["person_id"], "score": primary["score"],
           "relation": p["relation"] if p else "", "persona": p["persona"] if p else "",
           "notes": p["notes"] if p else "", "persona_text": persona_text(p) if p else "",
           "maybe": maybe, "far_search": bool(getattr(_SEARCH, "far", False)),
           "margin": primary["margin"], "candidates": primary["candidates"], "results": results, "rotation_used": rot,
           "say": say, "threshold": MATCH_THRESHOLD, "ms": int((time.time() - t0) * 1000)}
    want_tg = truthy(v.get("notify"), NOTIFY_DEFAULT) and (primary["known"] or NOTIFY_UNKNOWN)
    if want_tg:                                              # green box + name + match score (red for unknown faces)
        jpg = render_annotated(rimg, results, TELEGRAM_MAX_SIDE, show_score=True)
        out["telegram"] = notify_telegram(jpg, identify_caption(results, primary, p, maybe))
    if truthy(v.get("annotate")):
        out["annotated"] = draw_annotated(rimg, results)
    return jsonify(out)


@app.get("/people")
def people_list():
    with store.lock:
        items = sorted((public_person(p) for p in store.people.values()), key=lambda d: d["name"].casefold())
    return jsonify(ok=True, people=items)


@app.post("/people/<pid>")
def people_update(pid):
    with store.lock:
        p = store.people.get(pid)
        if not p:
            raise ApiError("No such person.", 404)
        data = request.get_json(silent=True) or request.values
        for field in ("name", "relation", "persona", "notes"):
            if field in data and data[field] is not None:
                val = str(data[field]).strip()
                if field == "name" and not val:
                    continue
                p[field] = val
        extra = (data.get("append_note") or "").strip() if hasattr(data, "get") else ""
        if extra:
            stamp = time.strftime("%Y-%m-%d")
            p["notes"] = (p["notes"] + " | " if p["notes"] else "") + f"{stamp}: {extra}"
        synced = store.save(pid)
        return jsonify(ok=True, person=public_person(p), synced=synced)


@app.delete("/people/<pid>")
def people_delete(pid):
    if not store.delete(pid):
        raise ApiError("No such person (or the database delete failed).", 404)
    return jsonify(ok=True)


@app.post("/reload")
def reload_people():
    store.load()
    return jsonify(ok=True, people=len(store.people))


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "8000"))
    if NO_AUTH:
        print("\n[auth] NO_AUTH=1: API key check is OFF. Only use this on your own network, never on Render.\n", flush=True)
    elif API_KEY_GENERATED:
        print(f"\n[auth] No API_KEY set, generated one for this run:\n       {API_KEY}\n", flush=True)
    print(f"[server] http://0.0.0.0:{port}  (open / for the test page)", flush=True)
    app.run(host="0.0.0.0", port=port, threaded=True)
elif API_KEY_GENERATED:
    print(f"[auth] No API_KEY env var set, using a generated key for this run: {API_KEY}", flush=True)