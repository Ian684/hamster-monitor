#!/usr/bin/env python3
"""Hamster tracker, phase 1: motion tracking with Picamera2 + OpenCV background subtraction.

- Finds the largest moving blob inside the cage ROI -> hamster position
- Optional 4-corner floor calibration turns an oblique view into top-down floor coordinates
- Keeps the last known position while the hamster sits still
- Logs 1 Hz positions + per-minute activity (active seconds, distance) to SQLite
- Writes data/latest.jpg (annotated), data/heatmap.png, data/tracker.json
- Streams annotated frames to /run/hamster/live.jpg while the dashboard is watching
- Saves raw frames to data/dataset/ for training a YOLO model later (phase 2)
"""
import json
import logging
import math
import os
import signal
import sqlite3
import statistics
import time
import tomllib
from collections import deque
from datetime import datetime, timedelta
from pathlib import Path

import cv2
import numpy as np
from libcamera import Transform
from picamera2 import Picamera2

from clips import ClipRecorder
from floor_geom import floor_shape
import darkness
from zonedefs import hex_bgr, zone_defs, zone_ids, zone_types

BASE = Path(__file__).resolve().parent
LIVE_DIR = Path("/run/hamster")  # tmpfs (RAM): live frames never touch the SD card; see /etc/tmpfiles.d/hamster.conf
LIVE_REQUEST = LIVE_DIR / "live.request"  # web.py touches this while someone is watching
PHOTO_REQUEST = LIVE_DIR / "photo.request"   # 拍照 on the page: write one clean full-size frame to photo.jpg
REC_REQUEST = LIVE_DIR / "rec.request"       # 錄影 on the page: its text = seconds to record (0 = stop)
PAUSE_FILE = BASE / "data" / "pause.json"   # {"until": epoch} written by web.py: hands in the cage, don't record
log = logging.getLogger("tracker")


def load_config():
    with open(BASE / "config.toml", "rb") as f:
        return tomllib.load(f)


def atomic_write(path, data):
    tmp = path.with_name(path.name + ".tmp")
    try:
        tmp.write_bytes(data)
    except FileNotFoundError:  # directory vanished: recreate and retry once
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp.write_bytes(data)
    os.replace(tmp, path)


def encode(img, ext=".jpg", quality=80):
    params = [cv2.IMWRITE_JPEG_QUALITY, quality] if ext == ".jpg" else []
    ok, buf = cv2.imencode(ext, img, params)
    return buf.tobytes() if ok else None


class MotionDetector:
    def __init__(self, cfg):
        self.cfg = cfg
        self.bg = cv2.createBackgroundSubtractorMOG2(
            history=cfg["bg_history"], varThreshold=cfg["bg_threshold"], detectShadows=False)
        self.kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        self.warmup = cfg["warmup_frames"]
        self.blobs = []   # (area share, box) of every moving blob in the last frame, the wheel area left out
        self.blob_skip = None   # bool mask (detector scale) of areas whose movement is never a hand (the wheel)
        self.roi = cfg.get("roi") or []   # cage outline, normalized [[x, y], ...]
        self.roi_mask = None

    def detect(self, frame):
        """Return the largest plausible moving blob as a normalized (x, y, w, h) box, or None."""
        s = self.cfg["process_scale"]
        small = cv2.resize(frame, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
        gray = cv2.GaussianBlur(cv2.cvtColor(small, cv2.COLOR_BGR2GRAY), (7, 7), 0)
        mask = self.bg.apply(gray)
        self.blobs = []
        if self.warmup > 0:  # let the background model settle first
            self.warmup -= 1
            return None
        h, w = mask.shape
        if len(self.roi) >= 3:  # ignore anything moving outside the cage
            if self.roi_mask is None:
                self.roi_mask = np.zeros_like(mask)
                pts = np.array([[x * w, y * h] for x, y in self.roi], np.int32)
                cv2.fillPoly(self.roi_mask, [pts], 255)
            mask = cv2.bitwise_and(mask, self.roi_mask)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self.kernel)
        mask = cv2.dilate(mask, self.kernel, iterations=2)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        lo, hi = self.cfg["min_area"] * w * h, self.cfg["max_area"] * w * h
        best, best_a = None, 0
        # every moving blob of some size, the wheel left out: used to notice a hand in the cage (hand_in_cage)
        hm = mask
        if self.blob_skip is not None and self.blob_skip.shape == mask.shape:
            hm = mask.copy()
            hm[self.blob_skip] = 0
        if cv2.countNonZero(hm) >= 0.05 * w * h:   # nothing big moving: skip the second pass
            for c in (contours if hm is mask else cv2.findContours(hm, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]):
                a = cv2.contourArea(c)
                if a >= 0.05 * w * h:
                    bx, by, bw_, bh_ = cv2.boundingRect(c)
                    self.blobs.append((a / (w * h), (bx / w, by / h, bw_ / w, bh_ / h)))
        for c in contours:
            a = cv2.contourArea(c)
            # too small = noise/bedding; too big = light switched on / camera bumped
            if lo <= a <= hi and a > best_a:
                best, best_a = c, a
        if best is None:
            return None
        x, y, bw, bh = cv2.boundingRect(best)
        return (x / w, y / h, bw / w, bh / h)


class FloorMap:
    """Image (normalized) <-> cage floor mapping.

    Without calibration this is the identity and the box centre is used (top-down camera).
    With 4 points on the floor (1 TL, 2 TR, 3 BR, 4 BL in image coords) and the real lengths of the four sides
    between them it is a perspective transform: the trapezoid the camera sees is flattened back into the real
    shape seen from straight above, so near and far get the same centimetres. The 4 points can be the cage
    floor's corners or 4 markers laid on the floor; the shape can be a rectangle or a trapezoid (1→2 parallel
    to 4→3, like a cage that is narrower at one end). Points outside the 4 points are extrapolated, not clipped.
    The bottom-centre of the box is used, since that is where the hamster touches the floor.

    Floor coordinates are 0..1 across the shape's bounding box; w_cm / h_cm give its real size.
    """

    def __init__(self, corners, sides_cm=None, fallback_cm=(30, 50)):
        self.H = None
        self.w_cm, self.h_cm = fallback_cm
        if corners and len(corners) == 4:
            real = floor_shape(sides_cm) if sides_cm else None
            if real is None:   # old config: a rectangle cage_w_cm x cage_h_cm
                w, h = fallback_cm
                real = [(0, 0), (w, 0), (w, h), (0, h)]
            xs, ys = [p[0] for p in real], [p[1] for p in real]
            x0, self.w_cm, self.h_cm = min(xs), max(xs) - min(xs), max(ys) - min(ys)
            dst = np.float32([[(px - x0) / self.w_cm, py / self.h_cm] for px, py in real])
            self.H = cv2.getPerspectiveTransform(np.float32(corners), dst)
            self.Hinv = np.linalg.inv(self.H)

    @property
    def calibrated(self):
        return self.H is not None

    def floor_point(self, box):
        x, y, w, h = box
        if self.H is None:
            return (x + w / 2, y + h / 2)
        q = cv2.perspectiveTransform(np.float32([[[x + w / 2, y + h]]]), self.H)[0][0]
        return (float(q[0]), float(q[1]))

    def image_point(self, pos):
        if self.H is None:
            return pos
        q = cv2.perspectiveTransform(np.float32([[pos]]), self.Hinv)[0][0]
        return (float(q[0]), float(q[1]))

    def floor_to_image(self, floor_img, W, H):
        """Warp a W x H floor-space image into camera space (for the heatmap overlay)."""
        if self.H is None:
            return floor_img
        N = np.diag([1 / W, 1 / H, 1.0])        # image px -> normalized
        S = np.diag([float(W), float(H), 1.0])  # floor unit square -> floor px
        M = S @ self.H @ N                     # image px -> floor px
        return cv2.warpPerspective(floor_img, M, (W, H), flags=cv2.WARP_INVERSE_MAP | cv2.INTER_LINEAR)


class Store:
    def __init__(self, db_path):
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(db_path, timeout=10)
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute("CREATE TABLE IF NOT EXISTS positions (ts TEXT PRIMARY KEY, x REAL, y REAL, moving INTEGER)")
        self.db.execute("CREATE TABLE IF NOT EXISTS activity (minute TEXT PRIMARY KEY, active_s INTEGER, distance_cm REAL)")
        # seconds per minute spent in each zone (nest/wheel/food/water/bath) and individual visits
        self.db.execute("CREATE TABLE IF NOT EXISTS zone_minutes (minute TEXT, zone TEXT, seconds INTEGER, PRIMARY KEY (minute, zone))")
        self.db.execute("CREATE TABLE IF NOT EXISTS zone_visits (start TEXT, zone TEXT, seconds INTEGER)")
        # AI per minute: how often it ran, how often it was sure it saw the hamster (>= yolo_conf),
        # how often it was unsure, and the confidence when sure (sum -> average, and best)
        # how much light there was (camera's own estimate) next to how well the AI did: where accuracy falls off
        self.db.execute("CREATE TABLE IF NOT EXISTS light_minutes (minute TEXT PRIMARY KEY, lux REAL, exp_ms REAL, "
                        "gain REAL, bright REAL, noise REAL)")
        # wheel turns counted from the white strips passing the top of the picture, and the distance they make
        self.db.execute("CREATE TABLE IF NOT EXISTS wheel_minutes (minute TEXT PRIMARY KEY, revs REAL, cm REAL)")
        self.db.execute("CREATE TABLE IF NOT EXISTS ai_minutes (minute TEXT PRIMARY KEY, runs INTEGER, seen INTEGER, "
                        "unsure INTEGER, conf_sum REAL, conf_max REAL)")
        # estimated eating / drinking events from the camera (see the 1 Hz block in main)
        self.db.execute("CREATE TABLE IF NOT EXISTS behavior_events (start TEXT, kind TEXT, seconds INTEGER)")
        try:   # average body movement during the event (eating check), for tuning eat_motion_min later
            self.db.execute("ALTER TABLE behavior_events ADD COLUMN motion REAL")
        except sqlite3.OperationalError:
            pass   # already there
        try:   # why the event was kept: the whole-event checks (judge_event), as JSON
            self.db.execute("ALTER TABLE behavior_events ADD COLUMN evidence TEXT")
        except sqlite3.OperationalError:
            pass
        # candidates the whole-event checks threw away (a hand turning the wheel, passing the spout ...)
        self.db.execute("CREATE TABLE IF NOT EXISTS behavior_dropped (start TEXT, kind TEXT, seconds INTEGER, "
                        "motion REAL, why TEXT, evidence TEXT)")
        self.db.commit()
        self.pending = []

    def behavior(self, start, kind, seconds, motion=None, evidence=None):
        """One estimated eating ("eat"), drinking ("drink") or wheel-running ("run") event."""
        self.db.execute("INSERT INTO behavior_events (start, kind, seconds, motion, evidence) VALUES (?,?,?,?,?)",
                        (start.isoformat(timespec="seconds"), kind, int(seconds), None if motion is None else round(motion, 4),
                         json.dumps(evidence, separators=(",", ":")) if evidence else None))
        self.db.commit()

    def behavior_dropped(self, start, kind, seconds, motion, why, evidence):
        self.db.execute("INSERT INTO behavior_dropped VALUES (?,?,?,?,?,?)",
                        (start.isoformat(timespec="seconds"), kind, int(seconds), None if motion is None else round(motion, 4),
                         why, json.dumps(evidence, separators=(",", ":"))))
        self.db.commit()

    def wheel_minute(self, minute, revs, cm):
        self.db.execute("INSERT OR REPLACE INTO wheel_minutes VALUES (?,?,?)", (minute, round(revs, 2), round(cm, 1)))
        self.db.commit()

    def light_minute(self, minute, s):
        n = s["n"]
        self.db.execute("INSERT OR REPLACE INTO light_minutes VALUES (?,?,?,?,?,?)",
                        (minute, round(s["lux"] / n, 2), round(s["exp"] / n, 2), round(s["gain"] / n, 2),
                         round(s["bright"] / n, 1), round(s["noise"] / n, 2)))
        self.db.commit()

    def ai_minute(self, minute, s):
        self.db.execute("INSERT OR REPLACE INTO ai_minutes VALUES (?,?,?,?,?,?)",
                        (minute, s["runs"], s["seen"], s["unsure"], round(float(s["conf_sum"]), 3), round(float(s["conf_max"]), 3)))
        self.db.commit()

    def position(self, ts, x, y, moving):
        self.pending.append((ts, x, y, int(moving)))

    def flush(self):
        if self.pending:
            self.db.executemany("INSERT OR REPLACE INTO positions VALUES (?,?,?,?)", self.pending)
            self.db.commit()
            self.pending.clear()

    def prune(self, keep_days):
        """1 Hz positions add ~86k rows a day; keep only recent ones (per-minute activity is kept forever)."""
        cutoff = (datetime.now() - timedelta(days=keep_days)).isoformat(timespec="seconds")
        n = self.db.execute("DELETE FROM positions WHERE ts < ?", (cutoff,)).rowcount
        self.db.commit()
        if n:
            log.info("pruned %d position rows older than %d days", n, keep_days)

    def activity(self, minute, active_s, distance_cm):
        self.db.execute("INSERT OR REPLACE INTO activity VALUES (?,?,?)", (minute, active_s, round(distance_cm, 1)))
        self.db.commit()

    def zone_minute(self, minute, zone, seconds):
        self.db.execute("INSERT OR REPLACE INTO zone_minutes VALUES (?,?,?)", (minute, zone, seconds))
        self.db.commit()

    def visit(self, start, zone, seconds):
        self.db.execute("INSERT INTO zone_visits VALUES (?,?,?)", (start.isoformat(timespec="seconds"), zone, seconds))
        self.db.commit()

    def close(self):
        self.flush()
        self.db.close()


def boxes_touch(a, b, margin=0.03):
    """True if two normalized (x, y, w, h) boxes overlap, allowing a small margin."""
    return not (a[0] > b[0] + b[2] + margin or b[0] > a[0] + a[2] + margin or
                a[1] > b[1] + b[3] + margin or b[1] > a[1] + a[3] + margin)


def decide(mode, y_now, y_moved, m_box, follow_box, since_ai, grace_s, follow_margin, ref_area=None, max_ratio=2.5):
    """Where is the hamster and did it move this frame?  -> (pos_box, event, via_follow, follow_box)

    motion (no model): any moving blob counts.
    yolo:   only the AI box; "moved" = the AI box shifted.
    hybrid: the AI decides. Movement counts only on/near the hamster the AI sees; a moving blob elsewhere
            (shadow, light change, bedding, a hand) is ignored. When the AI briefly loses the hamster (behind
            the bottle, motion blur), movement right next to where it was last seen still counts for grace_s
            seconds, and that spot follows the moving blob, so a running hamster isn't dropped.
            Only a blob about the hamster's size is followed (at most max_ratio x the box the AI last saw): a hand
            reaching in is a much bigger blob, and following it counted the hand as visits to the bowl / nest
            (11/16 22:33-22:37).
    """
    if mode == "motion":
        return m_box, m_box is not None, False, follow_box
    if mode == "yolo":
        return y_now, y_moved, False, follow_box
    if y_now is not None:
        return y_now, m_box is not None and boxes_touch(m_box, y_now), False, y_now
    if (m_box is not None and follow_box is not None and since_ai < grace_s and boxes_touch(m_box, follow_box, follow_margin)
            and (not ref_area or m_box[2] * m_box[3] <= max_ratio * ref_area)):
        return m_box, True, True, m_box
    return None, False, False, (follow_box if since_ai < grace_s else None)


def load_yolo(t_cfg):
    """Returns (mode, detector or None). Falls back to plain motion detection if the model is missing."""
    mode = t_cfg.get("detector", "motion")
    if mode not in ("yolo", "hybrid"):
        return "motion", None
    model_dir = BASE / t_cfg.get("yolo_model", "models/hamster_model")
    if not (model_dir / "model.onnx").exists():
        log.warning("detector=%s but no model at %s; using motion detection", mode, model_dir)
        return "motion", None
    from yolo_detector import YoloOnnx   # imported lazily: only needed once a model exists (ONNX Runtime)
    # ask the model for low-confidence boxes too: they are used to collect "hard" training photos;
    # only boxes >= yolo_conf count as the hamster
    low = min(t_cfg.get("yolo_conf", 0.45), t_cfg.get("hard_conf_low", 0.2))
    det = YoloOnnx(model_dir, imgsz=t_cfg.get("yolo_imgsz", 320), conf=low)
    log.info("YOLO model loaded from %s (classes: %s)", model_dir, det.names or "?")
    return mode, det


BEH_OF_TYPE = {"eat": "eat", "drink": "drink"}   # area type -> estimated behaviour while staying put there
# outline colours on the camera picture (BGR); load_zones fills in the user's colours from the zone list
ZONE_COLORS = {"nest": (255, 120, 190), "wheel": (80, 220, 80), "food": (0, 170, 255), "water": (255, 160, 40), "bath": (200, 200, 200)}


def load_zones(t_cfg, W, H):
    """[(kind, polygon in pixels)] from [tracker] zones = [{kind = "nest", poly = [[x, y], ...]}, ...].
    The water bottle is not a drawn zone any more: its area is the drinking spot under the marked spout
    (drink_spot), so an old drawn "water" polygon is ignored."""
    defs = zone_defs(t_cfg)   # the areas in the user's list (edited on the dashboard); a deleted one is ignored
    ZONE_COLORS.update({d["id"]: hex_bgr(d["color"]) for d in defs})
    # 喝水 areas (the built-in 水瓶 and added ones) are a circle under their spout, not an outline
    ids = {d["id"] for d in defs if d["type"] != "drink"}
    out = []
    for z in t_cfg.get("zones") or []:
        poly = z.get("poly") or []
        if z.get("kind") in ids and len(poly) >= 3:
            out.append((z["kind"], np.float32([[x * W, y * H] for x, y in poly])))
    return out


def note_photo_zone(dataset, photo, zone, moved=None):
    """Remember where the hamster was when a training photo was taken (dataset/zones.jsonl, one line per photo),
    so the photo manager can filter by place, and whether anything moved in the cage right then ("m": 1 / 0 -
    plain motion detection, not the AI: it can't be judged again later from a single photo)."""
    if not zone and moved is None:
        return
    rec = {"n": Path(photo).name}
    if zone:
        rec["z"] = zone
    if moved is not None:
        rec["m"] = int(bool(moved))
    try:
        with open(dataset / "zones.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps(rec) + "\n")
    except OSError:
        pass


def file_model_id(model_dir):
    """Same id as score_photos.py / web.py (yolo_detector.model_fingerprint)."""
    from yolo_detector import model_fingerprint
    try:
        return model_fingerprint(model_dir)
    except OSError:
        return None


def score_photo(yolo, mid, frame, dataset, photo):
    """The AI's verdict on THIS photo (not the live picture a moment before), right when it is saved:
    dataset/ai_live.jsonl {n, model, s: [conf, x, y, w, h (+ head conf, hx, hy, hw, hh)]}. Installing a new model
    makes score_photos.py score every photo again, replacing these."""
    if yolo is None or not mid:
        return
    from yolo_detector import hamster_and_head
    try:
        b, hd = hamster_and_head(yolo.detect(frame))
        s = [round(float(b[4]), 3), *[round(float(v), 4) for v in b[:4]]] if b else [0.0]
        if b and hd:
            s += [round(float(hd[4]), 3), *[round(float(v), 4) for v in hd[:4]]]
        with open(dataset / "ai_live.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps({"n": Path(photo).name, "model": mid, "s": s}) + "\n")
    except Exception as e:   # noqa: BLE001 - a missing score is only shown as "評分中"
        log.warning("photo scoring failed: %s", e)


def box_motion(g0, g1, b, W, H, margin=0.15, level=14):
    """Share (0..1) of pixels inside box b (normalized x, y, w, h, enlarged by margin) that changed by more than
    `level` grey levels between two blurred grey frames. ~0 = lying still, a few % = head/paws moving."""
    x, y, w, h = b[:4]
    x0, y0 = max(0, int((x - w * margin) * W)), max(0, int((y - h * margin) * H))
    x1, y1 = min(W, int((x + w * (1 + margin)) * W)), min(H, int((y + h * (1 + margin)) * H))
    if x1 - x0 < 5 or y1 - y0 < 5:
        return None
    return float((cv2.absdiff(g0[y0:y1, x0:x1], g1[y0:y1, x0:x1]) > level).mean())


def drink_spot(spout, W, H, custom=None, frac=0.09):
    """Where a hamster stands to drink: by default a circle just below the spout (it stands up under the bottle
    with its mouth on the spout); the editor can move and resize it ([tracker] drink_spot = [cx, cy, r], picture
    fractions, r of the width). Returns (cx, cy, r) in pixels, or None without a marked spout."""
    if spout is None:
        return None
    if isinstance(custom, list) and len(custom) == 3:
        return custom[0] * W, custom[1] * H, custom[2] * W
    r = frac * W
    return spout[0], spout[1] + r * 0.6, r


def zone_of_box(zones, b, W, H, spots):
    """Zone of a hamster box. A bottle's area is the spot under its marked spout (where it stands to drink;
    spots = {zone id: (cx, cy, r)}), but only where no other zone was drawn: a sand bath right under the spout
    stays the bath, so rolling in the sand is never taken for drinking."""
    cx, cy = (b[0] + b[2] / 2) * W, (b[1] + b[3] / 2) * H
    z = zone_at(zones, cx, cy)
    if z is None:
        for k, (sx, sy, r) in (spots or {}).items():
            if (cx - sx) ** 2 + (cy - sy) ** 2 <= r * r:
                return k
    return z


def wheel_turns(buf, now, strips):
    """(turns per second, clearly turning?) from the last ~2 s of brightness of a few vertical lines across the
    wheel area (buf = [(time, values)]). Each strip going by brightens the lines once, so the lines flicker at
    "strips going by per second". At night the exposure is long and a fast strip is smeared into a faint band, so
    instead of counting flashes (that found 0 turns, 11/15) this takes the strongest flicker frequency: a flicked
    wheel recorded 11/15 showed 1.9 -> 1.0 -> 0.4 Hz as it slowed down, ~11-13 turns of the 15 the owner counted
    (some turned before the recording began). Still lines (no flicker, or just noise) give 0."""
    if len(buf) < 20:
        return 0.0, False
    ts = np.array([t for t, _ in buf])
    seg = np.stack([v for _, v in buf])
    keep = ts >= now - 2.0
    ts, seg = ts[keep], seg[keep]
    if len(ts) < 16 or ts[-1] - ts[0] < 1.5 or seg.shape[1] == 0:
        return 0.0, False
    fps = (len(ts) - 1) / (ts[-1] - ts[0])
    # the whole picture getting brighter / darker (dawn, a lamp, the camera adjusting its exposure) changes every
    # line together: remove that common part, and any slow drift, so only the strips' own flicker is left
    # (11/15 06:38: at dawn the drift read as 0.2-0.5 turns/s with the wheel standing still)
    if seg.shape[1] >= 3:
        seg = seg - seg.mean(1, keepdims=True)
    tt = ts - ts.mean()
    slope = (tt[:, None] * (seg - seg.mean(0))).sum(0) / max(1e-9, float((tt ** 2).sum()))
    seg = seg - seg.mean(0) - tt[:, None] * slope
    amp = float(np.sqrt(np.mean(seg ** 2)))
    sp = np.abs(np.fft.rfft(seg * np.hanning(len(seg))[:, None], n=256, axis=0)) ** 2
    fr = np.fft.rfftfreq(256, 1 / fps)
    # up to 5.5 strips a second (2 strips: 2.75 turns/s, ~1.8 m/s on a 21 cm wheel, faster than a hamster runs);
    # higher is too close to the camera's 15 frames/s to trust (a stopping wheel once read 6.8/s there)
    band = (fr >= 0.4) & (fr <= fps / 2 - 0.3)          # noise level: the whole usable range
    if not band.any():
        return 0.0, False
    p, f = sp[band].sum(1), fr[band]
    ok = f <= min(5.5, fps / 2 - 0.5)                    # where the strips' frequency may be
    k = int(np.flatnonzero(ok)[p[ok].argmax()])
    snr = float(p[k] / (np.median(p) + 1e-9))
    if amp < 0.5 or snr < 3.5:   # standing still: the lines only show noise (amp ~0.1)
        return 0.0, False
    return float(f[k]) / strips, True


def lick_frame(cur, prev, sx, sy, x0, y0, r):
    """One frame of the licking check. cur / prev: grey crops (float32) of the area within 2r of the spout
    (top-left corner at x0, y0 in the picture), ~0.1 s apart. Returns (flick, near, out):
      flick - the square at the spout changed (tongue, steel ball, drops). The size of change that counts
              follows the light: 18 grey levels in daylight, down to 6 in the dark, and 4 (on 2 % of the
              pixels instead of 3 %) when the spout is almost black (a dark room AND a dark head over it)
      near  - share of pixels that changed right at the tip (within r/2)
      out   - share that changed further out (r..2r)
    Licking changes things AT the tip; a hamster just sitting / breathing / sniffing next to the spout changes
    the surroundings as much or more. Measured on the owner-labelled clips of 11/15."""
    yy, xx = np.mgrid[y0:y0 + cur.shape[0], x0:x0 + cur.shape[1]]
    dist = np.hypot(xx - sx, yy - sy)
    inner = dist < r
    cm = float(cur[inner].mean()) if inner.any() else 0.0
    almost_black = cm < 40
    m = np.abs(cur - prev) > (max(4.0, cm * 0.14) if almost_black else min(18.0, max(6.0, cm * 0.14)))
    flick = bool(inner.any() and m[inner].mean() >= (0.02 if almost_black else 0.03))
    near, far = dist < r / 2, dist >= r
    return flick, float(m[near].mean()) if near.any() else 0.0, float(m[far].mean()) if far.any() else 0.0


def cleaning_until(ep, now, clean_until, gap=120.0, hold=60.0, quiet=10.0):
    """A hand was noticed at `now`. ep = [start, last] of the current hand moment (a pause over `quiet` s starts a
    new one), updated in place. A new moment within `gap` s of the last = cleaning: until `hold` s after now; while
    cleaning, every hand pushes the end on. -> the new end of cleaning (in the past = not cleaning).
    Checked on data/hand.log 11/17-11/23 (gap 120, hold 60): 71 cleaning stretches, ~35 min a day; of the events
    recorded then, 26 fall inside: the 11/22 06:01-06:15 runs and 13:01-13:08 drinks / eats while cleaning, and two
    drinks at 11/22 01:51 where the hamster itself dashing past was taken for a hand."""
    if now - ep[1] > quiet:
        if now - ep[1] <= gap:
            clean_until = now + hold
        ep[0] = now
    elif now < clean_until:
        clean_until = now + hold
    ep[1] = now
    return clean_until


def hand_in_cage(blobs, ham_box, free=0.15, on_ham=0.3):
    """A big moving thing in the cage that is not the hamster: a hand / arm (maybe holding the hamster), or the
    whole picture changing with the light. The wheel's own area is already left out of the blobs (MotionDetector).
    Live 11/17: the hamster's own moving blob (dilated) reached 0.16 of the picture, always on the AI's box; a hand
    in the dark 0.17-0.23 with the hamster not seen; a light change 0.48-0.50. So a blob on the box the AI sees
    right now must be really big (on_ham) to be a hand holding it, anywhere else free is enough."""
    for a, b in blobs:
        if a >= on_ham or (a >= free and not (ham_box is not None and boxes_touch(b, ham_box))):
            return True
    return False


def spin_push_like(s, max_len=15):
    """The wheel's movement, second by second, looks like one push coasting to a stop (a hand, the nest or bedding
    knocking it): strongest at the very start, then (almost) only falling, ending low. A hamster running keeps it
    going, up and down. Longer than max_len seconds is never one push."""
    if len(s) < 3 or len(s) > max_len:
        return False
    pk = max(range(len(s)), key=lambda i: s[i])
    if pk > 1 or s[pk] <= 0:
        return False
    tail = s[pk:]
    falls = sum(1 for a, b in zip(tail, tail[1:]) if b <= a * 1.1)
    return falls >= 0.75 * (len(tail) - 1) and tail[-1] < 0.5 * s[pk]


def judge_event(kind, n, ev):
    """Keep a finished candidate event? -> (keep, reason). ev = what was collected second by second.
    run:   the hamster shown to be at the wheel: the AI sees it in the wheel during the turning, or it was last seen at
           the wheel a moment before (the camera sees only the wheel's bottom edge, and turning alone is also a hand,
           cleaning, or the wheel rocking). Not one push coasting down; not the hamster seen elsewhere (something
           else turned it)
    drink: the mouth / padded body box on the spout itself for at least half of it (not just near the circle:
           sitting in the sand bath next to the spout, or walking past it)"""
    if kind == "run":
        if not ev.get("ai_wheel") and not ev.get("near_wheel", 1):
            # 11/22 05:15 cleaning: 3 "runs" with the hamster unseen for 4-5 min and nowhere near the wheel. Every run
            # the owner labelled in 跑分 had the AI seeing it in the wheel. (old events have no near_wheel: kept as were)
            return False, "not_at_wheel"
        if spin_push_like(ev.get("spin", [])) and not ev.get("ai_wheel"):
            return False, "push"
        if ev.get("else", 0) >= max(2, n):
            return False, "seen_elsewhere"
        if ev.get("entry_far") and not ev.get("ai_wheel"):
            return False, "came_from_far"
        return True, "ok"
    if kind == "drink":
        on = ev.get("cov", 0) + ev.get("head", 0) - ev.get("both", 0)
        if on < max(2, 0.5 * n):
            return False, "not_on_spout"
        return True, "ok"
    return True, "ok"


def licking_second(hits, near_sum, out_sum, n, need):
    """A second counts as licking: enough flicker frames, and the change concentrated at the tip."""
    if not n:
        return False
    near, out = near_sum / n, out_sum / n
    return hits >= need and near >= 0.03 and near >= 0.9 * out


def bottle_of_box(b, W, H, spots, spouts, pad=0.15):
    """Which bottle the hamster is at, judged by the spout first: its box (a little bigger) covers that bottle's
    spout, or the box touches the bottle's circle at all. Not the box's centre: a hamster drinking from the side
    is long, its centre can be a body length away from the spout (and the box gets cut at the picture's edge)."""
    if not b:
        return None
    x0, y0 = (b[0] - b[2] * pad) * W, (b[1] - b[3] * pad) * H
    x1, y1 = (b[0] + b[2] * (1 + pad)) * W, (b[1] + b[3] * (1 + pad)) * H
    for k, (sx, sy) in (spouts or {}).items():
        if x0 <= sx <= x1 and y0 <= sy <= y1:
            return k
    for k, (cx, cy, r) in (spots or {}).items():
        nx, ny = min(max(cx, x0), x1), min(max(cy, y0), y1)   # the box's point nearest the circle's centre
        if (nx - cx) ** 2 + (ny - cy) ** 2 <= r * r:
            return k
    return None


def zone_at(zones, px, py):
    for kind, poly in zones:
        if cv2.pointPolygonTest(poly, (float(px), float(py)), False) >= 0:
            return kind
    return None


# layers of the dashboard picture the owner can switch off (首頁 › 攝影機 › 顯示); saved by web.py in data/overlay.json
# plus "zones_off": the areas (by id) not to draw - a drink area's spout square and circle go with it
OVERLAY_LAYERS = ("box", "head", "roi", "floor", "time")


def read_overlay(path):
    """{layer: shown?, "zones_off": set} from overlay.json (missing / broken = everything shown)."""
    try:
        d = json.loads(Path(path).read_text())
        out = {k: bool(d.get(k, True)) for k in OVERLAY_LAYERS}
        out["zones_off"] = {str(z) for z in (d.get("zones_off") or [])}
        return out
    except (OSError, ValueError, AttributeError, TypeError):
        return {**dict.fromkeys(OVERLAY_LAYERS, True), "zones_off": set()}


def annotate(frame, W, H, roi, floor_corners, box, pos_img, moving, paused=False, zones=(), ai_box=None, ai_conf=None,
             spouts=(), spout_r=0, head=None, spot=None, show=None, head_conf=None):
    view = frame.copy()
    show = show or {**dict.fromkeys(OVERLAY_LAYERS, True), "zones_off": set()}
    if not show["head"]:
        head = None
    if not show["box"]:
        ai_box = box = pos_img = None
    off = show.get("zones_off") or set()
    if off:
        zones = [(k, p) for k, p in zones if k not in off]
        spouts = [(k, s) for k, s in spouts if k not in off]
        spot = {k: s for k, s in (spot or {}).items() if k not in off}
    if not show["roi"]:
        roi = ()
    if not show["floor"]:
        floor_corners = None
    if head is not None:   # the head (head-trained model)
        x, y, bw, bh = int(head[0] * W), int(head[1] * H), int(head[2] * W), int(head[3] * H)
        cv2.rectangle(view, (x, y), (x + bw, y + bh), (216, 79, 255), 1, cv2.LINE_AA)
        hl = f"head {round(head_conf * 100)}%" if head_conf is not None else "head"
        cv2.putText(view, hl, (x + 2, y + bh + 12 if y + bh + 14 < H else max(10, y - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.38,
                    (216, 79, 255), 1, cv2.LINE_AA)   # under the head box: the body's label sits above
    for kind, (sx, sy) in spouts:   # the squares watched for licking (one per bottle)
        sx, sy = int(sx), int(sy)
        cv2.rectangle(view, (sx - spout_r, sy - spout_r), (sx + spout_r, sy + spout_r), ZONE_COLORS.get(kind, (255, 160, 40)), 1)
    for kind, (cx, cy, r) in (spot or {}).items():   # each bottle's drinking spot (counts as that bottle's area)
        cv2.circle(view, (int(cx), int(cy)), int(r), ZONE_COLORS.get(kind, (255, 160, 40)), 1, cv2.LINE_AA)
    if ai_box is not None:
        # what the AI sees right now, with how sure it is (shown even when the hamster sits still)
        x, y, bw, bh = int(ai_box[0] * W), int(ai_box[1] * H), int(ai_box[2] * W), int(ai_box[3] * H)
        cv2.rectangle(view, (x, y), (x + bw, y + bh), (88, 209, 48), 1, cv2.LINE_AA)
        label = f"hamster {round((ai_conf or 0) * 100)}%"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
        ty = y - 5 if y - th - 8 > 0 else y + bh + th + 5
        cv2.rectangle(view, (x, ty - th - 3), (x + tw + 6, ty + 3), (88, 209, 48), -1)
        cv2.putText(view, label, (x + 3, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1, cv2.LINE_AA)
        box = None if box is not None and tuple(box) == tuple(ai_box) else box   # don't draw it twice
    for kind, poly in zones:
        cv2.polylines(view, [np.int32(poly)], True, ZONE_COLORS[kind], 1)
        cv2.putText(view, kind, tuple(np.int32(poly[0]) + (3, 14)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, ZONE_COLORS[kind], 1)
    if len(roi) >= 3:
        cv2.polylines(view, [np.int32([[x * W, y * H] for x, y in roi])], True, (255, 200, 0), 1)
    if floor_corners and len(floor_corners) == 4:
        cv2.polylines(view, [np.int32([[x * W, y * H] for x, y in floor_corners])], True, (200, 0, 255), 1)
    if box and moving:
        x, y, bw, bh = int(box[0] * W), int(box[1] * H), int(box[2] * W), int(box[3] * H)
        cv2.rectangle(view, (x, y), (x + bw, y + bh), (88, 209, 48), 1, cv2.LINE_AA)
    if pos_img is not None:
        cv2.circle(view, (int(pos_img[0] * W), int(pos_img[1] * H)), 6,
                   (0, 255, 0) if moving else (0, 200, 255), -1)
    if show["time"]:
        cv2.putText(view, datetime.now().strftime("%H:%M:%S") + (" PAUSED" if paused else " MOVING" if moving else ""),
                    (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    return view


def pause_until():
    try:
        return float(json.loads(PAUSE_FILE.read_text()).get("until", 0))
    except (OSError, ValueError, AttributeError):
        return 0.0


def live_requested():
    try:
        return time.time() - LIVE_REQUEST.stat().st_mtime < 5
    except OSError:
        return False


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    cfg = load_config()
    t_cfg = cfg["tracker"]
    W, H = t_cfg["width"], t_cfg["height"]
    roi = t_cfg.get("roi") or []
    roi_px = np.float32([[x * W, y * H] for x, y in roi]) if len(roi) >= 3 else None
    floor_corners = t_cfg.get("floor") or []
    data = BASE / "data"
    dataset = data / "dataset"
    dataset.mkdir(parents=True, exist_ok=True)
    LIVE_DIR.mkdir(parents=True, exist_ok=True)

    rot = cfg["camera"]["rotation"]
    transform = Transform(hflip=1, vflip=1) if rot == 180 else Transform()
    cam = Picamera2()
    cam.configure(cam.create_video_configuration(
        main={"size": (W, H), "format": "RGB888"},  # RGB888 in picamera2 = BGR byte order, what OpenCV wants
        # full-sensor binned mode; the small 640x480 sensor mode is a centre crop and loses most of the cage
        sensor={"output_size": (1640, 1232)},
        transform=transform,
        controls={"FrameRate": t_cfg["fps"]}))
    cam.start()

    detector = MotionDetector(t_cfg)
    mode, yolo = load_yolo(t_cfg)
    yolo_mid = file_model_id(BASE / t_cfg.get("yolo_model", "models/hamster_model")) if yolo else None
    t_mbox = -1e9   # last time anything moved in the cage (plain motion detection)
    if yolo:
        from yolo_detector import hamster_and_head
    # head (models trained with "head" boxes, class 1): where its mouth is, for drinking / eating
    y_head, y_head_t, y_head_conf = None, -1e9, 0.0
    head_th = t_cfg.get("head_conf", 0.35)
    use_motion = mode in ("motion", "hybrid")
    y_interval = t_cfg.get("yolo_interval_s", 0.5)
    t_yolo = y_seen = -1e9
    y_box, y_conf, y_prev_c = None, 0.0, None
    follow_box = None                                   # where the hamster was last seen (AI), followed while the AI blinks
    # "lying still in the open": the AI keeps seeing the hamster in the same spot (outside the nest).
    # alerts.py warns when this lasts long (possible illness, injury, or torpor when it's cold).
    still_anchor, still_since = None, None
    beh = None                                          # current eating/drinking event: [kind, start, last mono, seconds]
    beh_still_cm = t_cfg.get("behavior_still_cm", 2.0)  # moved less than this in a second = staying put
    # short interruptions don't split one event: a meal / a drink has pauses (chewing, looking around), so they get
    # as long a gap as a zone visit (otherwise one visit to the bowl became 3-4 "meals"); running stops are short
    beh_gap = {"eat": t_cfg.get("eat_gap_s", 8), "drink": t_cfg.get("drink_gap_s", 8), "run": t_cfg.get("behavior_gap_s", 3)}
    beh_min_s = {"eat": t_cfg.get("eat_min_s", 5), "drink": t_cfg.get("drink_min_s", 3), "run": t_cfg.get("run_min_s", 3)}

    def beh_need(b):   # seconds an event needs to be kept; a run with only weak turns needs longer (see wheel_weak_min_s)
        return beh_min_s[b[0]] if b[0] != "run" or b[5] else max(beh_min_s["run"], wheel_weak_min_s)

    # A hand in the cage (cleaning, picking the hamster up, putting it back): the wheel gets knocked, the hamster
    # is carried over the bowl. Nothing is counted while a hand is there and for hand_hold_s after.
    hand_until = -1e9
    hand_spin_after_s = t_cfg.get("hand_spin_after_s", 20)   # a pushed wheel keeps turning: not the hamster for this long
    wheel_near_s = t_cfg.get("wheel_near_s", 30)   # a run starting this soon after it was seen at the wheel counts unseen
    hand_hold_s = t_cfg.get("hand_hold_s", 3)
    hand_free = t_cfg.get("hand_area", 0.15)          # share of the picture (see hand_in_cage)
    hand_on_ham = t_cfg.get("hand_area_on_hamster", 0.3)
    last_push_end = -1e9   # when the last run dropped as a push ended: the wheel coasting on after it is that push too
    # Second hand check, not fooled by a close / still hand: the share of the cage (wheel left out) that changed in
    # the last ~0.25 s. The background model soaks a big hand up within a second or two (11/17 19:57: hand over half
    # the picture, blob ~0); this does not. Measured 11/17: a hand 0.07-0.12, the hamster eating / drinking /
    # running / walking at most 0.05.
    hand_diff = t_cfg.get("hand_change_share", 0.06)
    hd_prev, hd_t, hd_keep, hand_diff_s = None, -1e9, None, 0.0
    hand_max_s = 0.0   # biggest moving blob this second (outside the wheel), written to data/hand.log for checking
    # Cleaning: the hand comes and goes, and is only noticed now and then (a still hand fades into the background).
    # 11/22 06:01-06:15: 11 "runs" were counted in the gaps between the moments it was noticed (the hand moving the
    # nest / the wheel). A second hand moment within clean_gap_s of the last one = cleaning: nothing counts until
    # clean_hold_s after the last hand (標註標準: hands in the cage = no label). See cleaning_until().
    clean_gap_s = t_cfg.get("clean_gap_s", 120)
    clean_hold_s = t_cfg.get("clean_hold_s", 60)
    hand_ep, clean_until = [-1e9, -1e9], -1e9

    def finish_beh(b):
        """A candidate event ended: keep it (judge_event) or put it with the dropped ones. -> kept?"""
        nonlocal last_push_end
        if b[3] < beh_need(b):
            return False
        ok, why = judge_event(b[0], b[3], b[6])
        if ok and b[0] == "run" and not b[5] and not b[6].get("ai_wheel") and \
                b[6].get("t0", 0) - max(last_push_end, hand_until) <= 8:
            ok, why = False, "after_push"   # weak turning right after a dropped push / a hand: the wheel still rolling
        elif ok and b[0] == "run" and not b[6].get("ai_wheel") and b[6].get("t0", 0) - hand_until <= 10:
            # the wheel started turning within 10 s of a hand: turned by that hand (often right above the picture,
            # 11/17 20:07: 39 s with the hamster nowhere in view), unless the AI really sees the hamster in the wheel
            ok, why = False, "after_hand"
        if not ok and why in ("push", "after_push"):
            last_push_end = b[2]
        ev = {k: v for k, v in b[6].items() if k != "t0"}
        ev["spin"] = [round(v, 3) for v in ev.get("spin", [])][:60]
        if ok:
            store.behavior(b[1], b[0], b[3], b[4] / b[3], ev)
        else:
            store.behavior_dropped(b[1], b[0], b[3], b[4] / b[3], why, ev)
            log.info("%s %ds at %s not counted (%s)", b[0], b[3], b[1].strftime("%H:%M:%S"), why)
        return ok
    # eating = at the bowl AND the body itself moves (head, paws, chewing, stuffing cheeks). Measured clips: a
    # hamster resting at the bowl changes ~0 % of the pixels in its box from one second to the next, eating ~2 %+.
    eat_motion_min = t_cfg.get("eat_motion_min", 0.01)
    g_prev_1hz = None                                   # last 1 Hz grey frame (body movement inside the AI box)
    # drinking: the spout of the water bottle (marked on the dashboard). Licking barely moves the body, so instead
    # a small square around the spout is compared frame by frame: the tongue, the steel ball and water drops make
    # it flicker many times a second while the hamster's head is there.
    spout = t_cfg.get("spout")
    spout = (spout[0] * W, spout[1] * H) if isinstance(spout, list) and len(spout) == 2 else None
    if "water" not in zone_ids(t_cfg):   # 水瓶 deleted from the area list: no water area / drinking estimate
        spout = None
    spout_r = max(6, int(t_cfg.get("spout_radius", 0.03) * W))
    # each bottle's area: a circle under its spout (moved / resized on the dashboard). {zone id: (cx, cy, r) px}
    spot = {"water": drink_spot(spout, W, H, t_cfg.get("drink_spot"))} if spout is not None else {}
    drink_flicker_min = t_cfg.get("drink_flicker_frames", 3)   # changed frames per second (of ~15) to count as licking
    # every bottle: the built-in 水瓶 + added areas of type 喝水, each with its own spout and circle
    # ([tracker] zones = [{kind = "c…", poly = [], spout = [x, y], spot = [cx, cy, r]}])
    types = zone_types(t_cfg)
    spouts = {"water": spout} if spout is not None else {}
    for z in t_cfg.get("zones") or []:
        s = z.get("spout")
        if types.get(z.get("kind")) == "drink" and z.get("kind") != "water" and isinstance(s, list) and len(s) == 2:
            spouts[z["kind"]] = (s[0] * W, s[1] * H)
            spot[z["kind"]] = drink_spot(spouts[z["kind"]], W, H, z.get("spot"))
    spout_prev, spout_hits = {}, {k: 0 for k in spouts}
    lick_sum = {k: [0.0, 0.0, 0] for k in spouts}   # per bottle, this second: near-tip share, further-out share, frames
    # running: decided by the wheel turning (white strips on it, see wheel_spin_min below), not by the hamster's
    # body shape (that version once counted grooming on the wheel as running)
    still_move_cm = t_cfg.get("still_move_cm", 3)       # moving less than this doesn't count as moving
    still_conf = t_cfg.get("still_conf", 0.55)          # only trust AI boxes this sure (not a look-alike object)
    new_ai_min = lambda: {"runs": 0, "seen": 0, "unsure": 0, "conf_sum": 0.0, "conf_max": 0.0}  # noqa: E731
    ai_min = new_ai_min()
    # light: every 5 s the camera's metadata (its lux estimate, exposure time, gain) + the picture's brightness/noise
    new_light = lambda: {"n": 0, "lux": 0.0, "exp": 0.0, "gain": 0.0, "bright": 0.0, "noise": 0.0}   # noqa: E731
    light, light_now, t_light = new_light(), None, -1e9
    ai_grace_s = t_cfg.get("ai_grace_s", 8)             # how long nearby movement still counts after the AI lost it
    ai_follow_margin = t_cfg.get("ai_follow_margin", 0.06)
    ai_follow_max = t_cfg.get("ai_follow_max_ratio", 2.5)   # a moving blob bigger than this x the hamster = not it (a hand)
    y_area = None   # size of the box the AI last saw the hamster in
    # Light switched on/off: for a second or two the camera's exposure is wrong and re-adjusting, the picture is
    # dark/grainy and the tunnel shadow looks like a hamster to the AI (11/16). A sudden jump in the whole picture's
    # brightness starts a short "settling" time in which AI sightings and motion are ignored.
    bright_ema, settle_until = None, -1e9
    light_jump = t_cfg.get("light_jump_ratio", 0.35)
    light_settle_s = t_cfg.get("light_settle_s", 4.0)
    # A hamster that was not seen for a while must be seen twice (overlapping, within a few seconds) before it
    # counts: one odd frame (a shadow, a reflection) then no longer starts a visit.
    y_pend = None   # (box, time) of a first sighting waiting for its confirmation
    y_confirm_after_s = t_cfg.get("ai_confirm_after_s", 10)
    y_th = t_cfg.get("yolo_conf", 0.45)                 # boxes at/above this are the hamster (hard_conf_low: see load_yolo)
    hard_interval = t_cfg.get("hard_interval_s", 60)
    hard_max = t_cfg.get("hard_max", 1000)
    t_hard = -1e9
    prelabel = t_cfg.get("prelabel", True)
    zones = load_zones(t_cfg, W, H)
    # wheel spinning: most of the wheel is above the picture, but its lowest part shows at the top edge, with white
    # strips stuck on it (11/15). When it turns the strips sweep through there: a big share of that area changes
    # from frame to frame. Spinning while the hamster isn't seen anywhere else = it is running in the wheel.
    run_zones = [(k, poly) for k, poly in zones if types.get(k) == "run"]
    # share of the wheel area that changes per frame. Measured 11/15 turning it by hand: 0.10-0.20 turning fast,
    # 0.02-0.09 turning slowly, 0.00-0.01 standing still
    wheel_spin_min = t_cfg.get("wheel_spin_share", 0.02)
    wheel_flick_min = t_cfg.get("wheel_flick_share", 0.005)   # the flicker only counts if the area changes this much
    wmask = None
    if run_zones:
        allp = np.concatenate([p for _, p in run_zones])
        wx0, wy0 = max(0, int(allp[:, 0].min())), max(0, int(allp[:, 1].min()))
        wx1, wy1 = min(W, int(allp[:, 0].max()) + 1), min(H, int(allp[:, 1].max()) + 1)
        m = np.zeros((wy1 - wy0, wx1 - wx0), np.uint8)
        for _, p in run_zones:
            cv2.fillPoly(m, [np.int32(p - [wx0, wy0])], 1)
        wmask = m.astype(bool) if m.any() else None
    spin_zone = run_zones[0][0] if run_zones else None
    if run_zones:
        _s = t_cfg["process_scale"]
        _sk = np.zeros((int(round(H * _s)), int(round(W * _s))), np.uint8)
        for _, _wp in run_zones:
            cv2.fillPoly(_sk, [np.int32(_wp * _s)], 1)
        detector.blob_skip = cv2.dilate(_sk, np.ones((5, 5), np.uint8)).astype(bool)
    _hk = np.zeros((H // 4, W // 4), np.uint8)
    if roi_px is not None:
        cv2.fillPoly(_hk, [np.int32(roi_px / 4)], 1)
    else:
        _hk[:] = 1
    for _, _wp in run_zones:
        cv2.fillPoly(_hk, [np.int32(_wp / 4)], 0)
    hd_keep = _hk.astype(bool) if _hk.any() else None
    wheel_hist, wheel_sum, wheel_spin_now, spin_streak = [], [0.0, 0], 0.0, 0
    # Weak vs strong turning. A hamster really in the wheel moves a lot of the wheel area at least now and then
    # (22:26 on 11/15: 8-39 %); a fast dash of a few seconds is strong too. Bedding or the nest moving makes only a
    # WEAK turn (strips "flicker", < wheel_strong_share of the area changes), and only for a moment (11/15 17:52 /
    # 17:58: 3 s). So a run with no strong second must last wheel_weak_min_s to count; the decision is made on the
    # whole run (start time, length and distance all kept), not on where the hamster was seen last - it can go from
    # the wheel to the bath in seconds.
    wheel_strong = t_cfg.get("wheel_strong_share", 0.04)
    wheel_weak_min_s = t_cfg.get("wheel_weak_min_s", 4)   # the false ones (11/15) were all 3 s
    spinning, spin_sure, strong_now, strong_until = False, False, False, -1e9
    spin_ham = spin_sure_ham = False   # the same, but not right after a hand (a pushed wheel)
    wheel_box = ((wx0 / W, wy0 / H, (wx1 - wx0) / W, (wy1 - wy0) / H) if run_zones else None)
    # distance: each strip passing a fixed spot = 1/strips of a turn; a turn = pi x diameter (set on the dashboard).
    # At 15 frames a second up to ~7 strips a second can be counted (faster ones are missed: fewer strips = higher
    # top speed). Sampled on a grid of spots in the wheel area; the spot that flickers most is used.
    wheel_d_cm = float(t_cfg.get("wheel_diameter_cm", 21))
    wheel_strips = max(1, int(t_cfg.get("wheel_strips", 2)))
    wbuf, wheel_revs_min, wheel_rps, wrec, run_revs = [], 0.0, 0.0, None, 0.0
    t_overlay, overlay = -1e9, None
    if wmask is not None:
        # 9 vertical lines across the wheel area (inner 15-85 % of its width), each averaged over the wheel's rows
        _cols = np.linspace(0.15, 0.85, 9) * wmask.shape[1]
        wlines = [(int(c), wmask[:, int(c)]) for c in _cols if wmask[:, int(c)].sum() >= 6]
    # zones cut off by the picture's edge (part of the sand bath / wheel is outside the camera's view) work like the
    # nest, with two safeguards learned the hard way (it once ran in the out-of-view wheel for 25 minutes and all of
    # it was counted as "bath", the last place it had been seen):
    #   - only when it disappeared NEAR THE PICTURE'S EDGE (vanishing in the middle = the AI just lost it)
    #   - at most offscreen_max_s per zone: out-of-view parts connect, so after a while we really don't know
    edge = t_cfg.get("zone_edge_px", 8)
    edge_zones = {k for k, poly in zones if k != "nest" and (poly[:, 0].min() <= edge or poly[:, 0].max() >= W - 1 - edge or
                                                             poly[:, 1].min() <= edge or poly[:, 1].max() >= H - 1 - edge)}
    offscreen_max_s = {"bath": 300, "food": 600, "wheel": float("inf"), **(t_cfg.get("offscreen_max_s") or {})}
    edge_frac = t_cfg.get("offscreen_edge", 0.08)   # "near the edge" = its box within 8 % of the picture's border
    near_edge = lambda b: b is not None and (b[0] <= edge_frac or b[1] <= edge_frac or   # noqa: E731
                                             b[0] + b[2] >= 1 - edge_frac or b[1] + b[3] >= 1 - edge_frac)
    left_at_edge = False   # did it disappear near the picture's edge?
    if edge_zones:
        log.info("zones cut by the picture edge (still counted while the hamster is out of view there): %s", sorted(edge_zones))
    visit_gap_s = t_cfg.get("visit_gap_s", 8)           # short gaps don't split a visit
    min_visit_s = t_cfg.get("min_visit_s", 3)
    last_seen_t, last_seen_zone, in_nest = -1e9, None, False
    # After a restart (reboot, calibration saved, update) a hamster asleep in the nest is invisible, so nothing
    # would be counted until it came out and went back in. Carry the nest state over from the last status file.
    try:
        prev = json.loads((data / "tracker.json").read_text())
        prev_age = time.time() - datetime.fromisoformat(prev["updated"]).timestamp()
        if prev.get("zone") == "nest" and prev_age < 600 and any(kind == "nest" for kind, _ in zones):
            in_nest, last_seen_zone = True, "nest"
            log.info("hamster was in the nest before the restart: still counting it as in the nest")
        elif prev.get("zone") in edge_zones and prev_age < 60:   # same for the hidden part of the bath / bowl / wheel
            last_seen_zone, left_at_edge = prev["zone"], True
            last_seen_t = time.monotonic() - prev_age   # its time limit keeps running across the restart
            log.info("hamster was in %s before the restart: still counting it there until it is seen", last_seen_zone)
    except (OSError, ValueError, KeyError, TypeError):
        pass
    zone_secs = {}
    visit = None                                        # [zone, start datetime, last monotonic, seconds]
    zone_now = None
    # the bottles' circles are counted on their own, next to the drawn areas: standing in the sand bath right under
    # the bottle is a stay in the bath AND at the bottle (two separate records that don't affect each other)
    bottle_now = last_seen_bottle = None
    bvisit = None                                       # [bottle zone, start datetime, last monotonic, seconds]
    floor = FloorMap(floor_corners, t_cfg.get("floor_sides_cm"), (t_cfg["cage_w_cm"], t_cfg["cage_h_cm"]))
    store = Store(BASE / cfg["log"]["db"])
    rec = ClipRecorder(cfg, BASE, W, H)                 # video clips while the hamster moves
    heat = np.zeros((H // 10, W // 10), np.float32)  # occupancy grid in floor coordinates

    stop = False

    def handle_stop(*_):
        nonlocal stop
        stop = True

    signal.signal(signal.SIGTERM, handle_stop)
    signal.signal(signal.SIGINT, handle_stop)

    pos = None              # last known position on the cage floor (0..1)
    box = None              # last detection box (normalized image coords)
    last_motion = 0.0
    minute_key = datetime.now().strftime("%Y-%m-%dT%H:%M")
    minute_active = 0
    minute_dist = 0.0
    total_saved = len(list(dataset.glob("motion_*.jpg"))) + len(list(dataset.glob("idle_*.jpg")))
    hard_saved = len(list(dataset.glob("hard_*.jpg")))
    t_sample = t_snap = t_heat = t_fps = time.monotonic()
    t_prune = -1e9  # prune once at start, then hourly
    keep_days = t_cfg.get("positions_keep_days", 30)
    t_live = 0.0
    t_raw = -1e9
    t_req, manual_until = -1e9, -1e9   # 拍照／錄影 requests from the page (checked 4x a second)
    paused_till = pause_until()
    paused = time.time() < paused_till
    last_sample_pos, t_last_sample = None, time.monotonic()
    t_ds_motion = t_ds_idle = -1e9  # allow the first dataset frame right away
    live_interval = 1.0 / t_cfg.get("stream_fps", 10)
    frames = 0
    fps = 0.0
    cw, ch = floor.w_cm, floor.h_cm   # real size of the floor coordinates (from the measured sides when given)
    log.info("started %dx%d @%dfps, detector=%s, roi=%s, floor calibrated=%s, dataset has %d images",
             W, H, t_cfg["fps"], mode, bool(roi), floor.calibrated, total_saved)

    # performance for 跑分: per-frame processing time (from a frame arriving to being done with it, not counting the
    # wait for the camera), AI inference time, and how many AI analyses really happen per second
    perf_proc, perf_ai, perf_ai_t, perf_t0 = deque(maxlen=600), deque(maxlen=200), deque(maxlen=400), None
    perf_proc_ai, y_ran = deque(maxlen=200), False   # full processing of the frames the AI looked at
    try:
        while not stop:
            if perf_t0 is not None:
                (perf_proc_ai if y_ran else perf_proc).append((time.perf_counter() - perf_t0) * 1000)
            if time.monotonic() - t_light >= 5:
                req = cam.capture_request()
                try:
                    frame, md = req.make_array("main"), req.get_metadata()
                finally:
                    req.release()
                t_light = time.monotonic()
                g_small = cv2.cvtColor(frame[::4, ::4], cv2.COLOR_BGR2GRAY).astype(np.float32)
                # grain: how much the same pixels change between two frames in a row (high gain in the dark = grainy);
                # median-based so a moving hamster doesn't count
                frame = cam.capture_array("main")
                g2 = cv2.cvtColor(frame[::4, ::4], cv2.COLOR_BGR2GRAY).astype(np.float32)
                noise = float(1.4826 * np.median(np.abs(g2 - g_small)) / 1.4142)
                light_now = {"lux": round(float(md.get("Lux") or 0), 2), "exp_ms": round((md.get("ExposureTime") or 0) / 1000, 1),
                             "gain": round(float(md.get("AnalogueGain") or 0) * float(md.get("DigitalGain") or 1), 2),
                             "bright": round(float(g_small.mean()), 1), "noise": round(noise, 2)}
                light_now["level"] = darkness.level(light_now["bright"], light_now["noise"])
                for k, v in (("lux", "lux"), ("exp", "exp_ms"), ("gain", "gain"), ("bright", "bright"), ("noise", "noise")):
                    light[k] += light_now[v]
                light["n"] += 1
            else:
                frame = cam.capture_array("main")
            now = time.monotonic()
            perf_t0 = time.perf_counter()
            frames += 1
            if wmask is not None:   # wheel area: compared with the frame two back (~0.13 s)
                wc = cv2.cvtColor(frame[wy0:wy1, wx0:wx1], cv2.COLOR_BGR2GRAY)
                if len(wheel_hist) == 2:
                    # a pixel "changed" if it moved by 40 % of the area's brightness (8..25 grey levels): with a
                    # fixed 25 the dim evening picture (brightness ~21, 11/15 15:49) showed a running hamster as ~0
                    wth = min(25.0, max(8.0, 0.4 * float(wc[wmask].mean())))
                    wheel_sum[0] += float((cv2.absdiff(wc, wheel_hist[0])[wmask] > wth).mean())
                    wheel_sum[1] += 1
                wheel_hist.append(wc)
                del wheel_hist[:-2]
                wbuf.append((now, np.array([float(wc[m, c].mean()) for c, m in wlines], np.float32)))
                del wbuf[:-90]   # ~6 s
                # calibration recording: `touch data/wheel_rec.flag` saves the wheel area for 60 s (data/wheel_rec.npz)
                if wrec is None and (data / "wheel_rec.flag").exists():
                    (data / "wheel_rec.flag").unlink(missing_ok=True)
                    wrec = {"t0": now, "frames": [], "ts": []}
                    log.info("wheel calibration recording started")
                if wrec is not None:
                    wrec["frames"].append(wc.copy()); wrec["ts"].append(now - wrec["t0"])
                    if now - wrec["t0"] >= 60:
                        np.savez_compressed(data / "wheel_rec.npz", frames=np.stack(wrec["frames"]), ts=np.array(wrec["ts"]),
                                            box=np.array([wx0, wy0, wx1, wy1]))
                        log.info("wheel calibration recording saved (%d frames)", len(wrec["frames"]))
                        wrec = None

            # whole-picture brightness (coarse grid, cheap): a sudden jump = the room light changed
            b_now = float(frame[::16, ::16].mean())
            if bright_ema is None:
                bright_ema = b_now
            elif abs(b_now - bright_ema) / max(bright_ema, 8.0) > light_jump:
                if now >= settle_until:
                    log.info("light changed (brightness %.0f -> %.0f): ignoring detections for a moment", bright_ema, b_now)
                settle_until = now + light_settle_s
                detector.warmup = t_cfg["warmup_frames"]   # relearn the background in the new light
                y_pend = None
            bright_ema += 0.15 * (b_now - bright_ema)
            settling = now < settle_until

            m_box = detector.detect(frame) if use_motion and not settling else None
            if use_motion and not settling:
                if detector.blobs:
                    hand_max_s = max(hand_max_s, max(a for a, _ in detector.blobs))
                if hand_in_cage(detector.blobs, y_box if now - y_seen < 2.0 else None, hand_free, hand_on_ham):
                    was = clean_until > now
                    clean_until = cleaning_until(hand_ep, now, clean_until, clean_gap_s, clean_hold_s)
                    if clean_until > now and not was:
                        log.info("hands in and out of the cage: cleaning, nothing counted until %d s after the last hand", clean_hold_s)
                    hand_until = max(now + hand_hold_s, clean_until)
            if hd_keep is not None and now - hd_t >= 0.25:
                hd_t = now
                _g = cv2.GaussianBlur(cv2.cvtColor(cv2.resize(frame, (W // 4, H // 4), interpolation=cv2.INTER_AREA),
                                                   cv2.COLOR_BGR2GRAY), (5, 5), 0)
                if hd_prev is not None:
                    _k = hd_keep
                    _hb = y_box if now - y_seen < 2.0 else None
                    if _hb is not None:   # the hamster itself (close to the camera it is big: 11/17 20:09 read 0.08)
                        _k = hd_keep.copy()
                        _x0, _y0 = int((_hb[0] - _hb[2] * 0.25) * W / 4), int((_hb[1] - _hb[3] * 0.25) * H / 4)
                        _x1, _y1 = int((_hb[0] + _hb[2] * 1.25) * W / 4) + 1, int((_hb[1] + _hb[3] * 1.25) * H / 4) + 1
                        _k[max(0, _y0):max(0, _y1), max(0, _x0):max(0, _x1)] = False
                    _f = float(np.count_nonzero((cv2.absdiff(_g, hd_prev) > 20) & _k)) / max(1, int(hd_keep.sum()))
                    hand_diff_s = max(hand_diff_s, _f)
                    if _f >= hand_diff:
                        if _f < 0.8:   # (most of the picture changing at once is the room light, not a hand)
                            was = clean_until > now
                            clean_until = cleaning_until(hand_ep, now, clean_until, clean_gap_s, clean_hold_s)
                            if clean_until > now and not was:
                                log.info("hands in and out of the cage: cleaning, nothing counted until %d s after the last hand",
                                         clean_hold_s)
                        hand_until = max(now + hand_hold_s, clean_until)
                hd_prev = _g
            if m_box is not None:
                t_mbox = now
            cage_moved = now - t_mbox < 1.0   # something moved in the cage in the last second (any blob, not the AI)

            # spout square: did it change since the last frame? (counted per second below, for the drinking estimate)
            # (compared with the frame two back: ~0.13 s at 15 fps, about what the checks were measured on)
            for sk, (sx, sy) in spouts.items():
                sx, sy = int(sx), int(sy)
                x0, y0 = max(0, sx - 2 * spout_r), max(0, sy - 2 * spout_r)
                crop = cv2.cvtColor(frame[y0:sy + 2 * spout_r, x0:sx + 2 * spout_r], cv2.COLOR_BGR2GRAY).astype(np.float32)
                hist = spout_prev.setdefault(sk, [])
                if len(hist) == 2 and crop.shape == hist[0].shape and crop.size:
                    flick, near, out = lick_frame(crop, hist[0], sx, sy, x0, y0, spout_r)
                    spout_hits[sk] += flick
                    ls = lick_sum[sk]
                    ls[0] += near; ls[1] += out; ls[2] += 1
                hist.append(crop)
                del hist[:-2]

            # AI detector: a few times a second is plenty and keeps the CPU cool
            y_moved = False
            y_unsure = None   # a low-confidence box from this frame's AI run, if any
            y_ran = False
            if yolo and now - t_yolo >= y_interval:
                t_yolo, y_ran = now, True
                _t_ai = time.perf_counter()
                found = yolo.detect(frame)
                perf_ai.append((time.perf_counter() - _t_ai) * 1000)
                perf_ai_t.append(now)
                if roi_px is not None:
                    # A box centred outside the drawn cage area is usually a reflection on the glass (the nest's mirror
                    # image at the top right, 11/15: 60 % sure and never moving). The hamster can still pass there,
                    # so such a box counts only while something actually moves inside it.
                    g_ai = None

                    def _keep(d):
                        nonlocal_g = g_ai
                        if cv2.pointPolygonTest(roi_px, (float((d[0] + d[2] / 2) * W), float((d[1] + d[3] / 2) * H)), True) >= -10:
                            return True
                        if g_prev_1hz is None or nonlocal_g is None:
                            return False
                        bm = box_motion(g_prev_1hz, nonlocal_g, d[:4], W, H)
                        return bm is not None and bm >= 0.03
                    if any(cv2.pointPolygonTest(roi_px, (float((d[0] + d[2] / 2) * W), float((d[1] + d[3] / 2) * H)), True) < -10 for d in found):
                        g_ai = cv2.GaussianBlur(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (5, 5), 0)
                    found = [d for d in found if _keep(d)]
                best, head = hamster_and_head(found)   # the head only with a model trained with head boxes
                if head is not None and best is not None and best[4] >= y_th and head[4] >= head_th:
                    y_head, y_head_t, y_head_conf = head[:4], now, float(head[4])
                ai_min["runs"] += 1   # per-minute AI statistics, saved with the activity (see Store.ai_minute)
                if best and best[4] >= y_th:
                    ai_min["seen"] += 1
                    ai_min["conf_sum"] += best[4]
                    ai_min["conf_max"] = max(ai_min["conf_max"], best[4])
                elif best:
                    ai_min["unsure"] += 1
                if best and best[4] < y_th:
                    y_unsure = best
                if best and best[4] >= y_th and settling:
                    best = None   # the picture is still adjusting to the light change: not trusted
                if best and best[4] >= y_th and now - y_seen > y_confirm_after_s:
                    # first sighting after a while: wait for a second, overlapping one
                    if not (y_pend and now - y_pend[1] <= 3.0 and boxes_touch(y_pend[0], best[:4])):
                        y_pend, best = (best[:4], now), None
                    else:
                        y_pend = None
                if best and best[4] >= y_th:
                    y_box, y_conf, y_seen = best[:4], best[4], now
                    y_area = max(1e-4, float(best[2]) * float(best[3]))
                    c = floor.floor_point(y_box)
                    if y_prev_c is not None:
                        y_moved = math.hypot((c[0] - y_prev_c[0]) * cw, (c[1] - y_prev_c[1]) * ch) >= t_cfg["min_step_cm"]
                    y_prev_c = c
            y_now = y_box if now - y_seen < 2.0 else None   # AI box is "current" for 2 s
            head_now = y_head if y_now is not None and now - y_head_t < 1.5 else None   # where its head is right now

            # Which box gives the position, and did the hamster move?
            #   motion (no model): any moving blob counts.
            #   yolo:   only the AI box; "moved" = the AI box shifted.
            #   hybrid: the AI decides. Movement counts only on/near the hamster the AI sees. A moving blob
            #           elsewhere (shadow, light change, bedding, a hand) is ignored. When the AI briefly loses the
            #           hamster (behind the bottle, motion blur), movement right next to where it was last seen
            #           still counts for a few seconds (ai_grace_s), following the hamster as it moves.
            pos_box, event, via_follow, follow_box = decide(mode, y_now, y_moved, m_box, follow_box,
                                                            now - y_seen, ai_grace_s, ai_follow_margin, y_area, ai_follow_max)

            if pos_box:
                box = pos_box
                pos = floor.floor_point(pos_box)   # distance is measured once a second below, not per frame
                last_seen_t = now
                last_seen_zone = zone_of_box(zones, pos_box, W, H, {})    # the drawn areas
                last_seen_bottle = bottle_of_box(pos_box, W, H, spot, spouts)   # at a bottle (水瓶 / added 喝水 areas)
            if event:
                last_motion = now
            moving = now - last_motion < 1.0

            # "hard" training photos: the AI was unsure, or something moved where the AI saw nothing.
            # These are exactly the frames the next training round needs.
            if y_ran and not paused and hard_saved < hard_max and now - t_hard >= hard_interval:
                missed = m_box is not None and y_now is None
                if y_unsure or missed:
                    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    name = dataset / f"hard_{'unsure' if y_unsure else 'missed'}_{stamp}.jpg"
                    cv2.imwrite(str(name), frame, [cv2.IMWRITE_JPEG_QUALITY, 92])
                    note_photo_zone(dataset, name, last_seen_zone or last_seen_bottle, moved=cage_moved)
                    score_photo(yolo, yolo_mid, frame, dataset, name)
                    # Pre-drawn box only when there is a real reason to think it's the hamster: the AI's own
                    # (unsure) guess, or movement right where the AI just saw it. A moving blob the AI never
                    # confirmed is often a shadow: no box, so the photo is either a clean "no hamster" example
                    # or gets its box drawn by hand while labelling.
                    guess = y_unsure[:4] if y_unsure else (m_box if via_follow else None)
                    if guess:
                        x, y, w, h = guess
                        name.with_suffix(".txt").write_text(f"0 {x + w / 2:.6f} {y + h / 2:.6f} {w:.6f} {h:.6f}\n")
                    t_hard = now
                    hard_saved += 1

            # 1 Hz: pause check, position sample, occupancy heatmap, activity seconds
            if now - t_sample >= 1.0:
                t_sample = now
                paused_till = pause_until()
                was_paused, paused = paused, time.time() < paused_till
                if was_paused and not paused:
                    # the cage may look different now (new bedding, moved toys): relearn the background
                    detector.warmup = t_cfg["warmup_frames"]
                    log.info("recording resumed")
                elif paused and not was_paused:
                    log.info("recording paused until %s", datetime.fromtimestamp(paused_till).strftime("%H:%M"))
                # Distance from 1 Hz samples: per-frame centroid jitter (the moving blob changes shape
                # every frame) used to add up to tens of metres. Count a step only if it is bigger than
                # jitter (min_step_cm) and physically possible (max_speed_cm_s); a larger jump means the
                # detector switched to another blob (reflection, bedding, a hand), so it is skipped.
                # how far the hamster moved since the last 1 Hz sample (used by the eating/drinking estimate)
                step_1hz = (math.hypot((pos[0] - last_sample_pos[0]) * cw, (pos[1] - last_sample_pos[1]) * ch)
                            if pos is not None and last_sample_pos is not None else None)
                if pos is not None and moving and not paused and last_sample_pos is not None:
                    dt = max(now - t_last_sample, 1e-3)
                    step = math.hypot((pos[0] - last_sample_pos[0]) * cw, (pos[1] - last_sample_pos[1]) * ch)
                    if t_cfg["min_step_cm"] <= step <= t_cfg.get("max_speed_cm_s", 50) * dt:
                        minute_dist += step
                last_sample_pos, t_last_sample = pos, now
                if spin_sure_ham and wheel_box is not None and now - last_seen_t >= 1.5:
                    pos = floor.floor_point(wheel_box)   # running out of sight: it is in the wheel
                if pos is not None and not paused:
                    store.position(datetime.now().isoformat(timespec="seconds"),
                                   round(float(pos[0]), 4), round(float(pos[1]), 4), moving)
                    # clamp: with marker calibration the hamster can be outside the 4 points (pos < 0 or > 1)
                    heat[min(max(int(pos[1] * heat.shape[0]), 0), heat.shape[0] - 1),
                         min(max(int(pos[0] * heat.shape[1]), 0), heat.shape[1] - 1)] += 1
                if moving and not paused:
                    minute_active += 1

                # how long has the AI seen the hamster sitting in one spot? (a short AI blink doesn't reset it)
                if y_now is not None and y_conf >= still_conf and not paused:
                    c = floor.floor_point(y_now)
                    if still_anchor is None or math.hypot((c[0] - still_anchor[0]) * cw,
                                                          (c[1] - still_anchor[1]) * ch) > still_move_cm:
                        still_anchor, still_since = c, now
                elif paused or now - y_seen > 60:
                    still_anchor, still_since = None, None

                wheel_spin_now = wheel_sum[0] / wheel_sum[1] if wheel_sum[1] else 0.0
                wheel_sum = [0.0, 0]
                wheel_rps, flicking = wheel_turns(wbuf, now, wheel_strips) if wmask is not None else (0.0, False)
                # a flicker with the wheel area itself unchanged is not the wheel: real turns (by hand or by the
                # hamster, 11/15) always changed >= 0.6 % of the area, while 41 "turns" at 12:13-12:18 and the
                # dawn ones came with ~0 % (light / noise rhythm)
                if wheel_spin_now < wheel_flick_min:
                    wheel_rps, flicking = 0.0, False
                # turning = a lot of the wheel area changing (slow / medium) OR the strips flickering at a steady
                # rate (fast: smeared strips change little but regularly); 2 s in a row, not the hamster walking past
                spin_streak = spin_streak + 1 if (wheel_spin_now >= wheel_spin_min or flicking) else 0
                strong_now = wheel_spin_now >= wheel_strong
                if strong_now:
                    strong_until = now + 10          # a strong turn: the run keeps counting through weak seconds
                spinning = spin_streak >= 2 and not paused
                # sure enough to put it in the wheel (area / position) right now: strong, or weak long enough
                spin_sure = spinning and (now < strong_until or spin_streak >= wheel_weak_min_s)
                # a hand in the cage, or just out of it: a turning wheel may be the hand's push (it keeps turning a
                # while after). Then turning alone does not put the hamster in the wheel (11/21 01:24: a hand turned
                # it and the hamster was logged "in the wheel" twice while it sat in the nest). Seeing it still counts.
                hand_spin = now < hand_until + hand_spin_after_s
                at_wheel = types.get(last_seen_zone) == "run"   # last seen in the wheel: it can be running out of sight
                spin_ham = spinning and not hand_spin and at_wheel
                spin_sure_ham = spin_sure and not hand_spin and at_wheel
                # the turns become running DISTANCE only in seconds that are recorded as running (below), so the
                # distance and the running record always agree (11/15: distance without any running looked wrong)
                if paused:
                    wheel_rps = 0.0
                if wheel_spin_now >= 0.01 or wheel_rps:   # per-second record for checking the wheel numbers (data/wheel.log)
                    try:
                        if (data / "wheel.log").exists() and (data / "wheel.log").stat().st_size > 5_000_000:
                            (data / "wheel.log").replace(data / "wheel.log.old")   # keep it small: one old copy
                        with open(data / "wheel.log", "a") as wl:
                            wl.write(f"{datetime.now():%H:%M:%S} spin {wheel_spin_now:.3f} spinning {int(spinning)} turns {wheel_rps:.3f}\n")
                    except OSError:
                        pass
                # zones: where is the hamster this second? (inside the nest = last seen at the nest door, then
                # out of sight)
                if (zones or spot) and not paused:   # spot = the water area under the spout
                    seen = now - last_seen_t < 1.5
                    if seen or spin_ham:   # the wheel turning = it came out of the nest and is running
                        in_nest = False
                    elif last_seen_zone == "nest" and not spin_sure_ham:
                        # gone from view at the nest door = gone into the nest, right away. Waiting nest_after_s first
                        # left a gap that ended the door visit, so every trip in was stored twice (door + inside):
                        # 11/16 17-20 h showed 112 nest visits, many of a few seconds
                        in_nest = True
                    # gone from view near the picture's edge, in a zone that continues outside it: still there (a while)
                    if seen:
                        left_at_edge = near_edge(box)
                    hidden_edge = (not seen and left_at_edge and last_seen_zone in edge_zones and
                                   now - last_seen_t <= offscreen_max_s.get(last_seen_zone, 600))
                    zone_now = "nest" if in_nest else (last_seen_zone if seen or hidden_edge
                                                       else spin_zone if spin_sure_ham else None)   # out of sight, wheel turning
                    if zone_now:
                        zone_secs[zone_now] = zone_secs.get(zone_now, 0) + 1
                        if visit and visit[0] == zone_now and now - visit[2] <= visit_gap_s:
                            visit[2] = now
                            visit[3] += 1
                        else:
                            if visit and visit[3] >= min_visit_s:
                                store.visit(visit[1], visit[0], visit[3])
                            visit = [zone_now, datetime.now(), now, 1]
                    elif visit and now - visit[2] > visit_gap_s:
                        if visit[3] >= min_visit_s:
                            store.visit(visit[1], visit[0], visit[3])
                        visit = None
                    # at a bottle (its circle), counted separately from the area above
                    bottle_now = last_seen_bottle if seen else None
                    if bottle_now:
                        zone_secs[bottle_now] = zone_secs.get(bottle_now, 0) + 1
                        if bvisit and bvisit[0] == bottle_now and now - bvisit[2] <= visit_gap_s:
                            bvisit[2] = now
                            bvisit[3] += 1
                        else:
                            if bvisit and bvisit[3] >= min_visit_s:
                                store.visit(bvisit[1], bvisit[0], bvisit[3])
                            bvisit = [bottle_now, datetime.now(), now, 1]
                    elif bvisit and now - bvisit[2] > visit_gap_s:
                        if bvisit[3] >= min_visit_s:
                            store.visit(bvisit[1], bvisit[0], bvisit[3])
                        bvisit = None

                # eating / drinking (camera estimate): the hamster the AI sees is at the food bowl / water bottle
                # and stays basically in one spot (head down at the bowl, mouth at the spout) long enough.
                # Just walking past doesn't count. Hamsters also carry food away in their cheeks and eat in the
                # nest, which the camera can't see, so "eat" is a lower bound.
                #
                # running on the wheel: the wheel is transparent, so the AI can see the hamster inside it.
                # Running = in the wheel zone + body stretched out (running posture; sitting or grooming looks
                # round) + staying in place on the cage floor (the wheel turns, the hamster doesn't travel)
                # + constant movement on the hamster itself (legs).
                seen_now = now - last_seen_t < 1.5
                # body movement: share of pixels inside the hamster's box that changed since a second ago
                g_now = cv2.GaussianBlur(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (5, 5), 0) if zones or spot else None
                body_ref = y_now if y_now is not None else (box if seen_now else None)
                body_motion = box_motion(g_prev_1hz, g_now, body_ref, W, H) if g_prev_1hz is not None and body_ref else None
                g_prev_1hz = g_now
                flicker, spout_hits = spout_hits, {k: 0 for k in spouts}
                licks, lick_sum = lick_sum, {k: [0.0, 0.0, 0] for k in spouts}
                # drinking = in a bottle's area (水瓶: the big circle) + the square at its spout flickering (tongue,
                # ball, drops). With a head-trained model its HEAD must also be at the spout (mouth on the spout).
                at_spout = None   # which bottle
                if spouts and head_now is not None:
                    hx, hy, hw, hh = head_now[:4]
                    at_spout = next((k for k, (sx, sy) in spouts.items() if (hx - hw * 0.4) * W <= sx <= (hx + hw * 1.4) * W
                                     and (hy - hh * 0.4) * H <= sy <= (hy + hh * 1.4) * H), None)
                elif seen_now and box is not None:
                    # drinking is decided by the spout (licking flickers its square); the circle only says the
                    # hamster is standing at that bottle. It has nothing to do with which AREA the stay is counted
                    # in: that is worked out on its own (a circle over the sand bath still counts the stay as bath).
                    at_spout = bottle_of_box(box, W, H, spot, spouts)
                # head in the food bowl (head-trained model): eating needs the head there, not just the body nearby
                head_zone = zone_at(zones, (head_now[0] + head_now[2] / 2) * W, (head_now[1] + head_now[3] / 2) * H) \
                    if head_now is not None and zones else None
                b_kind = None
                if spinning and (not seen_now or types.get(zone_now) == "run"):
                    b_kind = "run"   # the wheel turns and the hamster is in it or out of sight (above the picture)
                    body_motion = wheel_spin_now   # stored with the event: how much of the wheel area changed
                elif (zones or spot) and seen_now and not paused and step_1hz is not None:
                    # licking moves the head, and a box cut at the picture's edge jumps around: allow a bit more
                    if at_spout and step_1hz < beh_still_cm * 2:
                        # mouth on the spout (head model) is strong evidence by itself: a little flicker is enough
                        ls = licks[at_spout]
                        if (flicker[at_spout] >= max(1, drink_flicker_min - 1) if head_now is not None
                                else licking_second(flicker[at_spout], ls[0], ls[1], ls[2], drink_flicker_min)):
                            b_kind = "drink"
                            body_motion = flicker[at_spout] / max(1.0, fps or t_cfg["fps"])   # stored with the event: share of licking frames
                    elif types.get(zone_now) == "run":
                        pass   # running is decided only by the wheel turning (above): one rule whether it is seen or not
                    elif step_1hz < beh_still_cm:
                        b_kind = BEH_OF_TYPE.get(types.get(zone_now))
                        if b_kind == "eat" and (body_motion is None or body_motion < eat_motion_min):
                            b_kind = None   # sitting / resting at the bowl without moving is not eating
                        if b_kind == "eat" and head_now is not None and types.get(head_zone) != "eat":
                            b_kind = None   # the head model sees its head is not in a bowl
                        if b_kind == "drink" and zone_now in spouts:
                            b_kind = None   # spout marked: only the spout check above counts drinking
                hand_now = now < hand_until
                if hand_max_s >= 0.05 or hand_diff_s >= 0.03:
                    try:
                        if (data / "hand.log").exists() and (data / "hand.log").stat().st_size > 2_000_000:
                            (data / "hand.log").replace(data / "hand.log.old")
                        with open(data / "hand.log", "a") as hl:
                            hl.write(f"{datetime.now():%m-%d %H:%M:%S} blob {hand_max_s:.3f} change {hand_diff_s:.3f} ham {y_area or 0:.3f} "
                                     f"hand {int(hand_now)} seen {int(seen_now)}\n")
                    except OSError:
                        pass
                hand_max_s = hand_diff_s = 0.0
                if hand_now:
                    b_kind = None   # a hand in the cage: whatever turns / sits at the spout now is not the hamster's doing
                strong_sec = b_kind == "run" and strong_now
                # evidence for the whole-event check (judge_event)
                sec_ev = {}
                if b_kind == "run":
                    sec_ev = {"spin": wheel_spin_now, "ai_wheel": int(seen_now and types.get(zone_now) == "run")}
                elif b_kind == "drink":
                    cov = bool(seen_now and box is not None and bottle_of_box(box, W, H, {}, {at_spout: spouts[at_spout]}))
                    hd = head_now is not None
                    sec_ev = {"cov": int(cov), "head": int(hd), "both": int(cov and hd)}
                if b_kind and beh and beh[0] == b_kind and now - beh[2] <= beh_gap[b_kind]:
                    beh[2], beh[3] = now, beh[3] + 1
                    beh[4] += body_motion or 0.0
                    beh[5] = beh[5] or strong_sec
                elif b_kind:
                    if beh and finish_beh(beh) and beh[0] == "run":
                        wheel_revs_min += run_revs
                    run_revs = 0.0
                    ev0 = {"spin": []} if b_kind == "run" else {}
                    if b_kind == "run":
                        # where was it just before? seen in the lower half a moment ago = could not be in the wheel yet
                        ev0["entry_far"] = int(now - last_seen_t < 3.0 and box is not None and
                                               box[1] + box[3] / 2 > 0.6 and types.get(last_seen_zone) != "run")
                        # kept for tuning later: where / how long ago it was last seen when the wheel started
                        ev0.update(t0=now, last_zone=last_seen_zone, unseen_s=round(min(now - last_seen_t, 86400)))
                        # was it at the wheel just before? (last seen in the wheel area, within wheel_near_s)
                        ev0["near_wheel"] = int(types.get(last_seen_zone) == "run" and now - last_seen_t <= wheel_near_s)
                    beh = [b_kind, datetime.now(), now, 1, body_motion or 0.0, strong_sec, ev0]
                elif beh and now - beh[2] > beh_gap[beh[0]]:
                    if finish_beh(beh) and beh[0] == "run":
                        wheel_revs_min += run_revs
                    run_revs = 0.0
                    beh = None
                if b_kind and beh:
                    for k, v in sec_ev.items():
                        if k == "spin":
                            beh[6]["spin"].append(v)
                        else:
                            beh[6][k] = beh[6].get(k, 0) + v
                if beh and beh[0] == "run" and spinning and seen_now and types.get(zone_now) != "run" and not hand_now:
                    beh[6]["else"] = beh[6].get("else", 0) + 1   # wheel turning while the AI sees the hamster elsewhere
                # running distance: turns of the seconds counted as running; held back until the run is long enough
                # that it can't be a push coasting down (spin_push_like), else until the run is judged at its end
                if b_kind == "run":
                    run_revs += wheel_rps
                    if beh[3] >= max(beh_need(beh), 16):
                        wheel_revs_min, run_revs = wheel_revs_min + run_revs, 0.0

                key = datetime.now().strftime("%Y-%m-%dT%H:%M")
                if key != minute_key:
                    store.activity(minute_key, minute_active, minute_dist)
                    for z, s in zone_secs.items():
                        store.zone_minute(minute_key, z, s)
                    if ai_min["runs"]:
                        store.ai_minute(minute_key, ai_min)
                    if light["n"]:
                        store.light_minute(minute_key, light)
                        light = new_light()
                    if wheel_revs_min:
                        store.wheel_minute(minute_key, wheel_revs_min, wheel_revs_min * math.pi * wheel_d_cm)
                        wheel_revs_min = 0.0
                    store.flush()
                    minute_key, minute_active, minute_dist, zone_secs, ai_min = key, 0, 0.0, {}, new_ai_min()

            # dataset frames for phase-2 training (raw, unannotated image + optional pre-label box)
            if total_saved < t_cfg["dataset_max"] and not paused:
                stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                if event and now - t_ds_motion >= t_cfg["dataset_motion_interval_s"]:
                    name = dataset / f"motion_{stamp}.jpg"
                    cv2.imwrite(str(name), frame, [cv2.IMWRITE_JPEG_QUALITY, 92])
                    note_photo_zone(dataset, name, zone_now or bottle_now or last_seen_zone, moved=cage_moved)
                    score_photo(yolo, yolo_mid, frame, dataset, name)
                    if prelabel and pos_box:   # YOLO-format guess (class 0 = hamster, 1 = head) to correct while labeling
                        x, y, w, h = pos_box
                        lines = f"0 {x + w / 2:.6f} {y + h / 2:.6f} {w:.6f} {h:.6f}\n"
                        if head_now is not None:
                            x, y, w, h = head_now[:4]
                            lines += f"1 {x + w / 2:.6f} {y + h / 2:.6f} {w:.6f} {h:.6f}\n"
                        name.with_suffix(".txt").write_text(lines)
                    t_ds_motion = now
                    total_saved += 1
                elif not moving and now - t_ds_idle >= t_cfg["dataset_idle_interval_s"]:
                    cv2.imwrite(str(dataset / f"idle_{stamp}.jpg"), frame, [cv2.IMWRITE_JPEG_QUALITY, 92])
                    note_photo_zone(dataset, dataset / f"idle_{stamp}.jpg", zone_now or bottle_now or last_seen_zone, moved=cage_moved)
                    score_photo(yolo, yolo_mid, frame, dataset, dataset / f"idle_{stamp}.jpg")
                    t_ds_idle = now
                    total_saved += 1

            # video clip: record while the hamster moves. With a model, "event" is already AI-confirmed
            # (the hamster the AI sees, or right where it just saw it), so shadows don't start clips.
            fbox = None   # what the AI sees in this frame, kept beside the clip (drawn over it only on request)
            if y_now is not None:
                fbox = [round(float(v), 4) for v in y_now[:4]] + [round(float(y_conf), 2)]
                if head_now is not None:   # [.., hx, hy, hw, hh, head conf]
                    fbox += [round(float(v), 4) for v in head_now[:4]] + [round(y_head_conf, 2)]
            if now - t_req >= 0.25:
                t_req = now
                if PHOTO_REQUEST.exists():   # clean frame, best quality (no boxes, no zones drawn on it)
                    PHOTO_REQUEST.unlink(missing_ok=True)
                    shot = encode(frame, quality=95)
                    if shot:
                        atomic_write(LIVE_DIR / "photo.jpg", shot)
                if REC_REQUEST.exists():
                    try:
                        secs = float(REC_REQUEST.read_text().strip() or 0)
                    except (OSError, ValueError):
                        secs = 0.0
                    REC_REQUEST.unlink(missing_ok=True)
                    manual_until = now + max(0.0, min(secs, 300.0))
                    log.info("manual recording %s", f"for {secs:.0f} s" if secs > 0 else "stopped")
            manual_rec = now < manual_until and not paused
            rec.manual = manual_rec
            rec.feed(frame, now, event or (spinning and not paused) or manual_rec, [zone_now, bottle_now], paused, pos_box,
                     conf=y_conf if (y_ran and y_seen == now) else None, boxes=fbox,
                     head_conf=y_head_conf if (y_ran and y_head_t == now) else None)   # confidence of this frame's AI run, if it saw the hamster

            # clean frame (no overlays) for the cage-area editor on the dashboard
            if now - t_raw >= 5:
                t_raw = now
                raw = encode(frame, quality=88)
                if raw:
                    atomic_write(data / "raw.jpg", raw)

            want_snap = now - t_snap >= t_cfg["snapshot_interval_s"]
            want_live = now - t_live >= live_interval and live_requested()
            if want_snap or want_live:
                pos_img = floor.image_point(pos) if pos is not None else None
                if now - t_overlay >= 1:   # what the owner wants drawn on the picture (re-read once a second)
                    t_overlay, overlay = now, read_overlay(data / "overlay.json")
                view = annotate(frame, W, H, roi, floor_corners, box, pos_img, moving, paused, zones,
                                ai_box=y_now, ai_conf=y_conf if y_now is not None else None, spouts=list(spouts.items()), spout_r=spout_r,
                                head=head_now, spot=spot, show=overlay, head_conf=y_head_conf if head_now is not None else None)
                jpg = encode(view)
                if jpg and want_live:
                    t_live = now
                    atomic_write(LIVE_DIR / "live.jpg", jpg)
                if jpg and want_snap:
                    t_snap = now
                    atomic_write(data / "latest.jpg", jpg)
                    atomic_write(data / "tracker.json", json.dumps({
                        "x": pos and round(pos[0], 3), "y": pos and round(pos[1], 3),
                        "moving": moving, "fps": round(fps, 1),
                        "last_motion_s_ago": round(now - last_motion) if last_motion else None,
                        "minute_active_s": minute_active, "minute_distance_cm": round(minute_dist, 1),
                        "dataset_images": total_saved, "floor_calibrated": floor.calibrated,
                        "detector": mode, "ai_confidence": round(y_conf, 2) if y_now else None,
                        "head_confidence": round(y_head_conf, 2) if head_now is not None else None,
                        "paused_until": paused_till if paused else None,
                        "zone": zone_now or bottle_now, "hard_images": hard_saved, "light": light_now,
                        "wheel": {"spin": round(wheel_spin_now, 3), "spinning": bool(spinning), "rps": round(wheel_rps, 2),
                                  "cm_s": round(wheel_rps * math.pi * wheel_d_cm, 1)} if wmask is not None else None, "recording": rec.recording,
                        "manual_rec_s": round(manual_until - now) if now < manual_until and not paused else 0,
                        # seconds the hamster has been seen motionless outside the nest (0 = moving / not seen)
                        "still_s": round(now - still_since) if still_since is not None and last_seen_zone != "nest" else 0,
                        "still_zone": (last_seen_zone or last_seen_bottle) if still_since is not None else None,
                        "ai_seen_s_ago": round(now - y_seen) if y_seen > 0 else None,
                        "perf": {"proc_ms": round(statistics.median(perf_proc), 1) if perf_proc else None,
                                 "proc_ai_ms": round(statistics.median(perf_proc_ai), 1) if perf_proc_ai else None,
                                 "proc_ai_p95_ms": round(sorted(perf_proc_ai)[int(len(perf_proc_ai) * .95) - 1], 1) if len(perf_proc_ai) > 20 else None,
                                 "ai_ms": round(statistics.median(perf_ai), 1) if perf_ai else None,
                                 "ai_hz": round(sum(1 for x in perf_ai_t if now - x <= 60) / 60, 2),
                                 "cam_fps": round(fps, 1)},
                        "updated": datetime.now().isoformat(timespec="seconds")}).encode())

            # heatmap overlay once a minute
            if now - t_heat >= 60:
                t_heat = now
                if heat.max() > 0:
                    norm = cv2.resize((heat / heat.max() * 255).astype(np.uint8), (W, H),
                                      interpolation=cv2.INTER_LINEAR)
                    norm = floor.floor_to_image(cv2.GaussianBlur(norm, (31, 31), 0), W, H)
                    color = cv2.applyColorMap(norm, cv2.COLORMAP_JET)
                    png = encode(cv2.addWeighted(frame, 0.5, color, 0.5, 0), ".png")
                    if png:
                        atomic_write(data / "heatmap.png", png)

            if now - t_prune >= 3600:
                t_prune = now
                store.prune(keep_days)

            if now - t_fps >= 10:
                fps = frames / (now - t_fps)
                frames, t_fps = 0, now
    finally:
        log.info("stopping")
        store.activity(minute_key, minute_active, minute_dist)
        for z, s in zone_secs.items():
            store.zone_minute(minute_key, z, s)
        if ai_min["runs"]:
            store.ai_minute(minute_key, ai_min)
        if beh:
            finish_beh(beh)
        if visit and visit[3] >= min_visit_s:
            store.visit(visit[1], visit[0], visit[3])
        if bvisit and bvisit[3] >= min_visit_s:
            store.visit(bvisit[1], bvisit[0], bvisit[3])
        store.close()
        rec.stop()          # let the last clip finish encoding
        cam.stop()


if __name__ == "__main__":
    main()
