"""Tap-to-outline with MobileSAM (Segment Anything, small version) for the behaviour zones in the cage editor.

You tap on an object (the water bottle, the food bowl, ...); the AI cuts out the whole object around the tap.
More taps on the same object refine it ("this is part of it too"); a tap marked "not this" removes a part.
The AI also offers the object at a few sizes (e.g. just the bowl's rim / the bowl / the bowl with the bedding
around it); the editor's "換大小" button steps through them.

Models (models/mobilesam/, from huggingface.co/Acly/MobileSAM, MIT licence):
  mobile_sam_image_encoder.onnx  picture -> 64x64x256 "understanding" of the picture (the slow part, ~3 s on a Pi 5)
  sam_mask_decoder_multi.onnx     that + the taps -> 4 candidate masks (fast, ~0.2 s)
The encoder result is kept for the picture, so only the first tap on a new picture waits for it.
"""
import threading
from pathlib import Path

import cv2
import numpy as np

DIR = Path(__file__).resolve().parent / "models" / "mobilesam"
ENC, DEC = DIR / "mobile_sam_image_encoder.onnx", DIR / "sam_mask_decoder_multi.onnx"
SIZE = 1024   # SAM works on pictures whose long side is 1024
# largest believable size of each zone (share of the picture); bigger candidates are offered but not picked first
MAX_SHARE = {"water": 0.2, "food": 0.15, "wheel": 0.3, "nest": 0.25, "bath": 0.25}
_lock = threading.Lock()
_s = {"enc": None, "dec": None, "key": None, "emb": None, "scale": 1.0}


def available():
    return ENC.exists() and DEC.exists()


def _sessions():
    if _s["enc"] is None:
        import onnxruntime as ort
        opt = ort.SessionOptions()
        opt.intra_op_num_threads = 3   # leave a core for the camera tracker
        _s["enc"] = ort.InferenceSession(str(ENC), opt, providers=["CPUExecutionProvider"])
        _s["dec"] = ort.InferenceSession(str(DEC), opt, providers=["CPUExecutionProvider"])
    return _s["enc"], _s["dec"]


def _embedding(img, key):
    enc, _ = _sessions()
    if _s["key"] != key:
        H, W = img.shape[:2]
        scale = SIZE / max(H, W)
        rgb = cv2.cvtColor(cv2.resize(img, (round(W * scale), round(H * scale)), interpolation=cv2.INTER_LINEAR), cv2.COLOR_BGR2RGB)
        _s["emb"] = enc.run(None, {"input_image": rgb.astype(np.float32)})[0]   # normalising + padding happen inside
        _s["key"], _s["scale"] = key, scale
    return _s["emb"], _s["scale"]


def _to_poly(m, pos, W, H):
    """Mask -> polygon (0..1), keeping only the pieces that were tapped."""
    n, labels = cv2.connectedComponents(m)
    keep = np.zeros_like(m)
    for t in pos:
        i = labels[min(H - 1, int(t[1] * H)), min(W - 1, int(t[0] * W))]
        if i:
            keep[labels == i] = 1
    if not keep.any():   # the tap is just outside the mask (on an edge): take the biggest piece
        if n <= 1:
            return None
        keep[labels == 1 + int(np.argmax([(labels == i).sum() for i in range(1, n)]))] = 1
    keep = cv2.morphologyEx(keep, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)))
    cnts, _ = cv2.findContours(keep, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not cnts:
        return None
    c = cv2.convexHull(np.vstack(cnts)) if len(cnts) > 1 else cnts[0]
    eps = 0.006 * cv2.arcLength(c, True)
    poly = cv2.approxPolyDP(c, eps, True)
    while len(poly) > 20:
        eps *= 1.25
        poly = cv2.approxPolyDP(c, eps, True)
    if len(poly) < 3 or cv2.contourArea(poly) < 0.001 * W * H:
        return None
    return [[round(float(px) / W, 4), round(float(py) / H, 4)] for px, py in poly[:, 0, :]], cv2.contourArea(poly) / (W * H)


def outline(img, taps, kind=None, key=None):
    """img: BGR picture; taps: [[x, y, label], ...] with x, y in 0..1 and label 1 = "the object", 0 = "not this".
    key: identifies the picture (e.g. its file time) so the slow step is done once per picture.
    Returns ({"options": [{"poly", "area", "sure"}, ...] small -> big, "pick": index}, None) or (None, reason)."""
    H, W = img.shape[:2]
    pos = [t for t in taps if (t[2] if len(t) > 2 else 1)]
    if not pos:
        return None, "至少要點一下物品"
    with _lock:   # one at a time: the sessions and the cached picture are shared
        emb, scale = _embedding(img, key if key is not None else hash(img.tobytes()[::997]))
        _, dec = _sessions()
        pts = np.array([[[t[0] * W * scale, t[1] * H * scale] for t in taps] + [[0.0, 0.0]]], np.float32)
        lab = np.array([[float(t[2]) if len(t) > 2 else 1.0 for t in taps] + [-1.0]], np.float32)   # -1 = padding point
        masks, iou, _ = dec.run(None, {
            "image_embeddings": emb, "point_coords": pts, "point_labels": lab,
            "mask_input": np.zeros((1, 1, 256, 256), np.float32), "has_mask_input": np.zeros(1, np.float32),
            "orig_im_size": np.array([H, W], np.float32)})
    opts = []
    for k in range(masks.shape[1]):
        m = (masks[0, k] > 0).astype(np.uint8)
        if m.mean() > 0.7:
            continue
        r = _to_poly(m, pos, W, H)
        if r is None:
            continue
        poly, area = r
        if any(abs(area - o["area"]) < 0.004 for o in opts):   # practically the same as one we have
            continue
        opts.append({"poly": poly, "area": round(area, 4), "sure": round(float(iou[0, k]), 3)})
    if not opts:
        return None, "找不到物品，換個地方點，或點「不是這裡」排除多框的部分"
    opts.sort(key=lambda o: o["area"])
    limit = MAX_SHARE.get(kind, 0.3)
    fits = [i for i, o in enumerate(opts) if o["area"] <= limit]
    # the biggest believable one the AI is (nearly) as sure about as its best: the whole bottle, not just its cap
    if fits:
        best = max(opts[i]["sure"] for i in fits)
        pick = max(i for i in fits if opts[i]["sure"] >= best - 0.04)
    else:
        pick = 0
    return {"options": opts, "pick": pick}, None


def prepare(img, key):
    """Do the slow step now (when the editor opens), so the first tap answers quickly."""
    with _lock:
        _embedding(img, key)
