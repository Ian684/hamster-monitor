#!/usr/bin/env python3
"""考試照片: test one or more AI models on the owner's answer photos (data/exam/), never used for training.

Each exam photo has an answer file data/exam/<photo>.json:
    {"done": true, "boxes": [[cx, cy, w, h], ...], "heads": [[cx, cy, w, h], ...]}   (0-1; no boxes = no hamster)
For every model this counts, with the tracker's own thresholds:
    找到率   photos WITH a hamster where the AI's box overlaps the answer (IoU >= 0.5)
    框錯位置 photos with a hamster where the AI is sure but its box is somewhere else
    漏掉     photos with a hamster where the AI is not sure of anything
    誤認率   photos WITHOUT a hamster where the AI is sure there is one
    框準度   average overlap (IoU) of the boxes it found
    頭找到率 photos with a head answer where the AI's head box overlaps it (IoU >= 0.3)
split by brightness (夠亮 / 偏暗 / 太暗) and by area. Writes data/exam/results.json.

usage (on the Pi):  .venv/bin/python exam_run.py models/hamster_ncnn_model [models/hamster_ncnn_model.bak-...]
"""
import json
import os
import sys
import time
import tomllib
from datetime import datetime
from pathlib import Path

import cv2

BASE = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE))
import darkness  # noqa: E402
from yolo_detector import YoloNcnn, hamster_and_head  # noqa: E402

EXAM = BASE / "data" / "exam"
PROGRESS = Path("/run/hamster/exam.json")
IOU_BODY, IOU_HEAD = 0.5, 0.3


def xywh(c):   # (cx, cy, w, h) -> (x, y, w, h)
    return [c[0] - c[2] / 2, c[1] - c[3] / 2, c[2], c[3]]


def iou(a, b):
    ix = max(0.0, min(a[0] + a[2], b[0] + b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[1] + a[3], b[1] + b[3]) - max(a[1], b[1]))
    inter = ix * iy
    u = a[2] * a[3] + b[2] * b[3] - inter
    return inter / u if u > 0 else 0.0


def model_label(d):
    """version N (oldest = 1), or the owner's own name from 管理模型 → 改名."""
    import modelnames
    return modelnames.label(d)


def progress(**kw):
    try:
        PROGRESS.parent.mkdir(parents=True, exist_ok=True)
        tmp = PROGRESS.with_suffix(".tmp")
        tmp.write_text(json.dumps({**kw, "updated": time.time()}))
        os.replace(tmp, PROGRESS)
    except OSError:
        pass


def new_tally():
    return {"ham": 0, "hit": 0, "wrong": 0, "miss": 0, "empty": 0, "false": 0, "iou_sum": 0.0, "heads": 0, "head_hit": 0}


def add(t, r):
    if r["has"]:
        t["ham"] += 1
        t[r["o"]] += 1
        if r["o"] == "hit":
            t["iou_sum"] += r["iou"]
    else:
        t["empty"] += 1
        if r["o"] == "false":
            t["false"] += 1
    if r.get("head_has"):
        t["heads"] += 1
        t["head_hit"] += bool(r.get("head_ok"))


def finish(t):
    pct = lambda a, b: round(100 * a / b, 1) if b else None   # noqa: E731
    return {"ham": t["ham"], "empty": t["empty"], "heads": t["heads"],
            "found": pct(t["hit"], t["ham"]), "wrong": pct(t["wrong"], t["ham"]), "missed": pct(t["miss"], t["ham"]),
            "false": pct(t["false"], t["empty"]), "iou": round(100 * t["iou_sum"] / t["hit"], 1) if t["hit"] else None,
            "head": pct(t["head_hit"], t["heads"])}


def main():
    cfg = tomllib.load(open(BASE / "config.toml", "rb")).get("tracker", {})
    y_th, head_th = cfg.get("yolo_conf", 0.45), cfg.get("head_conf", 0.35)
    photos = []
    for j in sorted(EXAM.glob("*.json")):
        if j.name == "results.json":
            continue
        try:
            a = json.loads(j.read_text())
        except (OSError, ValueError):
            continue
        img = EXAM / (j.stem + ".jpg")
        if a.get("done") and img.exists():
            photos.append((img, a))
    dirs = [BASE / m for m in sys.argv[1:] if (BASE / m / "model.ncnn.param").exists()]
    if not photos or not dirs:
        progress(running=False, error="沒有標好答案的考試照片" if not photos else "找不到模型")
        return
    total, done = len(photos) * len(dirs), 0
    out = {"time": datetime.now().isoformat(timespec="seconds"), "threshold": y_th, "head_threshold": head_th,
           "photos": len(photos), "models": []}
    for d in dirs:
        det = YoloNcnn(d, conf=0.05, threads=2)
        overall, by_dark, by_zone, per = new_tally(), {}, {}, {}
        for img, a in photos:
            f = cv2.imread(str(img))
            done += 1
            if f is None:
                continue
            b, hd = hamster_and_head(det.detect(f))
            gt = [xywh(x) for x in a.get("boxes") or []]
            gh = [xywh(x) for x in a.get("heads") or []]
            sure = bool(b) and b[4] >= y_th
            r = {"has": bool(gt), "conf": round(float(b[4]), 3) if b else 0.0,
                 "box": [round(float(v), 4) for v in b[:4]] if b else None}
            if gt:
                ov = max((iou(b[:4], g) for g in gt), default=0.0) if b else 0.0
                r["iou"] = round(ov, 3)
                r["o"] = "hit" if sure and ov >= IOU_BODY else "wrong" if sure else "miss"
            else:
                r["o"] = "false" if sure else "ok"
            if gh:
                r["head_has"] = True
                hok = bool(hd) and sure and hd[4] >= head_th and max(iou(hd[:4], g) for g in gh) >= IOU_HEAD
                r["head_ok"] = hok
            if hd:
                r["head_conf"] = round(float(hd[4]), 3)
                r["head"] = [round(float(v), 4) for v in hd[:4]]
            per[img.name] = r
            add(overall, r)
            lv = darkness.level(a.get("bright"))
            if lv:
                add(by_dark.setdefault(lv, new_tally()), r)
            if a.get("zone"):
                add(by_zone.setdefault(a["zone"], new_tally()), r)
            if done % 5 == 0:
                progress(running=True, done=done, total=total)
        out["models"].append({"dir": d.name, "label": model_label(d), "overall": finish(overall),
                              "by_dark": {k: finish(v) for k, v in by_dark.items()},
                              "by_zone": {k: finish(v) for k, v in by_zone.items()}, "per": per})
    tmp = EXAM / "results.json.tmp"
    tmp.write_text(json.dumps(out, ensure_ascii=False))
    os.replace(tmp, EXAM / "results.json")
    progress(running=False, done=total, total=total)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:   # noqa: BLE001 - shown on the page
        progress(running=False, error=str(e)[:200])
        raise
