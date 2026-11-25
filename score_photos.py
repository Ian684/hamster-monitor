#!/usr/bin/env python3
"""Score training photos and video clips with the installed AI model, so the dashboard can show how sure the AI
is on each one (照片管理 / 錄影紀錄 › 信心低→高). Sorting by the lowest confidence shows exactly where the model is
weak: those photos are the ones to label carefully for the next training round.

Writes:
  data/dataset/ai_scores.json   {"model": <id of the model>, "scores": {photo name: [conf, x, y, w, h]}}
                                (conf 0 and no box = the AI found nothing)
  data/clips/<clip>.json        adds "ai_conf" {avg, max, n} to clips recorded before confidence was saved

Only photos/clips not yet scored by *this* model are done, so after installing a new model everything is re-scored.
Usage on the Pi:  cd ~/hamster && .venv/bin/python score_photos.py [--all]
The web page starts this automatically in the background when it finds unscored photos.
"""
import argparse
import fcntl
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import cv2

BASE = Path(__file__).resolve().parent
DATASET = BASE / "data" / "dataset"
CLIPS = BASE / "data" / "clips"
SCORES = DATASET / "ai_scores.json"
MODEL = BASE / "models" / "hamster_ncnn_model"
PROGRESS = Path("/run/hamster/scoring.json")
LOCK = Path("/run/hamster/scoring.lock")
SURE = 0.45   # same threshold the tracker uses for "this is the hamster"


def model_id():
    p = MODEL / "model.ncnn.bin"
    h = hashlib.md5()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:12]


def read_live(mid):
    """Scores the tracker wrote when it saved each photo, for model `mid` only."""
    out = []
    try:
        for line in (DATASET / "ai_live.jsonl").read_text(encoding="utf-8").splitlines():
            try:
                r = json.loads(line)
                if r.get("model") == mid and isinstance(r.get("s"), list):
                    out.append(r)
            except ValueError:
                pass
    except OSError:
        pass
    return out


def atomic_json(path, obj):
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False))
    os.replace(tmp, path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--all", action="store_true", help="re-score everything")
    a = ap.parse_args()
    if not (MODEL / "model.ncnn.param").exists():
        print("no model installed"); return
    LOCK.parent.mkdir(parents=True, exist_ok=True)
    lock = open(LOCK, "w")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)   # only one scorer at a time
    except BlockingIOError:
        print("already running"); return
    os.nice(15)   # the tracker comes first

    sys.path.insert(0, str(BASE))
    from yolo_detector import YoloNcnn, hamster_and_head
    det = YoloNcnn(MODEL, conf=0.05, threads=1)   # low threshold: report even very unsure guesses
    mid = model_id()
    data = {"model": mid, "scores": {}}
    if SCORES.exists() and not a.all:
        try:
            old = json.loads(SCORES.read_text())
            if old.get("model") == mid:
                data = old
        except (OSError, ValueError):
            pass
    if not a.all:   # scored by the tracker right when they were saved (data/dataset/ai_live.jsonl)
        for r in read_live(mid):
            data["scores"].setdefault(r["n"], r["s"])
    photos = sorted(p for p in DATASET.glob("*.jpg"))
    names = {p.name for p in photos}
    data["scores"] = {k: v for k, v in data["scores"].items() if k in names}   # forget deleted photos
    todo = [p for p in photos if p.name not in data["scores"]]
    clips = [p for p in sorted(CLIPS.glob("clip_*.json"))
             if a.all or json.loads(p.read_text()).get("ai_model") != mid]
    total, done, t0 = len(todo) + len(clips), 0, time.time()

    def progress():
        atomic_json(PROGRESS, {"running": True, "done": done, "total": total, "updated": time.time()})

    progress()
    for p in todo:
        img = cv2.imread(str(p))
        if img is not None:
            b, hd = hamster_and_head(det.detect(img))   # the whole hamster (+ its head with a head-trained model)
            s = [round(float(b[4]), 3), *[round(float(v), 4) for v in b[:4]]] if b else [0.0]
            if b and hd:   # [conf, x, y, w, h, head conf, head x, y, w, h]
                s += [round(float(hd[4]), 3), *[round(float(v), 4) for v in hd[:4]]]
            data["scores"][p.name] = s
        done += 1
        if done % 20 == 0:
            atomic_json(SCORES, data); progress()
    atomic_json(SCORES, data)

    for j in clips:   # clips: one frame per second; confidence of the frames where the AI is sure it's the hamster
        mp4 = j.with_suffix(".mp4")
        info = json.loads(j.read_text())
        cap = cv2.VideoCapture(str(mp4))
        fps = cap.get(cv2.CAP_PROP_FPS) or 10
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        confs, hconfs, frames = [], [], 0
        for k in range(0, n, max(1, int(round(fps)))):
            cap.set(cv2.CAP_PROP_POS_FRAMES, k)
            ok, f = cap.read()
            if not ok:
                break
            frames += 1
            b, hd = hamster_and_head(det.detect(f))
            if b and b[4] >= SURE:
                confs.append(float(b[4]))
                if hd:
                    hconfs.append(float(hd[4]))
        cap.release()
        info["ai_conf"] = ({"avg": round(sum(confs) / len(confs), 3), "max": round(max(confs), 3), "n": len(confs)}
                           if confs else {"avg": 0.0, "max": 0.0, "n": 0})
        info["ai_seen_ratio"] = round(len(confs) / frames, 2) if frames else 0
        info["head_conf"] = {"avg": round(sum(hconfs) / len(hconfs), 3) if hconfs else 0.0, "n": len(hconfs)}
        info["ai_model"] = mid
        atomic_json(j, info)
        done += 1
        progress()

    atomic_json(PROGRESS, {"running": False, "done": done, "total": total, "updated": time.time(),
                           "seconds": round(time.time() - t0)})
    print(json.dumps({"photos": len(todo), "clips": len(clips), "seconds": round(time.time() - t0)}))


if __name__ == "__main__":
    main()
