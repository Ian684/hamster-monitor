#!/usr/bin/env python3
"""Pack the collected training photos into a zip ready for labeling.

- Drops near-duplicate frames (perceptual hash), so you don't label the same picture 50 times
- Keeps each motion photo's pre-label box (YOLO .txt) next to it, plus classes.txt
- Adds a share of "idle" photos (often a sleeping or hidden hamster, or an empty cage)

Usage on the Pi:   cd ~/hamster && .venv/bin/python prepare_dataset.py [--max 600]
Result:            data/export/hamster_dataset.zip
"""
import argparse
import json
import random
import shutil
import zipfile
from pathlib import Path

import cv2
import numpy as np

BASE = Path(__file__).resolve().parent
SRC = BASE / "data" / "dataset"
OUT = BASE / "data" / "export"


def dhash(path, size=16):
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None
    small = cv2.resize(img, (size + 1, size), interpolation=cv2.INTER_AREA)
    return np.packbits(small[:, 1:] > small[:, :-1])


def pick(paths, max_n, min_dist):
    """Greedy: keep a photo only if it differs enough from every photo already kept."""
    kept, hashes = [], []
    for p in paths:
        h = dhash(p)
        if h is None:
            continue
        if all(np.unpackbits(h ^ k).sum() >= min_dist for k in hashes):
            kept.append(p)
            hashes.append(h)
            if len(kept) >= max_n:
                break
    return kept


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max", type=int, default=600, help="max photos in the pack")
    ap.add_argument("--idle-share", type=float, default=0.2, help="fraction of idle (no-motion) photos")
    ap.add_argument("--min-diff", type=int, default=12, help="hash bits that must differ (higher = fewer, more varied)")
    ap.add_argument("--names-file", help="only these photos (one file name per line; chosen on the dashboard)")
    ap.add_argument("--note", default="", help="what was chosen, written into README.txt")
    ap.add_argument("--first-file", help="photos labelled on the dashboard (one name per line): packed first")
    a = ap.parse_args()

    if a.names_file:   # the dashboard's filters picked the photos: keep all kinds it chose, hard ones first
        names = {n.strip() for n in Path(a.names_file).read_text().splitlines() if n.strip()}
        files = [SRC / n for n in sorted(names) if (SRC / n).exists()]
        random.seed(0)
        random.shuffle(files)
        # labelled on the dashboard first (finished answers), then hard / uploaded ones; dedup keeps whichever
        # comes first. Stable sort: still shuffled inside each group
        first = {n.strip() for n in Path(a.first_file).read_text().splitlines()} if a.first_file else set()
        files.sort(key=lambda p: (p.name not in first, not p.name.startswith(("hard_", "upload_"))))
        chosen = pick(files, a.max, a.min_diff) if a.min_diff > 0 else files[:a.max]
        motion = [p for p in files if p.name.startswith("motion_")]
        idle = [p for p in files if p.name.startswith("idle_")]
        hard = [p for p in files if p.name.startswith(("hard_", "upload_"))]
    else:
        motion = sorted(SRC.glob("motion_*.jpg"))
        idle = sorted(SRC.glob("idle_*.jpg"))
        hard = sorted(SRC.glob("hard_*.jpg"))   # AI unsure / missed: the most useful photos for round 2+
        random.seed(0)
        for group in (motion, idle, hard):
            random.shuffle(group)
        n_idle = int(a.max * a.idle_share)
        chosen = pick(hard, a.max // 2, a.min_diff)
        chosen += pick(motion, a.max - len(chosen) - min(n_idle, len(idle)), a.min_diff)
        chosen += pick(idle, a.max - len(chosen), a.min_diff)

    work = OUT / "hamster_dataset"
    shutil.rmtree(work, ignore_errors=True)
    work.mkdir(parents=True)
    prelabeled = 0
    for p in chosen:
        shutil.copy2(p, work / p.name)
        txt = p.with_suffix(".txt")
        if txt.exists():
            shutil.copy2(txt, work / txt.name)
            prelabeled += 1
    (work / "classes.txt").write_text("hamster\nhead\n")   # class 0 = the whole hamster, class 1 = its head
    best = BASE / "models" / "hamster_ncnn_model" / "best.pt"   # the Colab notebook continues from the model in use
    if best.exists():
        shutil.copy2(best, work / "best.pt")
    (work / "README.txt").write_text(
        "倉鼠訓練照片\n"
        "- 照片都已經在網頁上「畫框標註」好（打包時只選「已標好」）：這個 zip 可以直接拿去 Colab 訓練，不用再開 X-AnyLabeling\n"
        "- 還有沒標好的照片時：用 X-AnyLabeling 開啟這個資料夾，格式選 YOLO，類別有兩種：hamster（整隻）、head（頭）\n"
        "- 有倉鼠的照片：框一個 hamster（整隻，貼緊身體），看得到頭就再框一個 head（從鼻尖到耳朵後面）\n"
        "- 頭被擋住、背對鏡頭看不到臉：只框 hamster，不要猜頭的位置\n"
        "- motion_ 開頭的照片大多已經有預先畫好的 hamster 框，只要修正位置大小，再補 head\n"
        "- 照片裡沒有倉鼠：不要畫框（會當作背景）\n"
        "- 看不清楚、不確定的照片：直接刪掉\n"
        "- best.pt 是現在使用中模型的訓練權重：Colab 筆記本會從它接著訓練（不要刪）\n" + (f"\n這包挑選的條件：{a.note}\n" if a.note else ""), encoding="utf-8")

    zpath = OUT / "hamster_dataset.zip"
    with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED) as z:
        for f in sorted(work.iterdir()):
            z.write(f, f"hamster_dataset/{f.name}")
    shutil.rmtree(work)
    summary = {"photos": len(chosen), "prelabeled": prelabeled, "selected": len(motion) + len(idle) + len(hard),
               "source_hard": len(hard), "source_motion": len(motion),
               "source_idle": len(idle), "zip_mb": round(zpath.stat().st_size / 1e6, 1)}
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
