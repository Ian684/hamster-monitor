#!/usr/bin/env python3
"""Install a trained ONNX model zip (from the Colab notebook) and switch the tracker to AI detection.

Usage on the Pi:  cd ~/hamster && .venv/bin/python install_model.py hamster_ncnn_model.zip [--mode hybrid|yolo]
Steps: unzip -> check it loads and runs -> back up the old model -> move into place ->
       set [tracker] detector -> restart hamster-tracker. Nothing changes if the check fails.
"""
import argparse
import json
import re
import shutil
import subprocess
import sys
import tempfile
import time
import zipfile
from pathlib import Path

BASE = Path(__file__).resolve().parent
MODELS = BASE / "models"
TARGET = MODELS / "hamster_ncnn_model"
CONFIG = BASE / "config.toml"


def fail(msg):
    print(json.dumps({"ok": False, "error": msg}, ensure_ascii=False))
    sys.exit(1)


# Runs the model once in a separate process, so a broken model file can't take this installer down with it.
PROBE = r"""
import sys, time, json
sys.path.insert(0, sys.argv[3])
import cv2, numpy as np
from yolo_detector import YoloOnnx
det = YoloOnnx(sys.argv[1], imgsz=int(sys.argv[2]), conf=0.25)
img = cv2.imread(sys.argv[3] + "/data/latest.jpg")
if img is None:
    img = np.zeros((480, 640, 3), np.uint8)
det.detect(img)
t = time.perf_counter()
found = det.detect(img)
print(json.dumps({"ms": (time.perf_counter() - t) * 1000, "found": len(found), "names": det.names, "imgsz": det.imgsz}))
"""
SIZES = (640, 320, 416, 480, 512, 768, 960)   # Ultralytics exports at 640 unless told otherwise


def probe(src, imgsz):
    """(result dict, None) if the model runs at this size, else (None, reason)."""
    try:
        r = subprocess.run([sys.executable, "-c", PROBE, str(src), str(imgsz), str(BASE)],
                           capture_output=True, text=True, timeout=120)
    except subprocess.TimeoutExpired:
        return None, "測試超過 2 分鐘"
    if r.returncode == 0:
        try:
            return json.loads(r.stdout.strip().splitlines()[-1]), None
        except (ValueError, IndexError):
            pass
    if r.returncode < 0 or r.returncode == 139:
        return None, "當掉了（輸入大小不對或模型損壞）"
    return None, (r.stderr.strip().splitlines() or ["沒有回應"])[-1][-200:]


def read_imgsz_safe(d):
    sys.path.insert(0, str(BASE))
    from yolo_detector import read_imgsz
    return read_imgsz(d)


def same_files(a, b):
    return all((a / n).exists() and (b / n).exists() and (a / n).read_bytes() == (b / n).read_bytes()
               for n in ("model.onnx",))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("zip")
    ap.add_argument("--mode", default="hybrid", choices=["hybrid", "yolo"])
    a = ap.parse_args()

    tmp = Path(tempfile.mkdtemp(prefix="model_"))
    try:
        install(a, tmp)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)   # also after a failure (they used to pile up in /tmp)


def install(a, tmp):
    try:
        with zipfile.ZipFile(a.zip) as z:
            for n in z.namelist():   # refuse paths that would escape the temp folder
                if n.startswith("/") or ".." in Path(n).parts:
                    fail(f"zip 裡有不安全的路徑：{n}")
            z.extractall(tmp)
    except zipfile.BadZipFile:
        fail("不是有效的 zip 檔")
    params = list(tmp.rglob("model.onnx"))
    if len(params) != 1:
        fail("zip 裡找不到 model.onnx（請用自動訓練或 Colab 筆記本產生的 zip）")
    src = params[0].parent
    # the training weights (best.pt) let the next training continue from this model: keep them wherever they
    # are in the zip (next to model.onnx, one folder up, ...)
    pts = sorted(tmp.rglob("*.pt"), key=lambda q: (q.name != "best.pt", len(q.parts)))
    if pts and not (src / "best.pt").exists():
        shutil.copy2(pts[0], src / "best.pt")

    sys.path.insert(0, str(BASE))
    from yolo_detector import read_imgsz
    if TARGET.exists() and same_files(src, TARGET):
        added = bool(pts) and not (TARGET / "best.pt").exists()
        if added:   # same model uploaded again, now with its weights: just add them
            shutil.copy2(src / "best.pt", TARGET / "best.pt")
        print(json.dumps({"ok": True, "same": True, "imgsz": read_imgsz(TARGET), "mode": a.mode, "backup": None,
                          "best_pt": (TARGET / "best.pt").exists(),
                          "message": "這個模型已經安裝好了；已補上訓練權重 best.pt，之後可以接著訓練" if added
                          else "這個模型已經安裝好了，沒有變動" + ("" if (TARGET / "best.pt").exists() else "（zip 裡沒有 best.pt，之後不能接著訓練）")},
                         ensure_ascii=False))
        return

    # sanity check: loads, runs, reasonable speed. The input size comes from metadata.yaml; when the zip has
    # none (only the .param/.bin were zipped), find it by trying the usual sizes and write metadata.yaml.
    known = read_imgsz(src)
    res, why = None, None
    for size in ([known] if known else SIZES):
        res, why = probe(src, size)
        if res:
            break
    if not res:
        fail(f"模型在樹莓派上無法執行：{why}。請用 Colab 筆記本最後一格重新匯出，zip 裡要有 metadata.yaml")
    names = {int(k): v for k, v in (res["names"] or {0: "hamster"}).items()}
    if not known:   # class names from the network itself (2 outputs = hamster + head)
        (src / "metadata.yaml").write_text(
            "# written by install_model.py: the zip had no metadata.yaml, the input size was found by testing\n"
            f"task: detect\nimgsz:\n- {res['imgsz']}\n- {res['imgsz']}\nnames:\n"
            + "".join(f"  {k}: {v}\n" for k, v in sorted(names.items())), encoding="utf-8")
    ms, found = res["ms"], res["found"]

    MODELS.mkdir(exist_ok=True)
    backup = None
    if TARGET.exists():
        backup = MODELS / f"hamster_ncnn_model.bak-{time.strftime('%Y%m%d-%H%M%S')}"
        TARGET.rename(backup)
    shutil.copytree(src, TARGET)

    text = CONFIG.read_text(encoding="utf-8")
    text, n = re.subn(r'(?m)^detector\s*=\s*"[^"]*"', f'detector = "{a.mode}"', text, count=1)
    if n == 0:
        fail("config.toml 的 [tracker] 沒有 detector 設定")
    CONFIG.write_text(text, encoding="utf-8")
    subprocess.run(["sudo", "-n", "systemctl", "restart", "hamster-tracker"], check=False)
    print(json.dumps({"ok": True, "classes": names, "ms_per_frame": round(ms), "found_in_snapshot": found,
                      "imgsz": res["imgsz"], "imgsz_detected": not known,
                      "mode": a.mode, "backup": backup.name if backup else None,
                      "best_pt": (TARGET / "best.pt").exists()}, ensure_ascii=False))


if __name__ == "__main__":
    main()
