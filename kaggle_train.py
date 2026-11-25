# Runs ON KAGGLE (free GPU), pushed there by autotrain.py. Same training as train_hamster_colab.ipynb:
# photos + YOLO .txt labels (0 = hamster, 1 = head; no / empty .txt = no hamster) -> YOLO11n -> ONNX (run on the Pi with ONNX Runtime).
# Output in /kaggle/working: hamster_ncnn_model.zip (what the Pi installs) and result.json (scores).
import hashlib
import json
import re
import pathlib
import random
import shutil
import subprocess
import sys
import zipfile

subprocess.run([sys.executable, "-m", "pip", "-q", "install", "ultralytics"], check=True)
from ultralytics import YOLO  # noqa: E402

INP, TMP, OUT = pathlib.Path("/kaggle/input"), pathlib.Path("/kaggle/tmp"), pathlib.Path("/kaggle/working")
EXPECT = None   # set by autotrain.py: the marker file of the upload this run must use
if EXPECT and not any(p.name == EXPECT for p in INP.rglob("*")):
    raise SystemExit(f"STALE_DATASET: {EXPECT} not in the input (Kaggle still serves the previous upload)")
raw = TMP / "raw"
shutil.rmtree(TMP, ignore_errors=True)
raw.mkdir(parents=True)
for z in INP.rglob("*.zip"):   # the dataset is uploaded as one zip (Kaggle may or may not unpack it)
    zipfile.ZipFile(z).extractall(raw)
for p in INP.rglob("*"):       # files Kaggle already unpacked
    if p.is_file() and p.suffix.lower() in (".jpg", ".jpeg", ".png", ".txt", ".pt"):
        dst = raw / p.relative_to(INP)
        dst.parent.mkdir(parents=True, exist_ok=True)
        if not dst.exists():
            shutil.copy(p, dst)

imgs = sorted(p for p in raw.rglob("*") if p.suffix.lower() in (".jpg", ".jpeg", ".png"))
imgs = list({p.name: p for p in imgs}.values())   # the same photo twice (zip + unpacked): once
cls = lambda p: [l.split()[0] for l in p.with_suffix(".txt").read_text().splitlines() if l.split()] if p.with_suffix(".txt").exists() else []  # noqa: E731
n_ham = sum("0" in cls(p) for p in imgs)
n_head = sum("1" in cls(p) for p in imgs)
start = next(iter(raw.rglob("best.pt")), None)   # the last model's weights: fine-tune instead of starting over
base = YOLO(str(start)) if start else YOLO("yolo11n.pt")
train_head = n_head >= 50 or (start is not None and len(base.names) >= 2)
# continuing a model: keep the input size it was trained at (v4 = 640), so it keeps what it learned
imgsz = int((getattr(base, "ckpt", None) or {}).get("train_args", {}).get("imgsz", 320) or 320) if start else 320
if isinstance(imgsz, (list, tuple)):
    imgsz = int(imgsz[0])
print(f"photos {len(imgs)}, with hamster {n_ham}, with head {n_head}, start from {'last model' if start else 'yolo11n'}, head {train_head}, imgsz {imgsz}")

# the check photos (val) must stay the same from one training to the next, or each run picks its "best" weights with
# a different test and the results jump around. A photo's place is fixed by its name, so adding photos never moves
# the old ones. Photos from the same minute go to the same side: near-copies in both would make the check too easy.
def val_side(p):
    m = re.search(r"_(\d{8}_\d{4})\d{2}$", p.stem)
    return int(hashlib.md5((m.group(1) if m else p.stem).encode()).hexdigest(), 16) % 100 < 15
val = [p for p in imgs if val_side(p)]
train = [p for p in imgs if not val_side(p)]
if not val:   # tiny sets
    random.seed(0)
    val = [random.choice(imgs)]
    train = [p for p in imgs if p not in val]
n_val = len(val)
print(f"check photos (fixed by name): {n_val}, training photos: {len(train)}")
ds = TMP / "ds"
for split, items in (("val", val), ("train", train)):
    (ds / "images" / split).mkdir(parents=True)
    (ds / "labels" / split).mkdir(parents=True)
    for p in items:
        shutil.copy(p, ds / "images" / split / p.name)
        keep = ("0", "1") if train_head else ("0",)
        txt = p.with_suffix(".txt")
        lines = [l for l in (txt.read_text().splitlines() if txt.exists() else []) if l.split() and l.split()[0] in keep]
        (ds / "labels" / split / (p.stem + ".txt")).write_text("\n".join(lines) + ("\n" if lines else ""))
names = "  0: hamster\n  1: head\n" if train_head else "  0: hamster\n"
(ds / "data.yaml").write_text(f"path: {ds}\ntrain: images/train\nval: images/val\nnames:\n{names}")

# continuing the model in use: small steps, so new photos add to what it knows instead of retraining it away
# (each full-speed rerun moved the head on the bottle back and forth). From scratch: the usual settings.
tune = (dict(epochs=40, patience=15, optimizer="AdamW", lr0=0.0005, warmup_epochs=1, close_mosaic=10) if start
        else dict(epochs=120, patience=25))
base.train(data=str(ds / "data.yaml"), imgsz=imgsz, batch=32, project=str(TMP / "runs"), name="hamster", exist_ok=True,
           fliplr=0.5, flipud=0.5, degrees=10, mosaic=1.0, **tune)
best = YOLO(str(TMP / "runs/hamster/weights/best.pt"))
m = best.val(data=str(ds / "data.yaml"), imgsz=imgsz)
DEPLOY_IMGSZ = 480   # what the Pi runs: on ONNX Runtime 480 was as accurate as 640 on the 跑分 photos in about half the time
out = pathlib.Path(best.export(format="onnx", imgsz=DEPLOY_IMGSZ, simplify=True))   # runs on the Pi with ONNX Runtime
pkg = TMP / "hamster_ncnn_model"   # folder name kept from the NCNN days: the Pi's model folders all use it
pkg.mkdir(parents=True, exist_ok=True)
shutil.copy(out, pkg / "model.onnx")
import datetime as _dt   # metadata.yaml: the input size, class names and training date the Pi reads
(pkg / "metadata.yaml").write_text("".join([
    f"description: Ultralytics YOLO11n model trained on {ds / 'data.yaml'}\n",
    f"date: \"{_dt.datetime.now(_dt.timezone.utc).isoformat()}\"\n",
    f"task: detect\nimgsz:\n- {imgsz}\n- {imgsz}\nnames:\n",
    *(f"  {k}: {v}\n" for k, v in sorted(best.names.items()))]), encoding="utf-8")
shutil.copy(TMP / "runs/hamster/weights/best.pt", pkg / "best.pt")   # the next training starts from this
shutil.make_archive(str(OUT / "hamster_ncnn_model"), "zip", str(TMP), "hamster_ncnn_model")
(OUT / "result.json").write_text(json.dumps({
    "photos": len(imgs), "imgsz": imgsz, "with_hamster": n_ham, "with_head": n_head, "head": train_head, "fine_tuned": start is not None, "val": n_val,
    "map50": round(float(m.box.map50), 3), "map50_95": round(float(m.box.map), 3),
    "precision": round(float(m.box.mp), 3), "recall": round(float(m.box.mr), 3)}))
print("done", (OUT / "result.json").read_text())
