#!/usr/bin/env python3
"""全自動訓練 (Kaggle): labelled photos -> Kaggle (free GPU) trains -> the new model comes back -> exam -> keep or not.

  1 pack     every photo labelled on the dashboard + every photo used by earlier automatic trainings
             (data/trained/: kept so the training set only grows) + the last model's best.pt to fine-tune from
  2 upload   as a private Kaggle dataset  <user>/hamster-train-data   (new version each time)
  3 train    push kaggle_train.py as a private GPU notebook  <user>/hamster-train,  wait until it finishes
  4 download its output: hamster_ncnn_model.zip + result.json
  5 exam     exam_run.py: the model in use vs the new one, on the owner's exam photos
  6 decide   new one better (找到率 - 誤認率 higher) -> install it (install_model.py; the old one stays in 管理模型)
             not better / too few exam photos -> keep the old one; the new one is kept as an old model to try by hand
Progress in data/autotrain.json (shown on the dashboard). Needs ~/.kaggle/kaggle.json (uploaded on the dashboard).
usage:  .venv/bin/python autotrain.py            (the dashboard starts it)
"""
import json
import os
import shutil
import subprocess
import sys
import time
import zipfile
from datetime import datetime
from pathlib import Path

BASE = Path(__file__).resolve().parent
DATA = BASE / "data"
DATASET, EXAM, TRAINED = DATA / "dataset", DATA / "exam", DATA / "trained"
MODELS = BASE / "models"
STATE = DATA / "autotrain.json"
WORK = DATA / "autotrain_work"
KAGGLE = str(BASE / ".venv" / "bin" / "kaggle")
MIN_LABELED, MIN_EXAM = 30, 10
POLL_S, MAX_WAIT_S = 60, 5 * 3600
sys.path.insert(0, str(BASE))


def save(**kw):
    st = {}
    try:
        st = json.loads(STATE.read_text())
    except (OSError, ValueError):
        pass
    st.update(kw, updated=datetime.now().isoformat(timespec="seconds"), pid=os.getpid())
    tmp = STATE.with_suffix(".tmp")
    tmp.write_text(json.dumps(st, ensure_ascii=False))
    os.replace(tmp, STATE)


PAUSE = DATA / "autotrain.pause"   # made / removed by the dashboard (暫停 / 繼續)


def wait_if_paused(where):
    """Hold here while 暫停 is on. Kaggle itself can't be paused: while it trains, it goes on (and is picked up after)."""
    if not PAUSE.exists():
        return
    before = json.loads(STATE.read_text()).get("text", "") if STATE.exists() else ""
    save(paused=True, text=f"已暫停（{where}）")
    while PAUSE.exists():
        time.sleep(5)
    save(paused=False, text=before or "繼續")


def step(n, text):
    save(state="running", step=n, text=text)
    print(f"[{n}] {text}", flush=True)


def kaggle(*args, check=True, timeout=900):
    r = subprocess.run([KAGGLE, *args], capture_output=True, text=True, timeout=timeout)
    if check and r.returncode != 0:
        raise RuntimeError(f"kaggle {' '.join(args[:2])}: {(r.stderr or r.stdout).strip()[-300:]}")
    return (r.stdout or "") + (r.stderr or "")


def kaggle_user():
    return json.loads((Path.home() / ".kaggle" / "kaggle.json").read_text())["username"]


def notify(title, msg):
    try:
        import tomllib
        import alerts
        alerts.send(tomllib.load(open(BASE / "config.toml", "rb")), title, msg)
    except Exception:   # noqa: BLE001 - a phone note is a bonus
        pass


STAMP = None


def pack():
    """Labelled photos (+ what earlier trainings used), never photos right next to an exam photo."""
    import web   # the same rules the dashboard uses (labelled list, exam leak guard)
    labd = web._labeled()
    ets = web._exam_times()
    shutil.rmtree(WORK, ignore_errors=True)
    src = WORK / "data" / "photos"
    src.mkdir(parents=True)
    TRAINED.mkdir(parents=True, exist_ok=True)
    new = []
    for n in labd:
        p = DATASET / n
        if p.exists() and not web._near_exam(n, ets):
            shutil.copy2(p, src / n)
            txt = p.with_suffix(".txt")
            (src / txt.name).write_text(txt.read_text() if txt.exists() else "")
            new.append(n)
    old = 0
    for p in TRAINED.glob("*.jpg"):
        if not (src / p.name).exists() and not web._near_exam(p.name, ets):
            shutil.copy2(p, src / p.name)
            txt = p.with_suffix(".txt")
            (src / txt.name).write_text(txt.read_text() if txt.exists() else "")
            old += 1
    best = MODELS / "hamster_ncnn_model" / "best.pt"   # fine-tune the model in use when its weights came with it
    if best.exists():
        shutil.copy2(best, src / "best.pt")
    total = len(new) + old
    zpath = WORK / "data" / "hamster_train.zip"
    with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED) as z:
        for f in sorted(src.iterdir()):
            z.write(f, f.name)
    shutil.rmtree(src)
    # a marker only this upload has: Kaggle says "ready" for the previous version while it still processes the new
    # one, and a notebook started then trains on the old photos (11/17 22:15: 1477 instead of 1513 photos, same model)
    global STAMP
    STAMP = f"00_upload_{datetime.now():%Y%m%d_%H%M%S}.txt"   # "00_": first in Kaggle's file list (shown 20 a page, A-Z)
    (WORK / "data" / STAMP).write_text(STAMP)
    return new, old, total, best.exists()


class TrainError(RuntimeError):
    """A failure with a short message for the page and the raw details kept separately."""
    def __init__(self, msg, detail=""):
        super().__init__(msg)
        self.detail = detail


def explain_kaggle_log(log):
    """The Kaggle run's log -> one sentence a person can act on."""
    low = log.lower()
    if "stale_dataset" in low:
        return "Kaggle 還在用上一次的照片（新照片還沒處理好），請過幾分鐘再按開始。"
    if "name resolution" in low or "failed to resolve" in low or "no matching distribution found for ultralytics" in low:
        return "Kaggle 不讓訓練程式上網，裝不了訓練工具。請到 kaggle.com › Settings 完成「手機驗證」（Phone verification），再按開始。"
    if "quota" in low or "gpu" in low and "limit" in low:
        return "這週的 Kaggle 免費 GPU 時數用完了，等下週重置再練。"
    if "out of memory" in low or "cuda oom" in low:
        return "Kaggle 的 GPU 記憶體不夠，請少一點照片或稍後再試。"
    lines = [l.strip() for l in log.splitlines() if "error" in l.lower() and l.strip()]
    return "Kaggle 訓練失敗：" + (lines[-1][:80] if lines else "原因不明，請稍後再試")


def main():
    started = datetime.now().isoformat(timespec="seconds")
    save(state="running", step=1, text="準備中", started=started, result=None, error=None)
    user = kaggle_user()
    ds_id, k_id = f"{user}/hamster-train-data", f"{user}/hamster-train"

    step(1, "打包已標好的照片")
    new, old, total, fine = pack()
    if total < MIN_LABELED:
        raise RuntimeError(f"已標好的照片只有 {total} 張，至少要 {MIN_LABELED} 張才能訓練（到「管理訓練照片」點照片按「畫框標註」）")
    save(photos=total, photos_new=len(new), fine_tune=fine)

    wait_if_paused("還沒上傳照片，按繼續才會上傳")
    step(2, f"上傳 {total} 張照片到 Kaggle")
    meta = WORK / "data" / "dataset-metadata.json"
    meta.write_text(json.dumps({"title": "hamster-train-data", "id": ds_id, "licenses": [{"name": "CC0-1.0"}]}))
    exists = "hamster-train-data" in kaggle("datasets", "list", "--mine", "-s", "hamster-train-data", check=False)
    if exists:
        kaggle("datasets", "version", "-p", str(WORK / "data"), "-m", f"auto {started}", "-r", "zip")
    else:
        kaggle("datasets", "create", "-p", str(WORK / "data"), "-r", "zip")
    for _ in range(90):   # Kaggle processes the upload before a notebook can use it: wait for THIS upload's marker
        s = kaggle("datasets", "status", ds_id, check=False).lower()
        if "ready" in s and STAMP in kaggle("datasets", "files", ds_id, check=False):
            break
        time.sleep(10)
    else:
        raise RuntimeError("Kaggle 處理照片太久，請稍後再試")
    # these photos are now training data: keep them for the next rounds (even if deleted from the photo list)
    for n in new:
        shutil.copy2(DATASET / n, TRAINED / n)
        t = (DATASET / n).with_suffix(".txt")
        (TRAINED / n).with_suffix(".txt").write_text(t.read_text() if t.exists() else "")
    with open(DATA / "packed.txt", "a") as f:
        f.write("".join(n + "\n" for n in new))

    step(3, "Kaggle 用 GPU 訓練中（通常 15～40 分鐘）")
    kdir = WORK / "kernel"
    kdir.mkdir(parents=True)
    # the notebook checks it got this upload (and stops instead of training on old photos)
    (kdir / "kaggle_train.py").write_text((BASE / "kaggle_train.py").read_text(encoding="utf-8").replace(
        'EXPECT = None', f'EXPECT = "{STAMP}"', 1), encoding="utf-8")
    (kdir / "kernel-metadata.json").write_text(json.dumps({
        "id": k_id, "title": "hamster-train", "code_file": "kaggle_train.py", "language": "python",
        "kernel_type": "script", "is_private": True, "enable_gpu": True, "enable_internet": True,
        "dataset_sources": [ds_id], "competition_sources": [], "kernel_sources": []}))
    kaggle("kernels", "push", "-p", str(kdir))
    t0 = time.time()
    while True:
        time.sleep(POLL_S)
        wait_if_paused("Kaggle 那邊照樣在練，按繼續才會接著等結果")
        s = kaggle("kernels", "status", k_id, check=False).lower()
        mins = int((time.time() - t0) / 60)
        # queued = Kaggle has no free GPU yet (busy hours): say so instead of "training"
        save(text=f"Kaggle 排隊等 GPU 中（已 {mins} 分鐘，還沒開始練）" if "queued" in s else f"Kaggle 用 GPU 訓練中（已 {mins} 分鐘）")
        if "complete" in s:
            break
        if "error" in s or "cancel" in s:
            out = WORK / "out"
            kaggle("kernels", "output", k_id, "-p", str(out), check=False, timeout=600)
            logs = sorted(out.glob("*.log"))
            raw = logs[0].read_text(errors="replace") if logs else s
            try:   # the .log is a JSON list of {stream_name, data}
                raw = "".join(r.get("data", "") for r in json.loads(raw))
            except (ValueError, AttributeError, TypeError):
                pass
            raise TrainError(explain_kaggle_log(raw), raw[-3000:])
        if time.time() - t0 > MAX_WAIT_S:
            raise RuntimeError("Kaggle 訓練超過 5 小時還沒好，先停下來")

    wait_if_paused("Kaggle 練好了，按繼續才會下載新模型")
    step(4, "下載新模型")
    out = WORK / "out"
    kaggle("kernels", "output", k_id, "-p", str(out), timeout=900)
    mzip = out / "hamster_ncnn_model.zip"
    if not mzip.exists():
        raise RuntimeError("Kaggle 沒有產生模型檔")
    res = json.loads((out / "result.json").read_text()) if (out / "result.json").exists() else {}
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    cand = MODELS / f"hamster_ncnn_model.bak-{stamp}"   # until decided it sits with the old models
    with zipfile.ZipFile(mzip) as z:
        z.extractall(WORK / "cand")
    params = list((WORK / "cand").rglob("model.onnx"))
    if not params:
        raise RuntimeError("下載的模型裡沒有 model.onnx")
    shutil.copytree(params[0].parent, cand)
    (cand / "autotrain.json").write_text(json.dumps({"trained": stamp, **res}, ensure_ascii=False))

    bench = "--no-bench" not in sys.argv
    wait_if_paused("新模型下載好了，按繼續才會跑分、決定換不換")
    step(5, "跑分：新模型 vs 現在的模型" if bench else "不跑分（你選了只訓練）")
    exam_done = sum(1 for j in EXAM.glob("*.json") if j.name != "results.json" and json.loads(j.read_text()).get("done"))
    verdict, keep_new, cur_s, new_s = None, False, None, None
    if bench and exam_done >= MIN_EXAM:
        subprocess.run([sys.executable, str(BASE / "exam_run.py"), "models/hamster_ncnn_model", f"models/{cand.name}"],
                       cwd=BASE, timeout=1800)
        r = json.loads((EXAM / "results.json").read_text())
        by = {m["dir"]: m["overall"] for m in r["models"]}
        score = lambda o: (o.get("found") or 0) - (o.get("false") or 0) + 0.25 * (o.get("head") or 0)   # noqa: E731
        cur_o, new_o = by.get("hamster_ncnn_model", {}), by.get(cand.name, {})
        cur_s, new_s = round(score(cur_o), 1), round(score(new_o), 1)
        keep_new = new_s > cur_s
        verdict = {"current": cur_o, "new": new_o}
    step(6, "決定要不要換")
    if keep_new:
        r = subprocess.run([sys.executable, str(BASE / "install_model.py"), str(mzip)], cwd=BASE,
                           capture_output=True, text=True, timeout=600)
        if r.returncode != 0:
            raise RuntimeError("新模型安裝失敗：" + (r.stderr or r.stdout)[-300:])
        shutil.rmtree(cand, ignore_errors=True)   # installed: it is the model in use now
        msg = f"新模型比較好（分數 {new_s} vs {cur_s}），已經換上。舊的留在「管理模型」。"
    elif not bench:
        msg = "訓練完成（這次選了不跑分）。新模型放在「管理模型」，可以到「跑分」比較後再手動換。"
    elif verdict is None:
        msg = f"跑分照片只有 {exam_done} 張（至少 {MIN_EXAM} 張才能自動比較），所以沒有自動換。新模型放在「管理模型」，可以手動換。"
    else:
        msg = f"新模型沒有比較好（分數 {new_s} vs {cur_s}），保留現在的。新模型放在「管理模型」，可以手動換。"
    save(state="done", step=6, text=msg, result={"switched": keep_new, "exam": verdict, "score_new": new_s,
                                               "score_current": cur_s, "train": res, "model": cand.name})
    notify("倉鼠 AI 自動訓練完成", msg)
    shutil.rmtree(WORK, ignore_errors=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:   # noqa: BLE001 - shown on the dashboard
        msg = str(e).splitlines()[0][:160] if str(e) else "不明原因"
        save(state="error", error=msg, text="失敗：" + msg, detail=getattr(e, "detail", "") or str(e)[-3000:])
        notify("倉鼠 AI 自動訓練失敗", msg)
        raise
