#!/usr/bin/env python3
"""跑分: the numbers for a competition report, from data the owner checked by hand (never used for training).

Layer 1  model accuracy on the 跑分照片 (data/exam, owner's boxes):
         mAP50, mAP50-95 (COCO 101-point AP), precision / recall / F1 at the thresholds the tracker really uses,
         for the body and the head (a 1-class model has no head: shown as not applicable, compare the body only),
         and in the fixed brightness groups 夠亮 / 偏暗 / 太暗 (photo + hamster counts; no lux, it was not measured)
Layer 2  on the Pi: model inference ms per picture, model size; the running tracker's full processing delay,
         real AI analyses per second (not the camera fps), CPU and memory
Layer 3  the whole system on the 跑分影片 (data/bench/clips, owner's timeline of what really happened):
         drinking / eating / running events paired one-to-one (P / R / F1, counts, duration error),
         area entries / stays (進出／停留), false alarms per hour of tested video, and the wheel turns the system measures on the clip vs the turns the owner counted
Accuracy vs darkness: real photos by brightness, plus the same bright photos darkened step by step (synthetic,
with camera-like noise) to see where it breaks.
Writes data/bench/results.json; progress in /run/hamster/bench.json.
usage: .venv/bin/python bench_run.py models/hamster_model [models/hamster_model.bak-...]
"""
import json
import math
import os
import sqlite3
import statistics
import sys
import time
import tomllib
from datetime import datetime, timedelta
from pathlib import Path

import cv2
import numpy as np

BASE = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE))
import darkness  # noqa: E402
from yolo_detector import YoloOnnx  # noqa: E402

DATA = BASE / "data"
EXAM, BENCH = DATA / "exam", DATA / "bench"
PROGRESS = Path("/run/hamster/bench.json")
BINS = [0, 8, 15, 25, 35, 50, 75, 256]           # brightness groups (darkness.py: < 15 dark, < 35 dim)
FACTORS = [1, .8, .6, .45, .35, .27, .2, .15, .11, .08, .06, .04]   # synthetic darkening steps


def progress(**kw):
    try:
        PROGRESS.parent.mkdir(parents=True, exist_ok=True)
        tmp = PROGRESS.with_suffix(".tmp")
        tmp.write_text(json.dumps({**kw, "updated": time.time()}, ensure_ascii=False))
        os.replace(tmp, PROGRESS)
    except OSError:
        pass


def xywh(c):
    return [c[0] - c[2] / 2, c[1] - c[3] / 2, c[2], c[3]]


def iou(a, b):
    ix = max(0.0, min(a[0] + a[2], b[0] + b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[1] + a[3], b[1] + b[3]) - max(a[1], b[1]))
    inter = ix * iy
    u = a[2] * a[3] + b[2] * b[3] - inter
    return inter / u if u > 0 else 0.0


def ap(dets, gts, thr):
    """COCO-style 101-point AP. dets: [(img, conf, box)], gts: {img: [box]}."""
    npos = sum(len(v) for v in gts.values())
    if not npos:
        return None, [], []
    used = {k: [False] * len(v) for k, v in gts.items()}
    tp = []
    for img, _c, b in sorted(dets, key=lambda d: -d[1]):
        g = gts.get(img, [])
        best, bi = 0.0, -1
        for i, gb in enumerate(g):
            o = iou(b, gb)
            if o > best and not used[img][i]:
                best, bi = o, i
        if best >= thr:
            used[img][bi] = True
            tp.append(1)
        else:
            tp.append(0)
    ctp = np.cumsum(tp)
    rec = ctp / npos
    prec = ctp / np.arange(1, len(tp) + 1) if tp else np.array([])
    pts = []
    for r in np.linspace(0, 1, 101):
        m = prec[rec >= r] if len(rec) else []
        pts.append(float(m.max()) if len(m) else 0.0)
    return float(np.mean(pts)), rec.tolist(), prec.tolist()


def at_threshold(dets, gts, conf, thr=0.5):
    """Precision / recall / F1 with the tracker's own confidence threshold."""
    tp = fp = 0
    used = {k: [False] * len(v) for k, v in gts.items()}
    for img, c, b in sorted(dets, key=lambda d: -d[1]):
        if c < conf:
            continue
        g = gts.get(img, [])
        best, bi = 0.0, -1
        for i, gb in enumerate(g):
            o = iou(b, gb)
            if o > best and not used[img][i]:
                best, bi = o, i
        if best >= thr:
            used[img][bi] = True
            tp += 1
        else:
            fp += 1
    npos = sum(len(v) for v in gts.values())
    p = tp / (tp + fp) if tp + fp else None
    r = tp / npos if npos else None
    f1 = 2 * p * r / (p + r) if p and r else (0.0 if p is not None and r is not None else None)
    return {"p": p, "r": r, "f1": f1, "tp": tp, "fp": fp, "fn": npos - tp, "n_gt": npos}


def pct(v):
    return None if v is None else round(100 * v, 1)


def model_label(d):
    """version N (oldest = 1), or the owner's own name from 管理模型 → 改名."""
    import modelnames
    return modelnames.label(d)


def photos():
    out = []
    for j in sorted(EXAM.glob("*.json")):
        if j.name == "results.json":
            continue
        try:
            a = json.loads(j.read_text())
        except (OSError, ValueError):
            continue
        img = EXAM / (j.stem + ".jpg")
        if a.get("done") and img.exists():
            out.append((img, a))
    return out


def place_rule(cfg, W=640, H=480):
    """The tracker's own place rule for one box -> (area, bottle): the area the box's centre is in (the drawn
    areas), and the bottle whose spout the box (a bit bigger) covers or whose circle it touches. The same function
    is run on the owner's box and on the model's box, so the answer needs no extra labelling and is not the rule
    marking itself: only the box can differ."""
    import tracker as T
    from zonedefs import zone_ids, zone_types
    zones, types = T.load_zones(cfg, W, H), zone_types(cfg)
    spouts, spot = {}, {}
    sp = cfg.get("spout")
    if isinstance(sp, list) and len(sp) == 2 and "water" in zone_ids(cfg):
        spouts["water"] = (sp[0] * W, sp[1] * H)
        spot["water"] = T.drink_spot(spouts["water"], W, H, cfg.get("drink_spot"))
    for z in cfg.get("zones") or []:
        s = z.get("spout")
        if types.get(z.get("kind")) == "drink" and z.get("kind") != "water" and isinstance(s, list) and len(s) == 2:
            spouts[z["kind"]] = (s[0] * W, s[1] * H)
            spot[z["kind"]] = T.drink_spot(spouts[z["kind"]], W, H, z.get("spot"))
    return lambda b: (T.zone_of_box(zones, b, W, H, {}), T.bottle_of_box(b, W, H, spot, spouts))


def place_score(items, per, place, y_th):
    """位置正確率 on the 跑分照片 with a hamster: the place worked out from the model's most sure box (>= the
    tracker's threshold) vs from the owner's box. ok = same area AND same bottle (or none). Not found = wrong."""
    n = ok = found = 0
    by, wrong = {}, []
    for img, a in items:
        gb = [xywh(b) for b in a.get("boxes") or []]
        if not gb:
            continue
        g = max(gb, key=lambda b: b[2] * b[3])
        gz, gbt = place(g)
        p = per.get(img.name) or {}
        mb = p.get("box") if (p.get("conf") or 0) >= y_th else None
        mz, mbt = place(mb) if mb else (None, None)
        hit = mb is not None and gz == mz and gbt == mbt
        n += 1; ok += hit; found += mb is not None
        k = gz or "-"
        d = by.setdefault(k, {"n": 0, "ok": 0})
        d["n"] += 1; d["ok"] += hit
        if gbt:
            d = by.setdefault("bottle:" + gbt, {"n": 0, "ok": 0})
            d["n"] += 1; d["ok"] += mb is not None and gbt == mbt
        p["place"] = {"gt": [gz, gbt], "ai": [mz, mbt] if mb else None, "ok": hit}
        if not hit:
            wrong.append([img.name, [gz, gbt], [mz, mbt] if mb else None])
    if not n:
        return None
    return {"n": n, "ok": ok, "acc": pct(ok / n), "found": found, "acc_found": pct(ok / found) if found else None,
            "zones": {k: {**v, "acc": pct(v["ok"] / v["n"])} for k, v in by.items()}, "wrong": wrong[:40]}


def layer1(det, items, cfg, db, synth, step):
    y_th, head_th = cfg.get("yolo_conf", 0.45), cfg.get("head_conf", 0.35)
    gt = {0: {}, 1: {}}
    dets = {0: [], 1: []}
    per, ms = {}, []
    bright = {}
    for k, (img, a) in enumerate(items):
        f = cv2.imread(str(img))
        if f is None:
            continue
        n = img.name
        gt[0][n] = [xywh(b) for b in a.get("boxes") or []]
        gt[1][n] = [xywh(b) for b in a.get("heads") or []]
        t0 = time.perf_counter()
        found = det.detect(f)
        ms.append((time.perf_counter() - t0) * 1000)
        for x, y, w, h, s, c in found:
            if c in (0, 1):
                dets[c].append((n, float(s), [x, y, w, h]))
        b = a.get("bright")
        if b is None:
            b = float(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY).mean())
        bright[n] = b
        body = sorted([d for d in found if d[5] == 0], key=lambda d: -d[4])
        per[n] = {"conf": round(float(body[0][4]), 3) if body else 0.0,
                  "box": [round(float(v), 4) for v in body[0][:4]] if body else None, "bright": round(b, 1)}
        if k % 5 == 0:
            step(k, len(items))
    res = {"ms": round(statistics.median(ms), 1) if ms else None}
    for c, name, th in ((0, "body", y_th), (1, "head", head_th)):
        if c == 1 and getattr(det, "nc", 2) < 2:
            res[name] = None          # this model has no head class: not comparable
            continue
        a50, rec, prec = ap(dets[c], gt[c], 0.5)
        aps = [ap(dets[c], gt[c], t)[0] for t in np.arange(0.5, 0.96, 0.05)]
        op = at_threshold(dets[c], gt[c], th)
        res[name] = {"map50": pct(a50), "map": pct(float(np.mean(aps))) if a50 is not None else None,
                     "p": pct(op["p"]), "r": pct(op["r"]), "f1": pct(op["f1"]), "n_gt": op["n_gt"],
                     "tp": op["tp"], "fp": op["fp"], "fn": op["fn"], "threshold": th,
                     "pr": [[round(r, 3), round(p, 3)] for r, p in zip(rec[::max(1, len(rec) // 60)], prec[::max(1, len(prec) // 60)])]}
    # accuracy vs darkness. Fixed groups (darkness.py: brightness < 15 太暗, < 35 偏暗, else 夠亮), each with how
    # many photos / hamsters it holds; plus finer steps for the curve. No lux: the camera's number is an estimate.
    def _grp(names):
        sub_gt = {n: gt[0][n] for n in names}
        op = at_threshold([d for d in dets[0] if d[0] in names], sub_gt, y_th)
        return {"n": len(names), "n_gt": op["n_gt"], "p": pct(op["p"]), "r": pct(op["r"]), "f1": pct(op["f1"]),
                "tp": op["tp"], "fp": op["fp"], "fn": op["fn"]}
    groups = {}
    for lv in ("ok", "dim", "dark"):
        groups[lv] = _grp([n for n, b in bright.items() if darkness.level(b) == lv])
    bins = []
    for lo, hi in zip(BINS, BINS[1:]):
        names = [n for n, b in bright.items() if lo <= b < hi]
        bins.append({"lo": lo, "hi": hi, **(_grp(names) if names else {"n": 0, "n_gt": 0})})
    cutoff = None   # the darkest step (>= 5 hamsters) still finding >= 80 %, before one that doesn't
    for b in reversed(bins):
        if b.get("n_gt", 0) >= 5 and b.get("r") is not None:
            if b["r"] >= 80:
                cutoff = b
            else:
                break
    res["dark"] = {"groups": groups, "bins": bins, "cutoff": {"lo": cutoff["lo"]} if cutoff else None}
    try:
        res["place"] = place_score(items, per, place_rule(cfg), y_th)
    except Exception as e:   # noqa: BLE001 - no areas drawn / old config: the rest of 跑分 still counts
        res["place"] = {"error": str(e)[:200]}
    # synthetic darkening of bright photos with a hamster
    if synth:
        rng = np.random.default_rng(0)
        pick = [(img, a) for img, a in items if a.get("boxes") and bright.get(img.name, 0) >= 35][:40]
        sweep = []
        for f_ in FACTORS:
            sg, sd, br = {}, [], []
            for img, a in pick:
                f = cv2.imread(str(img))
                if f is None:
                    continue
                noise = rng.normal(0, 1.5 + 7 * (1 - f_), f.shape)   # dark = more gain = more grain
                g = np.clip(f.astype(np.float32) * f_ + noise, 0, 255).astype(np.uint8)
                br.append(float(cv2.cvtColor(g, cv2.COLOR_BGR2GRAY).mean()))
                sg[img.name] = [xywh(b) for b in a["boxes"]]
                sd += [(img.name, float(s), [x, y, w, h]) for x, y, w, h, s, c in det.detect(g) if c == 0]
            if not sg:
                break
            op = at_threshold(sd, sg, y_th)
            sweep.append({"f": f_, "bright": round(float(np.mean(br)), 1), "r": pct(op["r"]), "p": pct(op["p"]), "n": len(sg)})
        res["synth"] = sweep
    res["per"] = per
    return res


def segments_abs(info, lab):
    st = datetime.fromisoformat(info["start"])
    return st, st + timedelta(seconds=float(info.get("seconds") or 0)), [
        (s["t"], st + timedelta(seconds=float(s["s"])), st + timedelta(seconds=float(s["e"])))
        for s in lab.get("segments") or [] if float(s["e"]) > float(s["s"])]


def overlap(a0, a1, b0, b1):
    return max(0.0, (min(a1, b1) - max(a0, b0)).total_seconds())


def match(g, s):
    """One-to-one pairing of the owner's events g and the system's events s ([(start, end)]): every pair that
    overlaps in time is a candidate; the pair with the longest overlap is taken first, then the next one among
    events not used yet. Returns the pairs [(gi, si)]."""
    cand = sorted(((overlap(a, b, c, d), i, j) for i, (a, b) in enumerate(g) for j, (c, d) in enumerate(s)), reverse=True)
    ug, us, pairs = set(), set(), []
    for o, i, j in cand:
        if o > 0 and i not in ug and j not in us:
            ug.add(i)
            us.add(j)
            pairs.append((i, j))
    return pairs


def score(g, s):
    pairs = match(g, s)
    tp, n_g, n_s = len(pairs), len(g), len(s)
    p = tp / n_s if n_s else None
    r = tp / n_g if n_g else None
    f1 = 2 * tp / (n_g + n_s) if n_g + n_s else None   # = 2PR/(P+R); 0 when nothing matched
    dur = lambda x: (x[1] - x[0]).total_seconds()  # noqa: E731
    errs = [dur(s[j]) - dur(g[i]) for i, j in pairs]
    return {"gt": n_g, "sys": n_s, "tp": tp, "fp": n_s - tp, "fn": n_g - tp, "p": pct(p), "r": pct(r), "f1": pct(f1),
            "gt_s": round(sum(map(dur, g)), 1), "sys_s": round(sum(map(dur, s)), 1),
            "dur_err_s": round(sum(abs(e) for e in errs) / len(errs), 1) if errs else None,
            "dur_bias_s": round(sum(errs) / len(errs), 1) if errs else None}


def layer3(cfg):
    clips = []
    for lj in sorted((BENCH / "clips").glob("*.lab.json")) if (BENCH / "clips").exists() else []:
        try:
            lab = json.loads(lj.read_text())
            info = json.loads((lj.parent / lj.name.replace(".lab.json", ".json")).read_text())
        except (OSError, ValueError):
            continue
        if lab.get("checked"):
            clips.append((lj.name.replace(".lab.json", ""), info, lab))
    out = {"clips": len(clips), "hours": 0.0, "behavior": {}, "zones": {}, "wheel": [], "false_per_hour": None}
    if not clips:
        return out
    db = sqlite3.connect(DATA / "hamster.db")
    kinds = {"drink": "喝水", "eat": "吃東西", "run": "跑滾輪"}
    G = {k: [] for k in kinds}
    S = {k: [] for k in kinds}
    ZG, ZS = {}, {}
    zone_clip_s = 0.0
    total_s = 0.0
    for name, info, lab in clips:
        w0, w1, segs = segments_abs(info, lab)
        total_s += (w1 - w0).total_seconds()
        q0, q1 = (w0 - timedelta(hours=1)).isoformat(timespec="seconds"), w1.isoformat(timespec="seconds")
        ev = [(k, datetime.fromisoformat(s), datetime.fromisoformat(s) + timedelta(seconds=sec))
              for s, k, sec in db.execute("SELECT start, kind, seconds FROM behavior_events WHERE start>=? AND start<=?", (q0, q1))]
        for k in kinds:
            G[k] += [(a, b) for t_, a, b in segs if t_ == k]
            S[k] += [(max(a, w0), min(b, w1)) for t_, a, b in ev if t_ == k and b > w0 and a < w1]
        if any(t_.startswith("zone:") for t_, _, _ in segs):   # only clips where the owner marked the places
            zone_clip_s += (w1 - w0).total_seconds()
            zv = [(z, datetime.fromisoformat(s), datetime.fromisoformat(s) + timedelta(seconds=sec))
                  for s, z, sec in db.execute("SELECT start, zone, seconds FROM zone_visits WHERE start>=? AND start<=?", (q0, q1))]
            for t_, a, b in segs:
                if t_.startswith("zone:"):
                    ZG.setdefault(t_[5:], []).append((a, b))
            for z, a, b in zv:
                if b > w0 and a < w1:
                    ZS.setdefault(z, []).append((max(a, w0), min(b, w1)))
        if lab.get("turns") not in (None, ""):
            out["wheel"].append({"clip": name, "manual": float(lab["turns"]), "system": wheel_replay(name, cfg)})
    out["hours"] = round(total_s / 3600, 3)
    fp_total = 0
    for k, nm in kinds.items():
        sc = score(G[k], S[k])
        fp_total += sc["fp"]
        out["behavior"][k] = {**sc, "name": nm}
    out["false_per_hour"] = round(fp_total / (total_s / 3600), 2) if total_s else None
    out["zone_hours"] = round(zone_clip_s / 3600, 3)
    for z in sorted(set(ZG) | set(ZS)):
        sc = score(ZG.get(z, []), ZS.get(z, []))
        sc["stay_err"] = pct((sc["sys_s"] - sc["gt_s"]) / sc["gt_s"]) if sc["gt_s"] else None
        out["zones"][z] = sc
    circ = math.pi * float(cfg.get("wheel_diameter_cm", 21))
    out["wheel_circ_cm"] = round(circ, 1)
    for w in out["wheel"]:
        w["err"] = pct((w["system"] - w["manual"]) / w["manual"]) if w["manual"] and w["system"] is not None else None
        w["manual_m"] = round(w["manual"] * circ / 100, 2)
        w["system_m"] = round(w["system"] * circ / 100, 2) if w["system"] is not None else None
    return out


def layer2_live():
    """The tracker as it really runs: per-frame processing, AI inference, real AI analyses per second, camera
    frames per second (not the same thing), CPU and memory of the tracker process."""
    out = {"measured_at": datetime.now().isoformat(timespec="seconds")}
    try:
        tr = json.loads((DATA / "tracker.json").read_text())
        out.update(tr.get("perf") or {})
    except (OSError, ValueError):
        pass
    try:
        import subprocess
        pid = int(subprocess.run(["systemctl", "show", "-p", "MainPID", "--value", "hamster-tracker"],
                                 capture_output=True, text=True, timeout=10).stdout.strip() or 0)
        if pid:
            def ticks():
                v = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
                return int(v[11]) + int(v[12])
            t0, c0 = time.time(), ticks()
            time.sleep(3)
            cpu = (ticks() - c0) / os.sysconf("SC_CLK_TCK") / (time.time() - t0) * 100
            rss = next(int(l.split()[1]) for l in Path(f"/proc/{pid}/status").read_text().splitlines() if l.startswith("VmRSS"))
            mem = {l.split(":")[0]: int(l.split()[1]) for l in Path("/proc/meminfo").read_text().splitlines()[:3]}
            out.update(cpu_pct=round(cpu, 1), cores=os.cpu_count(), rss_mb=round(rss / 1024, 1),
                       mem_total_mb=round(mem.get("MemTotal", 0) / 1024))
    except Exception:   # noqa: BLE001 - optional numbers
        pass
    return out


def wheel_replay(name, cfg):
    """The wheel turns the tracker's own method measures on this clip (same strips, same rules)."""
    import tracker as T
    from zonedefs import zone_types
    mp4 = BENCH / "clips" / f"{name}.mp4"
    cap = cv2.VideoCapture(str(mp4))
    W, H = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 640, int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 480
    fps = cap.get(cv2.CAP_PROP_FPS) or 10
    zones, types = T.load_zones(cfg, W, H), zone_types(cfg)
    run = [p for k, p in zones if types.get(k) == "run"]
    if not run:
        return None
    allp = np.concatenate(run)
    x0, y0 = max(0, int(allp[:, 0].min())), max(0, int(allp[:, 1].min()))
    x1, y1 = min(W, int(allp[:, 0].max()) + 1), min(H, int(allp[:, 1].max()) + 1)
    m = np.zeros((y1 - y0, x1 - x0), np.uint8)
    for p in run:
        cv2.fillPoly(m, [np.int32(p - [x0, y0])], 1)
    m = m.astype(bool)
    cols = [(int(c), m[:, int(c)]) for c in np.linspace(0.15, 0.85, 9) * m.shape[1] if m[:, int(c)].sum() >= 6]
    strips = max(1, int(cfg.get("wheel_strips", 2)))
    flick_min = cfg.get("wheel_flick_share", 0.005)
    hist, buf, share, turns, i = [], [], [], 0.0, 0
    while True:
        ok, f = cap.read()
        if not ok:
            break
        t = i / fps
        i += 1
        wc = cv2.cvtColor(f[y0:y1, x0:x1], cv2.COLOR_BGR2GRAY)
        if len(hist) == 2:
            th = min(25.0, max(8.0, 0.4 * float(wc[m].mean())))
            share.append(float((cv2.absdiff(wc, hist[0])[m] > th).mean()))
        hist = (hist + [wc])[-2:]
        buf = (buf + [(t, np.array([float(wc[mm, c].mean()) for c, mm in cols], np.float32))])[-90:]
        if i % max(1, int(round(fps))) == 0:
            s = float(np.mean(share)) if share else 0.0
            share = []
            rps, _ = T.wheel_turns(buf, t, strips)
            if s >= flick_min:
                turns += rps
    return round(turns, 2)


def main():
    cfg = tomllib.load(open(BASE / "config.toml", "rb")).get("tracker", {})
    dirs = [BASE / m for m in sys.argv[1:] if (BASE / m / "model.onnx").exists()]
    items = photos()
    out = {"time": datetime.now().isoformat(timespec="seconds"), "photos": len(items), "models": []}
    db = sqlite3.connect(DATA / "hamster.db")
    total = len(items) * len(dirs) + (len(FACTORS) * 40 if items else 0)
    progress(running=True, text="量樹莓派實際運作的速度和資源（先量，跑分本身也會吃 CPU）", done=0, total=total)
    out["live"] = layer2_live()   # before the benchmark loads the CPU itself
    done = [0]
    for k, d in enumerate(dirs):
        det = YoloOnnx(d, conf=0.01, threads=2)

        def step(i, n, k=k):
            progress(running=True, text=f"第 {k + 1}/{len(dirs)} 個模型：照片 {i}/{n}", done=done[0] + i, total=total)
        r = layer1(det, items, cfg, db, synth=(k == 0), step=step) if items else {}
        done[0] += len(items)
        size = sum(p.stat().st_size for p in d.glob("model.onnx"))
        out["models"].append({"dir": d.name, "label": model_label(d), "size_mb": round(size / 1e6, 1),
                              "imgsz": det.imgsz, "nc": det.nc, **r})
    progress(running=True, text="影片：行為、停留與滾輪", done=total, total=total)
    out["system"] = layer3(cfg)
    BENCH.mkdir(parents=True, exist_ok=True)
    tmp = BENCH / "results.json.tmp"
    tmp.write_text(json.dumps(out, ensure_ascii=False))
    os.replace(tmp, BENCH / "results.json")
    progress(running=False, text="完成", done=total, total=total)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:   # noqa: BLE001 - shown on the page
        progress(running=False, error=str(e)[:300])
        raise
