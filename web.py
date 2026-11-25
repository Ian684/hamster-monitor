#!/usr/bin/env python3
"""Hamster dashboard: status, charts, camera snapshot / live MJPEG, heatmap, manual fan control."""
import ipaddress
import json
import os
import re
import shutil
import sqlite3
import subprocess
import sys
import tempfile
import threading
import time
import tomllib
from datetime import datetime, timedelta
from functools import wraps
from pathlib import Path

from flask import Flask, Response, jsonify, request, send_file, send_from_directory
from werkzeug.security import check_password_hash

import alerts
import darkness
import zonedefs

BASE = Path(__file__).resolve().parent
DATA = BASE / "data"
LIVE_DIR = Path("/run/hamster")
LIVE_FRAME = LIVE_DIR / "live.jpg"
LIVE_REQUEST = LIVE_DIR / "live.request"
OVERRIDE_FILE = DATA / "fan_override.json"
PAUSE_FILE = DATA / "pause.json"   # read by tracker.py: don't record while hands are in the cage
SERVICES = ["hamster-climate", "hamster-tracker", "hamster-web", "hamster-aim", "hamster-alerts"]

CONFIG_FILE = BASE / "config.toml"
with open(CONFIG_FILE, "rb") as f:
    CFG = tomllib.load(f)
WEB = CFG.get("web", {})
_cfg_mtime = CONFIG_FILE.stat().st_mtime

# Each MJPEG viewer holds one server thread for as long as it watches, so cap them
# and keep spare threads for normal requests (status polling, charts, buttons).
THREADS = 16
MAX_STREAMS = 6
STREAM_MAX_S = 10 * 60      # one viewer connection at most 10 min; the page reconnects by itself
STREAM_IDLE_S = 8           # no new frame for this long -> end the connection instead of idling
_streams = threading.BoundedSemaphore(MAX_STREAMS)

app = Flask(__name__, static_folder=str(BASE / "static"))

# ---------- make the page light on slow / lossy connections ----------
_GZ_TYPES = ("text/html", "text/css", "text/javascript", "application/javascript", "application/json", "application/manifest+json")
_gz_cache = {}   # (path, etag) -> compressed bytes, for files that never change between requests


@app.after_request
def slim(resp):
    """gzip text responses (page 108 KB -> ~25 KB, chart library 200 KB -> ~70 KB, chart data 80 KB -> ~8 KB)
    and let browsers keep the static files, so a weak link has far less to carry."""
    try:
        if request.path.startswith("/static/") and resp.status_code == 200:
            resp.headers["Cache-Control"] = "public, max-age=604800"
        if (resp.status_code != 200 or resp.mimetype not in _GZ_TYPES or "Content-Encoding" in resp.headers
                or "gzip" not in request.headers.get("Accept-Encoding", "") or resp.is_streamed and not resp.direct_passthrough):
            return resp
        import gzip
        key = (request.path, resp.headers.get("ETag")) if resp.direct_passthrough and resp.headers.get("ETag") else None
        data = _gz_cache.get(key) if key else None
        if data is None:
            resp.direct_passthrough = False
            raw = resp.get_data()
            if len(raw) < 1024:
                return resp
            data = gzip.compress(raw, 6)
            if key:
                if len(_gz_cache) > 20:
                    _gz_cache.clear()
                _gz_cache[key] = data
        resp.direct_passthrough = False
        resp.set_data(data)
        resp.headers["Content-Encoding"] = "gzip"
        resp.headers["Content-Length"] = str(len(data))
        resp.headers.add("Vary", "Accept-Encoding")
    except Exception as e:   # noqa: BLE001 - compression must never break a response
        app.logger.warning("gzip skipped for %s: %s", request.path, e)
    return resp


def current_cfg():
    """Re-read config.toml when it changes, so new fan thresholds show without restarting the web service."""
    global CFG, _cfg_mtime
    try:
        m = CONFIG_FILE.stat().st_mtime
        if m != _cfg_mtime:
            with open(CONFIG_FILE, "rb") as f:
                CFG = tomllib.load(f)
            _cfg_mtime = m
    except (OSError, tomllib.TOMLDecodeError):
        pass  # keep the last good config while the file is being edited
    return CFG


def int_arg(value, default, lo, hi):
    """Parse a number from a query/body field; bad input falls back to the default instead of a 500 error."""
    try:
        return min(max(int(value), lo), hi)
    except (TypeError, ValueError):
        return default


def ip_allowed(addr):
    """None = no allow-list configured; True/False = whether addr is on it (IPs or CIDR ranges)."""
    rules = current_cfg().get("web", {}).get("allow_ips", [])
    if not rules:
        return None
    try:
        ip = ipaddress.ip_address(addr)
    except ValueError:
        return False
    if ip.is_loopback:          # the Pi itself (maintenance scripts, local tests)
        return True
    for r in rules:
        try:
            if ip in ipaddress.ip_network(r, strict=False):
                return True
        except ValueError:
            continue
    return False


TUNNEL_HOSTS = (".ngrok-free.app", ".ngrok-free.dev", ".ngrok.app", ".ngrok.dev", ".ngrok.io", ".ts.net")


def via_tunnel():
    """True when the request came in through a public tunnel (ngrok, Tailscale Funnel).
    Tunnels add X-Forwarded-For and keep their own host name; a device on the allow-list talking to the Pi directly
    sends neither. (A device adding the header itself only puts itself behind the tunnel login, so it can't be abused.)"""
    h = request.headers
    host = (h.get("X-Forwarded-Host") or h.get("Host") or "").split(":")[0].lower()
    return bool(h.get("X-Forwarded-For")) or host.endswith(TUNNEL_HOSTS)


def tunnel_role():
    """Who is logged in through the tunnel: "edit" (the owner's tunnel password, [web] tunnel_edit_password_hash),
    "readonly" (the judges' password, [web] judge_password_hash), or None (no / wrong password).
    Both are set on the Pi with set_password.py --edit / --judge."""
    web = current_cfg().get("web", {})
    auth = request.authorization
    pw = (auth.password or "") if auth else ""
    if not pw:
        return None
    key = (web.get("tunnel_edit_password_hash", ""), web.get("judge_password_hash", ""), pw)
    if key not in _role_cache:   # hashing is slow on the Pi and the page asks many times a minute: remember the answer
        role = None
        if key[0] and check_password_hash(key[0], pw):
            role = "edit"
        elif key[1] and check_password_hash(key[1], pw):
            role = "readonly"
        if len(_role_cache) > 64:
            _role_cache.clear()
        _role_cache[key] = role
    return _role_cache[key]


_role_cache = {}


def access_readonly():
    return via_tunnel() and tunnel_role() != "edit"


def require_auth(fn):
    """Access control.
    - through a tunnel (ngrok): HTTP basic auth with one of two passwords (any user name):
      the edit password = everything, the judge password = read-only (anything but GET / HEAD is refused,
      so nothing can be changed or deleted). No password set = the tunnel is closed.
    - [web] allow_ips set: only those devices get in (no password); everyone else gets 403.
    - otherwise: optional HTTP basic auth (any user name), set with: .venv/bin/python set_password.py
    """
    @wraps(fn)
    def wrapper(*a, **kw):
        if via_tunnel():
            web = current_cfg().get("web", {})
            if not (web.get("judge_password_hash") or web.get("tunnel_edit_password_hash")):
                return Response("外部連線尚未開放", 403, {"Content-Type": "text/plain; charset=utf-8"})
            role = tunnel_role()
            if role is None:
                return Response("login required", 401, {"WWW-Authenticate": 'Basic realm="hamster"'})
            if role == "readonly" and request.method not in ("GET", "HEAD", "OPTIONS"):
                return jsonify(ok=False, readonly=True, error="唯讀模式：不能修改或刪除資料"), 403
            return fn(*a, **kw)
        allowed = ip_allowed(request.remote_addr or "")
        if allowed is False:
            return Response("這個裝置沒有權限開啟倉鼠小屋", 403, {"Content-Type": "text/plain; charset=utf-8"})
        if allowed is None:
            pw_hash = current_cfg().get("web", {}).get("password_hash", "")
            if pw_hash:
                auth = request.authorization
                if not auth or not check_password_hash(pw_hash, auth.password or ""):
                    return Response("login required", 401, {"WWW-Authenticate": 'Basic realm="hamster"'})
        return fn(*a, **kw)
    return wrapper


def db():
    con = sqlite3.connect(BASE / current_cfg()["log"]["db"], timeout=10)
    con.row_factory = sqlite3.Row
    return con


def read_json(path):
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return None


def since(hours):
    return (datetime.now() - timedelta(hours=hours)).isoformat(timespec="seconds")


@app.get("/")
@require_auth
def index():
    return send_from_directory(BASE / "templates", "index.html")


@app.get("/api/access")
@require_auth
def access_mode():
    """Read-only (the judges' login through the tunnel) or editable: shown at the top of the page."""
    return no_cache(jsonify(readonly=access_readonly(), tunnel=via_tunnel()))


def _age(p):
    """Seconds since a file was last written (None if it doesn't exist)."""
    try:
        return round(time.time() - p.stat().st_mtime)
    except OSError:
        return None


def _live_minute():
    """Activity of the minute tracker.py is still counting (not in the database until the minute ends)."""
    tr = read_json(DATA / "tracker.json")
    a = _age(DATA / "tracker.json")
    if not tr or a is None or a > 10:
        return 0, 0.0
    return tr.get("minute_active_s") or 0, tr.get("minute_distance_cm") or 0.0


_photo_cache = {"t": 0, "n": None}


def _photo_count():
    """Real number of training photos on disk (the tracker's own counter doesn't go down when photos are deleted)."""
    if time.time() - _photo_cache["t"] > 15:
        _photo_cache.update(t=time.time(), n=sum(1 for _ in (DATA / "dataset").glob("*.jpg")),
                            clips=sum(1 for _ in CLIPS.glob("clip_*.json")) if CLIPS.exists() else 0)
    return _photo_cache["n"]


def _versions(con):
    """Tiny change markers: the page compares them every status poll and reloads only the list that changed
    (notifications, zone visits, eat/drink/run, finished minutes, clips, photos), instead of waiting for a reload."""
    v = {}
    for k, t in (("alerts", "alert_log"), ("visits", "zone_visits"), ("beh", "behavior_events"), ("minute", "activity"),
                 ("climate", "climate")):
        try:
            v[k] = con.execute(f"SELECT COUNT(*) || '/' || COALESCE(MAX(rowid), 0) FROM {t}").fetchone()[0]
        except sqlite3.OperationalError:
            v[k] = None
    try:
        lab_mt = int((DATASET / "labeled.json").stat().st_mtime)
    except OSError:
        lab_mt = 0
    v["photos"] = f"{_photo_count()}/{lab_mt}"
    v["clips"] = _photo_cache.get("clips")
    return v


@app.get("/api/status")
@require_auth
def status():
    services = {}
    for s in SERVICES:
        r = subprocess.run(["systemctl", "is-active", s], capture_output=True, text=True)
        services[s] = r.stdout.strip()
    today = datetime.now().strftime("%Y-%m-%d")
    live_s, live_cm = _live_minute()
    with db() as con:
        try:
            act = con.execute("SELECT COALESCE(SUM(active_s),0) a, COALESCE(SUM(distance_cm),0) d "
                              "FROM activity WHERE minute >= ?", (today,)).fetchone()
            # + the minute being recorded right now, so the number moves live and matches 每日紀錄 / the chart
            activity_today = {"active_min": round((act["a"] + live_s) / 60), "distance_m": round((act["d"] + live_cm) / 100, 1)}
        except sqlite3.OperationalError:  # tracker never ran -> no table yet
            activity_today = {"active_min": 0, "distance_m": 0}
        ver = _versions(con)
    override = read_json(OVERRIDE_FILE)
    if override and float(override.get("until", 0)) <= time.time():
        override = None
    cfg = current_cfg()
    f = cfg["fan"]
    al = cfg.get("alerts", {})
    pause = read_json(PAUSE_FILE)
    pause_left = int(float(pause.get("until", 0)) - time.time()) if pause else 0
    return jsonify(
        pause={"remaining_s": pause_left, "until_text": pause.get("until_text")} if pause_left > 0 else None,
        alerts={"enabled": bool(al.get("enabled") and al.get("topic")), "topic": al.get("topic"),
                "server": al.get("ntfy_server", "https://ntfy.sh"),
                "rules": alert_rules(al)},
        climate=read_json(DATA / "status.json"),
        tracker=read_json(DATA / "tracker.json") if services["hamster-tracker"] == "active" else None,
        aim=read_json(LIVE_DIR / "aim.json") if services["hamster-aim"] == "active" else None,
        override=override,
        thresholds={k: f[k] for k in ("temp_on", "temp_off", "hum_on", "hum_off")},
        activity_today=activity_today,
        services=services,
        # how old each source is, measured here on the Pi (the phone's clock may be off): the page greys out
        # anything that stopped updating instead of showing an old number as if it were current
        ages={"climate_s": _age(DATA / "status.json"), "tracker_s": _age(DATA / "tracker.json"),
              "snapshot_s": _age(DATA / "latest.jpg")},
        photos=_photo_count(),
        ver=ver,
        now=datetime.now().isoformat(timespec="seconds"),
    )


@app.get("/api/climate")
@require_auth
def climate():
    hours = int_arg(request.args.get("hours"), 24, 1, 24 * 31)
    # raw per-minute rows up to a day, then 10-minute / hourly averages to keep charts light
    width = 16 if hours <= 24 else (15 if hours <= 24 * 7 else 13)
    with db() as con:
        rows = con.execute(
            f"SELECT substr(ts,1,{width}) AS t, ROUND(AVG(temp),1) AS temp, ROUND(AVG(hum)) AS hum, "
            f"MAX(fan) AS fan FROM climate WHERE ts >= ? GROUP BY t ORDER BY t", (since(hours),)).fetchall()
    return jsonify([dict(r) for r in rows])


@app.get("/api/activity")
@require_auth
def activity():
    hours = int_arg(request.args.get("hours"), 24, 1, 24 * 31)
    with db() as con:
        try:
            rows = con.execute(
                "SELECT substr(minute,1,13) AS hour, SUM(active_s) AS active_s, ROUND(SUM(distance_cm)) AS distance_cm "
                "FROM activity WHERE minute >= ? GROUP BY hour ORDER BY hour", (since(hours)[:16],)).fetchall()
        except sqlite3.OperationalError:
            rows = []
    rows = [dict(r) for r in rows]
    live_s, live_cm = _live_minute()   # add the minute being recorded to the current hour (same total as the home page)
    if live_s or live_cm:
        hour = datetime.now().strftime("%Y-%m-%dT%H")
        cur = next((r for r in rows if r["hour"] == hour), None)
        if cur is None:
            cur = {"hour": hour, "active_s": 0, "distance_cm": 0}
            rows.append(cur)
        cur["active_s"] = (cur["active_s"] or 0) + live_s
        cur["distance_cm"] = round((cur["distance_cm"] or 0) + live_cm)
    return jsonify(rows)


@app.post("/api/fan")
@require_auth
def fan():
    body = request.get_json(silent=True) or {}
    mode = body.get("mode")
    if mode == "auto":
        OVERRIDE_FILE.unlink(missing_ok=True)
        return jsonify(ok=True, override=None)
    if mode not in ("on", "off"):
        return jsonify(ok=False, error="mode must be on, off or auto"), 400
    try:
        if float(body.get("minutes", 30)) < 0:
            return jsonify(ok=False, error="minutes must be 0 (hold) or more"), 400
    except (TypeError, ValueError):
        pass   # not a number: int_arg uses the default
    minutes = int_arg(body.get("minutes"), 30, 0, 12 * 60)
    if minutes == 0:  # 0 = stay like this until someone presses "自動"
        override = {"mode": mode, "until": time.time() + 10 * 365 * 86400, "until_text": None, "hold": True}
    else:
        override = {"mode": mode, "until": time.time() + minutes * 60,
                    "until_text": (datetime.now() + timedelta(minutes=minutes)).strftime("%H:%M")}
    tmp = OVERRIDE_FILE.with_suffix(".tmp")
    tmp.write_text(json.dumps(override))
    os.replace(tmp, OVERRIDE_FILE)
    return jsonify(ok=True, override=override)


_ai_cache = {"t": 0, "v": None}


@app.get("/api/ai")
@require_auth
def ai_status():
    """Training-photo counts and model state (cached 30 s: counting files is not free)."""
    if time.time() - _ai_cache["t"] > 30:
        ds = DATA / "dataset"
        t = current_cfg().get("tracker", {})
        model = BASE / t.get("yolo_model", "models/hamster_ncnn_model")
        pack = DATA / "export" / "hamster_dataset.zip"
        _ai_cache["v"] = {
            "detector": t.get("detector", "motion"),
            "model_installed": (model / "model.ncnn.param").exists(),
            "motion_photos": len(list(ds.glob("motion_*.jpg"))),
            "idle_photos": len(list(ds.glob("idle_*.jpg"))),
            "prelabeled": len(list(ds.glob("motion_*.txt"))),
            "hard_photos": len(list(ds.glob("hard_*.jpg"))),
            "dataset_max": t.get("dataset_max", 3000),
            "pack": {"mb": round(pack.stat().st_size / 1e6, 1),
                     "at": datetime.fromtimestamp(pack.stat().st_mtime).strftime("%m/%d %H:%M")} if pack.exists() else None,
            "last24": None,
        }
        try:   # labelling progress: what is done, what is worth doing next, day / night balance
            labd, allp = _labeled(), _photos()
            trained = {q.name for q in (DATA / "trained").glob("*.jpg")} if (DATA / "trained").exists() else set()
            scores, moved, (y_th, low) = _ai_scores(), _photo_moved(), _class_th()
            br = _photo_bright()
            lab_dark, todo = {"ok": 0, "dim": 0, "dark": 0}, {"unsure": 0, "none": 0, "seen": 0, "pending": 0, "dark": 0}
            names = {q.name for q in allp}
            for q in allp:
                lv = darkness.level(br.get(q.name))
                if q.name in labd:
                    lab_dark[lv] = lab_dark.get(lv, 0) + 1
                    continue
                k = _photo_class(q.name, scores, moved, y_th, low).split("_")[0]
                todo[k] = todo.get(k, 0) + 1
                if lv == "dark":
                    todo["dark"] += 1
            ban = _banned(list(names))   # 禁止: inside a 跑分 video, never trained on: counted apart, not as 新標／未標
            labelled = len([n for n in labd if n in names and n not in ban] + [n for n in trained if n not in labd])
            _ai_cache["v"]["labels"] = {"labeled": labelled,
                                        "new": len([n for n in labd if n in names and n not in trained and n not in ban]),
                                        "trained": len(trained), "by_dark": lab_dark, "todo": todo, "banned": len(ban),
                                        "unlabeled": sum(1 for q in allp if q.name not in labd and q.name not in ban),
                                        "first_goal": 30, "next_goal": 10}
        except Exception:   # noqa: BLE001 - the panel works without it
            _ai_cache["v"]["labels"] = None
        try:   # how the AI did over the last 24 hours (recorded by tracker.py once a minute)
            with db() as con:
                r = con.execute("SELECT SUM(runs) runs, SUM(seen) seen, SUM(unsure) unsure, SUM(conf_sum) cs, MAX(conf_max) cm "
                                "FROM ai_minutes WHERE minute >= ?", (since(24)[:16],)).fetchone()
            if r and r["runs"]:
                _ai_cache["v"]["last24"] = {"runs": r["runs"], "seen": r["seen"], "unsure": r["unsure"],
                                            "avg_conf": round(r["cs"] / r["seen"], 3) if r["seen"] else None,
                                            "max_conf": round(r["cm"], 3) if r["seen"] else None}
        except sqlite3.OperationalError:
            pass
        _ai_cache["t"] = time.time()
    return jsonify(_ai_cache["v"])


PACK_CONF = ("all", "none", "range")   # AI confidence filter: everything / not recognised / between lo..hi %


def _pack_filter(src):
    """Read the pack filters from a dict (query args or JSON). Returns (filters, error)."""
    kinds = src.get("kinds", list(PHOTO_KINDS))
    if isinstance(kinds, list) and any(k in ("motion", "idle", "hard_unsure", "hard_missed") for k in kinds):
        kinds = list(PHOTO_KINDS)   # a page from before the new classes: everything
    if isinstance(kinds, str):
        kinds = [k for k in kinds.split(",") if k]
    if not isinstance(kinds, list) or not kinds or any(k not in PHOTO_KINDS for k in kinds):
        return None, "至少要選一種照片"
    conf = src.get("conf", "all")
    if conf not in PACK_CONF:
        return None, "信心條件不對"
    try:
        lo, hi = float(src.get("lo", 0)), float(src.get("hi", 100))
    except (TypeError, ValueError):
        return None, "信心範圍要填數字"
    if conf == "range" and not (0 <= lo <= hi <= 100):
        return None, "信心範圍要在 0～100% 之間，而且左邊不能比右邊大"
    dark = src.get("dark", list(darkness.NAMES))
    if isinstance(dark, str):
        dark = [k for k in dark.split(",") if k]
    if not isinstance(dark, list) or not dark or any(k not in darkness.NAMES for k in dark):
        return None, "至少要選一種亮度"
    lab = src.get("lab", list(PACK_LAB))
    if isinstance(lab, str):
        lab = [k for k in lab.split(",") if k]
    if isinstance(lab, list) and "yes" in lab:   # a page from before 已標／新標 were separate
        lab = [k for k in lab if k != "yes"] + ["trained", "new"]
    if not isinstance(lab, list) or not lab or any(k not in PACK_LAB for k in lab):
        return None, "至少要選一種標註狀態"
    srcs = src.get("src", list(PACK_SRC))
    if isinstance(srcs, str):
        srcs = [k for k in srcs.split(",") if k]
    if not isinstance(srcs, list) or not srcs or any(k not in PACK_SRC for k in srcs):
        return None, "至少要選一種來源"
    zones = src.get("zones")   # None = every place; else the zone ids (+ "none" = not in any area)
    if isinstance(zones, str):
        zones = [k for k in zones.split(",") if k]
    if zones is not None and (not isinstance(zones, list) or not zones):
        return None, "至少要選一個地點"
    day = str(src.get("day") or "")
    if day and not re.fullmatch(r"\d{8}", day):
        return None, "日期不對"

    def hm(v):
        m = re.fullmatch(r"(\d{1,2}):?(\d{2})", str(src.get(v) or "").strip())
        return f"{int(m.group(1)):02d}{m.group(2)}" if m and int(m.group(1)) < 24 and int(m.group(2)) < 60 else ""
    q = [x.lower().removesuffix(".jpg") for x in re.split(r"[\s,，、]+", str(src.get("q") or "")) if x][:200]
    return {"kinds": kinds, "conf": conf, "lo": lo, "hi": hi, "dark": dark, "lab": lab, "src": srcs, "zones": zones,
            "day": day, "t0": hm("t0"), "t1": hm("t1"), "q": q}, None


# 來源 = the file name's start (same groups as 訓練照片's filter)
PACK_SRC = {"motion": ("動時拍", ("motion_",)), "idle": ("定時拍", ("idle_",)),
            "hard": ("加強用", ("hard_unsure_", "hard_missed_")), "upload": ("自己加", ("upload_",))}


def _pack_where(f, name, pz):
    """The 來源／地點／日期／時段／檔名 filters for one photo name."""
    if not any(name.startswith(pre) for k in f.get("src", PACK_SRC) for pre in PACK_SRC[k][1]):
        return False
    if f.get("zones") is not None and (pz.get(name) or "none") not in f["zones"]:
        return False
    m = PHOTO_RE.match(name)
    if f.get("day") and (not m or m.group(2) != f["day"]):
        return False
    if (f.get("t0") or f.get("t1")) and m:
        s = m.group(3)[:4]
        a, b = f.get("t0") or "0000", f.get("t1") or "2359"
        if not (a <= s <= b if a <= b else (s >= a or s <= b)):
            return False
    if f.get("q") and not any(x in name.lower() for x in f["q"]):
        return False
    return True


PACK_LAB = {"trained": "已標", "new": "新標", "no": "未標"}


def _lab_states():
    """name -> trained (labelled, a training used it) / new (labelled since) ; others are 未標."""
    trained = {q.name for q in (DATA / "trained").glob("*.jpg")} if (DATA / "trained").exists() else set()
    st = {n: "new" for n in _labeled()}
    st.update({n: "trained" for n in trained})
    return st


_near_c = {"t": 0.0, "v": set()}


def _near_set():
    """Names of the photos next to a 跑分 photo / inside a 跑分 video; kept a few seconds (one preview asks often)."""
    if time.time() - _near_c["t"] > 5:
        ets = _exam_times()
        _near_c["v"] = {p.name for p in _photos() if _near_exam(p.name, ets)}
        _near_c["t"] = time.time()
    return _near_c["v"]


def _pack_select(f, ban_out=None):
    """Photos matching the filters, plus how many could not be judged yet (no AI score from the current model).
    ban_out (a list): gets the 禁止 photos (inside a 跑分 video / next to a 跑分 photo) the kind and brightness
    filters would have taken; they are never packed."""
    scores = _ai_scores()
    y_th, low = _class_th()
    moved = _photo_moved()
    out, unscored = [], 0
    br = _photo_bright() if len(f["dark"]) < len(darkness.NAMES) else {}
    labs = _lab_states()
    pz = _photo_zones()
    near = _near_set()
    for p in _photos():
        if _photo_class(p.name, scores, moved, y_th, low) not in f["kinds"]:
            continue
        if not _pack_where(f, p.name, pz):
            continue
        if p.name in near:   # almost the same picture as an exam photo: the test would be leaked
            if ban_out is not None and labs.get(p.name) != "trained" and \
                    (not br or darkness.level(br.get(p.name)) in f["dark"]):
                ban_out.append(p)
            continue
        if labs.get(p.name, "no") not in f["lab"]:
            continue
        if br and darkness.level(br.get(p.name)) not in f["dark"]:
            continue
        if f["conf"] != "all":
            s = scores.get(p.name)
            if s is None:
                unscored += 1
                continue
            c = s[0] * 100
            if f["conf"] == "none" and c >= y_th * 100:
                continue
            if f["conf"] == "range" and not f["lo"] <= c <= f["hi"]:
                continue
        out.append(p)
    return out, unscored, y_th


def _pack_note(f, y_th):
    kinds = "、".join(PHOTO_KINDS[k] for k in f["kinds"])
    conf = {"all": "", "none": f"，AI 沒認出（信心低於 {y_th * 100:.0f}%）",
            "range": f"，AI 信心 {f['lo']:g}～{f['hi']:g}%"}[f["conf"]]
    dark = "" if len(f["dark"]) == len(darkness.NAMES) else "，" + "、".join(darkness.NAMES[k] for k in f["dark"])
    lab = "" if len(f.get("lab", [])) == len(PACK_LAB) else "，只有" + "、".join(PACK_LAB[k] for k in f["lab"]) + "的"
    more = []
    if len(f.get("src", PACK_SRC)) < len(PACK_SRC):
        more.append("、".join(PACK_SRC[k][0] for k in f["src"]))
    if f.get("zones") is not None:
        more.append("地點：" + "、".join("沒在區域" if z == "none" else z for z in f["zones"]))
    if f.get("day"):
        more.append(f"{f['day'][4:6]}/{f['day'][6:]}")
    if f.get("t0") or f.get("t1"):
        t0, t1 = f.get("t0") or "0000", f.get("t1") or "2359"
        more.append(f"{t0[:2]}:{t0[2:]}～{t1[:2]}:{t1[2:]}")
    if f.get("q"):
        more.append(f"檔名含 {len(f['q'])} 個關鍵字")
    return kinds + conf + dark + lab + "".join("，" + m for m in more)


@app.get("/api/dataset/pack/preview")
@require_auth
def dataset_pack_preview():
    """How many photos the chosen filters would pack (before removing near-duplicates)."""
    f, err = _pack_filter(request.args)
    if err:
        return jsonify(ok=False, error=err), 400
    sel, unscored, y_th = _pack_select(f)
    by_kind = {k: 0 for k in PHOTO_KINDS}
    scores, moved, (y_th2, low) = _ai_scores(), _photo_moved(), _class_th()
    for p in sel:
        by_kind[_photo_class(p.name, scores, moved, y_th2, low)] += 1
    # how many of each brightness the OTHER filters leave (so the chips can show what picking them would give)
    br = _photo_bright()
    sel_all, _, _ = _pack_select({**f, "dark": list(darkness.NAMES)})
    by_dark = {k: 0 for k in darkness.NAMES}
    for p in sel_all:
        lv = darkness.level(br.get(p.name))
        if lv:
            by_dark[lv] += 1
    # like the brightness chips: what each 已標／新標／未標 chip would give with the other filters as they are
    labs = _lab_states()
    ban = []
    sel_lab, _, _ = _pack_select({**f, "lab": list(PACK_LAB)}, ban_out=ban)
    by_lab = {k: sum(labs.get(p.name, "no") == k for p in sel_lab) for k in PACK_LAB}
    by_lab["ban"] = len(ban)
    pz = _photo_zones()
    sel_src, _, _ = _pack_select({**f, "src": list(PACK_SRC)})
    by_src = {k: sum(p.name.startswith(PACK_SRC[k][1]) for p in sel_src) for k in PACK_SRC}
    sel_zone, _, _ = _pack_select({**f, "zones": None})
    by_zone = {}
    for p in sel_zone:
        z = pz.get(p.name) or "none"
        by_zone[z] = by_zone.get(z, 0) + 1
    sel_day, _, _ = _pack_select({**f, "day": ""})
    by_day = {}
    for p in sel_day:
        d = PHOTO_RE.match(p.name).group(2)
        by_day[d] = by_day.get(d, 0) + 1
    return no_cache(jsonify(ok=True, count=len(sel), by_kind=by_kind, by_dark=by_dark, by_lab=by_lab, by_src=by_src, by_zone=by_zone,
                            days=[{"day": d, "label": f"{d[4:6]}/{d[6:]}", "count": n} for d, n in sorted(by_day.items(), reverse=True)],
                            unscored=unscored, threshold=y_th,
                            scoring=_scoring_status(False), note=_pack_note(f, y_th)))


@app.post("/api/dataset/pack")
@require_auth
def dataset_pack():
    body = request.get_json(silent=True) or {}
    f, err = _pack_filter(body)
    if err:
        return jsonify(ok=False, error=err), 400
    sel, _unscored, y_th = _pack_select(f)
    if not sel:
        return jsonify(ok=False, error="沒有符合條件的照片"), 400
    max_n = int_arg(body.get("max"), 600, 10, 3000)
    dedup = body.get("dedup", True) is not False
    labd = _labeled()
    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as nf:
        nf.write("\n".join(p.name for p in sel))
    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as ff:   # labelled first: dedup keeps them
        ff.write("\n".join(p.name for p in sel if p.name in labd))
    try:
        r = subprocess.run([sys.executable, str(BASE / "prepare_dataset.py"), "--names-file", nf.name, "--max", str(max_n),
                            "--min-diff", "12" if dedup else "0", "--note", _pack_note(f, y_th), "--first-file", ff.name],
                           capture_output=True, text=True, timeout=600)
    finally:
        os.unlink(nf.name)
        os.unlink(ff.name)
    _ai_cache["t"] = 0
    if r.returncode == 0:
        try:
            with open(PACKED, "a") as pf:
                pf.write("".join(p.name + "\n" for p in sel))
        except OSError:
            pass
    try:
        return jsonify(ok=r.returncode == 0, note=_pack_note(f, y_th), **json.loads(r.stdout.strip().splitlines()[-1]))
    except (ValueError, IndexError):
        return jsonify(ok=False, error=(r.stderr or r.stdout)[-300:]), 500


@app.get("/download/train_hamster_colab.ipynb")
@require_auth
def colab_notebook():
    """The Colab training notebook, so it can be opened in Colab (File -> Upload notebook) from any device."""
    return send_file(BASE / "train_hamster_colab.ipynb", as_attachment=True, download_name="train_hamster_colab.ipynb",
                     mimetype="application/x-ipynb+json")


@app.get("/download/dataset.zip")
@require_auth
def dataset_download():
    p = DATA / "export" / "hamster_dataset.zip"
    return send_file(p, as_attachment=True, download_name="hamster_dataset.zip") if p.exists() else ("先按「打包訓練照片」", 404)


# ---------- training-photo manager ----------
DATASET = DATA / "dataset"
THUMBS = DATA / "thumbs"
PHOTO_RE = re.compile(r"^(motion|idle|hard_unsure|hard_missed|upload)_(\d{8})_(\d{6})\.jpg$")
# Photo classes = two layers:
#   AI (the installed model's verdict on the photo itself; a new model re-scores every photo, so this can change):
#     seen (>= yolo_conf) / unsure (>= hard_conf_low) / none / pending (not scored yet)
#   motion (plain motion detection when the photo was taken, not the AI; fixed): move / still
PHOTO_KINDS = {"seen_move": "AI 偵測到・有動靜", "seen_still": "AI 偵測到・沒動靜",
               "none_move": "AI 沒偵測到・有動靜", "none_still": "AI 沒偵測到・沒動靜",
               "unsure_move": "AI 不確定・有動靜", "unsure_still": "AI 不確定・沒動靜", "pending": "AI 評分中"}
PHOTO_FILTERS = {"seen": "AI 偵測到", "none": "AI 沒偵測到", "unsure": "AI 不確定"}


def _photo_class(name, scores, moved, y_th, low):
    s = scores.get(name)
    if s is None:
        return "pending"
    ai = "seen" if s[0] >= y_th else "unsure" if s[0] >= low else "none"
    m = moved.get(name)
    if m is None:   # taken before this was recorded: the old kind says it (motion / missed = something moved)
        m = 0 if name.startswith("idle_") else 1
    return f"{ai}_{'move' if m else 'still'}"


def _class_th():
    t = current_cfg().get("tracker", {})
    return t.get("yolo_conf", 0.45), t.get("hard_conf_low", 0.2)


def _photo_info(p):
    m = PHOTO_RE.match(p.name)
    boxes, heads = [], []
    txt = p.with_suffix(".txt")
    if txt.exists():
        try:
            for line in txt.read_text().splitlines():
                parts = line.split()
                if len(parts) == 5:   # class 0 = hamster, 1 = head; cx, cy, w, h (0-1)
                    (heads if parts[0] == "1" else boxes).append([round(float(v), 4) for v in parts[1:]])
        except (OSError, ValueError):
            pass
    d, t = m.group(2), m.group(3)
    return {"name": p.name, "kind": m.group(1), "day": d,
            "time": f"{d[4:6]}/{d[6:]} {t[:2]}:{t[2:4]}:{t[4:]}", "boxes": boxes, "heads": heads}


_scores = {"mt": None, "data": {}}
_model_id = {"mt": None, "id": None}


def _ai_scores():
    """AI confidence per training photo (score_photos.py), only if made by the model installed now."""
    p = DATASET / "ai_scores.json"
    try:
        mt = p.stat().st_mtime
    except OSError:
        mt = None
        _scores.update(mt=None, data={})
    if mt is not None and _scores["mt"] != mt:
        try:
            d = json.loads(p.read_text())
        except (OSError, ValueError):
            return _scores["data"]
        _scores.update(mt=mt, data=d.get("scores", {}) if d.get("model") == _current_model_id() else {})
    live = DATASET / "ai_live.jsonl"
    try:
        lmt = live.stat().st_mtime
    except OSError:
        return _scores["data"]
    if _scores.get("lmt") != (lmt, mt):
        merged, mid = dict(_scores["data"]), _current_model_id()
        for line in live.read_text(encoding="utf-8", errors="replace").splitlines():
            try:
                r = json.loads(line)
                if r.get("model") == mid and isinstance(r.get("s"), list):
                    merged.setdefault(r["n"], r["s"])
            except ValueError:
                pass
        _scores.update(lmt=(lmt, mt), merged=merged)
    return _scores["merged"]


def _current_model_id():
    import hashlib
    p = BASE / "models" / "hamster_ncnn_model" / "model.ncnn.bin"
    try:
        mt = p.stat().st_mtime
    except OSError:
        return None
    if _model_id["mt"] != mt:
        h = hashlib.md5()
        with open(p, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        _model_id.update(mt=mt, id=h.hexdigest()[:12])
    return _model_id["id"]


def _scoring_status(start_if_needed):
    """Progress of the background scorer; start it when there's something new to score."""
    st = read_json(Path("/run/hamster/scoring.json")) or {}
    running = bool(st.get("running")) and time.time() - st.get("updated", 0) < 120
    if start_if_needed and not running and _current_model_id():
        scores = _ai_scores()
        clip_missing = any((json.loads(p.read_text()).get("ai_model") != _current_model_id())
                           for p in list(CLIPS.glob("clip_*.json"))[:500]) if CLIPS.exists() else False
        if clip_missing or any(p.name not in scores for p in DATASET.glob("*.jpg")):
            subprocess.Popen([sys.executable, str(BASE / "score_photos.py")], cwd=BASE, start_new_session=True,
                             stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            running = True
            st = {"done": 0, "total": None}
    return {"running": running, "done": st.get("done"), "total": st.get("total")}


def _photos():
    return sorted((p for p in DATASET.glob("*.jpg") if PHOTO_RE.match(p.name)),
                  key=lambda p: PHOTO_RE.match(p.name).group(2, 3), reverse=True)   # newest first


@app.get("/api/dataset/photos")
@require_auth
def dataset_photos():
    """Training photos, newest first. ?kind=motion|idle|hard  ?day=YYYYMMDD  ?zone=wheel|...  ?dark=ok|dim|dark  ?offset ?limit"""
    kind, day, zone = request.args.get("kind", ""), request.args.get("day", ""), request.args.get("zone", "")
    motion = request.args.get("motion", "")   # move / still: the second layer, under an AI choice
    lab = request.args.get("lab", "")   # yes = labelled (trained = and trained on, new = not trained yet), no = not yet,
    # ban = 禁止 (inside a 跑分 video: never trained on, labelled or not); new / no leave those out
    trained = {q.name for q in (DATA / "trained").glob("*.jpg")} if (DATA / "trained").exists() else set()
    labd = set(_labeled()) | trained   # a photo a training used counts as labelled (its label went with it)
    dark = request.args.get("dark", "")
    # 檔名: one or more (parts of) file names, separated by spaces, commas or new lines
    qs = [x.lower().removesuffix(".jpg") for x in re.split(r"[\s,，、]+", request.args.get("q", "")) if x][:200]
    # 來源 = the file name's start: motion (動時拍) / idle (定時拍) / hard (加強用) / upload (自己加)
    src = request.args.get("src", "")
    pre = {"motion": ("motion_",), "idle": ("idle_",), "hard": ("hard_unsure_", "hard_missed_"), "upload": ("upload_",)}.get(src)
    # 時段 HH:MM–HH:MM of the day the photo was taken; from > to wraps past midnight (22:00–02:00)
    def hm(v):
        m = re.fullmatch(r"(\d{1,2}):?(\d{2})", request.args.get(v, "").strip())
        return f"{int(m.group(1)):02d}{m.group(2)}00" if m and int(m.group(1)) < 24 and int(m.group(2)) < 60 else None
    t0, t1 = hm("t0"), hm("t1")

    def in_time(p):
        if not t0 and not t1:
            return True
        s = PHOTO_RE.match(p.name).group(3)
        a, b = t0 or "000000", (t1[:4] + "59") if t1 else "235959"
        return a <= s <= b if a <= b else (s >= a or s <= b)
    pz = _photo_zones()
    br = _photo_bright()
    allp = _photos()
    days = {}
    for p in allp:
        d = PHOTO_RE.match(p.name).group(2)
        days[d] = days.get(d, 0) + 1
    scores = _ai_scores()
    moved, (y_th, low) = _photo_moved(), _class_th()
    cls = {p.name: _photo_class(p.name, scores, moved, y_th, low) for p in allp}
    ban = _banned([p.name for p in allp])
    sel = [p for p in allp
           if (not kind or cls[p.name].startswith(kind + "_")) and (not motion or cls[p.name].endswith("_" + motion))
           and (not lab or (lab == "yes" and p.name in labd) or (lab == "no" and p.name not in labd and p.name not in ban)
                or (lab == "trained" and p.name in trained) or (lab == "ban" and p.name in ban)
                or (lab == "new" and p.name in labd and p.name not in trained and p.name not in ban))
           and (not day or PHOTO_RE.match(p.name).group(2) == day)
           and (not zone or pz.get(p.name) == zone) and (not dark or darkness.level(br.get(p.name)) == dark)
           and (not qs or any(x in p.name.lower() for x in qs))
           and (not pre or p.name.startswith(pre)) and in_time(p)]
    if request.args.get("names") == "1":   # 全選: every photo these filters show
        return no_cache(jsonify(names=[p.name for p in sel]))
    if request.args.get("sort") == "conf":   # least sure first = where the model needs more training
        sel.sort(key=lambda p: (p.name not in scores, scores.get(p.name, [2])[0]))
    off, lim = int_arg(request.args.get("offset"), 0, 0, 100000), int_arg(request.args.get("limit"), 90, 1, 300)

    def info(p):
        d = _photo_info(p)
        s = scores.get(p.name)
        d["ai"] = None if s is None else {"conf": s[0], "box": s[1:5] if len(s) >= 5 else None,
                                          "head": s[6:10] if len(s) == 10 else None, "head_conf": s[5] if len(s) == 10 else None}
        d["zone"] = pz.get(p.name)
        d["cls"] = cls[p.name]
        d["labeled"] = p.name in labd
        d["trained"] = p.name in trained
        d["banned"] = p.name in ban
        d["bright"] = br.get(p.name)
        d["dark"] = darkness.level(d["bright"])
        return d
    return no_cache(jsonify(total=len(sel), all=len(allp), kinds=PHOTO_KINDS, filters=PHOTO_FILTERS, labeled_total=len(labd),
                            days=[{"day": d, "label": f"{d[4:6]}/{d[6:]}", "count": n} for d, n in sorted(days.items(), reverse=True)],
                            photos=[info(p) for p in sel[off:off + lim]], scoring=_scoring_status(True),
                            zone_defs=_zone_defs()))


_br = {"data": None}


def _photo_bright():
    """{photo name: mean brightness 0-255}, measured once per photo and kept in dataset/bright.json."""
    p = DATASET / "bright.json"
    if _br["data"] is None:
        try:
            _br["data"] = json.loads(p.read_text())
        except (OSError, ValueError):
            _br["data"] = {}
    data, new = _br["data"], 0
    for ph in _photos():
        if ph.name not in data:
            import cv2   # only needed for photos not measured yet
            g = cv2.imread(str(ph), cv2.IMREAD_REDUCED_GRAYSCALE_4)
            if g is not None:
                data[ph.name] = round(float(g.mean()), 1)
                new += 1
    if new:
        try:
            tmp = p.with_suffix(".tmp")
            tmp.write_text(json.dumps(data))
            os.replace(tmp, p)
        except OSError:
            pass
    return data


_pz = {"mt": None, "data": {}}


def _photo_zones():
    """{photo name: zone it was taken in} from dataset/zones.jsonl (written by tracker.py; photos from before
    this existed have no place)."""
    p = DATASET / "zones.jsonl"
    try:
        mt = p.stat().st_mtime
    except OSError:
        return {}
    if _pz["mt"] != mt:
        data, moved = {}, {}
        for line in p.read_text(encoding="utf-8", errors="replace").splitlines():
            try:
                r = json.loads(line)
                if r.get("z"):
                    data[r["n"]] = r["z"]
                if "m" in r:
                    moved[r["n"]] = r["m"]
            except (ValueError, KeyError, TypeError, AttributeError):
                pass
        _pz.update(mt=mt, data=data, moved=moved)
    return _pz["data"]


def _photo_moved():
    """{photo name: 1 / 0} did anything move in the cage when it was taken (zones.jsonl "m")."""
    _photo_zones()
    return _pz.get("moved", {})


def _photo_path(name):
    if not PHOTO_RE.match(name or ""):
        return None
    p = DATASET / name
    return p if p.exists() else None


# ---------- labelling on the dashboard (no X-AnyLabeling needed) + uploading your own photos ----------
def _clean_boxes(bs):
    """[[cx, cy, w, h], ...] 0-1 from the page; at most 6, tiny ones dropped."""
    out = []
    for b in (bs or [])[:6]:
        try:
            cx, cy, w, h = [min(1.0, max(0.0, float(v))) for v in b[:4]]
        except (TypeError, ValueError):
            continue
        if w >= 0.01 and h >= 0.01:
            out.append([round(cx, 4), round(cy, 4), round(w, 4), round(h, 4)])
    return out


_lab = {"mt": None, "data": {}}


def _labeled():
    """{photo name: when} of training photos the owner labelled on the dashboard (their .txt is the answer)."""
    p = DATASET / "labeled.json"
    try:
        mt = p.stat().st_mtime
    except OSError:
        return {}
    if _lab["mt"] != mt:
        try:
            _lab.update(mt=mt, data=json.loads(p.read_text()))
        except (OSError, ValueError):
            pass
    return _lab["data"]


def _set_labeled(name, on):
    d = dict(_labeled())
    if on:
        d[name] = datetime.now().isoformat(timespec="seconds")
    else:
        d.pop(name, None)
    tmp = DATASET / "labeled.json.tmp"
    tmp.write_text(json.dumps(d))
    os.replace(tmp, DATASET / "labeled.json")
    _ai_cache["t"] = 0


@app.post("/api/dataset/label")
@require_auth
def dataset_label():
    """Save the owner's boxes for a training photo as its YOLO label (0 = hamster, 1 = head; none = no hamster)."""
    body = request.get_json(silent=True) or {}
    p = _photo_path(str(body.get("name", "")))
    if not p:
        return jsonify(ok=False, error="找不到這張照片"), 404
    boxes, heads = _clean_boxes(body.get("boxes")), _clean_boxes(body.get("heads"))
    lines = "".join(f"0 {b[0]:.6f} {b[1]:.6f} {b[2]:.6f} {b[3]:.6f}\n" for b in boxes) + \
        "".join(f"1 {b[0]:.6f} {b[1]:.6f} {b[2]:.6f} {b[3]:.6f}\n" for b in heads)
    txt = p.with_suffix(".txt")
    try:   # what it was before, so 復原 on the page can put it back exactly (labelled or not, pre-drawn boxes or none)
        prev = {"txt": txt.read_text() if txt.exists() else None, "labeled": p.name in _labeled()}
    except OSError:
        prev = None
    txt.write_text(lines)   # empty file = "no hamster" (a background photo for training)
    _set_labeled(p.name, True)
    return jsonify(ok=True, prev=prev, banned=bool(_banned([p.name])))


LABEL_LINE = re.compile(r"^[01] [0-9.]+ [0-9.]+ [0-9.]+ [0-9.]+$")


@app.post("/api/dataset/label/restore")
@require_auth
def dataset_label_restore():
    """Undo a save: {name, txt (the label file as it was, or null = there was none), labeled (was it labelled)}."""
    body = request.get_json(silent=True) or {}
    p = _photo_path(str(body.get("name", "")))
    if not p:
        return jsonify(ok=False, error="找不到這張照片"), 404
    txt, old = p.with_suffix(".txt"), body.get("txt")
    if old is None:
        txt.unlink(missing_ok=True)
    else:
        old = str(old)[:20000]
        if not all(LABEL_LINE.match(l) for l in old.splitlines() if l.strip()):
            return jsonify(ok=False, error="標註內容不對"), 400
        txt.write_text(old)
    _set_labeled(p.name, bool(body.get("labeled")))
    return jsonify(ok=True)


@app.post("/api/photos/upload")
@require_auth
def photos_upload():
    """Your own photos (phone / computer) into the training photos (to=train) or the exam (to=exam)."""
    import cv2
    import numpy as np
    to = request.form.get("to", "train")
    if to not in ("train", "exam"):
        return jsonify(ok=False, error="要放哪裡？"), 400
    dest = EXAM if to == "exam" else DATASET
    dest.mkdir(parents=True, exist_ok=True)
    saved, bad = [], 0
    base = datetime.now()
    for k, fs in enumerate(request.files.getlist("photos")[:100]):
        img = cv2.imdecode(np.frombuffer(fs.read(20_000_000), np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            bad += 1
            continue
        h, w = img.shape[:2]
        s = min(1.0, 1280 / max(h, w))   # keep it about camera size
        if s < 1:
            img = cv2.resize(img, (round(w * s), round(h * s)), interpolation=cv2.INTER_AREA)
        t = base + timedelta(seconds=k)
        while (dest / f"upload_{t:%Y%m%d_%H%M%S}.jpg").exists():
            t += timedelta(seconds=1)
        name = f"upload_{t:%Y%m%d_%H%M%S}.jpg"
        cv2.imwrite(str(dest / name), img, [cv2.IMWRITE_JPEG_QUALITY, 92])
        bright = round(float(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).mean()), 1)
        if to == "exam":
            (EXAM / (name[:-4] + ".json")).write_text(json.dumps(
                {"done": False, "boxes": [], "heads": [], "bright": bright, "uploaded": True,
                 "added": datetime.now().isoformat(timespec="seconds")}))
        else:   # not a cage moment: nothing "moved"
            with open(DATASET / "zones.jsonl", "a", encoding="utf-8") as f:
                f.write(json.dumps({"n": name, "m": 0}) + "\n")
        saved.append(name)
    if saved and to == "train":
        _scoring_status(True)   # the tracker didn't take these: let the AI score them right away
    near = len(_banned(saved)) if to == "train" else 0
    return jsonify(ok=True, saved=len(saved), bad=bad, names=saved, near_exam=near)


# ---------- 考試照片: the owner's answer photos, never used for training (exam_run.py tests models on them) ----------
EXAM = DATA / "exam"
PACKED = DATA / "packed.txt"          # every photo that went into a training pack (those can't be exam photos)
EXAM_NEAR_S = 15                       # (was 120 until 2026-11-24) photos taken this close to an exam photo stay out of training packs too


def _packed():
    try:
        return set(PACKED.read_text().split())
    except OSError:
        return set()


def _exam_items():
    out = []
    for j in sorted(EXAM.glob("*.json")) if EXAM.exists() else []:
        if j.name == "results.json" or not (EXAM / (j.stem + ".jpg")).exists():
            continue
        try:
            a = json.loads(j.read_text())
        except (OSError, ValueError):
            continue
        m = PHOTO_RE.match(j.stem + ".jpg")
        d, tm = (m.group(2), m.group(3)) if m else ("", "000000")
        out.append({"name": j.stem + ".jpg", "done": bool(a.get("done")), "boxes": a.get("boxes") or [],
                    "heads": a.get("heads") or [], "zone": a.get("zone"), "bright": a.get("bright"),
                    "dark": darkness.level(a.get("bright")),
                    "time": f"{d[4:6]}/{d[6:]} {tm[:2]}:{tm[2:4]}:{tm[4:]}" if d else ""})
    return out


BENCH_NEAR_S = 15                      # (was 30 until 2026-11-24) and photos taken during a 跑分 video (or this close to it)


def _exam_times():
    """Times (epoch s) of the exam photos, plus (start, end) of each 跑分 video: training packs leave out photos
    taken right next to them, so the model is never tested on pictures it was trained on."""
    ts = []
    for p in EXAM.glob("*.jpg") if EXAM.exists() else []:
        m = PHOTO_RE.match(p.name)
        if m:
            try:
                ts.append(datetime.strptime(m.group(2) + m.group(3), "%Y%m%d%H%M%S").timestamp())
            except ValueError:
                pass
    for j in BCLIPS.glob("clip_*.json") if BCLIPS.exists() else []:
        if j.name.endswith(".lab.json"):
            continue
        info = read_json(j) or {}
        try:
            a = datetime.fromisoformat(info["start"]).timestamp()
        except (KeyError, ValueError, TypeError):
            continue
        ts.append((a, a + float(info.get("seconds") or 10)))
    return ts


def _near_exam(name, ts):
    m = PHOTO_RE.match(name)
    if not m or not ts:
        return False
    try:
        t0 = datetime.strptime(m.group(2) + m.group(3), "%Y%m%d%H%M%S").timestamp()
    except ValueError:
        return False
    return any(x[0] - BENCH_NEAR_S <= t0 <= x[1] + BENCH_NEAR_S if isinstance(x, tuple) else abs(t0 - x) <= EXAM_NEAR_S
               for x in ts)


def _banned(names=None):
    """禁止: photos taken inside a 跑分 video (or right next to a 跑分 photo) that no training used yet. They never go
    into a training pack (the model would be tested on pictures it learned), so labelling them is wasted work."""
    ets = _exam_times()
    trained = {q.name for q in (DATA / "trained").glob("*.jpg")} if (DATA / "trained").exists() else set()
    names = [p.name for p in _photos()] if names is None else names
    return {n for n in names if n not in trained and _near_exam(n, ets)}


def _training_hits(spans):
    """Training photos inside these time spans [(t0, t1) epoch s, padding already added]: (trained, labelled).
    trained = a training already used them (the model has seen these moments: the test is not fair);
    labelled = labelled, not trained yet: they turn 禁止 and the labelling work is not used."""
    if not spans:
        return 0, 0
    trained = {q.name for q in (DATA / "trained").glob("*.jpg")} if (DATA / "trained").exists() else set()
    labd = set(_labeled())
    tr = lab = 0
    for n in trained | labd:
        m = PHOTO_RE.match(n)
        if not m:
            continue
        try:
            t = datetime.strptime(m.group(2) + m.group(3), "%Y%m%d%H%M%S").timestamp()
        except ValueError:
            continue
        if any(a <= t <= b for a, b in spans):
            if n in trained:
                tr += 1
            else:
                lab += 1
    return tr, lab


def _exam_models():
    import modelnames
    vmap, out = modelnames.versions(MODELS), []
    for d in MODELS.glob("hamster_ncnn_model*"):
        if (d / "model.ncnn.param").exists():
            out.append({"dir": d.name, "label": modelnames.label(d, vmap), "current": d.name == "hamster_ncnn_model",
                        "version": vmap.get(d.name) or 0})
    out.sort(key=lambda m: (not m["current"], -m["version"]))   # in use, then the newest versions first
    return out


@app.get("/api/exam")
@require_auth
def exam_list():
    items = _exam_items()
    res = read_json(EXAM / "results.json")
    st = read_json(Path("/run/hamster/exam.json")) or {}
    running = bool(st.get("running")) and time.time() - st.get("updated", 0) < 120
    return no_cache(jsonify(items=items, models=_exam_models(), results=res,
                            run={"running": running, "done": st.get("done"), "total": st.get("total"), "error": st.get("error")},
                            near_s=EXAM_NEAR_S, zone_defs=_zone_defs()))


@app.post("/api/exam/add")
@require_auth
def exam_add():
    """Move photos from the training photos into the exam (their pre-drawn boxes become the starting answer)."""
    body = request.get_json(silent=True) or {}
    names = [str(n) for n in body.get("names") or [] if PHOTO_RE.match(str(n))][:500]
    packed = _packed()
    spans = []
    for n in names:
        if (DATASET / n).exists() and n not in packed:
            m = PHOTO_RE.match(n)
            t = datetime.strptime(m.group(2) + m.group(3), "%Y%m%d%H%M%S").timestamp()
            spans.append((t - EXAM_NEAR_S, t + EXAM_NEAR_S))
    hit_trained, hit_labelled = _training_hits(spans)
    hit_labelled -= sum(1 for n in names if n in _labeled() and n not in packed)   # the photos themselves move out
    if body.get("check"):
        return jsonify(ok=True, photos=len(spans), used=len(names) - len(spans), trained=hit_trained, labelled=max(0, hit_labelled))
    EXAM.mkdir(parents=True, exist_ok=True)
    pz, moved, br = _photo_zones(), _photo_moved(), _photo_bright()
    added, used = 0, 0
    for n in names:
        p = DATASET / n
        if not p.exists():
            continue
        if n in packed:   # already went to training: as an exam photo it would be a test the model has seen
            used += 1
            continue
        boxes, heads = [], []
        txt = p.with_suffix(".txt")
        if txt.exists():
            for line in txt.read_text().splitlines():
                parts = line.split()
                if len(parts) == 5:
                    (heads if parts[0] == "1" else boxes).append([round(float(v), 4) for v in parts[1:]])
        ans = {"done": n in _labeled(), "boxes": boxes, "heads": heads, "zone": pz.get(n), "moved": moved.get(n), "bright": br.get(n),
               "added": datetime.now().isoformat(timespec="seconds")}
        os.replace(p, EXAM / n)
        (EXAM / (p.stem + ".json")).write_text(json.dumps(ans, ensure_ascii=False))
        txt.unlink(missing_ok=True)
        (THUMBS / n).unlink(missing_ok=True)
        if n in _labeled():
            _set_labeled(n, False)
        added += 1
    _ai_cache["t"] = 0
    return jsonify(ok=True, added=added, used=used, total=len(_exam_items()), trained=hit_trained, labelled=max(0, hit_labelled))


@app.post("/api/exam/label")
@require_auth
def exam_label():
    """The owner's answer for one exam photo: body boxes and head boxes (cx, cy, w, h 0-1); none = no hamster."""
    body = request.get_json(silent=True) or {}
    n = str(body.get("name", ""))
    j = EXAM / (Path(n).stem + ".json")
    if not PHOTO_RE.match(n) or not j.exists():
        return jsonify(ok=False, error="找不到這張考試照片"), 404

    a = json.loads(j.read_text())
    a.update(done=True, boxes=_clean_boxes(body.get("boxes")), heads=_clean_boxes(body.get("heads")),
             labeled=datetime.now().isoformat(timespec="seconds"))
    tmp = j.with_suffix(".tmp")
    tmp.write_text(json.dumps(a, ensure_ascii=False))
    os.replace(tmp, j)
    return jsonify(ok=True)


@app.post("/api/exam/remove")
@require_auth
def exam_remove():
    """Back to the training photos (with the answer as its labels), e.g. a photo picked by mistake."""
    body = request.get_json(silent=True) or {}
    n, back = 0, []
    for name in [str(x) for x in body.get("names") or [] if PHOTO_RE.match(str(x))][:500]:
        p, j = EXAM / name, EXAM / (Path(name).stem + ".json")
        if not p.exists():
            continue
        try:
            a = json.loads(j.read_text())
        except (OSError, ValueError):
            a = {}
        lines = "".join(f"0 {b[0]} {b[1]} {b[2]} {b[3]}\n" for b in a.get("boxes") or []) + \
            "".join(f"1 {b[0]} {b[1]} {b[2]} {b[3]}\n" for b in a.get("heads") or [])
        os.replace(p, DATASET / name)
        if a.get("done"):
            (DATASET / name).with_suffix(".txt").write_text(lines)
            _set_labeled(name, True)
        elif lines:
            (DATASET / name).with_suffix(".txt").write_text(lines)
        j.unlink(missing_ok=True)
        n += 1
        back.append(name)
    _ai_cache["t"] = 0
    return jsonify(ok=True, removed=n, near_exam=len(_banned(back)))


@app.get("/exam/photo/<name>")
@require_auth
def exam_photo(name):
    p = EXAM / name
    return send_file(p, mimetype="image/jpeg", max_age=3600) if PHOTO_RE.match(name or "") and p.exists() else ("not found", 404)


_guess = {"id": None, "det": None}


@app.get("/api/exam/guess")
@require_auth
def exam_guess():
    """What the installed model sees on one exam photo (for 套用 AI 的框 while labelling): centre-format boxes."""
    import cv2
    n = request.args.get("name", "")
    p = EXAM / n
    if not PHOTO_RE.match(n) or not p.exists():
        return jsonify(ok=False, error="找不到這張"), 404
    mid = _current_model_id()
    if not mid:
        return jsonify(ok=False, error="還沒有安裝模型"), 400
    from yolo_detector import YoloNcnn, hamster_and_head
    if _guess["id"] != mid:
        _guess.update(id=mid, det=YoloNcnn(MODELS / "hamster_ncnn_model", conf=0.2, threads=1))
    img = cv2.imread(str(p))
    b, hd = hamster_and_head(_guess["det"].detect(img)) if img is not None else (None, None)
    c = lambda x: [round(x[0] + x[2] / 2, 4), round(x[1] + x[3] / 2, 4), round(x[2], 4), round(x[3], 4)]   # noqa: E731
    return jsonify(ok=True, box=c(b) if b else None, conf=round(float(b[4]), 3) if b else 0.0,
                   head=c(hd) if b and hd else None, head_conf=round(float(hd[4]), 3) if b and hd else None)


@app.post("/api/exam/run")
@require_auth
def exam_run():
    body = request.get_json(silent=True) or {}
    ok_dirs = {m["dir"] for m in _exam_models()}
    dirs = [d for d in body.get("models") or [] if d in ok_dirs][:4]
    if not dirs:
        return jsonify(ok=False, error="至少選一個模型"), 400
    if not any(i["done"] for i in _exam_items()):
        return jsonify(ok=False, error="還沒有標好答案的考試照片"), 400
    st = read_json(Path("/run/hamster/exam.json")) or {}
    if st.get("running") and time.time() - st.get("updated", 0) < 120:
        return jsonify(ok=False, error="考試正在進行中"), 409
    Path("/run/hamster").mkdir(parents=True, exist_ok=True)
    (Path("/run/hamster") / "exam.json").write_text(json.dumps({"running": True, "done": 0, "total": None, "updated": time.time()}))
    subprocess.Popen([sys.executable, str(BASE / "exam_run.py"), *[f"models/{d}" for d in dirs]], cwd=BASE,
                     start_new_session=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return jsonify(ok=True)


@app.get("/dataset/photo/<name>")
@require_auth
def dataset_photo(name):
    p = _photo_path(name)
    return send_file(p, mimetype="image/jpeg", max_age=86400) if p else ("not found", 404)


@app.get("/dataset/thumb/<name>")
@require_auth
def dataset_thumb(name):
    """Small cached copy for the grid (a 640x480 photo is ~40 KB; the thumb is ~6 KB)."""
    p = _photo_path(name)
    if not p:
        return "not found", 404
    t = THUMBS / name
    if not t.exists():
        import cv2   # only needed here; keeps web start-up light
        img = cv2.imread(str(p))
        if img is None:
            return "unreadable", 404
        THUMBS.mkdir(parents=True, exist_ok=True)
        h, w = img.shape[:2]
        small = cv2.resize(img, (240, round(240 * h / w)), interpolation=cv2.INTER_AREA)
        ok, buf = cv2.imencode(".jpg", small, [cv2.IMWRITE_JPEG_QUALITY, 78])
        if not ok:
            return "encode failed", 500
        tmp = t.with_suffix(".tmp")
        tmp.write_bytes(buf.tobytes())
        os.replace(tmp, t)
    return send_file(t, mimetype="image/jpeg", max_age=86400)


@app.post("/api/dataset/delete")
@require_auth
def dataset_delete():
    """Delete training photos (and their pre-label .txt). Body: {names: [...]} or {day: "YYYYMMDD"} or {all: true}."""
    body = request.get_json(silent=True) or {}
    if body.get("all") is True:
        targets = _photos()
    elif re.fullmatch(r"\d{8}", str(body.get("day", ""))):
        targets = [p for p in _photos() if PHOTO_RE.match(p.name).group(2) == body["day"]]
    elif isinstance(body.get("names"), list) and len(body["names"]) <= 5000:
        targets = [p for p in (_photo_path(str(n)) for n in body["names"]) if p]
    else:
        return jsonify(ok=False, error="要刪哪些照片？"), 400
    n = 0
    for p in targets:
        for f in (p, p.with_suffix(".txt"), THUMBS / p.name):
            try:
                f.unlink(missing_ok=True)
            except OSError:
                pass
        n += 1
    gone = {p.name for p in targets}
    d = _labeled()
    if gone & set(d):
        d = {k: v for k, v in d.items() if k not in gone}
        tmp = DATASET / "labeled.json.tmp"
        tmp.write_text(json.dumps(d))
        os.replace(tmp, DATASET / "labeled.json")
    _ai_cache["t"] = 0
    app.logger.warning("dataset delete from %s: %d photos", request.remote_addr, n)
    return jsonify(ok=True, deleted=n, left=len(_photos()))


# ---------- video clips (recorded by tracker.py via clips.py) ----------
CLIPS = DATA / "clips"
CLIP_RE = re.compile(r"^clip_(\d{8})_(\d{6})$")


def _clip_infos():
    """Finished clips (they get their .json once encoding is done), newest first."""
    out = []
    for j in CLIPS.glob("clip_*.json"):
        m = CLIP_RE.match(j.stem)
        if not m or not j.with_suffix(".mp4").exists():
            continue
        try:
            info = json.loads(j.read_text())
        except (OSError, ValueError):
            continue
        d, t = m.groups()
        info.update(name=j.stem, day=d, time=f"{d[4:6]}/{d[6:]} {t[:2]}:{t[2:4]}",
                    thumb=j.with_suffix(".jpg").exists(), boxes=j.with_suffix(".boxes").exists())
        out.append(info)
    out.sort(key=lambda x: x["name"], reverse=True)
    return out


@app.get("/api/clips")
@require_auth
def clips_list():
    """?day=YYYYMMDD  ?zone=wheel|food|water|bath|nest  ?offset ?limit"""
    day, zone = request.args.get("day", ""), request.args.get("zone", "")
    allc = _clip_infos()
    if request.args.get("lib") == "1":   # 設定 › 錄影紀錄: your own videos only once added (加入錄影紀錄)
        allc = [c for c in allc if not c.get("manual") or c.get("in_lib")]
    days = {}
    for c in allc:
        e = days.setdefault(c["day"], {"day": c["day"], "label": f"{c['day'][4:6]}/{c['day'][6:]}", "count": 0, "seconds": 0})
        e["count"] += 1
        e["seconds"] += c.get("seconds", 0)
    sel = [c for c in allc if (not day or c["day"] == day) and (not zone or zone in (c.get("zones") or {}))
           and (request.args.get("manual") not in ("0", "1") or bool(c.get("manual")) == (request.args.get("manual") == "1"))]
    if request.args.get("sort") == "conf":   # least sure first; clips not scored yet go last
        sel.sort(key=lambda c: (c.get("ai_conf") is None, (c.get("ai_conf") or {}).get("avg", 2)))
    if request.args.get("names") == "1":
        return no_cache(jsonify(names=[c["name"] for c in sel]))
    off, lim = int_arg(request.args.get("offset"), 0, 0, 100000), int_arg(request.args.get("limit"), 30, 1, 200)
    s = {}
    try:
        import clips as clipmod
        s = clipmod.clip_settings(current_cfg())
    except Exception:   # noqa: BLE001 - settings are only shown, never required
        pass
    used = sum(c.get("size", 0) for c in allc)
    try:
        free = shutil.disk_usage(DATA).free
    except OSError:
        free = 0
    return no_cache(jsonify(total=len(sel), all=len(allc), clips=sel[off:off + lim],
                            days=sorted(days.values(), key=lambda x: x["day"], reverse=True),
                            storage={"used_mb": round(used / 1e6), "cap_gb": s.get("max_gb"),
                                     "free_gb": round(free / 1e9, 1), "keep_days": s.get("keep_days")},
                            scoring=_scoring_status(True), zone_defs=_zone_defs(),
                            enabled=bool(s.get("enabled", True)) and shutil.which("ffmpeg") is not None))


def _clip_file(name, ext):
    if not CLIP_RE.match(name or ""):
        return None
    p = CLIPS / f"{name}{ext}"
    return p if p.exists() else None


@app.get("/clips/<name>.mp4")
@require_auth
def clip_video(name):
    p = _clip_file(name, ".mp4")
    # conditional=True answers Range requests: iPhone Safari needs them to play and seek in <video>
    return send_file(p, mimetype="video/mp4", conditional=True, max_age=86400) if p else ("not found", 404)


@app.get("/clips/<name>.jpg")
@require_auth
def clip_thumb(name):
    p = _clip_file(name, ".jpg")
    return send_file(p, mimetype="image/jpeg", max_age=86400) if p else ("not found", 404)


@app.get("/clips/<name>.boxes")
@require_auth
def clip_boxes(name):
    """Where the AI saw the hamster / its head in each frame of a clip (drawn over the video on request)."""
    p = _clip_file(name, ".boxes")
    return send_file(p, mimetype="application/json", max_age=86400) if p else ("not found", 404)


@app.post("/api/clips/delete")
@require_auth
def clips_delete():
    """Body: {names: [...]} or {day: "YYYYMMDD"} or {all: true}."""
    body = request.get_json(silent=True) or {}
    allc = _clip_infos()
    if body.get("all") is True:
        names = [c["name"] for c in allc]
    elif re.fullmatch(r"\d{8}", str(body.get("day", ""))):
        names = [c["name"] for c in allc if c["day"] == body["day"]]
    elif isinstance(body.get("names"), list) and len(body["names"]) <= 5000:
        names = [str(n) for n in body["names"] if CLIP_RE.match(str(n))]
    else:
        return jsonify(ok=False, error="要刪哪些影片？"), 400
    n = 0
    for name in names:
        hit = False
        for ext in (".mp4", ".jpg", ".json", ".boxes"):
            p = CLIPS / f"{name}{ext}"
            if p.exists():
                try:
                    p.unlink()
                    hit = True
                except OSError:
                    pass
        n += hit
    app.logger.warning("clip delete from %s: %d clips", request.remote_addr, n)
    return jsonify(ok=True, deleted=n, left=len(_clip_infos()))


@app.post("/api/model/upload")
@require_auth
def model_upload():
    f = request.files.get("model")
    if not f or not f.filename.lower().endswith(".zip"):
        return jsonify(ok=False, error="請選擇 Colab 下載的 hamster_ncnn_model.zip"), 400
    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "model.zip"
        f.save(path)
        r = subprocess.run([sys.executable, str(BASE / "install_model.py"), str(path)],
                           capture_output=True, text=True, timeout=300)
    _ai_cache["t"] = 0
    try:
        return jsonify(**json.loads(r.stdout.strip().splitlines()[-1]))
    except (ValueError, IndexError):
        if r.returncode < 0:   # the checker itself crashed (e.g. a model that cannot run on this Pi)
            return jsonify(ok=False, error=f"模型在樹莓派上測試時當掉了（訊號 {-r.returncode}），沒有安裝。請用 Colab 筆記本最後一格重新匯出"), 500
        return jsonify(ok=False, error=(r.stderr or r.stdout)[-300:] or "安裝程式沒有回應"), 500


# ---------- model manager: every model on the Pi (the one in use + the older ones kept by install_model.py) ----------
MODELS = BASE / "models"
MODEL_CUR = "hamster_ncnn_model"
MODEL_RE = re.compile(r"^hamster_ncnn_model(\.bak-(\d{8})-(\d{6}))?$")


def _model_dir(name):
    if not MODEL_RE.match(name or ""):
        return None
    d = MODELS / name
    return d if (d / "model.ncnn.param").exists() and (d / "model.ncnn.bin").exists() else None


def _model_info(d):
    import hashlib
    meta = (d / "metadata.yaml").read_text(encoding="utf-8", errors="replace") if (d / "metadata.yaml").exists() else ""
    m = re.search(r"(?m)^imgsz:\s*\n\s*-\s*(\d+)", meta)
    trained = None
    t = re.search(r'(?m)^date:\s*"?([0-9T:\-.+]+)', meta)
    if t:
        try:
            trained = datetime.fromisoformat(t.group(1)).astimezone().strftime("%m/%d %H:%M")
        except ValueError:
            pass
    h = hashlib.md5()
    with open(d / "model.ncnn.bin", "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    bm = MODEL_RE.match(d.name)
    retired = f"{bm.group(2)[4:6]}/{bm.group(2)[6:]} {bm.group(3)[:2]}:{bm.group(3)[2:4]}" if bm.group(1) else None
    return {"name": d.name, "current": d.name == MODEL_CUR, "id": h.hexdigest()[:12], "imgsz": int(m.group(1)) if m else None,
            "title": _model_title(d),
            "trained": trained, "added": datetime.fromtimestamp(d.stat().st_mtime).strftime("%m/%d %H:%M"), "retired": retired,
            "mb": round(sum(p.stat().st_size for p in d.iterdir() if p.is_file()) / 1e6, 1)}


def _model_title(d):
    """The owner's own name for a model (title.txt in its folder, so it follows the folder when models switch)."""
    try:
        return (Path(d) / "title.txt").read_text(encoding="utf-8").strip()[:30] or None
    except OSError:
        return None


@app.post("/api/models/rename")
@require_auth
def models_rename():
    body = request.get_json(silent=True) or {}
    d = _model_dir(body.get("name"))
    if not d:
        return jsonify(ok=False, error="找不到這個模型"), 404
    title = re.sub(r"\s+", " ", str(body.get("title") or "")).strip()[:30]
    f = d / "title.txt"
    if title:
        f.write_text(title, encoding="utf-8")
    elif f.exists():
        f.unlink()
    return jsonify(ok=True, title=title or None)


@app.get("/api/models")
@require_auth
def models_list():
    """The model in use first, then the older ones (newest first)."""
    dirs = [d for d in MODELS.iterdir() if d.is_dir() and _model_dir(d.name)] if MODELS.exists() else []
    import modelnames
    vmap = modelnames.versions(MODELS)
    out = [_model_info(d) for d in dirs]
    for m, d in zip(out, dirs):
        m.update(version=vmap.get(d.name), label=modelnames.label(d, vmap), never_used=modelnames.never_used(d))
        if m["never_used"]:
            m["retired"] = None   # its folder time is when it was trained, not when it was replaced
    out.sort(key=lambda m: (not m["current"], -(m["version"] or 0)))   # in use, then the newest versions first
    cur = next((m["id"] for m in out if m["current"]), None)
    seen = set()
    for m in out:   # copies: the same model kept twice (older copy = the duplicate)
        m["same_as_current"] = not m["current"] and m["id"] == cur
        m["duplicate"] = not m["current"] and not m["same_as_current"] and m["id"] in seen
        seen.add(m["id"])
    return no_cache(jsonify(models=out, detector=current_cfg().get("tracker", {}).get("detector", "motion")))


@app.post("/api/models/use")
@require_auth
def models_use():
    """Switch to an older model: it becomes the one in use, the one in use is kept as an older model."""
    name = (request.get_json(silent=True) or {}).get("name")
    src = _model_dir(name)
    if not src or name == MODEL_CUR:
        return jsonify(ok=False, error="找不到這個模型，或它已經在使用中"), 400
    sys.path.insert(0, str(BASE))
    from install_model import probe, read_imgsz_safe
    res, why = probe(src, read_imgsz_safe(src) or 640)
    if not res:
        return jsonify(ok=False, error=f"這個模型在樹莓派上跑不起來（{why}），沒有切換"), 400
    cur = MODELS / MODEL_CUR
    if cur.exists():
        cur.rename(MODELS / f"{MODEL_CUR}.bak-{time.strftime('%Y%m%d-%H%M%S')}")
    src.rename(cur)
    os.utime(cur)   # "added" = when it went into use
    text = CONFIG_FILE.read_text(encoding="utf-8")
    if re.search(r'(?m)^detector\s*=\s*"motion"', text):   # a model is chosen: use it (AI + motion)
        CONFIG_FILE.write_text(re.sub(r'(?m)^detector\s*=\s*"motion"', 'detector = "hybrid"', text, count=1), encoding="utf-8")
    subprocess.run(["sudo", "-n", "systemctl", "restart", "hamster-tracker"], capture_output=True, timeout=30)
    _ai_cache["t"] = 0
    app.logger.warning("model switched from %s: now using %s", request.remote_addr, name)
    return jsonify(ok=True, ms_per_frame=round(res["ms"]), imgsz=res["imgsz"])


@app.post("/api/models/delete")
@require_auth
def models_delete():
    name = (request.get_json(silent=True) or {}).get("name")
    d = _model_dir(name)
    if not d:
        return jsonify(ok=False, error="找不到這個模型"), 400
    if name == MODEL_CUR:
        return jsonify(ok=False, error="正在使用的模型不能刪除，先切換到別的模型"), 400
    shutil.rmtree(d)
    app.logger.warning("model deleted from %s: %s", request.remote_addr, name)
    return jsonify(ok=True)


@app.get("/download/model/<name>.zip")
@require_auth
def models_download(name):
    """The model as a zip in the same layout the upload button takes (hamster_ncnn_model/...)."""
    import io
    import zipfile
    d = _model_dir(name)
    if not d:
        return "找不到這個模型", 404
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        for p in sorted(d.iterdir()):
            if p.is_file():
                z.write(p, f"hamster_ncnn_model/{p.name}")
    buf.seek(0)
    info = _model_info(d)
    stamp = (info["trained"] or info["added"]).replace("/", "").replace(":", "").replace(" ", "_")
    return send_file(buf, mimetype="application/zip", as_attachment=True, download_name=f"hamster_ncnn_model_{stamp}_{info['id'][:6]}.zip")


@app.get("/raw.jpg")
@require_auth
def raw_frame():
    """Clean camera frame (no overlays) for the cage-area editor."""
    for p in (DATA / "raw.jpg", DATA / "latest.jpg"):
        if p.exists():
            return no_cache(send_file(p, mimetype="image/jpeg"))
    return "no camera frame yet", 404


@app.get("/api/calib")
@require_auth
def calib_get():
    t = current_cfg().get("tracker", {})
    return jsonify(roi=t.get("roi") or [], floor=t.get("floor") or [], zones=t.get("zones") or [], floor_cm=_floor_sides(t),
                   spout=_spout(t), drink_spot=_drink_spot(t), zone_defs=_zone_defs())


def _spout(t):
    """The water bottle's spout as [x, y] (0..1 of the picture), or None when not marked."""
    s = t.get("spout")
    return [float(s[0]), float(s[1])] if isinstance(s, list) and len(s) == 2 else None


def _drink_spot(t):
    """The drinking spot (water area) as [cx, cy, r] when moved/resized in the editor, else None (= default)."""
    s = t.get("drink_spot")
    return [float(v) for v in s] if isinstance(s, list) and len(s) == 3 else None




@app.get("/api/behavior")
@require_auth
def behavior():
    """Last-24h zone summary: time in nest (sleep), wheel, food, water, bath, plus visit counts."""
    hours = int_arg(request.args.get("hours"), 24, 1, 24 * 31)
    since_ts = since(hours)
    out = {k: {"seconds": 0, "visits": 0, "last": None} for k in _zone_ids()}
    with db() as con:
        try:
            for r in con.execute("SELECT zone, SUM(seconds) s FROM zone_minutes WHERE minute >= ? GROUP BY zone", (since_ts[:16],)):
                if r["zone"] in out:
                    out[r["zone"]]["seconds"] = r["s"]
            for r in con.execute("SELECT zone, COUNT(*) n, MAX(start) last FROM zone_visits WHERE start >= ? GROUP BY zone", (since_ts,)):
                if r["zone"] in out:
                    out[r["zone"]].update(visits=r["n"], last=r["last"])
            recent = [dict(r) for r in con.execute(
                "SELECT start, zone, seconds FROM zone_visits WHERE start >= ? ORDER BY start DESC LIMIT 12", (since_ts,))]
        except sqlite3.OperationalError:   # tracker hasn't created the tables yet
            recent = []
        est = {k: {"count": 0, "seconds": 0, "last": None} for k in ("eat", "drink", "run")}
        try:   # running distance from the wheel turns (tracker.py wheel_minutes)
            wheel_m = round((con.execute("SELECT COALESCE(SUM(cm),0) FROM wheel_minutes WHERE minute >= ?",
                                         (since_ts[:16],)).fetchone()[0] or 0) / 100, 1)
        except sqlite3.OperationalError:
            wheel_m = 0.0
        try:   # camera estimate of eating / drinking (tracker.py behavior_events)
            for r in con.execute("SELECT kind, COUNT(*) n, SUM(seconds) s, MAX(start) last FROM behavior_events "
                                 "WHERE start >= ? GROUP BY kind", (since_ts,)):
                if r["kind"] in est:
                    est[r["kind"]] = {"count": r["n"], "seconds": r["s"] or 0, "last": r["last"]}
        except sqlite3.OperationalError:
            pass
    # 回看: the clip that covers each event (its start inside the clip, or overlapping it), and where in it
    if recent:
        clips = [(c["start"], c["end"], c) for c in _clip_infos() if c.get("start") and c.get("end")]
        for r in recent:
            try:
                s0 = datetime.fromisoformat(r["start"])
            except (TypeError, ValueError):
                continue
            e0 = s0 + timedelta(seconds=float(r.get("seconds") or 0))
            for cs, ce, c in clips:
                a, b = datetime.fromisoformat(cs), datetime.fromisoformat(ce)
                if a <= e0 and b >= s0:
                    r["clip"] = {"name": c["name"], "day": c["day"], "time": c["time"], "seconds": c.get("seconds", 0),
                                 "thumb": c.get("thumb"), "boxes": c.get("boxes"), "manual": c.get("manual"),
                                 "offset": max(0.0, round((s0 - a).total_seconds() - 1, 1))}
                    break
    return jsonify(hours=hours, zones_set=alerts.zone_kinds(current_cfg()), summary=out, recent=recent, estimate=est, wheel_m=wheel_m,
                   zone_defs=_zone_defs())


def _daily(days):
    """One row per calendar day (newest first): activity, distance, each zone's time and visits,
    temperature/humidity, and how the AI did. Days with nothing recorded are included (has_data = False)."""
    today = datetime.now().date()
    first = today - timedelta(days=days - 1)
    d0 = first.isoformat()
    out = {}
    kinds = _zone_ids()
    for i in range(days):
        d = (today - timedelta(days=i)).isoformat()
        out[d] = {"day": d, "has_data": False, "active_min": 0, "distance_m": 0.0,
                  "zones": {k: {"seconds": 0, "visits": 0} for k in kinds},
                  "temp": None, "temp_min": None, "temp_max": None, "hum": None, "ai": None,
                  "eat": {"count": 0, "seconds": 0}, "drink": {"count": 0, "seconds": 0}, "run": {"count": 0, "seconds": 0},
                  "wheel_m": 0.0, "act_rows": 0, "ai_rows": 0}

    def q(con, sql):
        try:
            return con.execute(sql, (d0,)).fetchall()
        except sqlite3.OperationalError:   # table not created yet (e.g. no AI model so far)
            return []

    with db() as con:
        for r in q(con, "SELECT substr(minute,1,10) d, SUM(active_s) a, SUM(distance_cm) m, COUNT(*) n FROM activity "
                        "WHERE minute >= ? GROUP BY d"):
            if r["d"] in out:
                out[r["d"]].update(active_min=round((r["a"] or 0) / 60), distance_m=round((r["m"] or 0) / 100, 1), has_data=True,
                                   act_rows=r["n"])
        for r in q(con, "SELECT substr(minute,1,10) d, zone, SUM(seconds) s FROM zone_minutes WHERE minute >= ? GROUP BY d, zone"):
            if r["d"] in out and r["zone"] in kinds:
                out[r["d"]]["zones"][r["zone"]]["seconds"] = r["s"] or 0
                out[r["d"]]["has_data"] = True
        for r in q(con, "SELECT substr(start,1,10) d, zone, COUNT(*) n FROM zone_visits WHERE start >= ? GROUP BY d, zone"):
            if r["d"] in out and r["zone"] in kinds:
                out[r["d"]]["zones"][r["zone"]]["visits"] = r["n"]
        for r in q(con, "SELECT substr(minute,1,10) d, SUM(cm) c FROM wheel_minutes WHERE minute >= ? GROUP BY d"):
            if r["d"] in out:
                out[r["d"]]["wheel_m"] = round((r["c"] or 0) / 100, 1)
        for r in q(con, "SELECT substr(ts,1,10) d, AVG(temp) t, MIN(temp) tmin, MAX(temp) tmax, AVG(hum) h "
                        "FROM climate WHERE ts >= ? AND temp IS NOT NULL GROUP BY d"):
            if r["d"] in out:
                out[r["d"]].update(temp=round(r["t"], 1), temp_min=round(r["tmin"], 1), temp_max=round(r["tmax"], 1),
                                   hum=round(r["h"]), has_data=True)
        for r in q(con, "SELECT substr(start,1,10) d, kind, COUNT(*) n, SUM(seconds) s FROM behavior_events "
                        "WHERE start >= ? GROUP BY d, kind"):
            if r["d"] in out and r["kind"] in ("eat", "drink", "run"):
                out[r["d"]][r["kind"]] = {"count": r["n"], "seconds": r["s"] or 0}
        # the tracker writes one activity row and one ai_minutes row per minute it runs: far fewer activity rows than
        # AI minutes = that day's hamster records were cleared (or partly lost), not "the hamster did nothing"
        for r in q(con, "SELECT substr(minute,1,10) d, SUM(runs) runs, SUM(seen) seen, SUM(unsure) unsure, "
                        "SUM(conf_sum) cs, MAX(conf_max) cm, COUNT(*) n FROM ai_minutes WHERE minute >= ? GROUP BY d"):
            if r["d"] in out:
                out[r["d"]]["ai_rows"] = r["n"]
            if r["d"] in out and r["runs"]:
                out[r["d"]]["ai"] = {"runs": r["runs"], "seen": r["seen"], "unsure": r["unsure"],
                                     "avg_conf": round(r["cs"] / r["seen"], 3) if r["seen"] else None,
                                     "max_conf": round(r["cm"], 3) if r["seen"] else None}
    live_s, live_cm = _live_minute()   # today's row includes the minute being recorded (same as the home page)
    t = out.get(today.isoformat())
    if t and (live_s or live_cm):
        # re-derive from the exact sums so rounding matches the home page
        with db() as con:
            a = con.execute("SELECT COALESCE(SUM(active_s),0), COALESCE(SUM(distance_cm),0) FROM activity WHERE minute >= ?",
                            (today.isoformat(),)).fetchone()
        t.update(active_min=round((a[0] + live_s) / 60), distance_m=round((a[1] + live_cm) / 100, 1), has_data=True)
    return sorted(out.values(), key=lambda x: x["day"], reverse=True)


@app.get("/api/daily")
@require_auth
def daily():
    """?days=7|30 (1..366)"""
    days = int_arg(request.args.get("days"), 7, 1, 366)
    return no_cache(jsonify(days=_daily(days), zones_set=alerts.zone_kinds(current_cfg()), zone_defs=_zone_defs(),
                            detector=current_cfg().get("tracker", {}).get("detector", "motion")))


@app.get("/download/daily.csv")
@require_auth
def daily_csv():
    """The same table as a spreadsheet (opens in Excel / Google Sheets; UTF-8 with BOM so Chinese shows correctly)."""
    import csv
    import io
    days = int_arg(request.args.get("days"), 30, 1, 366)
    names = {d["id"]: d["name"] for d in _zone_defs()}
    buf = io.StringIO()
    w = csv.writer(buf)
    today = datetime.now().date().isoformat()

    def status(r):   # same rules as the table on the page
        tracked = r["active_min"] > 0 or any(z["seconds"] > 0 for z in r["zones"].values()) or bool(r["ai"] and r["ai"].get("runs"))
        cover = min(1, r["act_rows"] / r["ai_rows"]) if r["ai_rows"] > 30 else 1
        if not r["has_data"]:
            return "沒有紀錄", False
        if not tracked and r["day"] != today:
            return "相機追蹤沒在跑（只有溫濕度）", False
        if cover < 0.05:
            return "倉鼠紀錄已清除（只剩溫濕度和 AI 統計）", False
        if cover < 0.9:
            return f"資料不完整（約 {round(cover * 100)}% 的時間）", True
        return ("今天（還在記錄）" if r["day"] == today else "完整"), True

    w.writerow(["日期", "紀錄狀態", "活動(分鐘)", "移動距離(公尺)"] +
               [x for k in names for x in (f"{names[k]}(分鐘)", f"{names[k]}(次)")] +
               ["跑滾輪推估(次)", "跑滾輪推估(分鐘)", "吃東西推估(次)", "吃東西推估(分鐘)", "喝水推估(次)", "喝水推估(分鐘)", "滾輪距離(公尺)"] +
               ["平均溫度", "最低溫度", "最高溫度", "平均濕度", "AI辨識次數", "AI看到倉鼠", "AI不確定", "AI平均信心(%)", "AI最高信心(%)"])
    for r in sorted(_daily(days), key=lambda x: x["day"]):
        ai = r["ai"] or {}
        pct = lambda v: round(v * 100) if v is not None else ""   # noqa: E731
        st, ham = status(r)
        hv = (lambda v: v) if ham else (lambda v: "")   # no hamster records that day: blank, never a fake 0
        w.writerow([r["day"], st, hv(r["active_min"]), hv(r["distance_m"])] +
                   [hv(x) for k in names if k in r["zones"] for x in (round(r["zones"][k]["seconds"] / 60, 1), r["zones"][k]["visits"])] +
                   [hv(x) for k in ("run", "eat", "drink") for x in (r[k]["count"], round(r[k]["seconds"] / 60, 1))] + [hv(r.get("wheel_m", 0))] +
                   [r["temp"] if r["temp"] is not None else "", r["temp_min"] if r["temp_min"] is not None else "",
                    r["temp_max"] if r["temp_max"] is not None else "", r["hum"] if r["hum"] is not None else "",
                    ai.get("runs", ""), ai.get("seen", ""), ai.get("unsure", ""), pct(ai.get("avg_conf")), pct(ai.get("max_conf"))])
    data = ("﻿" + buf.getvalue()).encode("utf-8")
    resp = Response(data, mimetype="text/csv")
    resp.headers["Content-Disposition"] = f"attachment; filename=hamster_daily_{days}d_{datetime.now():%Y%m%d}.csv"
    return no_cache(resp)


def _points(value):
    """List of [x, y] pairs -> rounded pairs clamped to 0..1 (drops anything that isn't a pair of numbers)."""
    out = []
    for p in value if isinstance(value, list) else []:
        try:
            x, y = float(p[0]), float(p[1])
        except (TypeError, ValueError, IndexError, KeyError):
            continue
        if x == x and y == y:   # skip NaN
            out.append([round(min(max(x, 0.0), 1.0), 4), round(min(max(y, 0.0), 1.0), 4)])
    return out


@app.post("/api/calib")
@require_auth
def calib_set():
    body = request.get_json(silent=True)
    if body is None:
        app.logger.warning("calib save from %s: body is not JSON (%s)", request.remote_addr, request.content_type)
        return jsonify(ok=False, error="收到的資料格式不對，請重新整理網頁再試"), 400
    # the editor always sends both; a request without them must never wipe the calibration
    if not isinstance(body, dict) or not isinstance(body.get("roi"), list) or not isinstance(body.get("floor"), list):
        app.logger.warning("calib save from %s refused: roi/floor missing", request.remote_addr)
        return jsonify(ok=False, error="收到的資料缺少偵測範圍或籠底四角，請重新整理網頁再試"), 400
    if "zones" in body and not isinstance(body["zones"], list):
        return jsonify(ok=False, error="區域資料格式不對"), 400
    spout = None   # None = not sent (keep), [] = removed, [x, y] = the water bottle's spout
    if "spout" in body:
        sp = body["spout"]
        if sp in (None, []):
            spout = []
        else:
            pts = _points([sp]) if isinstance(sp, list) and len(sp) == 2 else []
            if len(pts) != 1:
                return jsonify(ok=False, error="出水口的位置格式不對"), 400
            spout = list(pts[0])
    spot = None   # None = not sent (keep), [] = back to the default circle, [cx, cy, r] = moved / resized
    if "drink_spot" in body:
        spot, err = _spot_value(body["drink_spot"])
        if err:
            return jsonify(ok=False, error=err), 400
    roi, floor = _points(body.get("roi")), _points(body.get("floor"))
    # an all-empty save wipes the whole calibration: refuse it unless the editor says it's on purpose
    # (a page that failed to load the settings once saved nothing over everything)
    if not roi and not floor and not body.get("zones") and not body.get("spout") and not body.get("clear_all"):
        t_now = current_cfg().get("tracker", {})
        if t_now.get("roi") or t_now.get("floor") or t_now.get("zones") or t_now.get("spout"):
            app.logger.warning("calib save from %s refused: everything empty", request.remote_addr)
            return jsonify(ok=False, error="收到的資料全部是空的，沒有儲存。請重新整理網頁、重開編輯器再試"), 409
    app.logger.warning("calib save from %s: roi %d pts, floor %d pts (raw lengths %s / %s)", request.remote_addr,
                       len(roi), len(floor), len(body.get("roi") or []), len(body.get("floor") or []))
    if roi and len(roi) < 3:
        return jsonify(ok=False, error=f"偵測範圍只收到 {len(roi)} 個點，至少要 3 個"), 400
    if floor and len(floor) != 4:
        return jsonify(ok=False, error=f"籠底收到 {len(floor)} 個點，要剛好 4 個"), 400
    sides = None
    if body.get("floor_cm") is not None:   # real lengths of the sides 1→2, 2→3, 3→4, 4→1 (cm)
        from floor_geom import floor_shape
        try:
            sides = [round(float(v), 1) for v in body["floor_cm"]]
        except (TypeError, ValueError):
            sides = []
        if len(sides) != 4 or not all(3 <= v <= 300 for v in sides):
            return jsonify(ok=False, error="籠底四邊的長度要填 4 個 3～300 之間的數字（公分）"), 400
        if floor_shape(sides) is None:
            return jsonify(ok=False, error="這四個長度拼不成梯形或長方形，請再量一次（1→2 和 4→3 要是平行的兩邊）"), 400
    zones = []
    types = zonedefs.zone_types(current_cfg().get("tracker", {}))
    for z in body.get("zones") or []:
        if not isinstance(z, dict) or z.get("kind") not in types or z.get("kind") == "water":
            continue   # the built-in water bottle is saved as spout / drink_spot (above)
        if types[z["kind"]] == "drink":
            # an added 喝水 area = its own bottle: a spout + the circle under it (no outline), like 水瓶
            sp = _points([z.get("spout")]) if z.get("spout") else []
            if not sp:
                continue
            zs, err = _spot_value(z.get("spot"))
            if err:
                return jsonify(ok=False, error=err), 400
            zones.append({"kind": z["kind"], "poly": [], "spout": list(sp[0]), **({"spot": zs} if zs else {})})
            continue
        poly = _points(z.get("poly"))
        if len(poly) >= 3:
            zones.append({"kind": z["kind"], "poly": poly})
    fmt = lambda pts: "[" + ", ".join(f"[{x}, {y}]" for x, y in pts) + "]"
    zline = _zones_line(zones)
    text = original = CONFIG_FILE.read_text(encoding="utf-8")
    text, n1 = re.subn(r"(?m)^roi = .*$", lambda _: "roi = " + fmt(roi), text, count=1)
    text, n2 = re.subn(r"(?m)^floor = .*$", lambda _: "floor = " + fmt(floor), text, count=1)
    if not (n1 and n2):
        return jsonify(ok=False, error="config.toml 裡找不到 roi / floor 設定"), 500
    n3 = 1 if "zones" not in body else 0   # zones not sent: keep the saved ones
    if "zones" in body:
        text, n3 = re.subn(r"(?m)^zones = .*$", lambda _: zline, text, count=1)
    if not n3:   # first time: add right after the floor line
        text = re.sub(r"(?m)^(floor = .*)$", lambda m: m.group(1) + "\n# 行為區域（用網頁「框選籠子範圍 → 區域」畫）\n" + zline, text, count=1)
    if sides:
        sline = "floor_sides_cm = [" + ", ".join(f"{v:g}" for v in sides) + "]"
        text, n4 = re.subn(r"(?m)^floor_sides_cm = .*$", lambda _: sline, text, count=1)
        if not n4:   # first time: add right after the floor line
            text = re.sub(r"(?m)^(floor = .*)$", lambda m: m.group(1) + "\n# 籠底 4 個點之間的實際長度 (cm)：1→2, 2→3, 3→4, 4→1（網頁「籠底四角」填）\n" + sline,
                          text, count=1)
    if spout is not None:
        spline = "spout = " + (f"[{spout[0]}, {spout[1]}]" if spout else "[]")
        text, n5 = re.subn(r"(?m)^spout = .*$", lambda _: spline, text, count=1)
        if not n5:   # first time: add right after the floor line
            text = re.sub(r"(?m)^(floor = .*)$", lambda m: m.group(1) + "\n# 水瓶出水口的位置（網頁「出水口」點），用來判斷喝水\n" + spline, text, count=1)
    if spot is not None:
        dline = "drink_spot = " + (f"[{spot[0]}, {spot[1]}, {spot[2]}]" if spot else "[]")
        text, n6 = re.subn(r"(?m)^drink_spot = .*$", lambda _: dline, text, count=1)
        if not n6:   # first time: add right after the floor line
            text = re.sub(r"(?m)^(floor = .*)$", lambda m: m.group(1) + "\n# 喝水位置（水瓶區域）[中心x, 中心y, 半徑]；[] = 出水口正下方的預設圓\n" + dline,
                          text, count=1)
    tomllib.loads(text)                     # never write a config that would not parse
    (DATA / "config.toml.bak-calib").write_text(original, encoding="utf-8")   # the previous calibration, just in case
    tmp = CONFIG_FILE.with_suffix(".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, CONFIG_FILE)
    subprocess.run(["sudo", "-n", "systemctl", "restart", "hamster-tracker"], capture_output=True, timeout=30)
    t = current_cfg().get("tracker", {})
    return jsonify(ok=True, roi=roi, floor=floor, zones=zones, floor_cm=sides or _floor_sides(t), spout=_spout(t),
                   drink_spot=_drink_spot(t))


def _spot_value(ds):
    """A drinking circle [cx, cy, r] (picture fractions, r of the width) -> (value, error); [] / None = default."""
    if ds in (None, []):
        return [], None
    try:
        cx, cy, r = (float(v) for v in ds)
    except (TypeError, ValueError):
        return None, "喝水位置的格式不對"
    if not (0 <= cx <= 1 and 0 <= cy <= 1 and 0.02 <= r <= 0.3):
        return None, "喝水位置要在畫面裡，大小在 2%～30% 之間"
    return [round(cx, 4), round(cy, 4), round(r, 4)], None


def _zone_ids():
    return zonedefs.zone_ids(current_cfg().get("tracker", {}))


def _zone_defs():
    return zonedefs.zone_defs(current_cfg().get("tracker", {}))


def _zones_line(zones):
    fmt = lambda pts: "[" + ", ".join(f"[{x}, {y}]" for x, y in pts) + "]"   # noqa: E731
    return "zones = [" + ", ".join(f'{{kind = "{z["kind"]}", poly = {fmt(z["poly"])}' +
                                   (f', spout = [{z["spout"][0]}, {z["spout"][1]}]' if z.get("spout") else "") +
                                   (f', spot = [{z["spot"][0]}, {z["spot"][1]}, {z["spot"][2]}]' if z.get("spot") else "") + "}"
                                   for z in zones) + "]"


OVERLAY_LAYERS = ("box", "head", "roi", "floor", "time")   # same list as tracker.py (+ "zones_off": area ids)


def _overlay():
    d = read_json(DATA / "overlay.json") or {}
    out = {k: bool(d.get(k, True)) for k in OVERLAY_LAYERS}
    out["zones_off"] = [z for z in (d.get("zones_off") or []) if isinstance(z, str) and zonedefs.ID_RE.match(z)]
    return out


@app.get("/api/overlay")
@require_auth
def overlay_get():
    """Which layers are drawn on the dashboard picture (the tracker re-reads the file every second)."""
    return no_cache(jsonify(_overlay()))


@app.post("/api/overlay")
@require_auth
def overlay_set():
    body = request.get_json(silent=True) or {}
    cur = _overlay()
    cur.update({k: bool(body[k]) for k in OVERLAY_LAYERS if k in body})
    if isinstance(body.get("zones_off"), list):
        cur["zones_off"] = sorted({z for z in body["zones_off"] if isinstance(z, str) and zonedefs.ID_RE.match(z)})[:24]
    tmp = DATA / "overlay.json.tmp"
    tmp.write_text(json.dumps(cur), encoding="utf-8")
    os.replace(tmp, DATA / "overlay.json")
    return jsonify(ok=True, **cur)


# ---------- 全自動訓練 (Kaggle): autotrain.py does the work; this starts it and shows how far it is ----------
AT_STATE = DATA / "autotrain.json"
KAGGLE_KEY = Path.home() / ".kaggle" / "kaggle.json"


def _at_state():
    st = read_json(AT_STATE) or {}
    pid = st.get("pid")
    alive = False
    if pid and st.get("state") == "running":
        try:
            os.kill(int(pid), 0)
            alive = True
        except (OSError, ValueError):
            pass
    if st.get("state") == "running" and not alive:   # the Pi restarted / the process died half way
        st.update(state="error", text="上次的自動訓練中斷了（樹莓派重開或程式停止），請重新開始")
    return st


@app.get("/api/autotrain")
@require_auth
def autotrain_status():
    names = {p.name for p in _photos()}
    labd = {n for n in _labeled() if n in names}   # a deleted photo is no longer "labelled" (same count as the AI panel)
    trained = {p.name for p in (DATA / "trained").glob("*.jpg")} if (DATA / "trained").exists() else set()
    exam_done = sum(1 for i in _exam_items() if i["done"])
    try:
        kuser = json.loads(KAGGLE_KEY.read_text()).get("username") if KAGGLE_KEY.exists() else None
    except (OSError, ValueError):
        kuser = None
    return no_cache(jsonify(**_at_state(), key=KAGGLE_KEY.exists(), kaggle_user=kuser,
                            kaggle_ready=(BASE / ".venv" / "bin" / "kaggle").exists(),
                            labeled=len(labd | trained), labeled_new=len(labd - trained - _banned(list(labd - trained))), trained=len(trained),
                            exam_done=exam_done, min_labeled=30, min_exam=10))


@app.post("/api/autotrain/key")
@require_auth
def autotrain_key():
    """Kaggle user name + API key typed in by the owner (kaggle.com -> Settings -> API). Stored only on the Pi."""
    d = request.get_json(silent=True) or {}
    d = {"username": str(d.get("username", "")).strip(), "key": str(d.get("key", "")).strip()}
    if not re.fullmatch(r"[A-Za-z0-9_-]{1,64}", d["username"]) or not re.fullmatch(r"[A-Za-z0-9_-]{8,200}", d["key"]):
        return jsonify(ok=False, error="帳號名稱或 API 金鑰格式不對（只能是英文、數字、底線、減號）"), 400
    KAGGLE_KEY.parent.mkdir(mode=0o700, exist_ok=True)
    tmp = KAGGLE_KEY.with_suffix(".tmp")
    tmp.write_text(json.dumps({"username": str(d["username"]), "key": str(d["key"])}))
    os.chmod(tmp, 0o600)
    os.replace(tmp, KAGGLE_KEY)
    tok = KAGGLE_KEY.parent / "access_token"   # the newer Kaggle tokens (KGAT_...) are read from here by the CLI
    if d["key"].startswith("KGAT_"):
        tok.write_text(d["key"])
        os.chmod(tok, 0o600)
    else:
        tok.unlink(missing_ok=True)
    return jsonify(ok=True, username=d["username"])


@app.post("/api/autotrain/start")
@require_auth
def autotrain_start():
    if _at_state().get("state") == "running":
        return jsonify(ok=False, error="自動訓練正在進行中"), 409
    if not KAGGLE_KEY.exists():
        return jsonify(ok=False, error="還沒上傳 Kaggle 金鑰"), 400
    (DATA / "autotrain.pause").unlink(missing_ok=True)
    AT_STATE.write_text(json.dumps({"state": "running", "step": 0, "text": "開始", "started": datetime.now().isoformat(timespec="seconds")}))
    log = open(DATA / "autotrain.log", "w")
    bench = (request.get_json(silent=True) or {}).get("bench", True) is not False
    p = subprocess.Popen([sys.executable, str(BASE / "autotrain.py")] + ([] if bench else ["--no-bench"]),
                         cwd=BASE, start_new_session=True, stdout=log, stderr=log)
    st = read_json(AT_STATE) or {}
    st["pid"] = p.pid
    AT_STATE.write_text(json.dumps(st))
    return jsonify(ok=True)


@app.post("/api/autotrain/pause")
@require_auth
def autotrain_pause():
    """暫停 / 繼續: the training holds at its next step (upload, waiting for Kaggle, download, 跑分)."""
    on = (request.get_json(silent=True) or {}).get("pause") is True
    flag = DATA / "autotrain.pause"
    if on:
        if _at_state().get("state") != "running":
            return jsonify(ok=False, error="現在沒有在訓練"), 409
        flag.write_text("1")
    else:
        flag.unlink(missing_ok=True)
    return jsonify(ok=True, paused=on)


@app.post("/api/autotrain/stop")
@require_auth
def autotrain_stop():
    """停止: end this training here. (A run already on Kaggle can't be called back; it finishes there unused.)"""
    st = read_json(AT_STATE) or {}
    pid = st.get("pid")
    if st.get("state") == "running" and pid:
        import signal
        for sig in (signal.SIGTERM, signal.SIGKILL):
            try:
                os.killpg(int(pid), sig)
            except (OSError, ValueError):
                break
            time.sleep(1.5)
    (DATA / "autotrain.pause").unlink(missing_ok=True)
    st.update(state="stopped", paused=False, text="已停止（照片和模型都沒有改變）", updated=datetime.now().isoformat(timespec="seconds"))
    AT_STATE.write_text(json.dumps(st, ensure_ascii=False))
    return jsonify(ok=True)


@app.post("/api/autotrain/import")
@require_auth
def autotrain_import():
    """Photos labelled elsewhere (e.g. the X-AnyLabeling zip used for an earlier model): kept as training data."""
    import zipfile
    f = request.files.get("zip")
    if not f:
        return jsonify(ok=False, error="沒有檔案"), 400
    tr = DATA / "trained"
    tr.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(suffix=".zip", delete=False) as tf:
        f.save(tf.name)
    n = lab = 0
    try:
        with zipfile.ZipFile(tf.name) as z:
            names = {Path(m).name: m for m in z.namelist() if not m.endswith("/")}
            for base, member in names.items():
                stem, ext = os.path.splitext(base)
                if ext.lower() not in (".jpg", ".jpeg", ".png") or base.startswith("."):
                    continue
                dst = tr / ("import_" + re.sub(r"[^A-Za-z0-9_.-]", "_", stem) + ".jpg")
                if dst.exists():
                    continue
                data = z.read(member)
                if ext.lower() == ".png":
                    import cv2
                    import numpy as np
                    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
                    if img is None:
                        continue
                    cv2.imwrite(str(dst), img, [cv2.IMWRITE_JPEG_QUALITY, 92])
                else:
                    dst.write_bytes(data)
                txt = names.get(stem + ".txt")
                lines = z.read(txt).decode("utf-8", "replace") if txt else ""
                dst.with_suffix(".txt").write_text(lines)
                n += 1
                lab += bool(lines.strip())
    except zipfile.BadZipFile:
        return jsonify(ok=False, error="不是 zip 檔"), 400
    finally:
        os.unlink(tf.name)
    return jsonify(ok=True, added=n, with_boxes=lab, total=len(list(tr.glob("*.jpg"))))


# ---------- 跑分 (benchmark): photos = the exam store (data/exam), videos + their timelines in data/bench/clips ----------
BENCH = DATA / "bench"
BCLIPS = BENCH / "clips"


def _bench_clip_infos():
    out = []
    for j in sorted(BCLIPS.glob("clip_*.json")) if BCLIPS.exists() else []:
        if j.name.endswith(".lab.json") or not j.with_suffix(".mp4").exists():
            continue
        try:
            info = json.loads(j.read_text())
        except (OSError, ValueError):
            continue
        lab = read_json(j.with_name(j.stem + ".lab.json")) or {}
        m = CLIP_RE.match(j.stem)
        d, tm = m.groups() if m else ("", "000000")
        out.append({"name": j.stem, "time": f"{d[4:6]}/{d[6:]} {tm[:2]}:{tm[2:4]}" if d else "", "seconds": info.get("seconds"),
                    "zones": info.get("zones") or {}, "checked": bool(lab.get("checked")), "draft": bool(lab.get("draft")),
                    "segments": lab.get("segments") or [], "turns": lab.get("turns"), "thumb": j.with_suffix(".jpg").exists()})
    return out


@app.get("/api/bench")
@require_auth
def bench_status():
    items = _exam_items()
    done = [i for i in items if i["done"]]
    bins = {"ok": 0, "dim": 0, "dark": 0}
    for i in done:
        if i.get("dark") in bins:
            bins[i["dark"]] += 1
    clips = _bench_clip_infos()
    st = read_json(Path("/run/hamster/bench.json")) or {}
    running = bool(st.get("running")) and time.time() - st.get("updated", 0) < 300
    return no_cache(jsonify(results=read_json(BENCH / "results.json"), models=_exam_models(), zone_defs=_zone_defs(),
                            run={"running": running, "text": st.get("text"), "done": st.get("done"), "total": st.get("total"),
                                 "error": st.get("error")},
                            photos={"total": len(items), "done": len(done), "with_ham": sum(1 for i in done if i["boxes"]),
                                    "with_head": sum(1 for i in done if i["heads"]), "by_dark": bins},
                            clips={"total": len(clips), "checked": sum(c["checked"] for c in clips),
                                   "turns": sum(1 for c in clips if c["turns"] not in (None, ""))}))


@app.post("/api/bench/run")
@require_auth
def bench_run():
    body = request.get_json(silent=True) or {}
    ok_dirs = {m["dir"] for m in _exam_models()}
    dirs = [d for d in body.get("models") or [] if d in ok_dirs]   # no limit: the owner picks as many as they like
    if not dirs:
        return jsonify(ok=False, error="至少選一個模型"), 400
    st = read_json(Path("/run/hamster/bench.json")) or {}
    if st.get("running") and time.time() - st.get("updated", 0) < 300:
        return jsonify(ok=False, error="跑分正在進行中"), 409
    Path("/run/hamster").mkdir(parents=True, exist_ok=True)
    (Path("/run/hamster") / "bench.json").write_text(json.dumps({"running": True, "text": "開始", "updated": time.time()}))
    subprocess.Popen([sys.executable, str(BASE / "bench_run.py"), *[f"models/{d}" for d in dirs]], cwd=BASE,
                     start_new_session=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return jsonify(ok=True)


@app.get("/api/bench/clips")
@require_auth
def bench_clips():
    return no_cache(jsonify(clips=_bench_clip_infos(), zone_defs=_zone_defs()))


@app.post("/api/bench/clips/add")
@require_auth
def bench_clips_add():
    """Copy recorded clips into the benchmark (a copy: the clip list may delete old clips by itself)."""
    body = request.get_json(silent=True) or {}
    names = [str(x) for x in body.get("names") or [] if CLIP_RE.match(str(x))][:200]
    names = [x for x in names if not (BCLIPS / f"{x}.mp4").exists() and (CLIPS / f"{x}.mp4").exists()]
    spans = []
    for x in names:   # the same window training leaves out: the clip plus BENCH_NEAR_S on each side
        info = read_json(CLIPS / f"{x}.json") or {}
        try:
            a = datetime.fromisoformat(info["start"]).timestamp()
        except (KeyError, ValueError, TypeError):
            continue
        spans.append((a - BENCH_NEAR_S, a + float(info.get("seconds") or 10) + BENCH_NEAR_S))
    hit_trained, hit_labelled = _training_hits(spans)
    if body.get("check"):
        return jsonify(ok=True, clips=len(names), trained=hit_trained, labelled=hit_labelled)
    BCLIPS.mkdir(parents=True, exist_ok=True)
    n = 0
    for name in names:
        for ext in (".mp4", ".json", ".jpg"):
            if (CLIPS / f"{name}{ext}").exists():
                shutil.copy2(CLIPS / f"{name}{ext}", BCLIPS / f"{name}{ext}")
        (BCLIPS / f"{name}.lab.json").write_text(json.dumps({"segments": [], "turns": None, "checked": False}))
        n += 1
    _ai_cache["t"] = 0
    return jsonify(ok=True, added=n, total=len(_bench_clip_infos()), trained=hit_trained, labelled=hit_labelled)


@app.post("/api/clips/frame")
@require_auth
def clip_frame_to_dataset():
    """截圖到訓練照片: the frame at {t} seconds of a recorded clip becomes an (unlabelled) training photo,
    named by the moment it was filmed, like the photos the tracker saves itself."""
    import cv2
    body = request.get_json(silent=True) or {}
    name = str(body.get("name", ""))
    if not CLIP_RE.match(name) or not (CLIPS / f"{name}.mp4").exists():
        return jsonify(ok=False, error="找不到這段影片"), 404
    try:
        sec = max(0.0, float(body.get("t", 0)))
    except (TypeError, ValueError):
        return jsonify(ok=False, error="時間不對"), 400
    cap = cv2.VideoCapture(str(CLIPS / f"{name}.mp4"))
    fps = cap.get(cv2.CAP_PROP_FPS) or 10
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(round(sec * fps)))
    ok, fr = cap.read()
    cap.release()
    if not ok or fr is None:
        return jsonify(ok=False, error="這個時間點讀不到畫面"), 400
    info = read_json(CLIPS / f"{name}.json") or {}
    try:
        t0 = datetime.fromisoformat(info["start"])
    except (KeyError, ValueError):
        t0 = datetime.strptime(name[5:], "%Y%m%d_%H%M%S")
    when = t0 + timedelta(seconds=sec)
    DATASET.mkdir(parents=True, exist_ok=True)
    while any((DATASET / f"{pf}_{when:%Y%m%d_%H%M%S}.jpg").exists() for pf in ("motion", "idle", "hard_unsure", "hard_missed", "upload")):
        when += timedelta(seconds=1)   # a photo of that very second exists already
    out = f"motion_{when:%Y%m%d_%H%M%S}.jpg"
    h, w = fr.shape[:2]
    if (w, h) != (640, 480):
        fr = cv2.resize(fr, (640, 480), interpolation=cv2.INTER_AREA)
    cv2.imwrite(str(DATASET / out), fr, [cv2.IMWRITE_JPEG_QUALITY, 92])
    with open(DATASET / "zones.jsonl", "a", encoding="utf-8") as f:   # a cage moment: something moved (it was recorded)
        f.write(json.dumps({"n": out, "m": 1}) + "\n")
    _ai_cache["t"] = 0
    near = _near_exam(out, _exam_times())   # right next to a 跑分 photo: training leaves it out (no leak)
    app.logger.info("clip frame %s +%.1fs -> %s", name, sec, out)
    return jsonify(ok=True, photo=out, time=when.strftime("%m/%d %H:%M:%S"), near_exam=near)


@app.post("/api/clips/to_lib")
@require_auth
def clips_to_lib():
    """加入錄影紀錄: your own videos (錄影) show in 設定 › 錄影紀錄 only after this."""
    n = already = 0
    for name in [str(x) for x in (request.get_json(silent=True) or {}).get("names") or [] if CLIP_RE.match(str(x))][:500]:
        j = CLIPS / f"{name}.json"
        info = read_json(j)
        if not info:
            continue
        if not info.get("manual") or info.get("in_lib"):
            already += 1
            continue
        info["in_lib"] = True
        tmp = j.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(info, ensure_ascii=False))
        os.replace(tmp, j)
        n += 1
    return jsonify(ok=True, added=n, already=already)


@app.post("/api/bench/clips/remove")
@require_auth
def bench_clips_remove():
    body = request.get_json(silent=True) or {}
    n = 0
    for name in [str(x) for x in body.get("names") or [] if CLIP_RE.match(str(x))]:
        for ext in (".mp4", ".json", ".jpg", ".lab.json"):
            p = BCLIPS / f"{name}{ext}"
            if p.exists():
                p.unlink()
                n += ext == ".mp4"
    return jsonify(ok=True, removed=n)


@app.post("/api/bench/clips/label")
@require_auth
def bench_clips_label():
    """The owner's timeline of what really happened in a clip: segments [{t, s, e}] (t = drink / eat / run /
    zone:<id>, seconds from the clip start), the wheel turns counted by hand, and whether it is checked."""
    body = request.get_json(silent=True) or {}
    name = str(body.get("name", ""))
    p = BCLIPS / f"{name}.lab.json"
    if not CLIP_RE.match(name) or not p.exists():
        return jsonify(ok=False, error="找不到這段影片"), 404
    ok_t = {"drink", "eat", "run"} | {f"zone:{z}" for z in _zone_ids()}
    segs = []
    for s in (body.get("segments") or [])[:200]:
        try:
            a, b = round(float(s["s"]), 1), round(float(s["e"]), 1)
        except (KeyError, TypeError, ValueError):
            continue
        if s.get("t") in ok_t and b > a >= 0:
            segs.append({"t": s["t"], "s": a, "e": b})
    turns = body.get("turns")
    try:
        turns = None if turns in (None, "") else round(max(0.0, float(turns)), 2)
    except (TypeError, ValueError):
        turns = None
    lab = {"segments": segs, "turns": turns, "checked": bool(body.get("checked")), "draft": False,
           "saved": datetime.now().isoformat(timespec="seconds")}
    tmp = p.with_suffix(".tmp")
    tmp.write_text(json.dumps(lab, ensure_ascii=False))
    os.replace(tmp, p)
    return jsonify(ok=True)


@app.get("/bench/clip/<name>.mp4")
@require_auth
def bench_clip_video(name):
    p = BCLIPS / f"{name}.mp4"
    return send_file(p, mimetype="video/mp4", conditional=True) if CLIP_RE.match(name or "") and p.exists() else ("not found", 404)


@app.get("/bench/clip/<name>.jpg")
@require_auth
def bench_clip_thumb(name):
    p = BCLIPS / f"{name}.jpg"
    return send_file(p, mimetype="image/jpeg") if CLIP_RE.match(name or "") and p.exists() else ("not found", 404)


@app.get("/download/bench_set.zip")
@require_auth
def bench_set_download():
    """跑分照片 in the standard YOLO layout, for Ultralytics' own val (手動跑分 in the Colab notebook)."""
    import io
    import zipfile
    buf = io.BytesIO()
    n = 0
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        for i in _exam_items():
            if not i["done"]:
                continue
            stem = Path(i["name"]).stem
            z.write(EXAM / i["name"], f"hamster_bench/images/test/{i['name']}")
            lines = "".join(f"0 {b[0]} {b[1]} {b[2]} {b[3]}\n" for b in i["boxes"]) + \
                "".join(f"1 {b[0]} {b[1]} {b[2]} {b[3]}\n" for b in i["heads"])
            z.writestr(f"hamster_bench/labels/test/{stem}.txt", lines)
            n += 1
        z.writestr("hamster_bench/data.yaml", "path: /content/hamster_bench\ntrain: images/test\nval: images/test\ntest: images/test\n"
                                              "names:\n  0: hamster\n  1: head\n")
        z.writestr("hamster_bench/README.txt",
                   f"倉鼠跑分照片 {n} 張（人工標註，從未用於訓練）\n格式：YOLO（images/test + labels/test，0 = hamster，1 = head）\n"
                   "用 Colab 訓練筆記本最後的「手動跑分」那一格上傳這個 zip，會用 Ultralytics 官方 val 算 mAP 並畫圖。\n")
    buf.seek(0)
    return send_file(buf, as_attachment=True, download_name="hamster_bench.zip", mimetype="application/zip")


@app.get("/api/wheel")
@require_auth
def wheel_get():
    t = current_cfg().get("tracker", {})
    return no_cache(jsonify(diameter_cm=float(t.get("wheel_diameter_cm", 21)), strips=int(t.get("wheel_strips", 2))))


@app.post("/api/wheel")
@require_auth
def wheel_set():
    """Wheel size and how many strips are stuck on it (for the running distance)."""
    body = request.get_json(silent=True) or {}
    try:
        d, n = float(body.get("diameter_cm")), int(body.get("strips"))
    except (TypeError, ValueError):
        return jsonify(ok=False, error="要填數字"), 400
    if not (5 <= d <= 60):
        return jsonify(ok=False, error="直徑要在 5～60 公分之間"), 400
    if not (1 <= n <= 24):
        return jsonify(ok=False, error="紙條數量要在 1～24 之間"), 400
    text = original = CONFIG_FILE.read_text(encoding="utf-8")
    for key, val in (("wheel_diameter_cm", f"{d:g}"), ("wheel_strips", str(n))):
        line = f"{key} = {val}"
        text, k = re.subn(rf"(?m)^{key} = .*$", lambda _: line, text, count=1)
        if not k:
            text = re.sub(r"(?m)^(floor = .*)$", lambda m: m.group(1) + "\n" + line, text, count=1)
    cfg = tomllib.loads(text)
    if cfg.get("tracker", {}).get("wheel_diameter_cm") is None:
        return jsonify(ok=False, error="config.toml 的格式跟預期的不一樣，沒有儲存"), 500
    (DATA / "config.toml.bak-wheel").write_text(original, encoding="utf-8")
    tmp = CONFIG_FILE.with_suffix(".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, CONFIG_FILE)
    subprocess.run(["sudo", "-n", "systemctl", "restart", "hamster-tracker"], capture_output=True, timeout=30)
    return jsonify(ok=True, diameter_cm=d, strips=n)


@app.get("/api/zones")
@require_auth
def zones_get():
    return no_cache(jsonify(defs=_zone_defs(), builtin=list(zonedefs.BUILTIN), max=zonedefs.MAX_ZONES, types=list(zonedefs.TYPES)))


@app.post("/api/zones")
@require_auth
def zones_set():
    """The area list: [{id (empty = new), name, color "#rrggbb"}], in display order. A deleted area's outline is
    removed from [tracker] zones; its old records stay in the database (they show again if it is added back)."""
    body = request.get_json(silent=True) or {}
    raw = body.get("defs")
    if body.get("reset"):
        raw = zonedefs.defaults()
    if not isinstance(raw, list) or not raw:
        return jsonify(ok=False, error="至少要留一個區域"), 400
    if len(raw) > zonedefs.MAX_ZONES:
        return jsonify(ok=False, error=f"最多 {zonedefs.MAX_ZONES} 個區域"), 400
    defs, seen, names = [], set(), set()
    for d in raw:
        if not isinstance(d, dict):
            return jsonify(ok=False, error="資料不對"), 400
        zid = str(d.get("id") or "")
        if not zid:   # new area: a fresh id that never matches an old one (so it starts with no records)
            zid = "c" + os.urandom(3).hex()
        if not zonedefs.ID_RE.match(zid) or zid in seen:
            return jsonify(ok=False, error="資料不對"), 400
        name = zonedefs.clean_name(d.get("name"))
        if not name:
            return jsonify(ok=False, error="每個區域都要有名字"), 400
        if name in names:
            return jsonify(ok=False, error=f"「{name}」重複了，名字不能一樣"), 400
        color = str(d.get("color") or "")
        if not zonedefs.COLOR_RE.match(color):
            return jsonify(ok=False, error="顏色不對"), 400
        ztype = zonedefs.BUILTIN_TYPE.get(zid) or d.get("type") or "plain"
        if ztype not in zonedefs.TYPES and zid not in zonedefs.BUILTIN:
            return jsonify(ok=False, error="區域類型不對"), 400
        defs.append({"id": zid, "name": name, "color": color.lower(), "type": ztype})
        seen.add(zid)
        names.add(name)
    t = current_cfg().get("tracker", {})
    kept = [z for z in t.get("zones") or [] if isinstance(z, dict) and z.get("kind") in seen and z.get("kind") != "water"]
    text = original = CONFIG_FILE.read_text(encoding="utf-8")
    dline = zonedefs.toml_line(defs)
    text, n = re.subn(r"(?m)^zone_defs = .*$", lambda _: dline, text, count=1)
    if not n:   # first time: add right after the floor line
        text = re.sub(r"(?m)^(floor = .*)$", lambda m: m.group(1) + "\n# 行為區域清單（網頁「編輯區域」改：名字、顏色、新增、刪除）\n" + dline, text, count=1)
    types = {d["id"]: d["type"] for d in defs}
    # a 喝水 area is its spout + circle (no outline); changed to another type, it has to be outlined again
    kept = [z for z in kept if (types.get(z["kind"]) == "drink") == bool(z.get("spout"))]
    zl = _zones_line([{"kind": z["kind"], "poly": z.get("poly") or [],
                       **({"spout": z["spout"]} if z.get("spout") else {}), **({"spot": z["spot"]} if z.get("spot") else {})}
                      for z in kept])
    text, n = re.subn(r"(?m)^zones = .*$", lambda _: zl, text, count=1)
    if not n:
        text = re.sub(r"(?m)^(zone_defs = .*)$", lambda m: m.group(1) + "\n" + zl, text, count=1)
    cfg = tomllib.loads(text)               # never write a config that would not parse
    if "zone_defs" not in cfg.get("tracker", {}):
        return jsonify(ok=False, error="config.toml 的格式跟預期的不一樣，沒有儲存"), 500
    (DATA / "config.toml.bak-zones").write_text(original, encoding="utf-8")
    tmp = CONFIG_FILE.with_suffix(".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, CONFIG_FILE)
    subprocess.run(["sudo", "-n", "systemctl", "restart", "hamster-tracker"], capture_output=True, timeout=30)
    return jsonify(ok=True, defs=_zone_defs(), zones=current_cfg().get("tracker", {}).get("zones") or [])


_sam_pic = {"sid": None, "img": None}


@app.post("/api/calib/autozone")
@require_auth
def calib_autozone():
    """Tap-to-outline for a behaviour zone. With MobileSAM installed (samzone.py) the AI cuts out the tapped object;
    otherwise the simpler colour/edge method (autozone.py).
    Body: {kind, taps: [[x, y, 1 = object | 0 = not this], ...], sid: editor session, prefetch: true = only get ready}.
    One editor session uses one fixed picture, so the slow AI step runs once per session, not per tap."""
    import cv2
    import samzone
    body = request.get_json(silent=True) or {}
    kind, sid = body.get("kind"), str(body.get("sid") or "")[:40]
    taps = []
    for t in body.get("taps") or []:
        try:
            x, y = min(max(float(t[0]), 0.0), 1.0), min(max(float(t[1]), 0.0), 1.0)
            taps.append([x, y, 0 if len(t) > 2 and not t[2] else 1])
        except (TypeError, ValueError, IndexError):
            return jsonify(ok=False, error="資料不對"), 400
    if kind not in _zone_ids() or len(taps) > 16 or (not taps and not body.get("prefetch")):
        return jsonify(ok=False, error="資料不對"), 400
    if _sam_pic["sid"] != sid or _sam_pic["img"] is None:
        img = cv2.imread(str(DATA / "raw.jpg"))
        if img is None:
            return jsonify(ok=False, error="還沒有相機畫面"), 503
        _sam_pic.update(sid=sid, img=img)
    img = _sam_pic["img"]
    if samzone.available():
        if body.get("prefetch"):
            samzone.prepare(img, sid)
            return jsonify(ok=True, ready=True)
        r, err = samzone.outline(img, taps, kind, key=sid)
        if r is None:
            return jsonify(ok=False, error=err), 200
        return jsonify(ok=True, ai=True, **r)
    if body.get("prefetch"):
        return jsonify(ok=True, ready=True)
    import autozone
    poly, _failed, why = autozone.outline_many(img, [t[:2] for t in taps if t[2]], kind)
    if poly is None:
        return jsonify(ok=False, error=why or "找不到物品的邊緣"), 200
    return jsonify(ok=True, ai=False, options=[{"poly": poly}], pick=0)


def _floor_sides(t):
    """Side lengths shown in the editor: the saved ones, or the old rectangle cage_w_cm x cage_h_cm."""
    s = t.get("floor_sides_cm")
    if isinstance(s, list) and len(s) == 4:
        return s
    w, h = t.get("cage_w_cm", 30), t.get("cage_h_cm", 50)
    return [w, h, w, h]


@app.post("/api/alerts/test")
@require_auth
def alerts_test():
    cfg = current_cfg()
    if not cfg.get("alerts", {}).get("topic"):
        return jsonify(ok=False, error="還沒有設定通知頻道"), 400
    rule = (request.get_json(silent=True) or {}).get("rule")
    if rule:   # preview one rule's notification (title and an example message)
        r = next((x for x in alerts.RULES if x[0] == rule), None)
        if not r:
            return jsonify(ok=False, error="沒有這個通知"), 400
        ok = alerts.send(cfg, f"（測試）{r[1]}", TEST_SAMPLES.get(rule, "這是測試通知，實際內容會依當時狀況而定。"), r[3], r[4])
    else:
        ok = alerts.send(cfg, "倉鼠小屋・測試通知", "收到這則就代表手機通知設定成功了。", 3, ["hamster", "tada"])
    return jsonify(ok=ok, error=None if ok else "傳送失敗，請確認樹莓派有網路")


TEST_SAMPLES = {   # example messages for the per-rule 測試 buttons (the real ones use live numbers)
    "temp_high": "籠子裡 31.2°C，超過 30°C",
    "temp_low": "籠子裡只有 14.6°C，低於 15°C",
    "hum_high": "濕度 78%，超過 75%",
    "hum_low": "濕度只有 28%，低於 30%。太乾燥倉鼠容易皮膚乾癢、呼吸道不舒服，可以在房間放一杯水或開加濕器（不要放在籠子裡）",
    "sensor": "DHT11 一直讀不到正確數值，請檢查接線",
    "services": "相機追蹤 沒有在運作",
    "inactive": "倉鼠已經 18 小時沒有被偵測到活動，去看看牠",
    "water": "已經 12 小時沒看到牠去水瓶。檢查一下水瓶有沒有水、出水口有沒有堵住，或水瓶移位後出水口要重新標",
    "drink": "已經 18 小時沒看到牠喝水（相機推估）。檢查水瓶有沒有水、出水口會不會出水（用手指按一下鋼珠）",
    "food": "已經 24 小時沒看到牠去食盆。看看食盆有沒有飼料、飼料是不是不新鮮，或牠是不是沒胃口",
    "eat": "已經 24 小時沒看到牠吃東西（相機推估）。倉鼠常把食物塞頰囊帶回窩裡吃，這只是參考；如果食盆的飼料幾天都沒少，就要注意",
    "wheel": "已經 24 小時沒看到牠去滾輪。可能是不舒服、腳受傷、滾輪卡住，也可能滾輪大半不在相機畫面裡",
    "run": "已經 24 小時沒看到牠跑滾輪（相機推估）。可能是不舒服、腳受傷或滾輪卡住",
    "nest": "已經 24 小時沒看到牠回窩。窩可能被弄亂、太熱太冷，或牠找到別的地方睡；也可能窩門口沒框好",
    "bath": "已經 48 小時沒看到牠去浴室。浴砂可能髒了或被換掉；有些倉鼠本來就很少用浴室",
    "still": "AI 看到牠在食盆旁已經 30 分鐘一動也不動。倉鼠平常會回窩裡睡，去看看牠有沒有在呼吸、身體是不是冰冷、叫牠有沒有反應",
    "not_seen": "AI 已經 24 小時沒看到倉鼠。可能一直躲在窩裡、鏡頭被擋住或太暗，也可能跑出籠子了，去看看牠",
    "fan_hot": "風扇已經連續開了 60 分鐘，籠子還是 28.4°C（27°C 以上才開風扇）。房間可能太熱、風扇沒在轉或被擋住，考慮開冷氣或把籠子移到陰涼的地方",
}


# ---------- alert thresholds, editable from 設定 › 手機通知 ----------
# key: (default, min, max, zero means "off" allowed, type, comment written into config.toml when the key is new)
ALERT_FIELDS = {
    "temp_high": (30.0, 20, 40, False, float, "溫度 >= 這個值持續 hold_minutes 就通知 (降到低 1°C 才算恢復)"),
    "temp_low": (15.0, 0, 25, False, float, "溫度 <= 這個值持續 hold_minutes 就通知"),
    "hum_high": (75, 40, 95, False, int, "濕度 >= 這個值持續 10 分鐘就通知"),
    "hold_minutes": (5, 1, 60, False, int, "狀況要持續多久才通知，避免短暫波動就吵你"),
    "repeat_minutes": (120, 10, 1440, False, int, "狀況一直沒改善，每隔多久再提醒一次"),
    "inactive_hours": (18, 1, 72, True, int, "倉鼠超過幾小時沒偵測到活動就通知 (0 = 不檢查)"),
    "water_hours": (12, 2, 48, True, int, "超過幾小時沒看到牠去水瓶（出水口下方）就通知 (要先標出水口；0 = 不檢查)"),
    "drink_hours": (0, 4, 72, True, int, "超過幾小時相機沒推估到牠喝水就通知 (0 = 不檢查，預設關閉；標好出水口較準)"),
    "food_hours": (24, 4, 72, True, int, "超過幾小時沒看到牠去食盆就通知 (要先框食盆區域；0 = 不檢查)"),
    "eat_hours": (0, 4, 72, True, int, "超過幾小時相機沒推估到牠吃東西就通知 (0 = 不檢查，預設關閉)"),
    "wheel_hours": (0, 4, 96, True, int, "超過幾小時沒看到牠去滾輪就通知 (0 = 不檢查，預設關閉；滾輪要在畫面裡)"),
    "run_hours": (0, 4, 96, True, int, "超過幾小時相機沒推估到牠跑滾輪就通知 (0 = 不檢查，預設關閉)"),
    "nest_hours": (0, 4, 72, True, int, "超過幾小時沒看到牠回窩就通知 (0 = 不檢查，預設關閉)"),
    "bath_hours": (0, 6, 168, True, int, "超過幾小時沒看到牠去浴室就通知 (0 = 不檢查，預設關閉)"),
    "still_minutes": (30, 5, 240, True, int, "AI 看到牠躺在窩外不動超過幾分鐘就通知 (0 = 不檢查)"),
    "anomaly_low": (0.4, 0.1, 0.9, False, float, "每日比對：低於前 7 天中位數的幾成就通知 (0.4 = 4 成)"),
    "anomaly_high": (2.5, 1.5, 5, False, float, "每日比對：喝水、活動高於平常幾倍就通知"),
    "hum_low": (30, 10, 60, True, int, "濕度 <= 這個值持續 hum_hold_minutes 就通知 (太乾燥；0 = 不檢查)"),
    "hum_hold_minutes": (10, 1, 60, False, int, "濕度太高／太低要持續多久才通知"),
    "sensor_minutes": (2, 1, 30, False, int, "溫濕度感測器讀不到要持續多久才通知"),
    "not_seen_hours": (24, 2, 96, True, int, "AI 超過幾小時完全沒看到倉鼠就通知 (0 = 不檢查)"),
    "fan_hot_minutes": (60, 10, 480, True, int, "風扇連續開了幾分鐘溫度還是太高就通知 (0 = 不檢查)"),
    "anomaly_days": (7, 3, 30, False, int, "每日比對：跟前幾天比"),
    "anomaly_min_days": (3, 2, 14, False, int, "每日比對：至少要有幾天紀錄才開始比"),
}
# the alert rules that can be switched off one by one (same keys as alerts.RULES)
RULE_SWITCHES = ["temp_high", "temp_low", "hum_high", "hum_low", "sensor", "services", "inactive", "water", "drink",
                 "food", "eat", "wheel", "run", "nest", "bath", "still",
                 "not_seen", "fan_hot"]
ANOMALY_ITEMS = ["active_min", "distance_m", "wheel_min", "food", "water", "eat", "drink", "run", "nest", "restless"]
EXTRA_COMMENTS = {
    "quiet_start": "勿擾時段開始 (只有太熱、太冷、躺著不動、風扇開很久還太熱會馬上通知；其他等結束一起送)",
    "quiet_end": "勿擾時段結束；兩個都空字串 = 不使用勿擾",
    "disabled": "關掉的通知 (網頁上每一項都有開關)",
    "anomaly_items": "每日比對要檢查的項目",
}
_cfg_write_lock = threading.Lock()


def alert_rules(al):
    """Current settings with defaults filled in (what the page shows and the alert program uses)."""
    r = {k: al.get(k, spec[0]) for k, spec in ALERT_FIELDS.items()}
    r["daily_summary"] = al.get("daily_summary", "09:00")
    r["quiet_start"], r["quiet_end"] = al.get("quiet_start", ""), al.get("quiet_end", "")
    r["disabled"] = [k for k in al.get("disabled", []) if k in RULE_SWITCHES]
    r["anomaly_items"] = [k for k in al.get("anomaly_items", ANOMALY_ITEMS) if k in ANOMALY_ITEMS]
    r["topic"] = al.get("topic", "")
    return r


def _toml_value(v):
    if isinstance(v, list):
        return "[" + ", ".join(_toml_value(x) for x in v) + "]"
    return f'"{v}"' if isinstance(v, str) else (f"{v:.1f}" if isinstance(v, float) and v == int(v) else str(v))


def set_alert_values(text, values):
    """Change keys inside [alerts] of config.toml, keeping every comment; append keys that aren't there yet."""
    m = re.search(r"(?ms)^\[alerts\][^\n]*\n(.*?)(?=^\[|\Z)", text)
    if not m:
        raise ValueError("config.toml 沒有 [alerts] 區段")
    section = m.group(1)
    for k, v in values.items():
        pat = re.compile(rf"(?m)^({re.escape(k)}\s*=\s*)(\[[^\]\n]*\]|\"[^\"\n]*\"|[^\s#\n]+)")
        if pat.search(section):
            section = pat.sub(lambda mm: mm.group(1) + _toml_value(v), section, count=1)
        else:
            comment = ALERT_FIELDS[k][5] if k in ALERT_FIELDS else EXTRA_COMMENTS.get(k, "")
            line = f"{k} = {_toml_value(v)}" + (f"    # {comment}" if comment else "")
            section = section.rstrip("\n") + "\n" + line + "\n" + ("\n" if section.endswith("\n\n") else "")
    return text[:m.start(1)] + section + text[m.end(1):]


@app.post("/api/alerts/rules")
@require_auth
def alerts_rules_save():
    body = request.get_json(silent=True) or {}
    new = {}
    for k, (default, lo, hi, zero_off, typ, _c) in ALERT_FIELDS.items():
        if k not in body:
            continue
        try:
            v = typ(float(body[k]))
        except (TypeError, ValueError):
            return jsonify(ok=False, error=f"{k} 要填數字"), 400
        if not (zero_off and v == 0) and not lo <= v <= hi:
            return jsonify(ok=False, error=f"{k} 要在 {lo}～{hi} 之間" + ("（或 0 = 關閉）" if zero_off else "")), 400
        new[k] = round(v, 2) if typ is float else v
    hhmm = re.compile(r"([01]\d|2[0-3]):[0-5]\d")
    for k, label in (("daily_summary", "每日摘要時間"), ("quiet_start", "勿擾開始時間"), ("quiet_end", "勿擾結束時間")):
        if k in body:
            s = str(body[k] or "")
            if s and not hhmm.fullmatch(s):
                return jsonify(ok=False, error=f"{label}格式要像 09:00"), 400
            new[k] = s
    for k, allowed, label in (("disabled", RULE_SWITCHES, "通知開關"), ("anomaly_items", ANOMALY_ITEMS, "每日比對項目")):
        if k in body:
            v = body[k]
            if not isinstance(v, list) or any(x not in allowed for x in v):
                return jsonify(ok=False, error=f"{label}的資料不對"), 400
            new[k] = [x for x in allowed if x in v]   # keep a stable order
    if "topic" in body:
        t = str(body["topic"] or "").strip()
        if not re.fullmatch(r"[A-Za-z0-9_-]{6,64}", t):
            return jsonify(ok=False, error="通知頻道只能用英文、數字、- 和 _，6～64 個字"), 400
        new["topic"] = t
    if not new:
        return jsonify(ok=False, error="沒有要改的設定"), 400
    with _cfg_write_lock:
        cur = alert_rules(current_cfg().get("alerts", {}))
        merged = {**cur, **new}
        if merged["temp_low"] > merged["temp_high"] - 3:
            return jsonify(ok=False, error="「太冷」的溫度要比「太熱」低至少 3°C"), 400
        if merged["hum_low"] and "hum_low" not in merged["disabled"] and merged["hum_low"] > merged["hum_high"] - 10:
            return jsonify(ok=False, error="「太乾燥」的濕度要比「太潮濕」低至少 10%"), 400
        if bool(merged["quiet_start"]) != bool(merged["quiet_end"]):
            return jsonify(ok=False, error="勿擾時段要同時設定開始和結束時間"), 400
        text = CONFIG_FILE.read_text(encoding="utf-8")
        try:
            out = set_alert_values(text, new)
            parsed = tomllib.loads(out)                       # never write a config the services can't read
            if any(parsed["alerts"].get(k) != v for k, v in new.items()):
                raise ValueError("寫入後讀回來的值不一致")
            if {k: v for k, v in parsed.items() if k != "alerts"} != {k: v for k, v in tomllib.loads(text).items() if k != "alerts"}:
                raise ValueError("會動到其他設定，已取消")
        except (ValueError, tomllib.TOMLDecodeError) as e:
            app.logger.warning("alert rules not saved: %s", e)
            return jsonify(ok=False, error=f"沒有儲存：{e}"), 500
        (DATA / "config.toml.bak-alerts").write_text(text, encoding="utf-8")
        tmp = CONFIG_FILE.with_suffix(".toml.tmp")
        tmp.write_text(out, encoding="utf-8")
        os.replace(tmp, CONFIG_FILE)
    app.logger.warning("alert rules changed from %s: %s", request.remote_addr, new)
    return jsonify(ok=True, rules=alert_rules(current_cfg().get("alerts", {})))


@app.get("/api/alerts/log")
@require_auth
def alerts_log():
    """Notifications sent recently (newest first), saved by alerts.send()."""
    limit = int_arg(request.args.get("limit"), 30, 1, 200)
    try:
        with db() as con:
            rows = [dict(r) for r in con.execute(
                "SELECT rowid AS id, ts, title, message, priority, sent FROM alert_log ORDER BY ts DESC LIMIT ?", (limit,))]
    except sqlite3.OperationalError:   # nothing sent yet since the history was added
        rows = []
    return no_cache(jsonify(rows))


@app.post("/api/alerts/log/delete")
@require_auth
def alerts_log_delete():
    """Remove notifications from 最近的通知: {ids: [...]} or {all: true}. (What was already pushed to phones stays there.)"""
    body = request.get_json(silent=True) or {}
    try:
        with db() as con:
            if body.get("all") is True:
                n = con.execute("DELETE FROM alert_log").rowcount
            else:
                ids = [int(i) for i in body.get("ids") or [] if str(i).isdigit()][:1000]
                n = con.executemany("DELETE FROM alert_log WHERE rowid = ?", [(i,) for i in ids]).rowcount if ids else 0
            con.commit()
    except sqlite3.OperationalError:
        n = 0
    return jsonify(ok=True, deleted=n)


@app.post("/api/pause")
@require_auth
def pause_recording():
    """Pause activity/position/training-photo recording for N minutes (0 = resume now)."""
    body = request.get_json(silent=True) or {}
    minutes = int_arg(body.get("minutes"), 10, 0, 120)
    if minutes == 0:
        PAUSE_FILE.unlink(missing_ok=True)
        return jsonify(ok=True, pause=None)
    start = time.time()
    if body.get("extend"):   # 再 10 分鐘: added to the pause that is running (cleaning takes longer than planned)
        cur = (read_json(PAUSE_FILE) or {}).get("until") or 0
        start = max(start, float(cur))
    until = min(start + minutes * 60, time.time() + 120 * 60)
    data = {"until": until, "until_text": datetime.fromtimestamp(until).strftime("%H:%M")}
    tmp = PAUSE_FILE.with_suffix(".tmp")
    tmp.write_text(json.dumps(data))
    os.replace(tmp, PAUSE_FILE)
    return jsonify(ok=True, pause=data)


@app.post("/api/aim")
@require_auth
def aim():
    """Start/stop the camera aiming preview (hamster-aim.service)."""
    action = (request.get_json(silent=True) or {}).get("action")
    if action not in ("start", "stop"):
        return jsonify(ok=False, error="action must be start or stop"), 400
    r = subprocess.run(["sudo", "-n", "systemctl", action, "hamster-aim"], capture_output=True, text=True, timeout=20)
    return jsonify(ok=r.returncode == 0, error=r.stderr.strip() or None)


def no_cache(resp):
    resp.headers["Cache-Control"] = "no-store"
    return resp


@app.post("/api/capture/photo")
@require_auth
def capture_photo():
    """拍照: ask the tracker for one clean full-size frame and send it back as a download."""
    LIVE_DIR.mkdir(parents=True, exist_ok=True)
    out = LIVE_DIR / "photo.jpg"
    t0 = time.time()
    (LIVE_DIR / "photo.request").touch()
    while time.time() - t0 < 4:
        try:
            if out.stat().st_mtime >= t0:
                break
        except OSError:
            pass
        time.sleep(0.05)
    else:   # tracker not running (e.g. 調整角度): the last clean frame, if it is recent
        out = DATA / "raw.jpg"
        try:
            if time.time() - out.stat().st_mtime > 30:
                return jsonify(ok=False, error="相機沒有在拍（相機追蹤沒開或正在調整角度）"), 503
        except OSError:
            return jsonify(ok=False, error="還沒有相機畫面"), 503
    # kept on the Pi in 相簿 (data/album) with a small thumbnail; the page shows / shares it from there
    ALBUM.mkdir(parents=True, exist_ok=True)
    name = f"photo_{datetime.now():%Y%m%d_%H%M%S}"
    dst = ALBUM / f"{name}.jpg"
    shutil.copyfile(out, dst)
    try:
        import cv2
        img = cv2.imread(str(dst))
        if img is not None:
            h, w = img.shape[:2]
            cv2.imwrite(str(ALBUM / f"{name}_t.jpg"), cv2.resize(img, (320, round(320 * h / w)), interpolation=cv2.INTER_AREA),
                        [cv2.IMWRITE_JPEG_QUALITY, 75])
    except Exception:   # noqa: BLE001 - the full photo is what matters
        pass
    return jsonify(ok=True, name=name)


ALBUM = DATA / "album"
ALBUM_RE = re.compile(r"^photo_(\d{8})_(\d{6})$")


@app.get("/api/album")
@require_auth
def album_list():
    out = []
    for p in sorted(ALBUM.glob("photo_*.jpg"), reverse=True) if ALBUM.exists() else []:
        m = ALBUM_RE.match(p.stem)
        if m:
            d, t = m.groups()
            out.append({"name": p.stem, "day": d, "time": f"{d[4:6]}/{d[6:]} {t[:2]}:{t[2:4]}"})
    return no_cache(jsonify(photos=out))


@app.get("/album/<name>.jpg")
@require_auth
def album_photo(name):
    small = name.endswith("_t")
    base = name[:-2] if small else name
    if not ALBUM_RE.match(base):
        return "not found", 404
    p = ALBUM / f"{base}_t.jpg" if small else ALBUM / f"{base}.jpg"
    if small and not p.exists():
        p = ALBUM / f"{base}.jpg"
    return send_file(p, mimetype="image/jpeg", max_age=86400) if p.exists() else ("not found", 404)


@app.get("/api/album/guess")
@require_auth
def album_guess():
    """The installed model on one 相簿 photo: centre-format body / head boxes (same as /api/exam/guess)."""
    import cv2
    n = request.args.get("name", "")
    if not ALBUM_RE.match(n) or not (ALBUM / f"{n}.jpg").exists():
        return jsonify(ok=False, error="找不到這張"), 404
    mid = _current_model_id()
    if not mid:
        return jsonify(ok=False, error="還沒有安裝模型"), 400
    from yolo_detector import YoloNcnn, hamster_and_head
    if _guess["id"] != mid:
        _guess.update(id=mid, det=YoloNcnn(MODELS / "hamster_ncnn_model", conf=0.2, threads=1))
    img = cv2.imread(str(ALBUM / f"{n}.jpg"))
    b, hd = hamster_and_head(_guess["det"].detect(img)) if img is not None else (None, None)
    c = lambda x: [round(x[0] + x[2] / 2, 4), round(x[1] + x[3] / 2, 4), round(x[2], 4), round(x[3], 4)]   # noqa: E731
    return jsonify(ok=True, box=c(b) if b else None, conf=round(float(b[4]), 3) if b else 0.0,
                   head=c(hd) if b and hd else None, head_conf=round(float(hd[4]), 3) if b and hd else None)


@app.post("/api/album/zip")
@require_auth
def album_zip():
    """Several photos / videos in one download: {kind: photo (相簿) | sys (system photos) | video, names: [...]}."""
    import zipfile
    body = request.get_json(silent=True) or {}
    kind, names = body.get("kind"), [str(n) for n in body.get("names") or []][:500]
    if kind == "photo":
        files = [ALBUM / f"{n}.jpg" for n in names if ALBUM_RE.match(n)]
    elif kind == "sys":
        files = [DATASET / n for n in names if PHOTO_RE.match(n)]
    elif kind == "video":
        files = [CLIPS / f"{n}.mp4" for n in names if CLIP_RE.match(n)]
    else:
        return jsonify(ok=False, error="要下載什麼？"), 400
    files = [f for f in files if f.exists()]
    if not files:
        return jsonify(ok=False, error="沒有可以下載的"), 404
    tmp = tempfile.NamedTemporaryFile(suffix=".zip", delete=False)
    with zipfile.ZipFile(tmp, "w", zipfile.ZIP_STORED) as z:   # jpg / mp4 are compressed already
        for f in files:
            z.write(f, f.name)
    tmp.close()
    resp = send_file(tmp.name, mimetype="application/zip", as_attachment=True,
                     download_name=f"hamster_{'videos' if kind == 'video' else 'photos'}_{datetime.now():%Y%m%d_%H%M}.zip")
    resp.call_on_close(lambda: os.unlink(tmp.name))
    return resp


@app.post("/api/dataset/accept")
@require_auth
def dataset_accept():
    """加入訓練 for many photos at once: each becomes 已標好 with the box it already has (the pre-drawn one, or the
    AI's when it was sure). Photos with no box at all are skipped: they may hold a hamster the AI missed."""
    names = [str(n) for n in (request.get_json(silent=True) or {}).get("names") or [] if PHOTO_RE.match(str(n))][:3000]
    labd, scores = _labeled(), _ai_scores()
    y_th = _class_th()[0]
    done = already = skipped = 0
    d = dict(labd)
    for n in names:
        p = DATASET / n
        if not p.exists():
            continue
        if n in labd:
            already += 1
            continue
        txt = p.with_suffix(".txt")
        lines = txt.read_text().strip() if txt.exists() else ""
        if not lines:
            s = scores.get(n)
            if s and s[0] >= y_th and len(s) >= 5:
                x, y, w, h = s[1:5]
                lines = f"0 {x + w / 2:.6f} {y + h / 2:.6f} {w:.6f} {h:.6f}"
                if len(s) == 10 and s[5] >= 0.35:
                    hx, hy, hw, hh = s[6:10]
                    lines += f"\n1 {hx + hw / 2:.6f} {hy + hh / 2:.6f} {hw:.6f} {hh:.6f}"
                txt.write_text(lines + "\n")
        if not lines:
            skipped += 1
            continue
        d[n] = datetime.now().isoformat(timespec="seconds")
        done += 1
    if done:
        tmp = DATASET / "labeled.json.tmp"
        tmp.write_text(json.dumps(d))
        os.replace(tmp, DATASET / "labeled.json")
    return jsonify(ok=True, added=done, already=already, skipped=skipped)


@app.post("/api/album/to_dataset")
@require_auth
def album_to_dataset():
    """加入訓練照片: your own photos (拍照) become unlabelled training photos, named by the moment they were taken.
    Each photo goes in once (album/added.json remembers which training photo it became)."""
    import cv2
    names = [str(n) for n in (request.get_json(silent=True) or {}).get("names", []) if ALBUM_RE.match(str(n))][:500]
    book = ALBUM / "added.json"
    done = read_json(book) or {}
    DATASET.mkdir(parents=True, exist_ok=True)
    added, already, near = [], 0, 0
    ets = _exam_times()
    for name in names:
        src = ALBUM / f"{name}.jpg"
        if not src.exists():
            continue
        if done.get(name) and (DATASET / done[name]).exists():
            already += 1
            continue
        img = cv2.imread(str(src))
        if img is None:
            continue
        if img.shape[:2] != (480, 640):   # the same size as the photos the tracker saves
            img = cv2.resize(img, (640, 480), interpolation=cv2.INTER_AREA)
        when = datetime.strptime(name[6:], "%Y%m%d_%H%M%S")
        while any((DATASET / f"{pf}_{when:%Y%m%d_%H%M%S}.jpg").exists() for pf in ("motion", "idle", "hard_unsure", "hard_missed", "upload")):
            when += timedelta(seconds=1)
        out = f"upload_{when:%Y%m%d_%H%M%S}.jpg"
        cv2.imwrite(str(DATASET / out), img, [cv2.IMWRITE_JPEG_QUALITY, 92])
        with open(DATASET / "zones.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps({"n": out, "m": 0}) + "\n")
        done[name] = out
        added.append(out)
        near += _near_exam(out, ets)
    tmp = book.with_suffix(".tmp")
    tmp.write_text(json.dumps(done))
    os.replace(tmp, book)
    _ai_cache["t"] = 0
    return jsonify(ok=True, added=len(added), already=already, near_exam=near)


@app.post("/api/album/delete")
@require_auth
def album_delete():
    names = [str(n) for n in (request.get_json(silent=True) or {}).get("names", []) if ALBUM_RE.match(str(n))][:2000]
    n = 0
    for name in names:
        for f in (ALBUM / f"{name}.jpg", ALBUM / f"{name}_t.jpg"):
            if f.exists():
                f.unlink()
                n += f.name == f"{name}.jpg"
    return jsonify(ok=True, deleted=n)


@app.post("/api/capture/video")
@require_auth
def capture_video():
    """錄影: {seconds: 5-300} records a clip into 錄影紀錄 (same as the automatic ones); {seconds: 0} stops."""
    secs = int_arg(str((request.get_json(silent=True) or {}).get("seconds", 30)), 30, 0, 300)
    if secs and secs < 5:
        secs = 5
    if float((read_json(PAUSE_FILE) or {}).get("until", 0) or 0) > time.time():
        return jsonify(ok=False, error="記錄暫停中，先恢復記錄才能錄影")
    LIVE_DIR.mkdir(parents=True, exist_ok=True)
    tmp = LIVE_DIR / "rec.request.tmp"
    tmp.write_text(str(secs))
    os.replace(tmp, LIVE_DIR / "rec.request")
    return jsonify(ok=True, seconds=secs)


@app.get("/snapshot.jpg")
@require_auth
def snapshot():
    """?small=1 gives a 400 px copy (~10 KB instead of ~46 KB); the page asks for it when the connection is slow."""
    p = DATA / "latest.jpg"
    if not p.exists():
        return "no snapshot yet", 404
    if request.args.get("small"):
        try:
            mt = p.stat().st_mtime
            if _small_snap["mt"] != mt:
                import cv2
                img = cv2.imread(str(p))
                if img is not None:
                    h, w = img.shape[:2]
                    ok, buf = cv2.imencode(".jpg", cv2.resize(img, (400, round(400 * h / w)), interpolation=cv2.INTER_AREA),
                                           [cv2.IMWRITE_JPEG_QUALITY, 60])
                    if ok:
                        _small_snap.update(mt=mt, data=buf.tobytes())
            if _small_snap["data"]:
                return no_cache(Response(_small_snap["data"], mimetype="image/jpeg"))
        except OSError:
            pass
    return no_cache(send_file(p, mimetype="image/jpeg"))


_small_snap = {"mt": 0, "data": None}


@app.get("/heatmap.png")
@require_auth
def heatmap():
    p = DATA / "heatmap.png"
    return no_cache(send_file(p, mimetype="image/png")) if p.exists() else ("no heatmap yet", 404)


@app.get("/stream.mjpg")
@require_auth
def stream():
    if not _streams.acquire(blocking=False):
        return no_cache(Response("too many live viewers", 503))
    LIVE_DIR.mkdir(parents=True, exist_ok=True)

    def frames():
        last = None
        start = last_frame = time.time()
        while True:
            now = time.time()
            if now - start > STREAM_MAX_S or now - last_frame > STREAM_IDLE_S:
                return  # free the thread; the page reconnects when it still wants video
            LIVE_REQUEST.touch()  # tells tracker.py someone is watching
            try:
                st = LIVE_FRAME.stat()
                if st.st_mtime_ns != last and now - st.st_mtime < 5:
                    last, last_frame = st.st_mtime_ns, now
                    jpg = LIVE_FRAME.read_bytes()
                    yield (b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: "
                           + str(len(jpg)).encode() + b"\r\n\r\n" + jpg + b"\r\n")
            except OSError:
                pass
            time.sleep(0.03)

    released = threading.Event()

    def release():  # runs when the response is closed: normal end, client gone, or never iterated
        if not released.is_set():
            released.set()
            _streams.release()

    resp = Response(frames(), mimetype="multipart/x-mixed-replace; boundary=frame")
    resp.call_on_close(release)
    return no_cache(resp)


_small_live = {"ns": None, "data": None}


@app.get("/live.jpg")
@require_auth
def live_frame():
    """One live frame. The page pulls frames one after another instead of holding an MJPEG stream open:
    a stream that dies (tracker restart, timeout, network hiccup) left the last frame frozen on screen with no
    error, and the page didn't notice for minutes. Pulling also adapts to a slow link by itself.

    ?since=<X-Frame-Ns of the frame the page already shows>: waits up to 1.5 s for a newer one, else answers 204.
    ?small=1: a 400 px copy (~12 KB instead of ~40 KB) for slow connections.
    Headers: X-Frame-Ns (frame id), X-Frame-Age-Ms (how old the frame is on the Pi when sent)."""
    LIVE_DIR.mkdir(parents=True, exist_ok=True)
    since = request.args.get("since", "")
    deadline = time.time() + 1.5
    t_touch = 0
    while True:
        if time.time() - t_touch > 0.5:
            LIVE_REQUEST.touch()   # tells tracker.py / aim.py someone is watching, so it keeps writing live frames
            t_touch = time.time()
        try:
            st = LIVE_FRAME.stat()
        except OSError:
            st = None
        if (st and str(st.st_mtime_ns) != since) or time.time() >= deadline:
            break
        time.sleep(0.03)
    if st is None:
        return no_cache(Response(status=204, headers={"X-Frame-Age-Ms": "-1"}))
    age_ms = str(int((time.time() - st.st_mtime) * 1000))
    if str(st.st_mtime_ns) == since:   # nothing new yet: tell the page how old its picture is
        return no_cache(Response(status=204, headers={"X-Frame-Ns": since, "X-Frame-Age-Ms": age_ms}))
    try:
        data = LIVE_FRAME.read_bytes()
    except OSError:
        return no_cache(Response(status=204, headers={"X-Frame-Age-Ms": "-1"}))
    if request.args.get("small"):
        if _small_live["ns"] != st.st_mtime_ns:
            import cv2
            import numpy as np
            img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
            if img is not None:
                h, w = img.shape[:2]
                ok, buf = cv2.imencode(".jpg", cv2.resize(img, (400, round(400 * h / w)), interpolation=cv2.INTER_AREA),
                                       [cv2.IMWRITE_JPEG_QUALITY, 60])
                if ok:
                    _small_live.update(ns=st.st_mtime_ns, data=buf.tobytes())
        if _small_live["ns"] == st.st_mtime_ns:
            data = _small_live["data"]
    resp = Response(data, mimetype="image/jpeg")
    resp.headers["X-Frame-Ns"] = str(st.st_mtime_ns)
    resp.headers["X-Frame-Age-Ms"] = age_ms
    return no_cache(resp)


if __name__ == "__main__":
    from waitress import serve
    serve(app, host=WEB.get("host", "0.0.0.0"), port=WEB.get("port", 8000), threads=THREADS)
