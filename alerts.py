#!/usr/bin/env python3
"""Phone alerts for the hamster cage via ntfy (https://ntfy.sh).

Every check_interval_s it looks at the climate status, the services and the activity log, and
pushes a notification when something stays wrong for hold_minutes, a reminder every
repeat_minutes while it stays wrong, and an "OK again" message when it recovers.
Optionally sends a daily summary of yesterday.
"""
import json
import logging
import signal
import sqlite3
import subprocess
import time
import tomllib
import urllib.request
from datetime import datetime, timedelta
from pathlib import Path

from zonedefs import BUILTIN, zone_ids, zone_names, zone_types

BASE = Path(__file__).resolve().parent
DATA = BASE / "data"
STATE_FILE = Path("/run/hamster/alerts.json")   # survives service restarts, resets on reboot
log = logging.getLogger("alerts")
KEEP = object()   # rule result: "in between thresholds, keep the previous state" (hysteresis)


def load_config():
    with open(BASE / "config.toml", "rb") as f:
        return tomllib.load(f)


def send(cfg, title, message, priority=3, tags=()):
    """Publish one notification. Returns True on success."""
    a = cfg["alerts"]
    body = json.dumps({"topic": a["topic"], "title": title, "message": message,
                       "priority": priority, "tags": list(tags)}).encode()
    req = urllib.request.Request(a.get("ntfy_server", "https://ntfy.sh").rstrip("/"), data=body,
                                 headers={"Content-Type": "application/json"}, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=15) as r:
            ok = 200 <= r.status < 300
    except OSError as e:
        log.warning("send failed: %s", e)
        ok = False
    else:
        log.info("sent: %s | %s", title, message.replace("\n", " / "))
    log_alert(cfg, title, message, priority, ok)
    return ok


def log_alert(cfg, title, message, priority, ok):
    """Keep every notification (also failed ones) so the dashboard can list them: 設定 › 手機通知 › 最近的通知.
    ok: True sent, False failed, 2 held for the end of the quiet hours."""
    try:
        con = sqlite3.connect(BASE / cfg["log"]["db"], timeout=10)
        with con:
            con.execute("CREATE TABLE IF NOT EXISTS alert_log (ts TEXT, title TEXT, message TEXT, priority INTEGER, sent INTEGER)")
            con.execute("INSERT INTO alert_log VALUES (?,?,?,?,?)",
                        (datetime.now().isoformat(timespec="seconds"), title, message, priority, int(ok)))
        con.close()
    except sqlite3.Error as e:
        log.warning("could not save alert to history: %s", e)


def read_json(path):
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return None


def service_state(name):
    return subprocess.run(["systemctl", "is-active", name], capture_output=True, text=True).stdout.strip()


def service_enabled(name):
    return subprocess.run(["systemctl", "is-enabled", name], capture_output=True, text=True).stdout.strip() == "enabled"


def db(cfg):
    con = sqlite3.connect(BASE / cfg["log"]["db"], timeout=10)
    con.row_factory = sqlite3.Row
    return con


# ---------------------------------------------------------------- rules
# Each rule returns (state, detail): True = problem, False = fine, KEEP = hold previous, None = can't tell.

def rule_temp_high(ctx, a):
    t = ctx["temp"]
    if t is None:
        return None, ""
    if t >= a["temp_high"]:
        return True, f"籠子裡 {t:.1f}°C，超過 {a['temp_high']}°C"
    return (False if t <= a["temp_high"] - 1 else KEEP), f"籠子裡 {t:.1f}°C"


def rule_temp_low(ctx, a):
    t = ctx["temp"]
    if t is None:
        return None, ""
    if t <= a["temp_low"]:
        return True, f"籠子裡只有 {t:.1f}°C，低於 {a['temp_low']}°C"
    return (False if t >= a["temp_low"] + 1 else KEEP), f"籠子裡 {t:.1f}°C"


def rule_hum_high(ctx, a):
    h = ctx["hum"]
    if h is None:
        return None, ""
    if h >= a["hum_high"]:
        return True, f"濕度 {h:.0f}%，超過 {a['hum_high']}%"
    return (False if h <= a["hum_high"] - 5 else KEEP), f"濕度 {h:.0f}%"


def rule_sensor(ctx, a):
    st = ctx["climate"]
    if ctx["services"]["hamster-climate"] != "active":
        return None, ""   # covered by the service rule
    if not st:
        return True, "讀不到溫濕度狀態"
    age = time.time() - datetime.fromisoformat(st["updated"]).timestamp()
    if age > 120:
        return True, f"溫濕度已經 {int(age // 60)} 分鐘沒有更新"
    if not st.get("ok"):
        return True, "DHT11 一直讀不到正確數值，請檢查接線"
    return False, "溫濕度感測器恢復正常"


def rule_services(ctx, a):
    down = [n for n, s in ctx["services"].items() if s != "active" and ctx["expected"][n]]
    names = {"hamster-climate": "溫濕度與風扇", "hamster-tracker": "相機追蹤", "hamster-web": "網頁"}
    if down:
        return True, "、".join(names.get(n, n) for n in down) + " 沒有在運作"
    return False, "所有服務都恢復運作"


def rule_inactive(ctx, a):
    hours = a.get("inactive_hours", 0)
    if not hours or ctx["services"].get("hamster-tracker") != "active" or ctx["last_active"] is None:
        return None, ""
    gap_h = (datetime.now() - ctx["last_active"]).total_seconds() / 3600
    if gap_h >= hours:
        return True, f"倉鼠已經 {gap_h:.0f} 小時沒有被偵測到活動，去看看牠"
    return False, "倉鼠又開始活動了"


# "Hasn't been to <zone> / hasn't <done> for N hours", for every zone and every camera estimate.
#   key: (setting, default hours (0 = off), zone it needs, words, what to check)
VISIT_RULES = {
    "water": ("water_hours", 12, "water", "去水瓶", "檢查一下水瓶有沒有水、出水口有沒有堵住，或水瓶移位後出水口要重新標"),
    "food": ("food_hours", 24, "food", "去食盆", "看看食盆有沒有飼料、飼料是不是不新鮮，或牠是不是沒胃口"),
    "wheel": ("wheel_hours", 0, "wheel", "去滾輪", "可能是不舒服、腳受傷、滾輪卡住，也可能滾輪大半不在相機畫面裡"),
    "nest": ("nest_hours", 0, "nest", "回窩", "窩可能被弄亂、太熱太冷，或牠找到別的地方睡；也可能窩門口沒框好"),
    "bath": ("bath_hours", 0, "bath", "去浴室", "浴砂可能髒了或被換掉；有些倉鼠本來就很少用浴室"),
}
# camera estimates: off by default (still being tuned); also quiet when nothing at all was estimated in 7 days,
# because then the estimate isn't working (spout not marked, wheel out of view...) and "never" would be a false alarm
EST_RULES = {
    "drink": ("drink_hours", 0, "water", "drink", "喝水", "檢查水瓶有沒有水、出水口會不會出水（用手指按一下鋼珠）"),
    "eat": ("eat_hours", 0, "food", "eat", "吃東西",
            "倉鼠常把食物塞頰囊帶回窩裡吃，這只是參考；如果食盆的飼料幾天都沒少，就要注意"),
    "run": ("run_hours", 0, "wheel", "run", "跑滾輪", "可能是不舒服、腳受傷或滾輪卡住"),
}


def _gap_rule(ctx, a, setting, default, zone, last, words, check, est_7d=None):
    hours = a.get(setting, default)
    if not hours or zone not in ctx["zones"] or ctx["services"].get("hamster-tracker") != "active" or not ctx["zone_since"]:
        return None, ""
    if (datetime.now() - ctx["zone_since"]).total_seconds() < hours * 3600:
        return None, ""   # not watched that long yet
    if est_7d is not None and not est_7d:
        return None, ""   # the estimate never fired this week: it isn't working, so it can't tell "never"
    gap_h = (datetime.now() - last).total_seconds() / 3600 if last else None
    if gap_h is None or gap_h >= hours:
        what = f"沒看到牠{words}" + ("（相機推估）" if est_7d is not None else "")
        return True, (f"已經 {gap_h:.0f} 小時{what}" if gap_h else f"超過 {hours} 小時{what}") + "。" + check
    return False, f"看到牠{words}了"


# a second bottle / bowl / wheel added on the dashboard (type 喝水 / 吃東西 / 跑滾輪) counts like the built-in one:
# drinking from the new bottle is drinking, not "hasn't been to the water bottle for 12 hours"
KIND_OF = {"water": "drink", "food": "eat", "wheel": "run", "nest": "nest"}


def same_kind(zones, types, zone):
    """The zones in use that do what `zone` does (the built-in one included); bath and plain areas only themselves."""
    k = KIND_OF.get(zone)
    if not k:
        return [zone] if zone in zones else []
    return [z for z in zones if types.get(z) == k]


def _visit_rule(key):
    setting, default, zone, words, check = VISIT_RULES[key]

    def rule(ctx, a):
        zs = same_kind(ctx["zones"], ctx["types"], zone)
        last = max((ctx["last_visit"][z] for z in zs if ctx["last_visit"].get(z)), default=None)
        return _gap_rule(ctx, a, setting, default, zs[0] if zs else zone, last, words, check)
    return rule


def _est_rule(key):
    setting, default, zone, kind, words, check = EST_RULES[key]

    def rule(ctx, a):
        zs = same_kind(ctx["zones"], ctx["types"], zone)
        return _gap_rule(ctx, a, setting, default, zs[0] if zs else zone, ctx["last_est"].get(kind), words, check,
                         est_7d=ctx["est_7d"].get(kind, 0))
    return rule


ZONE_WHERE = {"wheel": "滾輪", "food": "食盆旁", "water": "水瓶旁", "bath": "浴室", None: "籠子裡"}


def rule_still(ctx, a):
    """The AI keeps seeing the hamster lying in one spot outside the nest for still_minutes.

    Hamsters normally sleep inside the nest; lying still in the open for a long time can mean it is unwell or
    hurt, and when it's cold it can be torpor (a hibernation-like state that can be dangerous)."""
    mins = a.get("still_minutes", 30)
    tr = ctx["tracker"]
    if not mins or not tr or ctx["services"].get("hamster-tracker") != "active" or tr.get("detector") == "motion":
        return None, ""
    try:
        if time.time() - datetime.fromisoformat(tr["updated"]).timestamp() > 90:
            return None, ""   # stale status: the tracker isn't writing, the service rule covers that
    except (KeyError, ValueError):
        return None, ""
    s = tr.get("still_s") or 0
    if s >= mins * 60:
        sz = tr.get("still_zone")
        where = ZONE_WHERE[sz] if sz in ZONE_WHERE else ctx.get("zone_names", {}).get(sz, "籠子裡")
        msg = (f"AI 看到牠在{where}已經 {s // 60:.0f} 分鐘一動也不動。倉鼠平常會回窩裡睡，"
               "去看看牠有沒有在呼吸、身體是不是冰冷、叫牠有沒有反應")
        t = ctx["temp"]
        if t is not None and t < a.get("temp_low", 15) + 3:
            msg += f"。現在只有 {t:.1f}°C，太冷時倉鼠可能進入假冬眠，要讓籠子慢慢回暖"
        return True, msg
    return (False if s < 60 else KEEP), "牠又開始移動了"


def rule_hum_low(ctx, a):
    lim = a.get("hum_low", 30)
    h = ctx["hum"]
    if not lim or h is None:
        return None, ""
    if h <= lim:
        return True, (f"濕度只有 {h:.0f}%，低於 {lim}%。太乾燥倉鼠容易皮膚乾癢、呼吸道不舒服，"
                      "可以在房間放一杯水或開加濕器（不要放在籠子裡）")
    return (False if h >= lim + 5 else KEEP), f"濕度 {h:.0f}%"


def rule_not_seen(ctx, a):
    """The AI hasn't seen the hamster at all for not_seen_hours (needs an AI model and that much watching)."""
    hours = a.get("not_seen_hours", 24)
    tr = ctx["tracker"]
    if (not hours or ctx["services"].get("hamster-tracker") != "active" or not tr or tr.get("detector") == "motion"
            or not ctx["ai_since"]):
        return None, ""
    if (datetime.now() - ctx["ai_since"]).total_seconds() < hours * 3600:
        return None, ""   # the AI hasn't been watching that long yet
    last = ctx["ai_last_seen"]
    gap_h = (datetime.now() - last).total_seconds() / 3600 if last else None
    if gap_h is None or gap_h >= hours:
        return True, ((f"AI 已經 {gap_h:.0f} 小時沒看到倉鼠" if gap_h else f"AI 超過 {hours} 小時沒看到倉鼠") +
                      "。可能一直躲在窩裡、鏡頭被擋住或太暗，也可能跑出籠子了，去看看牠")
    return False, "AI 又看到倉鼠了"


def rule_fan_hot(ctx, a):
    """The fan has been running for fan_hot_minutes and the cage is still at/above the fan's switch-on temperature."""
    mins = a.get("fan_hot_minutes", 60)
    t, t_on = ctx["temp"], ctx["fan_temp_on"]
    if not mins or t is None or t_on is None:
        return None, ""
    since = ctx["fan_on_since"]
    if since is None:
        return False, "風扇已經停了，溫度降下來了"
    on_min = (datetime.now() - since).total_seconds() / 60
    if on_min >= mins and t >= t_on:
        return True, (f"風扇已經連續開了 {on_min:.0f} 分鐘，籠子還是 {t:.1f}°C（{t_on}°C 以上才開風扇）。"
                      "房間可能太熱、風扇沒在轉或被擋住，考慮開冷氣或把籠子移到陰涼的地方")
    return (False if t < t_on - 0.5 else KEEP), f"籠子 {t:.1f}°C"


RULES = [  # key, title, rule, priority, tags, hold: None = hold_minutes, number, or (config key, default)
    ("temp_high", "籠子太熱", rule_temp_high, 4, ["thermometer", "warning"], None),
    ("temp_low", "籠子太冷", rule_temp_low, 4, ["thermometer", "snowflake"], None),
    ("hum_high", "籠子太潮濕", rule_hum_high, 3, ["droplet"], ("hum_hold_minutes", 10)),
    ("hum_low", "籠子太乾燥", rule_hum_low, 3, ["droplet"], ("hum_hold_minutes", 10)),
    ("sensor", "溫濕度感測器異常", rule_sensor, 4, ["warning"], ("sensor_minutes", 2)),
    ("services", "倉鼠監控服務停止", rule_services, 4, ["rotating_light"], 2),
    ("inactive", "倉鼠很久沒動", rule_inactive, 4, ["hamster"], 0),
    ("water", "倉鼠很久沒去水瓶", _visit_rule("water"), 4, ["droplet", "warning"], 0),
    ("drink", "倉鼠很久沒喝水（推估）", _est_rule("drink"), 4, ["droplet", "warning"], 0),
    ("food", "倉鼠很久沒去食盆", _visit_rule("food"), 4, ["hamster", "warning"], 0),
    ("eat", "倉鼠很久沒吃東西（推估）", _est_rule("eat"), 4, ["hamster", "warning"], 0),
    ("wheel", "倉鼠很久沒去滾輪", _visit_rule("wheel"), 3, ["hamster"], 0),
    ("run", "倉鼠很久沒跑滾輪（推估）", _est_rule("run"), 3, ["hamster"], 0),
    ("nest", "倉鼠很久沒回窩", _visit_rule("nest"), 3, ["hamster"], 0),
    ("bath", "倉鼠很久沒去浴室", _visit_rule("bath"), 2, ["hamster"], 0),
    ("still", "倉鼠躺著很久沒動", rule_still, 4, ["hamster", "warning"], 0),
    ("not_seen", "很久沒看到倉鼠", rule_not_seen, 4, ["hamster", "mag"], 0),
    ("fan_hot", "風扇開很久還是太熱", rule_fan_hot, 4, ["thermometer", "warning"], 0),
]
RULE_KEYS = [r[0] for r in RULES]
# these still notify right away during quiet hours (the hamster's life may depend on it); everything else waits
URGENT = {"temp_high", "temp_low", "still", "fan_hot"}


def hold_minutes(a, hold):
    if hold is None:
        return a.get("hold_minutes", 5)
    if isinstance(hold, tuple):
        return a.get(hold[0], hold[1])
    return hold


def in_quiet(a, now=None):
    """Inside the quiet hours (勿擾時段)? quiet_start / quiet_end as "HH:MM"; empty = no quiet hours."""
    s, e = a.get("quiet_start", ""), a.get("quiet_end", "")
    if not s or not e or s == e:
        return False
    t = (now or datetime.now()).strftime("%H:%M")
    return s <= t < e if s < e else (t >= s or t < e)


def notify(cfg, state, key, title, message, priority=3, tags=()):
    """send(), except that during quiet hours non-urgent messages are kept and sent together when they end."""
    if key not in URGENT and in_quiet(cfg["alerts"]):
        state.setdefault("held", []).append([datetime.now().strftime("%m/%d %H:%M"), title, message])
        log_alert(cfg, title, message, priority, 2)   # 2 = held for the end of the quiet hours
        return True
    return send(cfg, title, message, priority, tags)


def zone_kinds(cfg):
    """Zones in use: the drawn ones, plus the water bottle when its spout is marked (its area is the spot under
    the spout, not a drawn zone)."""
    t = cfg.get("tracker", {})
    ids = zone_ids(t)   # areas deleted from the dashboard's list don't count
    kinds = [z.get("kind") for z in t.get("zones") or [] if z.get("kind") != "water" and z.get("kind") in ids
             and (len(z.get("poly") or []) >= 3 or z.get("spout"))]
    s = t.get("spout")
    return kinds + (["water"] if isinstance(s, list) and len(s) == 2 and "water" in ids else [])


def zone_name_map(cfg):
    return zone_names(cfg.get("tracker", {}))


def gather(cfg):
    services = {n: service_state(n) for n in ("hamster-climate", "hamster-tracker", "hamster-web")}
    expected = {n: (n != "hamster-tracker" or service_enabled(n)) for n in services}
    climate = read_json(DATA / "status.json")
    ok = bool(climate and climate.get("ok"))
    last_active = zone_since = ai_since = ai_last_seen = fan_on_since = None
    last_visit, last_est, est_7d = {}, {}, {}
    zones = zone_kinds(cfg)
    iso = lambda v: datetime.fromisoformat(v) if v else None   # noqa: E731
    try:
        with db(cfg) as con:
            row = con.execute("SELECT MAX(minute) m FROM activity WHERE active_s > 0").fetchone()
            if row and row["m"]:
                last_active = datetime.fromisoformat(row["m"])
            if zones:
                try:
                    # the last minute it was in each zone (not when a visit started: asleep in the nest for 10 hours
                    # is "in the nest now", not "hasn't been to the nest for 10 hours"), and since when zones are recorded
                    last_visit = {r["zone"]: iso(r["m"]) for r in con.execute("SELECT zone, MAX(minute) m FROM zone_minutes GROUP BY zone")}
                    row = con.execute("SELECT MIN(minute) m FROM zone_minutes").fetchone()
                    zone_since = iso(row["m"]) if row and row["m"] else None
                    # camera estimates (tracker.py behavior_events): last one of each kind, and how many this week
                    week = (datetime.now() - timedelta(days=7)).isoformat(timespec="seconds")
                    for r in con.execute("SELECT kind, MAX(start) s, SUM(start >= ?) n FROM behavior_events GROUP BY kind", (week,)):
                        last_est[r["kind"]], est_7d[r["kind"]] = iso(r["s"]), r["n"] or 0
                except sqlite3.OperationalError:
                    pass
            if climate and climate.get("fan"):   # fan on: since when? (the last reading with the fan off)
                row = con.execute("SELECT MAX(ts) t FROM climate WHERE fan = 0").fetchone()
                fan_on_since = iso(row["t"]) if row and row["t"] else iso(con.execute("SELECT MIN(ts) t FROM climate").fetchone()["t"])
            try:
                ai_since = iso(con.execute("SELECT MIN(minute) m FROM ai_minutes").fetchone()["m"])
                ai_last_seen = iso(con.execute("SELECT MAX(minute) m FROM ai_minutes WHERE seen > 0").fetchone()["m"])
            except sqlite3.OperationalError:
                pass
    except sqlite3.OperationalError:
        pass
    return {"services": services, "expected": expected, "climate": climate,
            "temp": climate["temp"] if ok else None, "hum": climate["hum"] if ok else None,
            "last_active": last_active, "zones": zones, "types": zone_types(cfg.get("tracker", {})), "last_visit": last_visit, "last_est": last_est, "est_7d": est_7d,
            "zone_since": zone_since, "tracker": read_json(DATA / "tracker.json"),
            "ai_since": ai_since, "ai_last_seen": ai_last_seen, "fan_on_since": fan_on_since,
            "fan_temp_on": cfg.get("fan", {}).get("temp_on"), "zone_names": zone_name_map(cfg)}


def day_metrics(con, day, types=None):
    """Totals for one calendar day: active minutes, wheel minutes, nest hours, food/water/bath visits."""
    d0, d1 = day.isoformat(), (day + timedelta(days=1)).isoformat()
    a_min, dist = con.execute("SELECT COALESCE(SUM(active_s),0)/60.0, COALESCE(SUM(distance_cm),0)/100.0 FROM activity "
                              "WHERE minute >= ? AND minute < ?", (d0, d1)).fetchone()
    m = {"active_min": a_min, "distance_m": dist}
    try:
        secs = dict(con.execute("SELECT zone, SUM(seconds) FROM zone_minutes WHERE minute >= ? AND minute < ? GROUP BY zone",
                                (d0, d1)).fetchall())
        visits = dict(con.execute("SELECT zone, COUNT(*) FROM zone_visits WHERE start >= ? AND start < ? GROUP BY zone",
                                  (d0, d1)).fetchall())
    except sqlite3.OperationalError:
        secs, visits = {}, {}
    try:   # camera estimates (tracker.py behavior_events)
        est = dict(con.execute("SELECT kind, COUNT(*) FROM behavior_events WHERE start >= ? AND start < ? GROUP BY kind",
                               (d0, d1)).fetchall())
    except sqlite3.OperationalError:
        est = {}
    m.update(wheel_min=secs.get("wheel", 0) / 60, nest_h=secs.get("nest", 0) / 3600,
             food=visits.get("food", 0), water=visits.get("water", 0), bath=visits.get("bath", 0),
             visits=visits, secs=secs,
             eat=est.get("eat", 0), drink=est.get("drink", 0), run=est.get("run", 0))
    if types:   # for the health checks: every area of the kind (a second bottle / bowl / wheel counts too)
        of = lambda k, d: sum(v for z, v in d.items() if types.get(z) == k)   # noqa: E731
        m.update(wheel_min=of("run", secs) / 60, food=of("eat", visits), water=of("drink", visits), nest_h=of("nest", secs) / 3600)
    return m


ANOMALY_NAMES = {"active_min": "活動時間", "distance_m": "移動距離", "wheel_min": "在滾輪的時間", "food": "去食盆次數",
                 "water": "去水瓶次數", "eat": "吃東西次數（推估）", "drink": "喝水次數（推估）", "run": "跑滾輪次數（推估）"}
ANOMALY_ZONE = {"wheel_min": "wheel", "food": "food", "water": "water", "eat": "food", "drink": "water", "run": "wheel"}
# everything the daily comparison can check; the settings page lets you untick any of them
ANOMALY_ITEMS = [*ANOMALY_NAMES, "nest", "restless"]


def anomaly_check(cfg, day):
    """Compare one day with the median of the anomaly_days (default 7) days before it. Returns warning lines.

    Needs at least anomaly_min_days (default 3) earlier days with data, so it stays quiet for the first days.
    Only the items in anomaly_items are checked (default: all).
    """
    import statistics
    a = cfg["alerts"]
    low, high = a.get("anomaly_low", 0.4), a.get("anomaly_high", 2.5)
    days, min_days = int(a.get("anomaly_days", 7)), int(a.get("anomaly_min_days", 3))
    items = set(a.get("anomaly_items", ANOMALY_ITEMS))
    types = zone_types(cfg.get("tracker", {}))
    with db(cfg) as con:
        today = day_metrics(con, day, types)
        past = [day_metrics(con, day - timedelta(days=i), types) for i in range(1, days + 1)]
    past = [p for p in past if p["active_min"] > 0]   # days the tracker actually ran
    if len(past) < min_days:
        return []
    zones = zone_kinds(cfg)
    lines = []
    for key, name in ANOMALY_NAMES.items():
        if key not in items:
            continue
        zone = ANOMALY_ZONE.get(key)
        if zone and not same_kind(zones, types, zone):
            continue
        base = statistics.median(p[key] for p in past)
        if base <= 0:
            continue
        v = today[key]
        if v < base * low:
            lines.append(f"{name}只有平常的 {v / base:.0%}（{v:.0f}，平常約 {base:.0f}）")
        elif key in ("water", "drink") and base >= 3 and v > base * high:
            lines.append(f"{name}是平常的 {v / base:.1f} 倍（{v:.0f}，平常約 {base:.0f}）——"
                         + ("一直去水瓶，可能是喝水變多，" if key == "water" else "") + "喝水突然變多可能是生病的徵兆")
    # slept far longer than usual: not wanting to come out is a common sign of feeling unwell
    if "nest" in zones and "nest" in items:
        base = statistics.median(p["nest_h"] for p in past)
        v = today["nest_h"]
        if base >= 1 and v > base * 1.5 and v - base >= 2:
            lines.append(f"在窩裡待了 {v:.1f} 小時，比平常（約 {base:.1f} 小時）多很多——一直不想出來可能是不舒服")
    # suddenly far more active than usual: stress, being disturbed, or discomfort
    base = statistics.median(p["active_min"] for p in past)
    v = today["active_min"]
    if "restless" in items and base >= 10 and v > base * high:
        lines.append(f"活動時間是平常的 {v / base:.1f} 倍（{v:.0f} 分，平常約 {base:.0f} 分）——"
                     "突然異常好動，可能是緊張、環境被打擾（例如換籠、噪音）或身體不舒服")
    return lines


def daily_summary(cfg, day):  # noqa: C901 - straightforward formatting
    """Text summary for one calendar day (datetime.date)."""
    d0, d1 = day.isoformat(), (day + timedelta(days=1)).isoformat()
    with db(cfg) as con:
        c = con.execute("SELECT MIN(temp) tmin, AVG(temp) tavg, MAX(temp) tmax, MIN(hum) hmin, MAX(hum) hmax, "
                        "AVG(fan) fan, COUNT(*) n FROM climate WHERE ts >= ? AND ts < ?", (d0, d1)).fetchone()
        try:
            act = con.execute("SELECT COALESCE(SUM(active_s),0) a, COALESCE(SUM(distance_cm),0) d "
                              "FROM activity WHERE minute >= ? AND minute < ?", (d0, d1)).fetchone()
        except sqlite3.OperationalError:
            act = {"a": 0, "d": 0}
    if not c["n"]:
        return None
    text = (f"溫度 {c['tmin']:.1f}～{c['tmax']:.1f}°C（平均 {c['tavg']:.1f}°C）\n"
            f"濕度 {c['hmin']:.0f}～{c['hmax']:.0f}%\n"
            f"風扇運轉 {c['fan'] * 100:.0f}% 的時間\n"
            f"倉鼠活動 {act['a'] / 60:.0f} 分鐘，移動 {act['d'] / 100:.1f} 公尺")
    zones = zone_kinds(cfg)
    if zones:
        with db(cfg) as con:
            m = day_metrics(con, day)
        # going to a spot vs. what the camera estimates it did there (shown separately, never mixed up)
        parts = [("nest", f"在窩裡 {m['nest_h']:.1f} 小時"),
                 ("wheel", f"在滾輪 {m['wheel_min']:.0f} 分鐘（跑滾輪推估 {m['run']} 次）"),
                 ("food", f"去食盆 {m['food']} 次（吃東西推估 {m['eat']} 次）"),
                 ("water", f"去水瓶 {m['water']} 次（喝水推估 {m['drink']} 次）"), ("bath", f"浴室 {m['bath']} 次"),
                 *[(k, f"{n} {m['visits'].get(k, 0)} 次（{m['secs'].get(k, 0) / 60:.0f} 分鐘）")
                   for k, n in zone_name_map(cfg).items() if k not in BUILTIN]]
        text += "\n" + "\n".join(t for k, t in parts if k in zones)
    return text


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    stop = False

    def handle_stop(*_):
        nonlocal stop
        stop = True

    signal.signal(signal.SIGTERM, handle_stop)
    signal.signal(signal.SIGINT, handle_stop)

    state = read_json(STATE_FILE) or {}
    log.info("started")
    while not stop:
        cfg = load_config()      # re-read each round so threshold edits apply without a restart
        a = cfg.get("alerts", {})
        if a.get("enabled") and a.get("topic"):
            now = time.time()
            ctx = gather(cfg)
            disabled = set(a.get("disabled", []))   # rules switched off in 設定 › 手機通知
            for key, title, rule, prio, tags, hold_override in RULES:
                st = state.setdefault(key, {"since": None, "notified": 0, "active": False})
                if key in disabled:
                    st.update(active=False, since=None)   # switched off: forget it quietly
                    continue
                bad, detail = rule(ctx, a)
                if bad is None:
                    continue
                if bad is KEEP:
                    bad = st["active"]
                hold = hold_minutes(a, hold_override) * 60
                if bad:
                    st["since"] = st["since"] or now
                    if not st["active"] and now - st["since"] >= hold:
                        if notify(cfg, state, key, title, detail, prio, tags):
                            st.update(active=True, notified=now)
                    elif st["active"] and now - st["notified"] >= a.get("repeat_minutes", 120) * 60:
                        if notify(cfg, state, key, title + "（仍未改善）", detail, prio, tags):
                            st["notified"] = now
                else:
                    if st["active"]:
                        notify(cfg, state, key, "已恢復正常", f"{title} → {detail}", 2, ["white_check_mark"])
                    st.update(active=False, since=None)

            # quiet hours just ended: send everything that was kept, as one message
            held = state.get("held") or []
            if held and not in_quiet(a):
                body = "\n".join(f"{t} {ti}：{m[:90]}" for t, ti, m in held[-20:])
                if len(held) > 20:
                    body = f"（只列最後 20 則，共 {len(held)} 則）\n" + body
                if send(cfg, f"勿擾時段內的通知（{len(held)} 則）", body, 3, ["bell"]):
                    state["held"] = []

            # daily summary of yesterday at the configured time
            at = a.get("daily_summary", "")
            today = datetime.now().date().isoformat()
            if at and datetime.now().strftime("%H:%M") >= at and state.get("summary_day") != today:
                yesterday = datetime.now().date() - timedelta(days=1)
                text = daily_summary(cfg, yesterday)
                if text is None or notify(cfg, state, "daily", "倉鼠小屋・昨天摘要", text, 2, ["hamster", "bar_chart"]):
                    state["summary_day"] = today
                    # possible illness: yesterday very different from its usual week
                    try:
                        lines = anomaly_check(cfg, yesterday)
                    except Exception as e:   # noqa: BLE001 - never let the check stop the alert loop
                        log.warning("anomaly check failed: %s", e)
                        lines = []
                    if lines:
                        notify(cfg, state, "daily", "倉鼠狀況跟平常不一樣", "昨天：\n" + "\n".join(lines) +
                               "\n\n如果持續好幾天，或牠看起來沒精神、不吃東西，建議帶去給獸醫看。", 4, ["hamster", "warning"])
            try:
                STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
                STATE_FILE.write_text(json.dumps(state))
            except OSError:
                pass

        end = time.monotonic() + a.get("check_interval_s", 30)
        while not stop and time.monotonic() < end:
            time.sleep(0.5)


if __name__ == "__main__":
    main()
