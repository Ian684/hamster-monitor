"""The list of behaviour areas (zones) the user can edit on the dashboard: name, colour, add, delete.

Saved in config.toml as [tracker] zone_defs = [{id = "nest", name = "窩門口", color = "#ff69b4"}, ...].
Without that line the five defaults are used. The five built-in ids keep their special meaning in tracker.py
(nest door, wheel running, food eating, water = the circle under the spout, bath); areas the user adds get
ids like "c1a2b3" and are plain "time spent / visits" areas. Polygons stay in [tracker] zones (kind = id).
"""
import re

DEFAULTS = [("nest", "窩門口", "#ff69b4"), ("wheel", "滾輪", "#30d158"), ("food", "食盆", "#ffd60a"),
            ("water", "水瓶", "#64d2ff"), ("bath", "浴室", "#c8c8cd")]
BUILTIN = tuple(k for k, _, _ in DEFAULTS)
# what is estimated in an area: the built-in ones are fixed; an added area can be any of these
BUILTIN_TYPE = {"nest": "nest", "wheel": "run", "food": "eat", "water": "drink", "bath": "plain"}
TYPES = ("plain", "eat", "drink", "run")   # 一般 (time / visits only), 吃東西, 喝水, 跑滾輪
MAX_ZONES = 12
ID_RE = re.compile(r"^[a-z][a-z0-9_]{0,15}$")
COLOR_RE = re.compile(r"^#[0-9a-fA-F]{6}$")


def clean_name(s, fallback=""):
    """Short, no control characters and nothing that could break the page's HTML."""
    return re.sub(r"[\x00-\x1f\x7f<>&\"'`\\]", "", str(s or "")).strip()[:10] or fallback


def defaults():
    return [{"id": k, "name": n, "color": c, "type": BUILTIN_TYPE[k]} for k, n, c in DEFAULTS]


def zone_defs(t):
    """The zone list from the [tracker] section (dict). Always valid; falls back to the defaults."""
    raw = t.get("zone_defs")
    if not isinstance(raw, list) or not raw:
        out = defaults()
        # a custom area drawn with the first version (kind = "custom", its own name)
        for z in t.get("zones") or []:
            if isinstance(z, dict) and z.get("kind") == "custom":
                out.append({"id": "custom", "name": clean_name(z.get("name"), "自訂區域"), "color": "#bf5af2", "type": "plain"})
                break
        return out
    out, seen = [], set()
    for d in raw:
        if not isinstance(d, dict):
            continue
        zid = str(d.get("id") or "")
        if not ID_RE.match(zid) or zid in seen:
            continue
        color = str(d.get("color") or "")
        builtin_default = next((c for k, _, c in DEFAULTS if k == zid), "#bf5af2")
        out.append({"id": zid, "name": clean_name(d.get("name"), dict((k, n) for k, n, _ in DEFAULTS).get(zid, "區域")),
                    "color": color.lower() if COLOR_RE.match(color) else builtin_default,
                    "type": BUILTIN_TYPE.get(zid) or (d.get("type") if d.get("type") in TYPES else "plain")})
        seen.add(zid)
    return out[:MAX_ZONES] or defaults()


def zone_ids(t):
    return [d["id"] for d in zone_defs(t)]


def zone_names(t):
    return {d["id"]: d["name"] for d in zone_defs(t)}


def zone_types(t):
    return {d["id"]: d["type"] for d in zone_defs(t)}


def hex_bgr(c):
    """'#rrggbb' -> (b, g, r) for OpenCV."""
    c = c.lstrip("#")
    return int(c[4:6], 16), int(c[2:4], 16), int(c[0:2], 16)


def toml_line(defs):
    """The config line for the list (names are already cleaned: no quotes or backslashes)."""
    return "zone_defs = [" + ", ".join(f'{{id = "{d["id"]}", name = "{d["name"]}", color = "{d["color"]}", type = "{d["type"]}"}}'
                                        for d in defs) + "]"
