"""Model names: version 1, 2, 3 ... from the oldest to the newest trained model.

The number belongs to the model itself (its weights' fingerprint, saved in models/versions.json), not to its folder:
switching models renames folders, but a model keeps its number. A model seen for the first time gets the next
number (several new ones: in the order they were trained). The owner's own name (title.txt, 管理模型 → 改名)
always wins over the number.
"""
import json
import os
import re
from datetime import datetime
from pathlib import Path

def model_dirs(models):
    return [d for d in Path(models).glob("hamster_model*") if d.is_dir() and (d / "model.onnx").exists()]


def model_id(d):
    from yolo_detector import model_fingerprint
    return model_fingerprint(d)


def trained_at(d):
    """When it was trained (metadata.yaml date), else when its folder appeared."""
    try:
        m = re.search(r'(?m)^date:\s*"?([0-9T:\-.+]+)', (Path(d) / "metadata.yaml").read_text(encoding="utf-8", errors="replace"))
        if m:
            return datetime.fromisoformat(m.group(1)).timestamp()
    except (OSError, ValueError):
        pass
    return Path(d).stat().st_mtime


def versions(models):
    """{folder name: version number} for every model folder; new models are numbered and remembered."""
    models = Path(models)
    f = models / "versions.json"
    try:
        known = json.loads(f.read_text())
    except (OSError, ValueError):
        known = {}
    dirs = model_dirs(models)
    ids = {d.name: model_id(d) for d in dirs}
    new = sorted((d for d in dirs if ids[d.name] not in known), key=trained_at)
    if new:
        n = max(known.values(), default=0)
        for d in new:
            if ids[d.name] not in known:
                n += 1
                known[ids[d.name]] = n
        tmp = f.with_suffix(".tmp")
        tmp.write_text(json.dumps(known))
        os.replace(tmp, f)
    return {name: known[i] for name, i in ids.items()}


def title(d):
    try:
        return (Path(d) / "title.txt").read_text(encoding="utf-8").strip()[:30] or None
    except OSError:
        return None


def label(d, vmap=None):
    """What the model is called everywhere: the owner's name, else 'version N'."""
    d = Path(d)
    t = title(d)
    if t:
        return t
    n = (vmap if vmap is not None else versions(d.parent)).get(d.name)
    return f"version {n}" if n else d.name


def never_used(d):
    """Made by 全自動訓練 and not switched to yet (its folder name is its training time, not a 'replaced at' time)."""
    try:
        j = json.loads((Path(d) / "autotrain.json").read_text())
        return Path(d).name.endswith("bak-" + str(j.get("trained")))
    except (OSError, ValueError):
        return False
