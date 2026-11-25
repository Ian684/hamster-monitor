"""Video clips of the hamster moving, for the dashboard (紀錄 › 影片).

tracker.py calls ClipRecorder.feed() with every camera frame. A clip:
  - starts when the hamster moves (with AI installed: only when the AI also sees the hamster),
  - includes the few seconds *before* that (pre-roll), so you see how it started,
  - ends after it has been still for post_s seconds, and is split at max_s,
  - is thrown away if there were fewer than min_active_s seconds of real movement (bedding flicker, light change).

Files live in data/clips/:  clip_YYYYMMDD_HHMMSS.mp4  (+ .jpg thumbnail, + .json details,
+ .boxes: where the AI saw the hamster / its head in each frame, drawn over the video only when the viewer
asks for it - the video itself stays clean so it can still be used for training)
Video is H.264 in MP4 (plays in Safari / Chrome / iPhone), encoded by ffmpeg in a background thread so the
tracker loop never waits on it. The Pi 5 has no hardware H.264 encoder; libx264 "ultrafast" at 640x480, 10 fps
costs a fraction of one core. Old clips are deleted automatically to stay under max_gb.
"""
import json
import logging
import os
import queue
import shutil
import subprocess
import threading
import time
from collections import deque
from datetime import datetime
from pathlib import Path

import cv2

log = logging.getLogger("clips")

DEFAULTS = {
    "enabled": True,
    "fps": 10,              # recorded frame rate (tracker runs at 15; 10 is smooth enough and lighter)
    "pre_s": 3,             # keep this much from before the movement started
    "post_s": 5,            # stop after the hamster has been still this long
    "max_s": 120,           # split longer activity into clips of this length
    "min_active_s": 2,      # discard clips with less real movement than this
    "crf": 28,              # H.264 quality: lower = better and bigger (23..30 is sensible)
    "max_gb": 10,           # total size limit for all clips; the oldest are deleted first
    "keep_days": 30,        # and nothing older than this
    "min_free_gb": 3,       # never fill the SD card: below this, delete old clips / stop recording
}


def clip_settings(cfg):
    s = dict(DEFAULTS)
    s.update(cfg.get("clips", {}))
    return s


class _Writer(threading.Thread):
    """One ffmpeg process per clip, fed raw BGR frames through a pipe from a queue."""

    def __init__(self, path, w, h, fps, crf):
        super().__init__(daemon=True)
        self.path, self.part = path, path.with_suffix(".part.mp4")
        self.q = queue.Queue(maxsize=int(fps * 4))
        self.dropped = 0
        self.ok = False
        cmd = ["nice", "-n", "10", "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
               "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{w}x{h}", "-r", str(fps), "-i", "-",
               "-c:v", "libx264", "-preset", "ultrafast", "-tune", "zerolatency", "-crf", str(crf),
               # a key frame every second: dragging the seek bar shows the picture at once (the default was one per
               # 25 s = one per clip, so the player had to decode from the start for every position)
               "-g", str(int(round(fps))), "-keyint_min", str(int(round(fps))),
               "-pix_fmt", "yuv420p", "-threads", "2", "-movflags", "+faststart", str(self.part)]
        self.proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
        self.start()

    def put(self, frame):
        try:
            self.q.put_nowait(frame)
            return True
        except queue.Full:
            self.dropped += 1   # encoder fell behind: skip a frame rather than slow the tracker down
            return False

    def close(self):
        self.q.put(None)

    def run(self):
        try:
            while True:
                f = self.q.get()
                if f is None:
                    break
                self.proc.stdin.write(f.tobytes())
            self.proc.stdin.close()
            err = self.proc.stderr.read().decode(errors="replace").strip()
            rc = self.proc.wait(timeout=60)
            if rc == 0 and self.part.exists():
                os.replace(self.part, self.path)
                self.ok = True
            else:
                log.warning("ffmpeg failed (%s): %s", rc, err[-300:])
        except (OSError, subprocess.SubprocessError, ValueError) as e:
            log.warning("clip writer error: %s", e)
        finally:
            if not self.ok:
                self.part.unlink(missing_ok=True)


class ClipRecorder:
    def __init__(self, cfg, base_dir, w, h):
        self.s = clip_settings(cfg)
        self.dir = Path(base_dir) / "data" / "clips"
        self.dir.mkdir(parents=True, exist_ok=True)
        self.w, self.h = w, h
        self.enabled = bool(self.s["enabled"]) and shutil.which("ffmpeg") is not None
        if self.s["enabled"] and not self.enabled:
            log.warning("ffmpeg not installed: video clips are off (sudo apt install ffmpeg)")
        self.step = 1.0 / self.s["fps"]
        self.pre = deque(maxlen=int(self.s["pre_s"] * self.s["fps"]) + 1)
        self.t_frame = -1e9
        self.manual = False       # set by tracker.py while 錄影 (recording by hand) is on
        self.cur = None           # the clip being recorded
        self.finishing = []       # clips whose encoder is still flushing
        self.t_house = -1e9
        self.t_low_warn = -1e9
        for p in self.dir.glob("*.part.mp4"):   # leftovers of a crash
            p.unlink(missing_ok=True)

    # ---------- called by tracker.py ----------
    @property
    def recording(self):
        return self.cur is not None

    def feed(self, frame, now, trigger, zone, paused, box=None, conf=None, boxes=None, head_conf=None):
        """frame: BGR image; now: monotonic seconds; trigger: hamster moving (and seen) this frame;
        conf: the AI's confidence if it ran on this frame and saw the hamster;
        boxes: what the AI sees now, [x, y, w, h, conf] (+ [hx, hy, hw, hh] of the head), 0-1, or None."""
        if not self.enabled:
            return
        if conf is not None and self.cur:
            c = self.cur
            c["conf_n"] += 1
            c["conf_sum"] += float(conf)
            c["conf_max"] = max(c["conf_max"], float(conf))
        if head_conf is not None and self.cur:   # the head, when the model found it in this AI run
            self.cur["head_n"] += 1
            self.cur["head_sum"] += float(head_conf)
        self._reap()
        if now - self.t_house >= 600:
            self.t_house = now
            self.housekeep()
        if paused:
            self._finish()
            self.pre.clear()
            return
        if now - self.t_frame < self.step:
            if trigger and self.cur:
                self.cur["last_trigger"] = now
            return
        # advance by exactly one step (not "= now"): with 15 fps in and 10 fps out, "= now" keeps only every
        # 2nd frame (7.5 fps) and the clip plays too fast. Don't build up a backlog after a pause.
        self.t_frame = max(self.t_frame + self.step, now - self.step)
        stamped = frame.copy()
        _stamp(stamped)

        if self.cur is None:
            self.pre.append((stamped, boxes))
            if trigger:
                self._begin(now, box)
            return

        c = self.cur
        if self.manual:
            c["manual"] = True
        if c["w"].put(stamped):
            c["boxes"].append(boxes)   # one entry per frame really in the video, so it stays in step
        c["frames"] += 1
        if trigger:
            c["last_trigger"] = now
            c["active_frames"] += 1
            if box is not None and c["thumb"] is None:
                c["thumb"] = stamped
        for z in zone if isinstance(zone, (list, tuple)) else [zone]:   # can be in an area and at a bottle at once
            if z:
                c["zones"][z] = c["zones"].get(z, 0) + self.step
        if now - c["last_trigger"] >= self.s["post_s"]:
            self._finish()
        elif now - c["t0"] >= (330 if c.get("manual") else self.s["max_s"]):   # 錄影 by hand: one clip (up to 5 min)
            still_active = now - c["last_trigger"] < 1.0
            self._finish()
            if still_active:            # keep going in a new clip, no gap
                self._begin(now, box)

    def stop(self):
        self._finish()
        for c in self.finishing:
            c["w"].join(timeout=60)
        self._reap()

    # ---------- internals ----------
    def _begin(self, now, box):
        if self._free_gb() < self.s["min_free_gb"]:
            self.housekeep()
            if self._free_gb() < self.s["min_free_gb"]:
                if now - self.t_low_warn >= 3600:
                    self.t_low_warn = now
                    log.warning("not recording: only %.1f GB free (min_free_gb = %s)", self._free_gb(), self.s["min_free_gb"])
                return
        start = datetime.now()
        name = f"clip_{start:%Y%m%d_%H%M%S}"
        path = self.dir / f"{name}.mp4"
        if path.exists():
            return
        try:
            w = _Writer(path, self.w, self.h, self.s["fps"], self.s["crf"])
        except OSError as e:
            log.warning("cannot start ffmpeg: %s", e)
            self.enabled = False
            return
        pre = list(self.pre)
        self.pre.clear()
        fb = [b for f, b in pre if w.put(f)]
        self.cur = {"name": name, "path": path, "w": w, "t0": now, "start": start, "frames": len(pre),
                    "pre_frames": len(pre), "active_frames": 1, "last_trigger": now, "zones": {},
                    "thumb": pre[-1][0] if pre else None, "conf_n": 0, "conf_sum": 0.0, "conf_max": 0.0, "boxes": fb,
                    "head_n": 0, "head_sum": 0.0}

    def _finish(self):
        c, self.cur = self.cur, None
        if c is None:
            return
        c["w"].close()
        c["end"] = datetime.now()
        self.finishing.append(c)

    def _reap(self):
        """Encoders that have finished: keep the clip (thumbnail + details) or throw it away."""
        for c in list(self.finishing):
            if c["w"].is_alive():
                continue
            self.finishing.remove(c)
            active_s = c["active_frames"] * self.step
            if not c["w"].ok or active_s < self.s["min_active_s"]:
                c["path"].unlink(missing_ok=True)
                continue
            seconds = round(c["frames"] / self.s["fps"], 1)
            if c["thumb"] is not None:
                t = c["thumb"]
                small = cv2.resize(t, (320, round(320 * t.shape[0] / t.shape[1])), interpolation=cv2.INTER_AREA)
                cv2.imwrite(str(c["path"].with_suffix(".jpg")), small, [cv2.IMWRITE_JPEG_QUALITY, 75])
            info = {"name": c["name"], "start": c["start"].isoformat(timespec="seconds"),
                    "end": c["end"].isoformat(timespec="seconds"), "seconds": seconds,
                    "active_s": round(active_s, 1), "zones": {k: round(v) for k, v in c["zones"].items() if v >= 1},
                    "size": c["path"].stat().st_size, "dropped_frames": c["w"].dropped}
            if c.get("manual"):
                info["manual"] = True   # 自己錄的: shown first in 影片, never deleted for being old
            if c["conf_n"]:   # how sure the AI was that this is the hamster, over the clip
                info["ai_conf"] = {"avg": round(c["conf_sum"] / c["conf_n"], 3), "max": round(c["conf_max"], 3), "n": c["conf_n"]}
                # head: average over the AI runs that found it, and in how many of the runs that saw the body
                info["head_conf"] = {"avg": round(c["head_sum"] / c["head_n"], 3) if c["head_n"] else 0.0, "n": c["head_n"]}
            if any(c["boxes"]):   # the boxes, per frame (small: ~10-30 KB a clip)
                bx = c["path"].with_suffix(".boxes")
                bx.with_suffix(".boxes.tmp").write_text(json.dumps(
                    {"fps": self.s["fps"], "f": c["boxes"]}, separators=(",", ":")))
                os.replace(bx.with_suffix(".boxes.tmp"), bx)
                info["boxes"] = True
            tmp = c["path"].with_suffix(".json.tmp")
            tmp.write_text(json.dumps(info, ensure_ascii=False))
            os.replace(tmp, c["path"].with_suffix(".json"))
            log.info("clip saved: %s, %.0f s (%d KB)%s", c["name"], seconds, info["size"] // 1024,
                     f", {c['w'].dropped} frames dropped" if c["w"].dropped else "")

    def _free_gb(self):
        try:
            return shutil.disk_usage(self.dir).free / 1e9
        except OSError:
            return 0.0

    def housekeep(self):
        """Delete clips older than keep_days, then oldest first until under max_gb and min_free_gb."""
        clips = sorted(p for p in self.dir.glob("clip_*.mp4") if not p.name.endswith(".part.mp4"))   # never one being written
        cutoff = time.time() - self.s["keep_days"] * 86400
        total = sum(p.stat().st_size for p in clips)
        cap = self.s["max_gb"] * 1e9
        for p in clips:
            if self.cur and p == self.cur["path"]:
                continue
            old = p.stat().st_mtime < cutoff
            full = total > cap or self._free_gb() < self.s["min_free_gb"]
            if not (old or full):
                break
            if not full and _is_manual(p):   # recorded by hand: kept however old (only a full card removes it)
                continue
            total -= p.stat().st_size
            for f in (p, p.with_suffix(".jpg"), p.with_suffix(".json"), p.with_suffix(".boxes")):
                f.unlink(missing_ok=True)


def _is_manual(mp4):
    try:
        return bool(json.loads(mp4.with_suffix(".json").read_text()).get("manual"))
    except (OSError, ValueError):
        return False


def _stamp(img):
    """Small date/time in the corner, so a clip shows when it happened."""
    # top right: outside the glass, nothing there to see. (Bottom left, where it was, is the water spout:
    # the time covered the mouth at the bottle when labelling drinking.)
    txt = datetime.now().strftime("%m/%d %H:%M:%S")
    (tw, th), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    org = (img.shape[1] - tw - 8, th + 8)
    cv2.putText(img, txt, org, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(img, txt, org, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
