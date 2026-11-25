#!/usr/bin/env python3
"""Camera aiming helper: live preview with guides, a focus loupe and a sharpness score.

Started from the dashboard ("調整角度"); runs as the hamster-aim service and stops by itself
after 10 minutes. Frames go to /run/hamster/live.jpg, which web.py streams as MJPEG.
"""
import json
import os
import signal
import time
import tomllib
from pathlib import Path

import cv2
import numpy as np
from libcamera import Transform
from picamera2 import Picamera2

BASE = Path(__file__).resolve().parent
LIVE_DIR = Path("/run/hamster")  # tmpfs created by /etc/tmpfiles.d/hamster.conf (not /dev/shm: logind wipes that on logout)
MAX_RUN_S = 10 * 60
W, H = 1024, 768          # bigger than the tracker's frames: easier to judge focus by eye
SENSOR_FULL_FOV = (1640, 1232)  # binned full-sensor mode; smaller modes are centre crops (narrower view)


def atomic_write(path, data):
    tmp = path.with_name(path.name + ".tmp")
    try:
        tmp.write_bytes(data)
    except FileNotFoundError:  # directory vanished (e.g. tmpfs cleaned): recreate and retry once
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp.write_bytes(data)
    os.replace(tmp, path)


def draw_guides(img, roi, floor):
    h, w = img.shape[:2]
    t = max(1, w // 640)
    line = (255, 255, 255)
    for i in (1, 2):  # rule-of-thirds grid
        cv2.line(img, (w * i // 3, 0), (w * i // 3, h), line, t, cv2.LINE_AA)
        cv2.line(img, (0, h * i // 3), (w, h * i // 3), line, t, cv2.LINE_AA)
    cx, cy, arm = w // 2, h // 2, 14 * t
    cv2.line(img, (cx - arm, cy), (cx + arm, cy), (0, 220, 255), 2 * t, cv2.LINE_AA)
    cv2.line(img, (cx, cy - arm), (cx, cy + arm), (0, 220, 255), 2 * t, cv2.LINE_AA)
    if len(roi) >= 3:
        cv2.polylines(img, [np.int32([[x * w, y * h] for x, y in roi])], True, (255, 200, 0), 2 * t, cv2.LINE_AA)
    if len(floor) == 4:
        cv2.polylines(img, [np.int32([[x * w, y * h] for x, y in floor])], True, (200, 0, 255), 2 * t, cv2.LINE_AA)


def main():
    with open(BASE / "config.toml", "rb") as f:
        cfg = tomllib.load(f)
    t_cfg = cfg.get("tracker", {})
    roi, floor = t_cfg.get("roi") or [], t_cfg.get("floor") or []
    LIVE_DIR.mkdir(parents=True, exist_ok=True)

    cam = Picamera2()
    transform = Transform(hflip=1, vflip=1) if cfg["camera"]["rotation"] == 180 else Transform()
    cam.configure(cam.create_video_configuration(
        main={"size": (W, H), "format": "RGB888"}, sensor={"output_size": SENSOR_FULL_FOV},
        transform=transform, controls={"FrameRate": 15}))
    cam.start()

    stop = False

    def handle_stop(*_):
        nonlocal stop
        stop = True

    signal.signal(signal.SIGTERM, handle_stop)
    signal.signal(signal.SIGINT, handle_stop)

    started = time.time()
    sharp_avg = None
    t_info = 0.0
    try:
        while not stop and time.time() - started < MAX_RUN_S:
            frame = cam.capture_array("main")
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            # focus is judged on the centre, where the cage floor should be
            centre = cv2.resize(gray[H // 4: 3 * H // 4, W // 4: 3 * W // 4], (320, 240), interpolation=cv2.INTER_AREA)
            sharp = float(cv2.Laplacian(centre, cv2.CV_64F).var())
            sharp_avg = sharp if sharp_avg is None else sharp_avg * 0.8 + sharp * 0.2
            bright = float(gray.mean())

            view = frame.copy()
            draw_guides(view, roi, floor)
            # 2x focus loupe of the centre, bottom-right
            lw, lh, m = W * 3 // 8, H * 3 // 8, W // 64
            src = frame[H // 2 - lh // 4: H // 2 + lh // 4, W // 2 - lw // 4: W // 2 + lw // 4]
            view[H - lh - m: H - m, W - lw - m: W - m] = cv2.resize(src, (lw, lh), interpolation=cv2.INTER_NEAREST)
            cv2.rectangle(view, (W - lw - m - 1, H - lh - m - 1), (W - m, H - m), (255, 255, 255), 2)
            cv2.putText(view, "2x", (W - lw - m + 8, H - lh - m + 26), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                        (255, 255, 255), 2, cv2.LINE_AA)

            ok, jpg = cv2.imencode(".jpg", view, [cv2.IMWRITE_JPEG_QUALITY, 80])
            if ok:
                atomic_write(LIVE_DIR / "live.jpg", jpg.tobytes())
            now = time.time()
            if now - t_info >= 0.5:
                t_info = now
                # clean frame for the cage-area editor, so it shows the angle you are setting right now
                ok_raw, raw = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
                if ok_raw:
                    atomic_write(BASE / "data" / "raw.jpg", raw.tobytes())
                atomic_write(LIVE_DIR / "aim.json", json.dumps({
                    "sharpness": round(sharp_avg, 1), "brightness": round(bright),
                    "remaining_s": int(MAX_RUN_S - (now - started)), "updated": now}).encode())
    finally:
        (LIVE_DIR / "aim.json").unlink(missing_ok=True)
        # leave the final, guide-free view as the dashboard snapshot so it matches the new camera position
        if "frame" in locals():
            ok, jpg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
            if ok:
                atomic_write(BASE / "data" / "latest.jpg", jpg.tobytes())
                atomic_write(BASE / "data" / "raw.jpg", jpg.tobytes())   # the cage-area editor uses this
        cam.stop()


if __name__ == "__main__":
    main()
