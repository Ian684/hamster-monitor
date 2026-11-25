#!/usr/bin/env python3
"""Draw the home-screen icon (iPhone "Add to Home Screen") into static/.

Run once on the Pi:  cd ~/hamster && .venv/bin/python make_icons.py
"""
from pathlib import Path

import cv2
import numpy as np

STATIC = Path(__file__).resolve().parent / "static"


def hamster_icon(size):
    s = size / 64.0          # artwork is designed on a 64x64 grid (same as the browser-tab icon)
    big = 4                  # draw 4x larger, then shrink = smooth edges
    n = size * big

    def p(x, y):
        return int(x * s * big), int(y * s * big)

    def r(v):
        return int(v * s * big)

    # sunset gradient, like the default wallpaper (BGR)
    top, bottom = np.array([201, 163, 122], float), np.array([44, 98, 184], float)
    t = np.linspace(0, 1, n)[:, None, None]
    img = np.broadcast_to(top * (1 - t) + bottom * t, (n, n, 3)).astype(np.uint8).copy()
    fur, belly, ink = (91, 163, 227), (216, 234, 251), (34, 34, 34)
    cv2.circle(img, p(17, 20), r(8), fur, -1, cv2.LINE_AA)                      # ears
    cv2.circle(img, p(47, 20), r(8), fur, -1, cv2.LINE_AA)
    cv2.circle(img, p(17, 20), r(4), (150, 170, 240), -1, cv2.LINE_AA)
    cv2.circle(img, p(47, 20), r(4), (150, 170, 240), -1, cv2.LINE_AA)
    cv2.ellipse(img, p(32, 38), (r(22), r(19)), 0, 0, 360, fur, -1, cv2.LINE_AA)  # head/body
    cv2.ellipse(img, p(32, 45), (r(13), r(10)), 0, 0, 360, belly, -1, cv2.LINE_AA)
    cv2.circle(img, p(24, 34), r(3), ink, -1, cv2.LINE_AA)                      # eyes
    cv2.circle(img, p(40, 34), r(3), ink, -1, cv2.LINE_AA)
    cv2.circle(img, p(25, 33), r(1), (255, 255, 255), -1, cv2.LINE_AA)
    cv2.circle(img, p(41, 33), r(1), (255, 255, 255), -1, cv2.LINE_AA)
    cv2.ellipse(img, p(32, 40), (r(2.2), r(1.6)), 0, 0, 360, (120, 110, 230), -1, cv2.LINE_AA)  # nose
    return cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)


if __name__ == "__main__":
    STATIC.mkdir(exist_ok=True)
    for name, size in (("apple-touch-icon.png", 180), ("icon-192.png", 192), ("icon-512.png", 512)):
        cv2.imwrite(str(STATIC / name), hamster_icon(size))
        print("wrote", STATIC / name)
