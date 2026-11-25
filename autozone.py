"""Tap-to-outline for the behaviour zones (wheel, food bowl, water bottle, nest, bath) in the cage editor.

You tap once on an object in the camera picture; this finds the object's outline around that point and returns
it as a polygon (0..1 picture coordinates) that you can then fine-tune by dragging the points.

How:
  1. The picture's edges are found (Canny, after evening out the brightness so dark night pictures work too).
  2. From the tap, a region grows over neighbouring pixels of similar colour, but never across an edge. Growing
     step by step from neighbour to neighbour follows smooth shading (a round yellow bottle that is light on one
     side and dark on the other) yet stops at the object's outline.
  3. This is tried with several strictness levels; the result that is biggest without "leaking" into the
     background (a sudden jump in size, or most of the picture) wins.
  4. Round things (wheel, food bowl) get an ellipse fitted to them; the others a smoothed outline.
Objects that are partly outside the picture are outlined up to the picture's edge.
"""
import cv2
import numpy as np

ROUND = {"wheel", "food"}   # zones that are usually round / oval in the picture
# largest believable size of each kind (share of the picture): anything bigger has leaked into the background
MAX_SHARE = {"water": 0.15, "food": 0.12, "wheel": 0.25, "nest": 0.2, "bath": 0.2}


def _prep(img):
    lab = cv2.cvtColor(cv2.GaussianBlur(img, (5, 5), 0), cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8)).apply(l)   # even out dark / bright parts
    lab = cv2.merge([l, a, b])
    gray = cv2.bilateralFilter(l, 7, 40, 7)
    edges = cv2.Canny(gray, 30, 90)
    edges = cv2.dilate(edges, np.ones((2, 2), np.uint8))
    return lab, edges


def _color_wall(lab, sx, sy, ab_max=16, l_max=70):
    """Pixels whose colour is clearly different from the tapped spot (hue/saturation, with room for shading).
    They act as walls, so the region can't creep into a neighbouring object of another colour through a gap."""
    H, W = lab.shape[:2]
    p = lab[max(0, sy - 4):sy + 5, max(0, sx - 4):sx + 5].reshape(-1, 3).astype(np.float32)
    ref = np.median(p, axis=0)
    d_ab = np.hypot(lab[..., 1].astype(np.float32) - ref[1], lab[..., 2].astype(np.float32) - ref[2])
    d_l = np.abs(lab[..., 0].astype(np.float32) - ref[0])
    return (d_ab > ab_max) | (d_l > l_max)


def _grow(lab, edges, sx, sy, tol, wall=None):
    H, W = edges.shape
    mask = np.zeros((H + 2, W + 2), np.uint8)
    walls = edges > 0
    if wall is not None:
        walls = walls | wall
    mask[1:-1, 1:-1] = walls.astype(np.uint8)               # edges (and other colours) are walls the region can't cross
    mask[1 + sy, 1 + sx] = 0
    cv2.floodFill(lab.copy(), mask, (sx, sy), 0, (tol, tol, tol), (tol, tol, tol),
                  8 | cv2.FLOODFILL_MASK_ONLY | (2 << 8))   # neighbour-to-neighbour (follows shading)
    region = (mask[1:-1, 1:-1] == 2).astype(np.uint8)
    # close the small gaps the edge walls leave inside the object, fill holes
    region = cv2.morphologyEx(region, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)))
    cnts, _ = cv2.findContours(region, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None
    filled = np.zeros_like(region)
    cv2.drawContours(filled, [max(cnts, key=cv2.contourArea)], -1, 1, -1)
    return filled


def _snap_tap(edges, sx, sy):
    """A tap that lands exactly on an edge line would grow nothing: move it to the nearest non-edge pixel."""
    if not edges[sy, sx]:
        return sx, sy
    ys, xs = np.nonzero(edges[max(0, sy - 8):sy + 9, max(0, sx - 8):sx + 9] == 0)
    if not len(xs):
        return sx, sy
    i = np.argmin((xs + max(0, sx - 8) - sx) ** 2 + (ys + max(0, sy - 8) - sy) ** 2)
    return int(xs[i] + max(0, sx - 8)), int(ys[i] + max(0, sy - 8))


def outline_many(img, taps, kind=None):
    """Several taps on the same object (e.g. the bottle's cap and its body): outline each and join them.
    Returns (polygon or None, number of taps that found nothing, reason of the last failure)."""
    H, W = img.shape[:2]
    m = np.zeros((H, W), np.uint8)
    failed, why = 0, None
    for x, y in taps:
        poly, err = outline(img, x, y, kind)
        if poly is None:
            failed, why = failed + 1, err
            continue
        cv2.fillPoly(m, [np.int32([[px * W, py * H] for px, py in poly])], 1)
    if not m.any():
        return None, failed, why
    cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    hull = cv2.convexHull(np.vstack(cnts))
    eps = 0.008 * cv2.arcLength(hull, True)
    poly = cv2.approxPolyDP(hull, eps, True)
    while len(poly) > 16:
        eps *= 1.3
        poly = cv2.approxPolyDP(hull, eps, True)
    return [[round(float(px) / W, 4), round(float(py) / H, 4)] for px, py in poly[:, 0, :]], failed, why


def _rim_ellipse(edges, sx, sy, W, H, max_share=0.3):
    """Round things filled with bedding (a food bowl) have no smooth inside to grow over, but their rim is a long
    curved edge line. Fit ellipses to the long edge curves near the tap and keep the best one around the tap."""
    r = int(0.3 * W)
    x0, y0 = max(0, sx - r), max(0, sy - r)
    cnts, _ = cv2.findContours(edges[y0:sy + r, x0:sx + r].copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    best, best_score = None, 0
    for c in cnts:
        if len(c) < 60:
            continue
        c = c + (x0, y0)
        (ex, ey), (ew, eh), ang = cv2.fitEllipse(c)
        area = np.pi * ew * eh / 4 / (W * H)
        if not 0.004 < area < max_share or min(ew, eh) < 0.25 * max(ew, eh):
            continue
        # the tap must be inside, and the curve must really lie on the ellipse
        pts = cv2.ellipse2Poly((int(ex), int(ey)), (int(ew / 2), int(eh / 2)), int(ang), 0, 360, 10).reshape(-1, 1, 2)
        if cv2.pointPolygonTest(pts, (float(sx), float(sy)), False) < 0:
            continue
        fit = np.mean([abs(cv2.pointPolygonTest(pts, (float(p[0][0]), float(p[0][1])), True)) for p in c[::5]])
        if fit > 6:
            continue
        score = len(c) / (1 + fit)
        if score > best_score:
            best, best_score = pts, score
    if best is None:
        return None
    best[:, 0, 0] = np.clip(best[:, 0, 0], 0, W - 1)
    best[:, 0, 1] = np.clip(best[:, 0, 1], 0, H - 1)
    return best


def outline(img, x, y, kind=None, max_share=None):
    """img: BGR picture; x, y: tap point (0..1); kind: zone kind (shape hint and size limit).
    Returns ([[x, y], ...] in 0..1, None) or (None, reason)."""
    max_share = max_share or MAX_SHARE.get(kind, 0.3)
    H, W = img.shape[:2]
    lab, edges = _prep(img)
    sx, sy = _snap_tap(edges, int(min(max(x, 0), 1) * (W - 1)), int(min(max(y, 0), 1) * (H - 1)))
    wall = _color_wall(lab, sx, sy)

    best, prev_area = None, 0
    for tol in (2, 3, 4, 5, 7, 9):
        r = _grow(lab, edges, sx, sy, tol, wall)
        if r is None:
            continue
        cnts, _ = cv2.findContours(r, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        area = cv2.contourArea(cv2.convexHull(max(cnts, key=cv2.contourArea))) / (W * H)   # size of the final outline
        if area > max_share or (prev_area > 0.004 and area > prev_area * 3.5):
            break                                             # leaked into the background: keep the previous one
        if area >= 0.0015:
            best, prev_area = r, area

    c = None
    if kind in ROUND and (best is None or prev_area < 0.006):   # bowl full of bedding: use its rim
        c = _rim_ellipse(edges, sx, sy, W, H, max_share)
    if c is None:
        if best is None:
            return None, "找不到物品的邊緣，請點在物品比較中間、顏色比較明顯的地方，或手動點"
        cnts, _ = cv2.findContours(best, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        c = max(cnts, key=cv2.contourArea)
    if kind in ROUND and len(c) >= 5:
        (ex, ey), (ew, eh), ang = cv2.fitEllipse(c)
        # an ellipse much bigger than the region means the region was a thin rim / partly hidden: use the hull then
        if np.pi * ew * eh / 4 < cv2.contourArea(cv2.convexHull(c)) * 1.8 and np.pi * ew * eh / 4 < max_share * W * H:
            pts = cv2.ellipse2Poly((int(ex), int(ey)), (int(ew / 2), int(eh / 2)), int(ang), 0, 360, 24)
            pts[:, 0] = np.clip(pts[:, 0], 0, W - 1)
            pts[:, 1] = np.clip(pts[:, 1], 0, H - 1)
            c = pts.reshape(-1, 1, 2)
    hull = cv2.convexHull(c)
    eps = 0.01 * cv2.arcLength(hull, True)
    poly = cv2.approxPolyDP(hull, eps, True)
    while len(poly) > 16:
        eps *= 1.3
        poly = cv2.approxPolyDP(hull, eps, True)
    if len(poly) < 3:
        return None, "找不到物品的邊緣"
    return [[round(float(px) / W, 4), round(float(py) / H, 4)] for px, py in poly[:, 0, :]], None
