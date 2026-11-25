#!/usr/bin/env python3
"""YOLO (Ultralytics YOLO11/YOLOv8 exported to NCNN) object detector for the Raspberry Pi CPU.

Usage:
    det = YoloNcnn("models/hamster_ncnn_model", imgsz=320, conf=0.4)
    for x, y, w, h, score, cls in det.detect(bgr_frame):   # boxes normalized 0..1 to the frame
        ...

Export a trained model with:  yolo export model=best.pt format=ncnn imgsz=320
That produces a folder with model.ncnn.param, model.ncnn.bin and metadata.yaml.
"""
import re
from pathlib import Path

import cv2
import ncnn
import numpy as np


def read_names(model_dir):
    """Class names from Ultralytics' metadata.yaml (no yaml dependency needed)."""
    meta = Path(model_dir) / "metadata.yaml"
    names = {}
    if meta.exists():
        block = re.search(r"(?ms)^names:\s*\n((?:\s+\d+:.*\n?)+)", meta.read_text(encoding="utf-8"))
        if block:
            for k, v in re.findall(r"^\s+(\d+):\s*(.+?)\s*$", block.group(1), re.M):
                names[int(k)] = v.strip("'\"")
    return names


HAMSTER, HEAD = 0, 1   # classes: 0 = the whole hamster, 1 = its head (models trained with head boxes)


def hamster_and_head(found):
    """From detect()'s boxes: (the surest hamster box or None, the head box inside it or None).
    With a hamster-only model every box is class 0, so the head is always None and nothing else changes."""
    found = sorted(found, key=lambda f: -f[4])
    ham = next((f for f in found if f[5] == HAMSTER), None)
    if ham is None:
        return None, None
    x, y, w, h = ham[:4]
    for f in found:
        if f[5] == HEAD:
            cx, cy = f[0] + f[2] / 2, f[1] + f[3] / 2
            if x - 0.15 * w <= cx <= x + 1.15 * w and y - 0.15 * h <= cy <= y + 1.15 * h:
                return ham, f
    return ham, None


def read_imgsz(model_dir):
    """Input size the model was exported with (metadata.yaml 'imgsz:' block), or None if unknown."""
    meta = Path(model_dir) / "metadata.yaml"
    if meta.exists():
        m = re.search(r"(?m)^imgsz:\s*\n\s*-\s*(\d+)", meta.read_text(encoding="utf-8"))
        if m:
            return int(m.group(1))
    return None


class YoloNcnn:
    def __init__(self, model_dir, imgsz=320, conf=0.4, iou=0.5, threads=2):
        d = Path(model_dir)
        # An NCNN export only works at the size it was exported with: feeding 320 to a 640 model
        # crashes the whole process (segfault), so the model's own size always wins.
        imgsz = read_imgsz(d) or imgsz
        self.imgsz, self.conf, self.iou = imgsz, conf, iou
        self.net = ncnn.Net()
        self.net.opt.use_vulkan_compute = False
        self.net.opt.num_threads = threads
        if self.net.load_param(str(d / "model.ncnn.param")) or self.net.load_model(str(d / "model.ncnn.bin")):
            raise RuntimeError(f"cannot load NCNN model from {d}")
        self.names = read_names(d)
        # how many classes the network really outputs (a zip without metadata.yaml was labelled "hamster" only,
        # 11/15, although the model also finds heads): one blank run tells, and unnamed ones get the usual names
        try:
            self.nc = len(self._raw(np.full((self.imgsz, self.imgsz, 3), 114, np.uint8))) - 4
        except Exception:
            self.nc = len(self.names) or 1
        for k, v in {HAMSTER: "hamster", HEAD: "head"}.items():
            if k < self.nc:
                self.names.setdefault(k, v)

    def _raw(self, img):
        mat = ncnn.Mat.from_pixels(np.ascontiguousarray(img), ncnn.Mat.PixelType.PIXEL_BGR2RGB, self.imgsz, self.imgsz)
        mat.substract_mean_normalize([], [1 / 255.0] * 3)
        ex = self.net.create_extractor()
        ex.input("in0", mat)
        _, out = ex.extract("out0")
        pred = np.array(out)                      # (4 + num_classes, num_anchors)
        return pred[0] if pred.ndim == 3 else pred

    def _letterbox(self, bgr):
        h0, w0 = bgr.shape[:2]
        s = self.imgsz
        r = min(s / h0, s / w0)
        nw, nh = round(w0 * r), round(h0 * r)
        top, left = (s - nh) // 2, (s - nw) // 2
        canvas = np.full((s, s, 3), 114, np.uint8)
        canvas[top:top + nh, left:left + nw] = cv2.resize(bgr, (nw, nh), interpolation=cv2.INTER_LINEAR)
        return canvas, r, left, top

    def detect(self, bgr):
        """Return [(x, y, w, h, score, class_id), ...] with x/y/w/h normalized to the input frame."""
        h0, w0 = bgr.shape[:2]
        img, r, left, top = self._letterbox(bgr)
        pred = self._raw(img)
        boxes, scores = pred[:4].T, pred[4:].T    # cx, cy, w, h in letterboxed pixels; class scores 0..1
        cls = scores.argmax(1)
        conf = scores[np.arange(len(cls)), cls]
        keep = conf >= self.conf
        if not keep.any():
            return []
        boxes, conf, cls = boxes[keep], conf[keep], cls[keep]
        # back to original-frame pixels
        x = (boxes[:, 0] - boxes[:, 2] / 2 - left) / r
        y = (boxes[:, 1] - boxes[:, 3] / 2 - top) / r
        w, h = boxes[:, 2] / r, boxes[:, 3] / r
        xywh = np.stack([x, y, w, h], 1)
        idx = cv2.dnn.NMSBoxesBatched(xywh.tolist(), conf.tolist(), cls.tolist(), self.conf, self.iou)
        res = []
        for i in np.array(idx).flatten():
            bx, by, bw, bh = (float(v) for v in xywh[i])   # plain floats: numpy float32 breaks json.dumps later
            bx, by = max(0.0, bx), max(0.0, by)
            bw, bh = min(bw, w0 - bx), min(bh, h0 - by)
            res.append((bx / w0, by / h0, bw / w0, bh / h0, float(conf[i]), int(cls[i])))
        return sorted(res, key=lambda d: -d[4])


if __name__ == "__main__":   # quick manual test:  python yolo_detector.py MODEL_DIR IMAGE
    import sys
    import time
    det = YoloNcnn(sys.argv[1], imgsz=int(sys.argv[3]) if len(sys.argv) > 3 else 320, conf=0.25)
    img = cv2.imread(sys.argv[2])
    det.detect(img)                                   # warm-up
    t = time.perf_counter()
    found = det.detect(img)
    print(f"{(time.perf_counter() - t) * 1000:.0f} ms, {len(found)} objects")
    for x, y, w, h, s, c in found:
        print(f"  {det.names.get(c, c):12s} {s:.2f}  x={x:.3f} y={y:.3f} w={w:.3f} h={h:.3f}")
