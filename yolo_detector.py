#!/usr/bin/env python3
"""YOLO (Ultralytics YOLO11 / YOLOv8 exported to ONNX) object detector for the Raspberry Pi CPU, on ONNX Runtime.

Usage:
    det = YoloOnnx("models/hamster_ncnn_model", conf=0.4)
    for x, y, w, h, score, cls in det.detect(bgr_frame):   # boxes normalized 0..1 to the frame
        ...

Export a trained model with:  yolo export model=best.pt format=onnx imgsz=640 simplify=True
and save it as model.onnx in the model folder (next to best.pt and metadata.yaml).
(The folder names still say "ncnn": they predate the switch to ONNX Runtime and are kept so nothing else moves.)
"""
import ast
import re
from pathlib import Path

import cv2
import numpy as np

MODEL_FILE = "model.onnx"


def has_model(model_dir):
    return (Path(model_dir) / MODEL_FILE).exists()


def weights_file(model_dir):
    """The file a model's fingerprint (id, version number) is taken from. Models from before the switch keep the id
    of their original weights file, so their version numbers don't change; newer ones use model.onnx."""
    d = Path(model_dir)
    old = d / "model.ncnn.bin"
    return old if old.exists() else d / MODEL_FILE


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


class YoloOnnx:
    def __init__(self, model_dir, imgsz=320, conf=0.4, iou=0.5, threads=2):
        import onnxruntime as ort   # imported here: only the processes that run a model need it
        d = Path(model_dir)
        so = ort.SessionOptions()
        so.intra_op_num_threads = threads
        so.inter_op_num_threads = 1
        self.sess = ort.InferenceSession(str(d / MODEL_FILE), sess_options=so, providers=["CPUExecutionProvider"])
        inp = self.sess.get_inputs()[0]
        self.input_name = inp.name
        # the export has a fixed input size: the model's own size always wins over the one asked for
        shape = inp.shape
        self.imgsz = int(shape[2]) if isinstance(shape[2], int) else (read_imgsz(d) or imgsz)
        self.conf, self.iou = conf, iou
        self.names = read_names(d)
        try:   # Ultralytics stores the class names in the ONNX file too
            meta = self.sess.get_modelmeta().custom_metadata_map
            if meta.get("names"):
                self.names = {int(k): str(v) for k, v in ast.literal_eval(meta["names"]).items()}
        except Exception:
            pass
        out = self.sess.get_outputs()[0].shape   # (1, 4 + num_classes, num_anchors)
        self.nc = int(out[1]) - 4 if isinstance(out[1], int) else (len(self.names) or 1)
        for k, v in {HAMSTER: "hamster", HEAD: "head"}.items():
            if k < self.nc:
                self.names.setdefault(k, v)

    def _raw(self, img):
        x = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        x = np.ascontiguousarray(x.transpose(2, 0, 1)[None])
        pred = self.sess.run(None, {self.input_name: x})[0]   # (1, 4 + num_classes, num_anchors)
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
    det = YoloOnnx(sys.argv[1], conf=0.25)
    img = cv2.imread(sys.argv[2])
    det.detect(img)                                   # warm-up
    t = time.perf_counter()
    found = det.detect(img)
    print(f"{(time.perf_counter() - t) * 1000:.0f} ms, {len(found)} objects")
    for x, y, w, h, s, c in found:
        print(f"  {det.names.get(c, c):12s} {s:.2f}  x={x:.3f} y={y:.3f} w={w:.3f} h={h:.3f}")
