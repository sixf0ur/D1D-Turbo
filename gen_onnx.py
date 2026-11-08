#!/usr/bin/env python3
"""Generate drum one-shots (kick / snare / hat, 0.512 s @ 32 kHz) with the D1D-Turbo
ONNX model. Needs only numpy, onnxruntime and soundfile -- no torch, no GPU.

  python gen_onnx.py                                  # 4 per class, settings from the model
  python gen_onnx.py --cls snare --n 8 --seed 1
  python gen_onnx.py --model path/to/model.onnx --bench

The sampler defaults (Euler steps, CFG weight) are stored in the ONNX metadata
(keys `steps`, `cfg_w`; D1D-Turbo: 3 steps, cfg 1.0 = 3 net passes per sample).

Sampler = rectified-flow Euler, t: 0 (noise) -> 1 (sound), with classifier-free
guidance against the null class 3 when cfg_w != 1. One sample at a time: on a
Ryzen 5 5500, batch 1 ran 124 ms/pass/sample and batch 2 186 (cache), so CFG's
conditional and null passes are two calls, not one batch.
"""
import argparse, os, time
import numpy as np
import onnxruntime as ort
import soundfile as sf

SR, L = 32000, 16384
CLASSES = ["kick", "snare", "hat"]
NULL = len(CLASSES)             # class 3 = "no class", used only for CFG
PEAK_OUT = 0.95
HERE = os.path.dirname(os.path.abspath(__file__))


def default_model():
    """model.onnx next to this script, else in the current directory."""
    p = os.path.join(HERE, "model.onnx")
    return p if os.path.exists(p) else "model.onnx"


def session(path, threads=0):
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    if threads:
        so.intra_op_num_threads = threads
    return ort.InferenceSession(path, so, providers=["CPUExecutionProvider"])


def model_defaults(sess):
    """(steps, cfg_w) from the ONNX metadata; (3, 1.0) if the model has none."""
    meta = sess.get_modelmeta().custom_metadata_map
    return int(meta.get("steps", 3)), float(meta.get("cfg_w", 1.0))


def sample(sess, cls, n, steps, cfg_w, rng=None, noise=None):
    """Returns (n, L) float32 waveforms at ~unit RMS (peak-normalise to write).
    noise: optional (n, 1, L) float32 start noise."""
    rng = rng or np.random.default_rng()
    x = noise.copy() if noise is not None else rng.standard_normal((n, 1, L), dtype=np.float32)
    dt = 1.0 / steps
    yc, yu = np.array([cls], np.int64), np.array([NULL], np.int64)
    for j in range(n):
        xj = x[j:j + 1]
        for i in range(steps):
            t = np.full(1, i * dt, np.float32)          # 0, 1/steps, ...; never 1
            v = sess.run(None, {"x": xj, "t": t, "y": yc})[0]
            if cfg_w != 1.0:
                vu = sess.run(None, {"x": xj, "t": t, "y": yu})[0]
                v = vu + cfg_w * (v - vu)
            xj = xj + v * dt
        x[j:j + 1] = xj
    return x[:, 0]


def write(w, path):
    """Peak-normalise (the model's output level is arbitrary) and write 16-bit PCM."""
    peak = float(np.abs(w).max())
    sf.write(path, w / peak * PEAK_OUT if peak > 1e-6 else w, SR, subtype="PCM_16")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--model", default=None, help="ONNX file (default: model.onnx next to this script)")
    ap.add_argument("--cls", default=None, choices=CLASSES, help="one class (default: all three)")
    ap.add_argument("--all", action="store_true", help="all classes (the default)")
    ap.add_argument("--n", type=int, default=4, help="samples per class")
    ap.add_argument("--steps", type=int, default=None, help="Euler steps (default: from the model)")
    ap.add_argument("--cfg_w", type=float, default=None, help="CFG weight (default: from the model)")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--threads", type=int, default=0, help="0 = onnxruntime default")
    ap.add_argument("--out", default="out")
    ap.add_argument("--bench", action="store_true", help="print time per sample")
    args = ap.parse_args()
    path = args.model or default_model()
    sess = session(path, args.threads)
    d_steps, d_cfg = model_defaults(sess)
    steps = args.steps or d_steps
    cfg_w = args.cfg_w if args.cfg_w is not None else d_cfg
    rng = np.random.default_rng(args.seed)
    os.makedirs(args.out, exist_ok=True)
    print(f"model {path}  steps={steps}  cfg_w={cfg_w}")
    for name in ([args.cls] if args.cls else CLASSES):
        t0 = time.time()
        w = sample(sess, CLASSES.index(name), args.n, steps, cfg_w, rng)
        dt = time.time() - t0
        for i in range(args.n):
            write(w[i], os.path.join(args.out, f"{name}_{i:02d}.wav"))
        msg = f"{args.n} x {name}"
        if args.bench:
            passes = steps * (2 if cfg_w != 1.0 else 1)
            msg += f"  |  {dt / args.n * 1000:.0f} ms/sample, {passes} net passes/sample"
        print(msg)
    print(f"-> {os.path.abspath(args.out)}")


if __name__ == "__main__":
    main()
