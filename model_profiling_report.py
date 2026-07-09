"""
Model Profiling Report
Metrics: params, FLOPs, model size, latency
Models:
  - ML : HOG + SVM  (hog_svm_base_model.joblib)
  - DL : MobileNetV3 (mobilenet_v3_binary_final.keras)
"""

import os
import time
import platform
import numpy as np
import joblib
import cv2
from pathlib import Path
from skimage.feature import hog

# ── paths ────────────────────────────────────────────────────────────────────
BASE = Path(__file__).resolve().parent
ML_MODEL_PATH = BASE / "ML/HOG/Base/hog_svm_base_outputs/hog_svm_base_model.joblib"
DL_MODEL_PATH = BASE / "DL/MobileNetV3/Opt/mobilenet_v3_binary_final.keras"
OUTPUT_PATH   = BASE / "model_profiling_report.txt"

IMAGE_SIZE    = (300, 300)
LATENCY_RUNS  = 100   # warm + repeated runs for stable latency

# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def file_size_mb(path: Path) -> float:
    return path.stat().st_size / (1024 ** 2)

def extract_hog(img_bgr: np.ndarray) -> np.ndarray:
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, IMAGE_SIZE)
    feat = hog(
        img_resized,
        orientations=9,
        pixels_per_cell=(16, 16),
        cells_per_block=(2, 2),
        channel_axis=-1,
    )
    return feat

def dummy_image_bgr() -> np.ndarray:
    return (np.random.rand(*IMAGE_SIZE, 3) * 255).astype(np.uint8)

# ─────────────────────────────────────────────────────────────────────────────
# ML  — HOG + SVM
# ─────────────────────────────────────────────────────────────────────────────

def profile_ml():
    print("  Loading HOG + SVM model ...")
    pipeline = joblib.load(ML_MODEL_PATH)

    svm = pipeline.named_steps["classifier"]
    scaler = pipeline.named_steps["scaler"]

    # ── params ───────────────────────────────────────────────────────────────
    n_sv          = svm.n_support_.sum()           # total support vectors
    n_features    = svm.shape_fit_[1]              # HOG feature dim after scaler
    n_classes     = len(svm.classes_)
    svm_params    = n_sv * n_features              # SV matrix elements
    scaler_params = n_features * 2                 # mean + var per feature
    total_params  = svm_params + scaler_params

    # ── FLOPs ────────────────────────────────────────────────────────────────
    # RBF kernel per SV: ||x - sv||^2  →  n_features sub + n_features mul + (n_features-1) add  ≈ 3*n_features
    # Weighted sum over SVs: n_sv mul + n_sv add
    # HOG extraction (approx per pixel block):
    #   pixels_per_cell=16x16, cells_per_block=2x2, orientations=9
    #   gradient magnitude+angle per pixel: ~6 ops
    #   binning per pixel: ~9 ops
    #   normalization per block: ~(4*9)*2 ops
    h, w = IMAGE_SIZE
    pixels = h * w
    hog_flops = pixels * (6 + 9) + (h // 16 - 1) * (w // 16 - 1) * (4 * 9 * 2)
    scaler_flops = n_features * 2                                   # (x-mean)/std
    kernel_flops = n_sv * (3 * n_features + 2)                      # per sample
    total_flops  = hog_flops + scaler_flops + kernel_flops

    # ── model size ───────────────────────────────────────────────────────────
    size_mb = file_size_mb(ML_MODEL_PATH)

    # ── latency ──────────────────────────────────────────────────────────────
    dummy = dummy_image_bgr()
    feat  = extract_hog(dummy).reshape(1, -1)
    # warmup
    for _ in range(5):
        pipeline.predict(feat)
    times = []
    for _ in range(LATENCY_RUNS):
        t0 = time.perf_counter()
        feat_fresh = extract_hog(dummy_image_bgr()).reshape(1, -1)
        pipeline.predict(feat_fresh)
        times.append(time.perf_counter() - t0)
    latency_ms = np.mean(times) * 1000
    latency_std_ms = np.std(times) * 1000

    return {
        "n_support_vectors": int(n_sv),
        "n_features": int(n_features),
        "n_classes": int(n_classes),
        "total_params": int(total_params),
        "total_flops": int(total_flops),
        "size_mb": size_mb,
        "latency_mean_ms": latency_ms,
        "latency_std_ms": latency_std_ms,
        "kernel": svm.kernel,
        "C": svm.C,
        "gamma": svm.gamma,
        "n_sv_per_class": svm.n_support_.tolist(),
    }

# ─────────────────────────────────────────────────────────────────────────────
# DL  — MobileNetV3
# ─────────────────────────────────────────────────────────────────────────────

def profile_dl():
    print("  Loading MobileNetV3 model (TensorFlow) ...")
    import tensorflow as tf

    model = tf.keras.models.load_model(DL_MODEL_PATH)

    # ── params ───────────────────────────────────────────────────────────────
    total_params     = model.count_params()
    trainable_params = sum(np.prod(v.shape) for v in model.trainable_variables)
    non_trainable    = total_params - trainable_params

    # ── FLOPs ────────────────────────────────────────────────────────────────
    try:
        from tensorflow.python.framework.convert_to_constants import (
            convert_variables_to_constants_v2_as_graph,
        )

        @tf.function(input_signature=[tf.TensorSpec([1, *IMAGE_SIZE, 3], tf.float32)])
        def forward(x):
            return model(x, training=False)

        concrete = forward.get_concrete_function()
        frozen_func, graph_def = convert_variables_to_constants_v2_as_graph(concrete)
        with tf.Graph().as_default() as graph:
            tf.graph_util.import_graph_def(graph_def, name="")
            run_meta = tf.compat.v1.RunMetadata()
            opts = tf.compat.v1.profiler.ProfileOptionBuilder.float_operation()
            flops_obj = tf.compat.v1.profiler.profile(
                graph=graph, run_meta=run_meta, cmd="op", options=opts
            )
        total_flops = flops_obj.total_float_ops
        flops_source = "tf.compat.v1.profiler (graph frozen)"
    except Exception as e:
        print(f"    tf.profiler fallback ({e})")
        # MobileNetV3Large MACs at input resolution (H/224)^2 * baseline_MACs
        # Baseline MNV3L 224x224 ≈ 219M MACs; scale by (300/224)^2
        scale = (IMAGE_SIZE[0] / 224) ** 2
        total_flops = int(219e6 * scale * 2)   # MACs * 2 = FLOPs
        flops_source = "analytical estimate (MNV3L 224 baseline scaled to 300x300)"

    # ── model size ───────────────────────────────────────────────────────────
    size_mb = file_size_mb(DL_MODEL_PATH)

    # ── latency ──────────────────────────────────────────────────────────────
    dummy_tf = tf.ones([1, *IMAGE_SIZE, 3], dtype=tf.float32)
    # warmup
    for _ in range(5):
        model(dummy_tf, training=False)
    times = []
    for _ in range(LATENCY_RUNS):
        t0 = time.perf_counter()
        model(dummy_tf, training=False)
        times.append(time.perf_counter() - t0)
    latency_ms = np.mean(times) * 1000
    latency_std_ms = np.std(times) * 1000

    return {
        "total_params": int(total_params),
        "trainable_params": int(trainable_params),
        "non_trainable_params": int(non_trainable),
        "total_flops": total_flops,
        "flops_source": flops_source,
        "size_mb": size_mb,
        "latency_mean_ms": latency_ms,
        "latency_std_ms": latency_std_ms,
        "tf_version": None,
    }

# ─────────────────────────────────────────────────────────────────────────────
# REPORT
# ─────────────────────────────────────────────────────────────────────────────

def fmt_flops(f):
    if f is None:
        return "N/A"
    if f >= 1e9:
        return f"{f/1e9:.3f} GFLOPs"
    if f >= 1e6:
        return f"{f/1e6:.3f} MFLOPs"
    return f"{f:,} FLOPs"

def fmt_params(p):
    if p >= 1e6:
        return f"{p/1e6:.4f} M  ({p:,})"
    if p >= 1e3:
        return f"{p/1e3:.2f} K  ({p:,})"
    return f"{p:,}"

def write_report(ml: dict, dl: dict):
    import datetime, tensorflow as tf

    lines = []
    lines.append("MODEL PROFILING REPORT")
    lines.append("=" * 60)
    lines.append(f"Generated   : {datetime.datetime.now():%Y-%m-%d %H:%M:%S}")
    lines.append(f"Platform    : {platform.system()} {platform.release()}")
    lines.append(f"Python      : {platform.python_version()}")
    lines.append(f"TensorFlow  : {tf.__version__}")
    lines.append(f"Latency runs: {LATENCY_RUNS} (full pipeline per sample)")
    lines.append("")

    # ── ML ────────────────────────────────────────────────────────────────────
    lines.append("=" * 60)
    lines.append("MODEL 1 : HOG + SVM (Machine Learning)")
    lines.append("=" * 60)
    lines.append(f"Model path              : {ML_MODEL_PATH}")
    lines.append("")
    lines.append("[ Parameters ]")
    lines.append(f"  SVM kernel            : {ml['kernel'].upper()}")
    lines.append(f"  SVM C                 : {ml['C']}")
    lines.append(f"  SVM gamma             : {ml['gamma']}")
    lines.append(f"  Support vectors total : {ml['n_support_vectors']:,}")
    lines.append(f"  SV per class          : {ml['n_sv_per_class']}")
    lines.append(f"  HOG feature dim       : {ml['n_features']:,}")
    lines.append(f"  Total params (SV mat + scaler): {fmt_params(ml['total_params'])}")
    lines.append("")
    lines.append("[ FLOPs (inference, 1 image) ]")
    lines.append(f"  HOG extraction + scaler + RBF kernel")
    lines.append(f"  Total FLOPs           : {fmt_flops(ml['total_flops'])}")
    lines.append(f"  Note: FLOPs = HOG ({fmt_flops(300*300*15)}) + scaler + RBF kernel per SV")
    lines.append("")
    lines.append("[ Model Size ]")
    lines.append(f"  File size             : {ml['size_mb']:.4f} MB")
    lines.append("")
    lines.append("[ Latency (CPU, full pipeline: HOG + scale + predict) ]")
    lines.append(f"  Mean latency          : {ml['latency_mean_ms']:.3f} ms")
    lines.append(f"  Std  latency          : {ml['latency_std_ms']:.3f} ms")
    lines.append(f"  Throughput (est.)     : {1000/ml['latency_mean_ms']:.1f} img/sec")
    lines.append("")

    # ── DL ────────────────────────────────────────────────────────────────────
    lines.append("=" * 60)
    lines.append("MODEL 2 : MobileNetV3Large (Deep Learning)")
    lines.append("=" * 60)
    lines.append(f"Model path              : {DL_MODEL_PATH}")
    lines.append("")
    lines.append("[ Parameters ]")
    lines.append(f"  Total params          : {fmt_params(dl['total_params'])}")
    lines.append(f"  Trainable params      : {fmt_params(dl['trainable_params'])}")
    lines.append(f"  Non-trainable params  : {fmt_params(dl['non_trainable_params'])}")
    lines.append("")
    lines.append("[ FLOPs (inference, 1 image @ 300x300) ]")
    lines.append(f"  Total FLOPs           : {fmt_flops(dl['total_flops'])}")
    lines.append(f"  Source                : {dl['flops_source']}")
    lines.append("")
    lines.append("[ Model Size ]")
    lines.append(f"  File size             : {dl['size_mb']:.4f} MB")
    lines.append("")
    lines.append("[ Latency (forward pass only, GPU if available) ]")
    lines.append(f"  Mean latency          : {dl['latency_mean_ms']:.3f} ms")
    lines.append(f"  Std  latency          : {dl['latency_std_ms']:.3f} ms")
    lines.append(f"  Throughput (est.)     : {1000/dl['latency_mean_ms']:.1f} img/sec")
    lines.append("")

    # ── COMPARISON TABLE ──────────────────────────────────────────────────────
    lines.append("=" * 60)
    lines.append("COMPARISON SUMMARY")
    lines.append("=" * 60)
    lines.append(f"{'Metric':<35} {'HOG+SVM':>12} {'MobileNetV3':>14}")
    lines.append("-" * 63)
    lines.append(f"{'Params (total)':<35} {ml['total_params']/1e6:>11.4f}M {dl['total_params']/1e6:>13.4f}M")
    flops_ml_str = f"{ml['total_flops']/1e6:.3f}M"
    flops_dl_str = f"{dl['total_flops']/1e9:.3f}G" if dl['total_flops'] else "N/A"
    lines.append(f"{'FLOPs (1 img)':<35} {flops_ml_str:>12} {flops_dl_str:>14}")
    lines.append(f"{'Model size (MB)':<35} {ml['size_mb']:>12.4f} {dl['size_mb']:>14.4f}")
    lines.append(f"{'Latency mean (ms)':<35} {ml['latency_mean_ms']:>12.3f} {dl['latency_mean_ms']:>14.3f}")
    lines.append(f"{'Throughput (img/s)':<35} {1000/ml['latency_mean_ms']:>12.1f} {1000/dl['latency_mean_ms']:>14.1f}")
    lines.append("")
    lines.append("Notes:")
    lines.append("  HOG+SVM latency includes HOG feature extraction.")
    lines.append("  MobileNetV3 latency is forward-pass only (excludes disk I/O).")
    lines.append("  FLOPs for SVM are approximate (RBF kernel MACs).")

    report = "\n".join(lines)
    print(report)
    OUTPUT_PATH.write_text(report, encoding="utf-8")
    print(f"\nReport saved → {OUTPUT_PATH}")

# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Profiling ML model (HOG + SVM) ...")
    ml_stats = profile_ml()
    print("  Done.\n")

    print("Profiling DL model (MobileNetV3) ...")
    dl_stats = profile_dl()
    print("  Done.\n")

    write_report(ml_stats, dl_stats)
