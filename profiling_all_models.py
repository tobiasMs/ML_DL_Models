"""
Comprehensive Model Profiling — All ML & DL Models
Metrics: params, FLOPs, model size, latency
"""

import os
import time
import platform
import datetime
import warnings
import numpy as np
import joblib
import cv2
from pathlib import Path
from skimage.feature import hog, local_binary_pattern

warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

BASE       = Path(__file__).resolve().parent
OUTPUT_DIR = BASE / "profiling_outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

IMAGE_SIZE   = (300, 300)
LATENCY_RUNS = 100

# ─────────────────────────────────────────────────────────────────────────────
# REGISTRY
# ─────────────────────────────────────────────────────────────────────────────

ML_MODELS = [
    {
        "name": "HOG + SVM",
        "variant": "Base",
        "path": BASE / "ML/HOG/Base/hog_svm_base_outputs/hog_svm_base_model.joblib",
        "feature": "HOG",
        "report": BASE / "ML/HOG/Base/hog_svm_base_outputs/hog_svm_base_report.txt",
    },
    {
        "name": "HOG + SVM",
        "variant": "Opt",
        "path": BASE / "ML/HOG/Opt/hog_svm_opt_outputs/hog_svm_opt_model.joblib",
        "feature": "HOG",
        "report": BASE / "ML/HOG/Opt/hog_svm_opt_outputs/hog_svm_opt_report.txt",
    },
    {
        "name": "LBP + Random Forest",
        "variant": "Base",
        "path": BASE / "ML/RandomForest/Base/lbp_random_forest_base_outputs/lbp_random_forest_base_model.joblib",
        "feature": "LBP",
        "report": BASE / "ML/RandomForest/Base/lbp_random_forest_base_outputs/lbp_random_forest_base_report.txt",
    },
    {
        "name": "LBP + Random Forest",
        "variant": "Opt",
        "path": BASE / "ML/RandomForest/Opt/lbp_random_forest_opt_outputs/lbp_random_forest_opt_model.joblib",
        "feature": "LBP",
        "report": BASE / "ML/RandomForest/Opt/lbp_random_forest_opt_outputs/lbp_random_forest_opt_report.txt",
    },
    {
        "name": "HSV Histogram + Logistic Regression",
        "variant": "Base",
        "path": BASE / "ML/LogisticRegression/Base/hsv_logistic_regression_base_outputs/hsv_logistic_regression_base_model.joblib",
        "feature": "HSV",
        "report": BASE / "ML/LogisticRegression/Base/hsv_logistic_regression_base_outputs/hsv_logistic_regression_base_report.txt",
    },
    {
        "name": "HSV Histogram + Logistic Regression",
        "variant": "Opt",
        "path": BASE / "ML/LogisticRegression/Opt/hsv_logistic_regression_opt_outputs/hsv_logistic_regression_opt_model.joblib",
        "feature": "HSV",
        "report": BASE / "ML/LogisticRegression/Opt/hsv_logistic_regression_opt_outputs/hsv_logistic_regression_opt_report.txt",
    },
]

DL_MODELS = [
    {
        "name": "MobileNetV3Large",
        "variant": "Base",
        "path": BASE / "DL/MobileNetV3/Base/mobilenet_v3_binary_final.keras",
        "report": BASE / "DL/MobileNetV3/Base/training_report_journal_friendly.txt",
    },
    {
        "name": "MobileNetV3Large",
        "variant": "Opt",
        "path": BASE / "DL/MobileNetV3/Opt/mobilenet_v3_binary_final.keras",
        "report": BASE / "DL/MobileNetV3/Opt/training_report_journal_friendly.txt",
    },
    {
        "name": "ResNet50",
        "variant": "Base",
        "path": BASE / "DL/ResNet/Base/resnet50_binary_final.keras",
        "report": BASE / "DL/ResNet/Base/training_report_resnet50.txt",
    },
    {
        "name": "ResNet50",
        "variant": "Opt",
        "path": BASE / "DL/ResNet/Opt/resnet50_binary_final.keras",
        "report": BASE / "DL/ResNet/Opt/training_report_resnet50_journal_friendly.txt",
    },
    {
        "name": "EfficientNetB0",
        "variant": "Base",
        "path": BASE / "DL/EfficientNet/Base/efficientnetb0_binary_final.keras",
        "report": BASE / "DL/EfficientNet/Base/training_report_journal_friendly.txt",
    },
    {
        "name": "EfficientNetB0",
        "variant": "Opt",
        "path": BASE / "DL/EfficientNet/Opt/efficientnetb0_binary_final.keras",
        "report": BASE / "DL/EfficientNet/Opt/training_report_journal_friendly.txt",
    },
]

# ─────────────────────────────────────────────────────────────────────────────
# FEATURE EXTRACTORS
# ─────────────────────────────────────────────────────────────────────────────

def dummy_bgr():
    return (np.random.rand(*IMAGE_SIZE, 3) * 255).astype(np.uint8)

def extract_hog(img_bgr):
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_res = cv2.resize(img_rgb, IMAGE_SIZE)
    feat = hog(img_res, orientations=9, pixels_per_cell=(16, 16),
               cells_per_block=(2, 2), channel_axis=-1)
    return feat.reshape(1, -1)

def extract_lbp(img_bgr):
    img_rgb   = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_res   = cv2.resize(img_rgb, IMAGE_SIZE)
    gray      = cv2.cvtColor(img_res, cv2.COLOR_RGB2GRAY)
    radius    = 3
    n_points  = 8 * radius
    lbp       = local_binary_pattern(gray, n_points, radius, method="uniform")
    hist, _   = np.histogram(lbp.ravel(), bins=np.arange(0, n_points + 3),
                              range=(0, n_points + 2))
    return hist.reshape(1, -1).astype(float)

def extract_hsv(img_bgr):
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_res = cv2.resize(img_rgb, IMAGE_SIZE)
    hsv     = cv2.cvtColor(img_res, cv2.COLOR_RGB2HSV)
    h = cv2.calcHist([hsv], [0], None, [32], [0, 180]).flatten()
    s = cv2.calcHist([hsv], [1], None, [32], [0, 256]).flatten()
    v = cv2.calcHist([hsv], [2], None, [32], [0, 256]).flatten()
    hist = np.concatenate([h, s, v]).astype(float)
    hist /= (hist.sum() + 1e-7)
    return hist.reshape(1, -1)

EXTRACTORS = {"HOG": extract_hog, "LBP": extract_lbp, "HSV": extract_hsv}

# ─────────────────────────────────────────────────────────────────────────────
# FLOPs HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def hog_flops():
    h, w = IMAGE_SIZE
    pixels = h * w
    grad_flops = pixels * 6
    bin_flops  = pixels * 9
    n_cells_h  = h // 16
    n_cells_w  = w // 16
    n_blocks   = (n_cells_h - 1) * (n_cells_w - 1)
    norm_flops = n_blocks * (4 * 9 * 2)
    return grad_flops + bin_flops + norm_flops

def lbp_flops():
    h, w = IMAGE_SIZE
    pixels   = h * w
    n_points = 24
    return pixels * (n_points * 3 + n_points + 1)

def hsv_flops():
    h, w = IMAGE_SIZE
    pixels = h * w
    return pixels * 3 + 32 * 3

def svm_flops(svm, n_features):
    n_sv = svm.n_support_.sum()
    scaler_ops = n_features * 2
    kernel_ops = n_sv * (3 * n_features + 2)
    return int(scaler_ops + kernel_ops)

def rf_flops(rf, n_features):
    n_trees = rf.n_estimators
    depth   = rf.max_depth if rf.max_depth else int(np.log2(n_features) * 2)
    nodes_per_tree = 2 ** depth
    ops_per_node   = 2
    return int(n_trees * nodes_per_tree * ops_per_node)

def lr_flops(lr, n_features):
    scaler_ops = n_features * 2
    dot_ops    = n_features * 2
    return int(scaler_ops + dot_ops)

# ─────────────────────────────────────────────────────────────────────────────
# ML PROFILER
# ─────────────────────────────────────────────────────────────────────────────

def get_clf_info(model):
    if hasattr(model, "named_steps"):
        steps = list(model.named_steps.keys())
        clf   = model.named_steps[steps[-1]]
        has_scaler = "scaler" in steps
    else:
        clf        = model
        has_scaler = False
    return clf, has_scaler

def profile_ml_model(entry):
    path    = entry["path"]
    feature = entry["feature"]
    extractor = EXTRACTORS[feature]

    model = joblib.load(path)
    clf, has_scaler = get_clf_info(model)
    clf_type = type(clf).__name__

    dummy = dummy_bgr()
    feat  = extractor(dummy)
    n_features = feat.shape[1]

    # ── params ──────────────────────────────────────────────────────────────
    if clf_type == "SVC":
        n_sv       = clf.n_support_.sum()
        sv_params  = int(n_sv * n_features)
        sc_params  = int(n_features * 2) if has_scaler else 0
        total_params = sv_params + sc_params
        extra = {
            "kernel": clf.kernel.upper(),
            "C": clf.C,
            "gamma": clf.gamma,
            "n_sv": int(n_sv),
            "n_sv_per_class": clf.n_support_.tolist(),
        }
    elif clf_type == "RandomForestClassifier":
        total_params = sum(
            t.tree_.node_count * 3 for t in clf.estimators_
        )
        extra = {
            "n_estimators": clf.n_estimators,
            "max_depth": clf.max_depth,
            "n_features_in": int(clf.n_features_in_),
        }
    elif clf_type == "LogisticRegression":
        total_params = int(clf.coef_.size + clf.intercept_.size)
        sc_params    = int(n_features * 2) if has_scaler else 0
        total_params += sc_params
        extra = {
            "C": clf.C,
            "solver": clf.solver,
            "max_iter": clf.max_iter,
        }
    else:
        total_params = 0
        extra = {}

    # ── FLOPs ────────────────────────────────────────────────────────────────
    feat_flops = {"HOG": hog_flops, "LBP": lbp_flops, "HSV": hsv_flops}[feature]()
    if clf_type == "SVC":
        clf_flops = svm_flops(clf, n_features)
    elif clf_type == "RandomForestClassifier":
        clf_flops = rf_flops(clf, n_features)
    elif clf_type == "LogisticRegression":
        clf_flops = lr_flops(clf, n_features)
    else:
        clf_flops = 0
    total_flops = feat_flops + clf_flops

    # ── model size ───────────────────────────────────────────────────────────
    size_mb = path.stat().st_size / (1024 ** 2)

    # ── latency ──────────────────────────────────────────────────────────────
    for _ in range(5):
        feat_w = extractor(dummy_bgr())
        model.predict(feat_w)

    times = []
    for _ in range(LATENCY_RUNS):
        t0 = time.perf_counter()
        f  = extractor(dummy_bgr())
        model.predict(f)
        times.append(time.perf_counter() - t0)

    return {
        "name": entry["name"],
        "variant": entry["variant"],
        "clf_type": clf_type,
        "feature_type": feature,
        "n_features": int(n_features),
        "total_params": int(total_params),
        "feat_flops": int(feat_flops),
        "clf_flops": int(clf_flops),
        "total_flops": int(total_flops),
        "size_mb": size_mb,
        "latency_mean_ms": float(np.mean(times) * 1000),
        "latency_std_ms": float(np.std(times) * 1000),
        **extra,
    }

# ─────────────────────────────────────────────────────────────────────────────
# DL PROFILER
# ─────────────────────────────────────────────────────────────────────────────

def get_dl_flops(model, tf):
    try:
        from tensorflow.python.framework.convert_to_constants import (
            convert_variables_to_constants_v2_as_graph,
        )

        @tf.function(input_signature=[tf.TensorSpec([1, *IMAGE_SIZE, 3], tf.float32)])
        def forward(x):
            return model(x, training=False)

        concrete = forward.get_concrete_function()
        _, graph_def = convert_variables_to_constants_v2_as_graph(concrete)
        with tf.Graph().as_default() as graph:
            tf.graph_util.import_graph_def(graph_def, name="")
            run_meta = tf.compat.v1.RunMetadata()
            opts = tf.compat.v1.profiler.ProfileOptionBuilder.float_operation()
            opts["output"] = "none"
            flops_obj = tf.compat.v1.profiler.profile(
                graph=graph, run_meta=run_meta, cmd="op", options=opts
            )
        return int(flops_obj.total_float_ops), "tf.compat.v1.profiler"
    except Exception as e:
        return None, f"unavailable ({e})"

def profile_dl_model(entry, tf):
    path  = entry["path"]
    model = tf.keras.models.load_model(path)

    total_params     = model.count_params()
    trainable_params = sum(int(np.prod(v.shape)) for v in model.trainable_variables)
    non_trainable    = total_params - trainable_params

    total_flops, flops_src = get_dl_flops(model, tf)

    size_mb = path.stat().st_size / (1024 ** 2)

    dummy_tf = tf.ones([1, *IMAGE_SIZE, 3], dtype=tf.float32)
    for _ in range(5):
        model(dummy_tf, training=False)

    times = []
    for _ in range(LATENCY_RUNS):
        t0 = time.perf_counter()
        model(dummy_tf, training=False)
        times.append(time.perf_counter() - t0)

    return {
        "name": entry["name"],
        "variant": entry["variant"],
        "total_params": int(total_params),
        "trainable_params": int(trainable_params),
        "non_trainable_params": int(non_trainable),
        "total_flops": total_flops,
        "flops_source": flops_src,
        "size_mb": size_mb,
        "latency_mean_ms": float(np.mean(times) * 1000),
        "latency_std_ms": float(np.std(times) * 1000),
    }

# ─────────────────────────────────────────────────────────────────────────────
# FORMATTING
# ─────────────────────────────────────────────────────────────────────────────

def fmt_flops(f):
    if f is None:
        return "N/A"
    if f >= 1e9:
        return f"{f/1e9:.4f} GFLOPs"
    if f >= 1e6:
        return f"{f/1e6:.4f} MFLOPs"
    return f"{f:,} FLOPs"

def fmt_params(p):
    if p >= 1e9:
        return f"{p/1e9:.4f} B  ({p:,})"
    if p >= 1e6:
        return f"{p/1e6:.4f} M  ({p:,})"
    if p >= 1e3:
        return f"{p/1e3:.2f} K  ({p:,})"
    return f"{p:,}"

def fmt_flops_short(f):
    if f is None:
        return "N/A"
    if f >= 1e9:
        return f"{f/1e9:.4f}G"
    if f >= 1e6:
        return f"{f/1e6:.3f}M"
    return f"{f:,}"

def fmt_params_short(p):
    if p >= 1e6:
        return f"{p/1e6:.4f}M"
    if p >= 1e3:
        return f"{p/1e3:.2f}K"
    return str(p)

# ─────────────────────────────────────────────────────────────────────────────
# INDIVIDUAL REPORT BUILDER
# ─────────────────────────────────────────────────────────────────────────────

def ml_model_section(r):
    lines = []
    tag = f"{r['name']} ({r['variant']})"
    lines.append("=" * 60)
    lines.append(f"MODEL : {tag}")
    lines.append("=" * 60)
    lines.append(f"Classifier type         : {r['clf_type']}")
    lines.append(f"Feature extraction      : {r['feature_type']}")
    lines.append(f"Feature dimensions      : {r['n_features']:,}")
    lines.append("")

    lines.append("[ Parameters ]")
    if r["clf_type"] == "SVC":
        lines.append(f"  Kernel                : {r.get('kernel','')}")
        lines.append(f"  C                     : {r.get('C','')}")
        lines.append(f"  Gamma                 : {r.get('gamma','')}")
        lines.append(f"  Support vectors       : {r.get('n_sv',0):,}  per class: {r.get('n_sv_per_class','')}")
    elif r["clf_type"] == "RandomForestClassifier":
        lines.append(f"  n_estimators          : {r.get('n_estimators','')}")
        lines.append(f"  max_depth             : {r.get('max_depth','')}")
        lines.append(f"  n_features_in         : {r.get('n_features_in',''):,}")
    elif r["clf_type"] == "LogisticRegression":
        lines.append(f"  C (regularization)    : {r.get('C','')}")
        lines.append(f"  Solver                : {r.get('solver','')}")
        lines.append(f"  Max iterations        : {r.get('max_iter','')}")
    lines.append(f"  Total params          : {fmt_params(r['total_params'])}")
    lines.append("")

    lines.append("[ FLOPs (inference, 1 image) ]")
    lines.append(f"  Feature extraction    : {fmt_flops(r['feat_flops'])}")
    lines.append(f"  Classifier            : {fmt_flops(r['clf_flops'])}")
    lines.append(f"  Total                 : {fmt_flops(r['total_flops'])}")
    if r["clf_type"] == "RandomForestClassifier":
        lines.append(f"  Note: RF FLOPs = n_trees * nodes_per_tree * comparisons (approx)")
    lines.append("")

    lines.append("[ Model Size ]")
    lines.append(f"  File size             : {r['size_mb']:.4f} MB")
    lines.append("")

    lines.append(f"[ Latency (CPU, full pipeline incl. feature extraction) ]")
    lines.append(f"  Mean                  : {r['latency_mean_ms']:.3f} ms")
    lines.append(f"  Std                   : {r['latency_std_ms']:.3f} ms")
    lines.append(f"  Throughput            : {1000/r['latency_mean_ms']:.1f} img/sec")
    lines.append("")
    return lines

def dl_model_section(r):
    lines = []
    tag = f"{r['name']} ({r['variant']})"
    lines.append("=" * 60)
    lines.append(f"MODEL : {tag}")
    lines.append("=" * 60)
    lines.append(f"Input size              : {IMAGE_SIZE[0]} x {IMAGE_SIZE[1]} x 3")
    lines.append("")

    lines.append("[ Parameters ]")
    lines.append(f"  Total params          : {fmt_params(r['total_params'])}")
    lines.append(f"  Trainable             : {fmt_params(r['trainable_params'])}")
    lines.append(f"  Non-trainable         : {fmt_params(r['non_trainable_params'])}")
    lines.append("")

    lines.append("[ FLOPs (inference, 1 image @ 300x300) ]")
    lines.append(f"  Total FLOPs           : {fmt_flops(r['total_flops'])}")
    lines.append(f"  Source                : {r['flops_source']}")
    lines.append("")

    lines.append("[ Model Size ]")
    lines.append(f"  File size             : {r['size_mb']:.4f} MB")
    lines.append("")

    lines.append("[ Latency (CPU forward-pass only) ]")
    lines.append(f"  Mean                  : {r['latency_mean_ms']:.3f} ms")
    lines.append(f"  Std                   : {r['latency_std_ms']:.3f} ms")
    lines.append(f"  Throughput            : {1000/r['latency_mean_ms']:.1f} img/sec")
    lines.append("")
    return lines

# ─────────────────────────────────────────────────────────────────────────────
# COMPARISON TABLE
# ─────────────────────────────────────────────────────────────────────────────

def comparison_table(ml_results, dl_results):
    lines = []
    lines.append("=" * 90)
    lines.append("COMPARISON SUMMARY — ALL MODELS")
    lines.append("=" * 90)
    header = f"{'Model':<42} {'Params':>10} {'FLOPs':>12} {'Size(MB)':>9} {'Lat(ms)':>9} {'Img/s':>7}"
    lines.append(header)
    lines.append("-" * 90)

    lines.append("--- MACHINE LEARNING ---")
    for r in ml_results:
        label = f"{r['name']} [{r['variant']}]"
        row = (
            f"{label:<42}"
            f" {fmt_params_short(r['total_params']):>10}"
            f" {fmt_flops_short(r['total_flops']):>12}"
            f" {r['size_mb']:>9.3f}"
            f" {r['latency_mean_ms']:>9.2f}"
            f" {1000/r['latency_mean_ms']:>7.1f}"
        )
        lines.append(row)

    lines.append("")
    lines.append("--- DEEP LEARNING ---")
    for r in dl_results:
        label = f"{r['name']} [{r['variant']}]"
        row = (
            f"{label:<42}"
            f" {fmt_params_short(r['total_params']):>10}"
            f" {fmt_flops_short(r['total_flops'] or 0):>12}"
            f" {r['size_mb']:>9.3f}"
            f" {r['latency_mean_ms']:>9.2f}"
            f" {1000/r['latency_mean_ms']:>7.1f}"
        )
        lines.append(row)

    lines.append("")
    lines.append("Notes:")
    lines.append("  ML latency  : full pipeline (feature extraction + classifier)")
    lines.append("  DL latency  : forward pass only (no disk I/O)")
    lines.append("  SVM FLOPs   : approximate (RBF kernel MACs per support vector)")
    lines.append("  RF FLOPs    : approximate (decision nodes traversed per tree)")
    lines.append("  LR FLOPs    : dot product + sigmoid (exact)")
    lines.append("  DL FLOPs    : counted via tf.compat.v1.profiler frozen graph")
    return lines

# ─────────────────────────────────────────────────────────────────────────────
# WRITE INDIVIDUAL TEXT REPORTS
# ─────────────────────────────────────────────────────────────────────────────

def write_individual_report(r, is_dl=False, tf_version="N/A"):
    safe_name = r["name"].replace(" ", "_").replace("+", "").replace("/", "_")
    fname = OUTPUT_DIR / f"profile_{safe_name}_{r['variant'].lower()}.txt"

    lines = []
    lines.append(f"MODEL PROFILE: {r['name']} ({r['variant']})")
    lines.append("=" * 60)
    lines.append(f"Generated   : {datetime.datetime.now():%Y-%m-%d %H:%M:%S}")
    lines.append(f"Platform    : {platform.system()} {platform.release()}")
    lines.append(f"Python      : {platform.python_version()}")
    if is_dl:
        lines.append(f"TensorFlow  : {tf_version}")
    lines.append(f"Latency runs: {LATENCY_RUNS}")
    lines.append("")

    if is_dl:
        lines += dl_model_section(r)
    else:
        lines += ml_model_section(r)

    text = "\n".join(lines)
    fname.write_text(text, encoding="utf-8")
    print(f"    Saved → {fname.name}")

# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import tensorflow as tf
    tf_version = tf.__version__

    # suppress tf profiler stdout
    import io, contextlib

    def silent_profile_dl(entry):
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            r = profile_dl_model(entry, tf)
        return r

    print("\n===== MACHINE LEARNING MODELS =====")
    ml_results = []
    for entry in ML_MODELS:
        tag = f"{entry['name']} [{entry['variant']}]"
        print(f"  Profiling {tag} ...")
        r = profile_ml_model(entry)
        ml_results.append(r)
        write_individual_report(r, is_dl=False)

    print("\n===== DEEP LEARNING MODELS =====")
    dl_results = []
    for entry in DL_MODELS:
        tag = f"{entry['name']} [{entry['variant']}]"
        print(f"  Profiling {tag} ...")
        r = silent_profile_dl(entry)
        dl_results.append(r)
        write_individual_report(r, is_dl=True, tf_version=tf_version)

    # ── master report ───────────────────────────────────────────────────────
    print("\n===== WRITING MASTER REPORT =====")
    master_lines = []
    master_lines.append("MASTER MODEL PROFILING REPORT — ALL MODELS")
    master_lines.append("=" * 60)
    master_lines.append(f"Generated   : {datetime.datetime.now():%Y-%m-%d %H:%M:%S}")
    master_lines.append(f"Platform    : {platform.system()} {platform.release()}")
    master_lines.append(f"Python      : {platform.python_version()}")
    master_lines.append(f"TensorFlow  : {tf_version}")
    master_lines.append(f"Latency runs: {LATENCY_RUNS} per model")
    master_lines.append(f"Input size  : {IMAGE_SIZE[0]}x{IMAGE_SIZE[1]} px (all models)")
    master_lines.append("")

    master_lines.append("=" * 60)
    master_lines.append("MACHINE LEARNING MODELS")
    master_lines.append("=" * 60)
    for r in ml_results:
        master_lines += ml_model_section(r)

    master_lines.append("=" * 60)
    master_lines.append("DEEP LEARNING MODELS")
    master_lines.append("=" * 60)
    for r in dl_results:
        master_lines += dl_model_section(r)

    master_lines += comparison_table(ml_results, dl_results)

    master_path = OUTPUT_DIR / "master_profiling_report.txt"
    master_text = "\n".join(master_lines)
    master_path.write_text(master_text, encoding="utf-8")

    # print summary table
    print()
    print("\n".join(comparison_table(ml_results, dl_results)))
    print(f"\nMaster report → {master_path}")
    print(f"Individual reports → {OUTPUT_DIR}/")
