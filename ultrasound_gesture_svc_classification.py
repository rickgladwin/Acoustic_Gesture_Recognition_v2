#Author - Keshav Bimbraw

import os
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import joblib

import config

# TODO: use skmetal for MPS optimization
# https://www.reddit.com/r/askdatascience/comments/1ua4oiq/project_skmetal_dropin_gpu_acceleration_for/


# ----------------------------
# Helpers
# ----------------------------
def ensure_dir(path_or_file: str):
    if not path_or_file:
        return
    d = path_or_file if os.path.isdir(path_or_file) else os.path.dirname(path_or_file)
    if d:
        os.makedirs(d, exist_ok=True)

def dump_json(obj, path):
    if not path:
        return
    ensure_dir(path)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)

def save_confusion_matrix_png(y_true, y_pred, path):
    if not path:
        return
    ensure_dir(path)
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation="nearest")
    ax.set_title("Confusion Matrix")
    fig.colorbar(im, ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    fig.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close(fig)

# ----------------------------
# Data loading
# ----------------------------
def load_subject_arrays(root, mode, subject):
    """
    Loads X_m_train.npy, X_m_test.npy, y_m_train.npy, y_m_test.npy
    from <root>/<mode>/<subject> and flattens X for SVM.
    """
    d = os.path.join(root, mode, subject)
    x_train = np.load(os.path.join(d, "X_m_train.npy"))
    x_test  = np.load(os.path.join(d, "X_m_test.npy"))
    y_train = np.load(os.path.join(d, "y_m_train.npy"))
    y_test  = np.load(os.path.join(d, "y_m_test.npy"))

    y_train = y_train.astype(np.int64).ravel()
    y_test  = y_test.astype(np.int64).ravel()

    # Ensure shape: (N,H,W[,C]) -> (N, H*W*C)
    if x_train.ndim == 3: x_train = x_train[..., np.newaxis]
    if x_test.ndim  == 3: x_test  = x_test[..., np.newaxis]
    x_train = x_train.reshape(x_train.shape[0], -1).astype("float32")
    x_test  = x_test.reshape(x_test.shape[0], -1).astype("float32")

    num_classes = int(max(y_train.max(), y_test.max()) + 1)
    return (x_train, y_train), (x_test, y_test), num_classes

# ----------------------------
# Main
# ----------------------------
def main():
    ap = argparse.ArgumentParser(description="SVM gesture classifier (single subject) with metrics + artifacts.")
    # Paths / data
    ap.add_argument("--root", type=str,
        # default=r"C:\Users\bimbr\Documents\Mirror_Paper\Data_Upload",
        default=config.default_dataset_path,
        help="Root folder containing 'mirror' and 'perp'.")
    ap.add_argument("--mode", type=str, choices=["mirror", "perp"], default="mirror",
        help="Dataset mode.")
    ap.add_argument("--subject", type=str, default="Subject_1",
        help="Subject folder name.")

    # Kernels allowed by the paper
    ap.add_argument("--kernel", type=str, choices=["linear", "rbf"], default="rbf",
        help="SVM kernel (paper uses only 'linear' and 'rbf').")
    ap.add_argument("--C", type=float, default=10.0, help="Regularization C.")
    ap.add_argument("--gamma", type=str, default="scale",
        help="Gamma for rbf ('scale','auto', or float). Ignored for linear.")
    ap.add_argument("--max-iter", type=int, default=2000, help="Max iterations (-1 for no limit).")
    ap.add_argument("--class-weight", type=str, default="",
        help="'' for None, or 'balanced' to rebalance by class frequency.")
    ap.add_argument("--no-scale", action="store_true", help="Disable StandardScaler (not recommended).")

    # Save / load
    ap.add_argument("--save-model", type=str, default="", help="Path to save .joblib model.")
    ap.add_argument("--out", type=str, default="", help="Path to save metrics JSON.")
    ap.add_argument("--cm", type=str, default="", help="Path to save confusion matrix PNG.")
    ap.add_argument("--load-model", type=str, default="", help="Load a .joblib model and skip training.")

    args = ap.parse_args()

    print(f"-- Loading data from: {args.root}...")

    # Data
    (x_train, y_train), (x_test, y_test), num_classes = load_subject_arrays(
        args.root, args.mode, args.subject
    )
    print(f"-- loaded {len(x_train)} training samples and {len(x_test)} test samples for {num_classes} classes")
    
    # input_shape = (args.image_size, args.image_size, 1)
    # print(f"-- input shape: {input_shape}")

    # Build / load
    if args.load_model and os.path.isfile(args.load_model):
        print(f"Loading SVM model from: {args.load_model}")
        clf = joblib.load(args.load_model)
        trained = True
    else:
        # gamma handling (rbf only): allow numeric strings (e.g., "0.01")
        # gamma_param = None  # ignored for linear -- throws error
        # sklearn.utils._param_validation.InvalidParameterError: The 'gamma' parameter of SVC must be a str among {'auto', 'scale'} or a float in the range [0.0, inf). Got None instead.
        gamma_param = args.gamma
        if args.kernel == "rbf":
            gamma_param = args.gamma
            try:
                gamma_param = float(args.gamma)
            except ValueError:
                # keep 'scale' or 'auto'
                pass

        svc = SVC(
            kernel=args.kernel,
            C=args.C,
            gamma=gamma_param,                 # None for linear; 'scale'/'auto'/float for rbf
            class_weight=(None if args.class_weight == "" else args.class_weight),
            max_iter=args.max_iter,
            verbose=True,
        )
        steps = []
        if not args.no_scale:
            steps.append(("scaler", StandardScaler(with_mean=True, with_std=True)))
        steps.append(("svc", svc))
        clf = Pipeline(steps)

        print(f"Started SVM training (kernel={args.kernel})…")
        clf.fit(x_train, y_train)
        print("Training finished.")
        trained = False

    # Predict & Metrics
    print(f"\n[{args.mode}/{args.subject}] Predicting and calculating metrics...")
    y_pred = clf.predict(x_test)
    acc = accuracy_score(y_test, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(
        y_test, y_pred, average="macro", zero_division=0
    )

    print(f"\n[{args.mode}/{args.subject}]   Test Accuracy: {acc:.4f}")
    print(f"[{args.mode}/{args.subject}] Random Accuracy: {1/num_classes:.4f}")
    print(f"[{args.mode}/{args.subject}] Macro Precision: {prec:.4f}  Macro Recall: {rec:.4f}  Macro F1: {f1:.4f}")

    # Artifacts
    if args.cm:
        save_confusion_matrix_png(y_test, y_pred, args.cm)
        print(f"Saved confusion matrix to: {args.cm}")

    if args.save_model and not trained:
        ensure_dir(args.save_model)
        joblib.dump(clf, args.save_model)
        print(f"Saved model to: {args.save_model}")

    if args.out:
        result = {
            "model": f"svc_{args.kernel}",
            "mode": args.mode,
            "subject": args.subject,
            "n_classes": int(num_classes),
            "metrics": {
                "accuracy": float(acc),
                "precision_macro": float(prec),
                "recall_macro": float(rec),
                "f1_macro": float(f1),
            },
            "confusion_matrix_path": args.cm if args.cm else "",
            "params": {
                "kernel": args.kernel,
                "C": args.C,
                "gamma": args.gamma if args.kernel == "rbf" else "ignored",
                "max_iter": args.max_iter,
                "class_weight": (None if args.class_weight == "" else args.class_weight),
                "standardize": (not args.no_scale),
            },
        }
        dump_json(result, args.out)
        print(f"Saved metrics JSON to: {args.out}")

if __name__ == "__main__":
    main()
