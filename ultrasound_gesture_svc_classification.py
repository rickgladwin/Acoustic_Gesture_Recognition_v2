#Author - Keshav Bimbraw

import os
import json
import argparse
from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt
from fontTools.ttLib.tables import TupleVariation
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import joblib
import skmetal


import config
from ultrasound_gesture_cnn_classification import is_apple_silicon, get_mac_system_info, train_test_duration_display
from ultrasound_gesture_vit_classification import create_caption_from_details
from utilities import ensure_dir, dump_json
from visualizations import set_global_matplotlib_font, save_confusion_matrix_png


# TODO: use skmetal for MPS optimization
# https://www.reddit.com/r/askdatascience/comments/1ua4oiq/project_skmetal_dropin_gpu_acceleration_for/


# ----------------------------
# Helpers
# ----------------------------
# def ensure_dir(path_or_file: str):
#     if not path_or_file:
#         return
#     d = path_or_file if os.path.isdir(path_or_file) else os.path.dirname(path_or_file)
#     if d:
#         os.makedirs(d, exist_ok=True)
# 
# def dump_json(obj, path):
#     if not path:
#         return
#     ensure_dir(path)
#     with open(path, "w", encoding="utf-8") as f:
#         json.dump(obj, f, indent=2)

# def save_confusion_matrix_png(y_true, y_pred, path, cm_title: str|None=None, details: dict|None=None):
#     if not path:
#         return
#     ensure_dir(path)
#     cm = confusion_matrix(y_true, y_pred)
#     
#     set_global_matplotlib_font()
# 
#     if details is not None:
#         caption = create_caption_from_details(details)
#     else:
#         caption = ""
#     caption_font_size = 10
# 
#     fig, ax = plt.subplots()
#     im = ax.imshow(cm, interpolation="nearest")
#     if cm_title is not None:
#         ax.set_title(cm_title)
#     else:
#         ax.set_title("Confusion Matrix")
#     fig.colorbar(im, ax=ax)
#     ax.set_xlabel(f"Predicted\n\n{caption}", fontdict={'size': caption_font_size})
#     ax.set_ylabel("True")
#     # set the class labels on the x and y axes explicitly
#     ax.set_xticks(np.arange(len(np.unique(y_pred))))
#     ax.set_yticks(np.arange(len(np.unique(y_true))))
#     fig.tight_layout()
#     plt.savefig(path, dpi=150)
#     plt.close(fig)


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
    
    # TODO: use Hilbert curve in place of flatten to vectorize image data?
    #  NOTE: this may not make a difference if the SVC algorithm calculates based
    #  on every value in the datapoint in relation to every other (that is, the
    #  pixel values are treated as independent features). There may be a way to
    #  make use of the spatial relationships between pixels, but SVC may not be the
    #  right architecture/algorithm to make use of this information.

    # Ensure shape: (N,H,W[,C]) -> (N, H*W*C)
    if x_train.ndim == 3: x_train = x_train[..., np.newaxis]
    if x_test.ndim  == 3: x_test  = x_test[..., np.newaxis]
    x_train = x_train.reshape(x_train.shape[0], -1).astype("float32")
    x_test  = x_test.reshape(x_test.shape[0], -1).astype("float32")

    num_classes = int(max(y_train.max(), y_test.max()) + 1)
    return (x_train, y_train), (x_test, y_test), num_classes


@skmetal.accelerate
def pipeline_from_steps(steps_list: list[tuple]):
    # returns a scikit-learn Pipeline object from a list of tuples,
    # where each tuple has ("step_name", <sklearn model or algorithm>)
    # wrapper activates skmetal acceleration for sklearn
    # example speeds (bottleneck may be due to training set size, or
    # the package may not be working, or may not help with the kind of
    # operations carried out by SVC), 1000 training samples:
    # -- with skmetal:
    # training: 0:01:08.750565
    # testing:  0:00:57.744026
    # -- without skmetal:
    # training: 0:01:00.973862
    # testing:  0:00:59.614771
    # example speeds, 2000 training samples:
    # -- with skmetal:
    # training: 0:03:25.432516
    # testing:  0:02:04.517311
    # -- without skmetal:
    # training: 0:04:13.336055
    # testing:  0:01:50.181202

    return Pipeline(steps_list)


# ----------------------------
# Main
# ----------------------------
def main():
    file_datetime = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    default_max_iterations: int = 2000 # 2000
    default_mode: str = "perp" # ["perp", "mirror"]
    # NOTE: rbf can consume more RAM than linear. In case of exit code 137 (Out Of Memory error),
    # reduce the number of training samples and/or switch to a linear kernel.
    default_kernel: str = "linear" # one of ["linear", "rbf"]
    default_training_set_size: int = 1000 # -1 for unrestricted, 4800 in the actual set
    default_subject_id: str = "4"
    default_metrics_filepath: str = f"results/metrics/svc/metrics_svc_subject_{default_subject_id}_{default_kernel}_kernel_{file_datetime}.json"
    default_confusion_matrix_filepath: str = f"results/figs/svc/cm_svc_subject_{default_subject_id}_{default_kernel}_kernel_{file_datetime}.png"

    # empty string for save or load model will skip save or load
    # default_save_model: str = f"results/models/svc/svc_{default_mode}_subject_{default_subject_id}_{default_training_set_size}_training_samples_{file_datetime}.keras"
    default_save_model: str = ""
    # default_load_model: str = f"results/models/svc/svc_perp_subject_2_1000_training_samples_20260717_191619.keras"
    default_load_model: str = ""

    ap = argparse.ArgumentParser(description="SVM gesture classifier (single subject) with metrics + artifacts.")
    # Paths / data
    ap.add_argument("--root", type=str,
        # default=r"C:\Users\bimbr\Documents\Mirror_Paper\Data_Upload",
        default=config.default_dataset_path,
        help="Root folder containing 'mirror' and 'perp'.")
    ap.add_argument("--mode", type=str, choices=["mirror", "perp"], default=default_mode,
        help="Dataset mode.")
    ap.add_argument("--subject", type=str, default=f"Subject_{default_subject_id}",
        help="Subject folder name.")

    # Kernels allowed by the paper
    ap.add_argument("--kernel", type=str, choices=["linear", "rbf"], default=default_kernel,
        help="SVM kernel (paper uses only 'linear' and 'rbf').")
    ap.add_argument("--C", type=float, default=10.0, help="Regularization C.")
    ap.add_argument("--gamma", type=str, default="scale",
        help="Gamma for rbf ('scale','auto', or float). Ignored for linear.")
    # ap.add_argument("--max-iter", type=int, default=2000, help="Max iterations (-1 for no limit).")
    ap.add_argument("--max-iter", type=int, default=default_max_iterations, help="Max iterations (-1 for no limit).")
    ap.add_argument("--class-weight", type=str, default="",
        help="'' for None, or 'balanced' to rebalance by class frequency.")
    ap.add_argument("--no-scale", action="store_true", help="Disable StandardScaler (not recommended).")

    # Save / load
    ap.add_argument("--save-model", type=str, default="", help="Path to save .joblib model.")
    ap.add_argument("--out", type=str, default=default_metrics_filepath, help="Path to save metrics JSON.")
    ap.add_argument("--cm", type=str, default=default_confusion_matrix_filepath, help="Path to save confusion matrix PNG.")
    ap.add_argument("--load-model", type=str, default="", help="Load a .joblib model and skip training.")

    args = ap.parse_args()

    print(f"-- Loading data from: {args.root}...")

    # Data
    (x_train, y_train), (x_test, y_test), num_classes = load_subject_arrays(
        args.root, args.mode, args.subject
    )
    # x_train data are flattened image vectors of size length x width
    # i.e. (length)^2, so sqrt(len(<flattened image vector>)) == length
    image_dimension = int(np.sqrt(x_train.shape[1])) 
    print(f"x_train.shape: {x_train.shape}")
    print(f"-- loaded {len(x_train)} training samples and {len(x_test)} test samples for {num_classes} classes")
    
    # Reduce dataset size (uses less RAM)
    if default_training_set_size != -1:
        # shuffle training data
        np.random.seed(42)
        indices = np.random.permutation(len(x_train))
        x_train = x_train[indices]
        y_train = y_train[indices]
        # truncate shuffled training data
        x_train = x_train[:default_training_set_size]
        y_train = y_train[:default_training_set_size]

        print(f"-- truncated training set to first {default_training_set_size} samples")

    # input_shape = (args.image_size, args.image_size, 1)
    # print(f"-- input shape: {input_shape}")

    # training_details is used to label result plots
    training_details: dict = {
        "mode": args.mode,
        "subject": args.subject,
        "kernel": args.kernel,
        "image_dimensions": f"{image_dimension}x{image_dimension}",
        "training_set_size": default_training_set_size,
    }

    # Build / load
    if args.load_model and os.path.isfile(args.load_model):
        print(f"Loading SVM model from: {args.load_model}")
        clf = joblib.load(args.load_model)
        trained = True
    else:
        # gamma handling (rbf only): allow numeric strings (e.g., "0.01")
        # gamma_param = None # ignored for linear -- throws error
        # sklearn.utils._param_validation.InvalidParameterError: The 'gamma' parameter of SVC must be a str among {'auto', 'scale'} or a float in the range [0.0, inf). Got None instead.
        gamma_param = args.gamma
        if args.kernel == "rbf":
            gamma_param = args.gamma
            try:
                gamma_param = float(args.gamma)
            except ValueError:
                # keep 'scale' or 'auto'
                pass

        # configure a Support Vector Classifier instance
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
        # clf = Pipeline(steps)
        clf = pipeline_from_steps(steps)

        print(f"Started SVM training (kernel={args.kernel})…")
        training_start = datetime.now()
        clf.fit(x_train, y_train)
        training_end = datetime.now()
        training_duration = train_test_duration_display(training_end - training_start)
        print("Training finished.")
        print(f"training duration: {training_duration}")
        training_details['training_duration'] = str(training_duration)
        trained = False

    # Predict & Metrics
    print(f"\n[{args.mode}/{args.subject}] Predicting and calculating metrics on {len(x_test)} test samples...")
    test_start = datetime.now()
    y_pred = clf.predict(x_test)
    test_end = datetime.now()
    test_duration = train_test_duration_display(test_end - test_start)
    training_details['testing_set_size'] = len(x_test)
    training_details['testing_duration'] = str(test_duration)
    print("Testing finished.")
    print(f"testing duration: {test_duration}")
    acc = accuracy_score(y_test, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(
        y_test, y_pred, average="macro", zero_division=0
    )

    print(f"\n[{args.mode}/{args.subject}]   Test Accuracy: {acc:.4f}")
    print(f"[{args.mode}/{args.subject}] Random Accuracy: {1/num_classes:.4f}")
    print(f"[{args.mode}/{args.subject}] Macro Precision: {prec:.4f}  Macro Recall: {rec:.4f}  Macro F1: {f1:.4f}")

    # Artifacts
    if args.cm:
        if is_apple_silicon():
            # get Mac system info
            mac_system_info: dict[str, str] = get_mac_system_info()
    
            training_details['processor_type'] = mac_system_info["processor_type"]
            training_details['cpu_cores'] = mac_system_info["cpu_cores"]
            training_details['gpu_cores'] = mac_system_info["gpu_cores"]

        confusion_matrix_title: str = "SVC Confusion Matrix"
        training_details['test_accuracy'] = f"{acc:.4f}"
        save_confusion_matrix_png(y_test, y_pred, path=args.cm, cm_title=confusion_matrix_title, details=training_details)
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
