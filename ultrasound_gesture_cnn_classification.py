#Author - Keshav Bimbraw

import os
import json
import argparse
from datetime import datetime, timedelta

import numpy as np
import tensorflow as tf
from keras.src.callbacks import History
from sklearn.model_selection import train_test_split
from tensorflow import keras
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
import subprocess
import platform

from tf_explain.callbacks.grad_cam import GradCAMCallback

import config
from utilities import is_apple_silicon, get_mac_system_info, set_seed, ensure_dir, dump_json
from visualizations import train_test_duration_display, save_confusion_matrix_png, plot_history_separately


# ----------------------------
# Helpers
# ----------------------------

# ----------------------------
# tqdm progress (epoch + batch bars)
# ----------------------------
class TqdmProgress(keras.callbacks.Callback):
    def __init__(self, enable=True):
        super().__init__()
        self.enable = enable
        self.epoch_bar = None
        self.batch_bar = None

    def on_train_begin(self, logs=None):
        if not self.enable: return
        total_epochs = self.params.get("epochs", None)
        self.epoch_bar = tqdm(total=total_epochs, desc="Epochs", position=0, leave=True)

    def on_epoch_begin(self, epoch, logs=None):
        if not self.enable: return
        total_steps = self.params.get("steps", None)
        self.batch_bar = tqdm(total=total_steps, desc=f"Epoch {epoch+1}/{self.params.get('epochs','?')}",
                              position=1, leave=False)

    def on_train_batch_end(self, batch, logs=None):
        if not self.enable or self.batch_bar is None: return
        self.batch_bar.update(1)
        if logs:
            self.batch_bar.set_postfix({
                "loss": f"{logs.get('loss', 0):.4f}",
                "acc": f"{logs.get('accuracy', 0):.4f}"
            })

    def on_epoch_end(self, epoch, logs=None):
        if not self.enable: return
        if self.batch_bar is not None:
            self.batch_bar.close()
            self.batch_bar = None
        if logs:
            tqdm.write(
                f"Epoch {epoch+1} done | "
                f"loss={logs.get('loss', 0):.4f} "
                f"acc={logs.get('accuracy', 0):.4f} "
                f"val_loss={logs.get('val_loss', 0):.4f} "
                f"val_acc={logs.get('val_accuracy', 0):.4f}"
            )
        if self.epoch_bar is not None:
            self.epoch_bar.update(1)

    def on_train_end(self, logs=None):
        if self.batch_bar is not None:
            self.batch_bar.close()
        if self.epoch_bar is not None:
            self.epoch_bar.close()

# ----------------------------
# Data loading
# ----------------------------
def load_subject_arrays(root: str, mode: str, subject: str, image_size: int):
    """
    Loads and preprocesses ultrasound data for training and testing. This function handles loading of data
    from disk, ensures data consistency, resizes images, normalizes the data, and returns the processed
    datasets along with the number of classes. It is specifically used for preparing ultrasound images for
    machine learning tasks, such as classification or segmentation.

    Parameters:
    root: str
        The root directory containing the dataset.
    mode: str
        The mode of data capture, one of ["mirror", "perp"], depending on the orientation of the ultrasound probe.
    subject: str
        Identifier of the subject whose data is to be loaded, one of ["Subject_1", "Subject_2", "Subject_3", "Subject_4", "Subject_5", "Subject_6"].
    image_size: int
        Desired width and height for image resizing. The function assumes a square dimension.

    Returns:
    tuple
        A tuple consisting of:
        - (x_train, y_train): Processed training dataset (features, labels).
        - (x_test, y_test): Processed testing dataset (features, labels).
        - num_classes: The total number of unique gesture classes in the dataset (12, for the 2025 mirror/perp dataset).
    """
    # TODO: perform preprocessing that:
    # - averages out the "static" noise in the ultrasound video from frame to frame (*? is there a way to
    #   mathematically determine what the frequency and shape of this noise is, in order to:
    #   -- filter it out/average over it
    #   -- augment the dataset with similar noise that should be ignored
    #   )
    # - combines data from different subjects

    # example folder path to data:
    # /home/username/ultrasound_gesture_data/mirror/Subject_1/
    d = os.path.join(root, mode, subject)
    x_train = np.load(os.path.join(d, "X_m_train.npy"))
    x_test  = np.load(os.path.join(d, "X_m_test.npy"))
    y_train: np.ndarray = np.load(os.path.join(d, "y_m_train.npy"))
    y_test: np.ndarray  = np.load(os.path.join(d, "y_m_test.npy"))

    y_train = y_train.astype(np.int64).ravel()
    y_test  = y_test.astype(np.int64).ravel()

    # Ensure channel dim (N, H, W, 1)
    if x_train.ndim == 3: x_train = x_train[..., np.newaxis]
    if x_test.ndim  == 3: x_test  = x_test[..., np.newaxis]

    # Resize to image_size (keeps your ViT/CNN parity if you want 320)
    # TODO: look at converting to tensors that can be used with MPS optimization
    x_train = tf.image.resize(tf.convert_to_tensor(x_train), (image_size, image_size)).numpy()
    x_test  = tf.image.resize(tf.convert_to_tensor(x_test ), (image_size, image_size)).numpy()

    # Normalize to [0,1]
    if x_train.dtype != np.float32:
        x_train = x_train.astype("float32"); x_test = x_test.astype("float32")
    maxv = max(float(x_train.max()), 1.0)
    x_train /= maxv; x_test /= maxv

    num_classes = int(max(y_train.max(), y_test.max()) + 1)
    return (x_train, y_train), (x_test, y_test), num_classes


# ----------------------------
# CNN model
# ----------------------------
def build_cnn(input_shape, num_classes,
              filters=(16, 16, 16, 16, 16), kernel_size=3, pool_size=2,
              dense_units=64, dropout=0.5, lr=1e-3):
    """
    A lightweight 2D CNN stack similar to your original:
      [Conv-BN-ReLU + MaxPool] x L  ->  Flatten -> Dense -> Dropout -> Softmax
    """
    chan_dim = -1
    inputs = keras.layers.Input(shape=input_shape)
    x = inputs
    for f in filters:
        x = keras.layers.Conv2D(f, (kernel_size, kernel_size), padding="same", activation="relu")(x)
        x = keras.layers.BatchNormalization(axis=chan_dim)(x)
        x = keras.layers.MaxPooling2D(pool_size=(pool_size, pool_size))(x)
        
    # TODO: look at depthwise maxpooling somewhere in this 2D CNN, in the Conv2D layers, far enough along that filters have
    #  been established, so that collections of similar filters can be pooled in a way that sums over "noisy" variations on
    #  the same filter (see Géron - depthwise maxpooling)

    # TODO: look at transfer learning strategies that use the weights from a trained 2D CNN (the layers before this
    #  point in the model) and use that as the starting point when training on another subject's data, to see if
    #  training will be faster/more accurate
    
    # TODO: look at different ways of combining/augmenting/using transfer learning in ways that separate the two parts
    #  of the CNN model – the filters and the Dense/ANN layers. Include different ways of combining the data from
    #  different subjects – train the different segments of the model(s) on the original, moving-average, and
    #  combined data. 
    
    x = keras.layers.Flatten()(x)
    x = keras.layers.Dense(dense_units, activation="relu")(x)
    x = keras.layers.BatchNormalization(axis=chan_dim)(x)
    # TODO: try additional dropout layers (have a look at where dropout is used in other 2D CNN and similar visual classification architectures)
    x = keras.layers.Dropout(dropout)(x)
    
    # TODO: add additional learning layers (the Dense/ANN layers) to the model (with dropout and other methods to avoid
    #  overtraining)
    
    outputs = keras.layers.Dense(num_classes)(x)  # logits
    model = keras.Model(inputs, outputs)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=lr),
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=[keras.metrics.SparseCategoricalAccuracy(name="accuracy")],
    )
    return model


# ----------------------------
# Main
# ----------------------------
def main():
    # TODO: consider creating a method to save the history objects themselves as data for specific runs.
    #  That would make the eventual dataset more useful and repeatable.
    
    # TODO: consider: more training isn't always better. What we want is training that further incorporates
    #  the signal in the data (and all the relevant information) in the model's representation, and further ignores the noise.
    #  How well any given model can do this in the ideal case depends on the architecture. So what we're
    #  looking to do when choosing when to stop training is to stop when the model's architecture has been given
    #  properties that fully (up to a theoretical limit) represent the signal and filter out the noise.
    #  With that in mind, it will help us to have the theoretical limit in advance, rather than running multiple
    #  trials or circling around a maximum accuracy (or another metric) incrementally. If we reach that theoretical
    #  limit or an acceptable distance from it, stop.
    
    # TODO: look at using some measure of distance from the human-generated attention map to the ViT's generated attention
    #  map as a loss function or a component of a loss function. (this is probably its own paper, but if you have
    #  access to both for this study, at least do the distance function and mention this idea in the paper).

    # TODO: implement tf-explain on the CNN model
    # https://github.com/sicara/tf-explain
    
    file_datetime = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    default_subject_id: str = "4"
    default_mode: str = "perp" # ["perp", "mirror"]
    default_image_size: int = 240 # was 320. Source images are 640x640, but these take a long time to process
    default_epochs: int = 10 # 200 to 0.9558 accuracy
    default_batch_size: int = 64
    default_explain_method: str = "GradCAM" #
    default_filters: list[int] = [16,16,16,16,16]
    default_dense_units: int = 64
    default_progress: str = "none" # ["tqdm", "none"]
    default_learning_rate: float = 1e-5 # 5e-5 # default 1e-3
    default_dropout_rate: float = 0.5
    # empty string for save or load model will skip save or load
    # default_save_model: str = f"results/models/cnn/cnn_{default_mode}_subject_{default_subject_id}_{default_epochs}_epochs_{file_datetime}.keras"
    default_save_model: str = ""
    default_load_model: str = f"results/models/cnn/cnn_perp_subject_4_200_epochs_20260719_133931.keras"
    # default_load_model: str = ""
    default_metrics_filepath: str = f"results/metrics/cnn/metrics_cnn_subject_{default_subject_id}_{default_epochs}_epochs_{file_datetime}.json"
    default_confusion_matrix_filepath: str = f"results/figs/cnn/cm_cnn_subject_{default_subject_id}_{default_epochs}_epochs_{file_datetime}.png"
    
    # reducing learning rate from 1e-3 to 1e-4 resulted in a smoother validation accuracy curve during training
    
    # TODO: introduce learning rate decay to reduce variability in the validation accuracy during training?
    # TODO: add test run details to confusion matrix image via arguments
    
    ap = argparse.ArgumentParser(description=f"CNN gesture classifier (subject {default_subject_id}) with progress bars + metrics.")
    # Paths / data
    ap.add_argument("--root", type=str,
        # default=r"C:\Users\bimbr\Documents\Mirror_Paper\Data_Upload",
        default=config.default_dataset_path,
        help="Root folder containing 'mirror' and 'perp'.")
    ap.add_argument("--mode", type=str, choices=["mirror", "perp"], default=default_mode,
        help="Dataset mode: mirror or perp.")
    ap.add_argument("--subject", type=str, default=f"Subject_{default_subject_id}",
        help="Subject folder name.")
    ap.add_argument("--image-size", type=int, default=default_image_size,
        help="Model input size (pixels).")
    # Training
    ap.add_argument("--epochs", type=int, default=default_epochs, help="Number of training epochs.")
    ap.add_argument("--batch-size", type=int, default=default_batch_size, help="Batch size.")
    ap.add_argument("--seed", type=int, default=42, help="Random seed.")
    ap.add_argument("--val-split", type=float, default=0.1, help="Validation split from training set.")
    # ap.add_argument("--progress", type=str, choices=["tqdm", "none"], default="tqdm",
    ap.add_argument("--progress", type=str, choices=["tqdm", "none"], default=default_progress,
        help="tqdm progress bars (tqdm) or Keras logs only (none).")
    ap.add_argument("--explain-method", type=str, default=default_explain_method, help="CNN attention visualization, e.g. 'GradCAM'")
    ap.add_argument("--explain-output-folder", type=str, default="results/attention/cnn/")
    # Model knobs (optional)
    ap.add_argument("--filters", type=int, nargs="+", default=default_filters, help="Conv filters per block.")
    ap.add_argument("--dense", type=int, default=default_dense_units, help="Units in the penultimate dense layer.")
    ap.add_argument("--dropout", type=float, default=default_dropout_rate, help="Dropout rate.")
    ap.add_argument("--lr", type=float, default=default_learning_rate, help="Adam learning rate.")
    # Save / load
    ap.add_argument("--load-model", type=str, default=default_load_model, help="Path to an existing .keras model to load (skip training if provided).")
    ap.add_argument("--save-model", type=str, default=default_save_model, help="Path to save trained model, e.g., results/cnn_mirror_subject1.keras")
    ap.add_argument("--out", type=str, default=default_metrics_filepath, help="Path to save metrics JSON, e.g., results/subject1_cnn.json")
    ap.add_argument("--cm", type=str, default=default_confusion_matrix_filepath, help="Path to save confusion matrix PNG, e.g., results/figs/subject1_cnn_cm.png")

    args = ap.parse_args()
    set_seed(args.seed)
 
    # Load data
    print(f"-- Loading data from: {args.root}...")

    (x_train, y_train), (x_test, y_test), num_classes = load_subject_arrays(
        args.root, args.mode, args.subject, args.image_size
    )
    print(f"-- loaded {len(x_train)} training samples and {len(x_test)} test samples for {num_classes} classes")

    input_shape = (args.image_size, args.image_size, 1)
    print(f"-- input shape: {input_shape}")
    
    # TODO: perform Integrated Gradient attribution for each image in each class
    # TODO: combine the IG attribution maps for all images in each class
    # TODO: compare the IG attribution maps for each class:
    #  - to each other
    #  - to the ViT attention maps for each class
    #  - to combined differencing maps
    #  - to human expert maps

    # The raw training arrays are grouped by class, so Keras' validation_split
    # would take a non-representative tail slice. Use an explicit stratified split.
    x_train_fit, x_val, y_train_fit, y_val = train_test_split(
        x_train,
        y_train,
        test_size=args.val_split,
        random_state=args.seed,
        stratify=y_train,
        shuffle=True,
    )
    print(f"-- train/val split: {len(x_train_fit)} train samples, {len(x_val)} val samples")

    model: keras.Model
    
    # TODO: train a CNN model on 240x240 images and save it

    # Build or load
    if args.load_model and os.path.isfile(args.load_model):
        print(f"Loading model from: {args.load_model}")
        model = keras.models.load_model(args.load_model, compile=False)
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=args.lr),
            loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=[keras.metrics.SparseCategoricalAccuracy(name="accuracy")],
        )
        trained = True
    else:
        model = build_cnn(
            input_shape=input_shape, num_classes=num_classes,
            filters=tuple(args.filters), dense_units=args.dense,
            dropout=args.dropout, lr=args.lr
        )
        trained = False

    # Progress setup
    callbacks = []
    verbose = 0 if args.progress == "tqdm" else 2
    if args.progress == "tqdm":
        callbacks.append(TqdmProgress(enable=True))
    # CNN attention visualization
    if args.explain_method == "GradCAM":
        callbacks.append(GradCAMCallback(
            validation_data=(x_val, y_val),
            class_index=0,
            output_dir=args.explain_output_folder,
        ))

    # training_details is used to label result plots
    training_details: dict = {
        "mode": args.mode,
        "subject": args.subject,
        "image_dimensions": f"{args.image_size}x{args.image_size}",
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.lr,
        "dropout": args.dropout,
        "val_split": args.val_split,
    }
    
    # Train
    history: History|None = None
    if not trained:
        print(f"-- training model for '{args.mode}/{args.subject}'...")
        train_start_datetime = datetime.now()

        history: History = model.fit(
            # x_train, y_train,
            x_train_fit, y_train_fit,
            batch_size=args.batch_size,
            epochs=args.epochs,
            # validation_split=args.val_split,
            validation_data=(x_val, y_val),
            callbacks=callbacks,
            verbose=verbose
        )
        training_details['training_set_size'] = len(x_train_fit)
        train_end_datetime = datetime.now()
        training_duration = train_test_duration_display(train_end_datetime - train_start_datetime)
        training_details['training_duration'] = training_duration
        
        # TODO: add max validation accuracy and max validation accuracy epoch to training details
        # TODO: add this in all relevant places

        # Note: Use 'acc' instead of 'accuracy' if you are using an older Keras version
        acc_key = 'accuracy' if 'accuracy' in history.history else 'acc'
        max_validation_acc = max(history.history[acc_key])
        max_validation_acc_epoch = history.history[acc_key].index(max_validation_acc) + 1
    
        training_details['max_val_acc'] = f"{max_validation_acc:.4f}"
        training_details['max_val_acc_epoch'] = max_validation_acc_epoch

        print(f"-- training complete.")
        print(f"-- training duration: {training_duration}")

    model.summary()
    
    # Evaluate
    test_start = datetime.now()
    logits = model.predict(x_test, verbose=0)
    y_pred = np.argmax(logits, axis=1)
    test_end = datetime.now()
    test_duration = train_test_duration_display(test_end - test_start)
    training_details['testing_set_size'] = len(x_test)
    training_details['testing_duration'] = str(test_duration)
    
    acc = accuracy_score(y_test, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(
        y_test, y_pred, average="macro", zero_division=0
    )

    print(f"\n[{args.mode}/{args.subject}]   Test Accuracy: {acc:.4f}")
    print(f"[{args.mode}/{args.subject}] Random Accuracy: {1/num_classes:.4f}")
    print(f"[{args.mode}/{args.subject}] Macro Precision: {prec:.4f}  Macro Recall: {rec:.4f}  Macro F1: {f1:.4f}")
    
    if not trained and history is not None:
        if is_apple_silicon():
            # get Mac system info
            mac_system_info: dict[str, str] = get_mac_system_info()

            training_details['processor_type'] = mac_system_info["processor_type"]
            training_details['cpu_cores'] = mac_system_info["cpu_cores"]
            training_details['gpu_cores'] = mac_system_info["gpu_cores"]

            training_details['test_accuracy'] = f"{acc:.4f}"

        print(f"plotting training history...")

        loss_title: str = "CNN Model Loss Over Epochs"
        accuracy_title: str = "CNN Model Accuracy Over Epochs"
        
        # use scientific notation format for learning rate in filepaths
        # (don't use decimal point)
        learning_rate_string: str = f"{args.lr:.2e}"
        learning_rate_string = learning_rate_string.replace(".", "p")
        history_plot_filename: str = f"results/figs/cnn/history_cnn_{args.epochs}_epochs_{learning_rate_string}_lr_{file_datetime}"

        plot_history_separately(training_history=history, loss_plot_title=loss_title, acc_plot_title=accuracy_title, details=training_details, save_plots=True, plot_filename=history_plot_filename)
        # plot_history_together(history)
        print(f"Saved training history plots to: {history_plot_filename}")

    # Save artifacts
    if args.cm:
        confusion_matrix_title: str = f"CNN Confusion Matrix"
        save_confusion_matrix_png(y_test, y_pred, args.cm, cm_title=confusion_matrix_title, details=training_details)
        print(f"Saved confusion matrix to: {args.cm}")

    if args.save_model:
        ensure_dir(args.save_model)
        model.save(args.save_model)
        print(f"Saved model to: {args.save_model}")

    if args.out:
        result = {
            "model": "cnn",
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
                "image_size": args.image_size,
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "seed": args.seed,
                "val_split": args.val_split,
                "filters": args.filters,
                "dense": args.dense,
                "dropout": args.dropout,
                "lr": args.lr,
            },
        }
        dump_json(result, args.out)
        print(f"Saved metrics JSON to: {args.out}")


if __name__ == "__main__":
    main()

# ---- References ----
#
# TODO: get the source references for LLM responses as well
# Claude Code running qwen3.6 (2026) "Format a python timedelta as a string" [LLM chat]. 2026–07–26 
# Google Gemini 3 (2026) "How to get the attention maps for all layers from an existing vision transformer in tensorflow" [LLM chat]. 2026–07–27 

