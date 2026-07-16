#Author - Keshav Bimbraw

import os
import json
import argparse
import numpy as np
import pandas as pd
import tensorflow as tf
from keras.src.callbacks import History
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import matplotlib.pyplot as plt
# import torch

# Progress bar (console "trackbar")
from tqdm.auto import tqdm

import config


# ============================
# Helpers
# ============================
def set_seed(seed: int):
    """Make runs reproducible across numpy and TF."""
    import random
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def ensure_dir(p):
    """Create parent directory for a file path if it does not exist."""
    if p:
        os.makedirs(os.path.dirname(p), exist_ok=True)


def save_confusion_matrix_png(y_true, y_pred, path):
    """Save a simple confusion matrix figure to PNG."""
    print(f"-- saving confusion matrix to '{path}'...")
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

def dump_json(obj, path):
    """Dump a JSON file with nice indentation."""
    print(f"-- saving JSON results to '{path}'...")
    if not path:
        return
    ensure_dir(path)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)

# ============================
# Keras Callback: tqdm progress bars
# ============================
class TqdmProgress(keras.callbacks.Callback):
    """
    Two-level progress:
      - Outer bar: epochs
      - Inner bar: batches within each epoch (the "trackbar" you asked for)
    """
    def __init__(self, enable=True):
        super().__init__()
        self.enable = enable
        self.epoch_bar = None
        self.batch_bar = None

    def on_train_begin(self, logs=None):
        if not self.enable:
            return
        total_epochs = self.params.get("epochs", None)
        self.epoch_bar = tqdm(total=total_epochs, desc="Epochs", position=0, leave=True)

    def on_epoch_begin(self, epoch, logs=None):
        if not self.enable:
            return
        # Create/refresh the per-epoch batch bar
        total_steps = self.params.get("steps", None)  # number of batches in an epoch
        # Note: steps can be None if TF infers it; in practice with numpy arrays it’s len(x_train)//batch_size
        self.batch_bar = tqdm(
            total=total_steps, desc=f"Epoch {epoch+1}/{self.params.get('epochs','?')}",
            position=1, leave=False
        )

    def on_train_batch_end(self, batch, logs=None):
        if not self.enable or self.batch_bar is None:
            return
        self.batch_bar.update(1)
        # Optionally show batch-level loss/acc in the bar postfix
        if logs:
            self.batch_bar.set_postfix({
                "loss": f"{logs.get('loss', 0):.4f}",
                "acc": f"{logs.get('accuracy', 0):.4f}"
            })

    def on_epoch_end(self, epoch, logs=None):
        if not self.enable:
            return
        # Close the inner bar and write a one-line summary with val metrics
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
        if not self.enable:
            return
        if self.batch_bar is not None:
            self.batch_bar.close()
        if self.epoch_bar is not None:
            self.epoch_bar.close()

# ============================
# ViT building blocks
# ============================
class Patches(keras.layers.Layer):
    """Split the input image into non-overlapping patches."""
    # TODO: ** add attention biasing layer
    # TODO: split ViT into multiple channels, one for each degree of freedom OR combination of DOF for each gesture
    # TODO: recombine DOF channels before the fully connected (MLP) layer(s) (or whatever is used after the ViT)
    # 
    # TODO: try different patch sizes
    # TODO: try a different method for splitting the image into patches, e.g. using convolutions or contouring
    #  to pick out significant regions/shapes
    # TODO: contouring is an outdated method for priming or masking patches for ViTs. 
    #  But a human expert could assign masking starting points for a multi-channel ViT, where each channel
    #  could include a mask for an individual muscle or for a set of muscles involved in synergistic movement of
    #  a given DOF.
    def __init__(self, patch_size):
        super().__init__()
        self.patch_size = patch_size
    def call(self, images):
        patches = tf.image.extract_patches(
            images=images,
            sizes=[1, self.patch_size, self.patch_size, 1],
            strides=[1, self.patch_size, self.patch_size, 1],
            rates=[1, 1, 1, 1],
            padding="VALID",
        )
        batch = tf.shape(patches)[0]
        return tf.reshape(patches, [batch, -1, patches.shape[-1]])

class PatchEncoder(keras.layers.Layer):
    """Linear projection + learnable positional embeddings."""
    def __init__(self, num_patches, projection_dim):
        super().__init__()
        self.num_patches = num_patches
        self.projection = keras.layers.Dense(units=projection_dim)
        self.position_embedding = keras.layers.Embedding(
            input_dim=num_patches, output_dim=projection_dim
        )
    def call(self, patch):
        positions = tf.range(start=0, limit=self.num_patches, delta=1)
        return self.projection(patch) + self.position_embedding(positions)

def mlp(x, hidden_units, dropout_rate):
    """Transformer MLP block."""
    for units in hidden_units:
        x = keras.layers.Dense(units, activation="relu")(x)
        x = keras.layers.Dropout(dropout_rate)(x)
    return x

def build_vit(input_shape, num_classes,
              patch_size=32, projection_dim=64, num_heads=8,
              transformer_layers=6, transformer_units=(128, 64),
              mlp_head_units=(512, 256)):
    """Build a compact ViT classifier (no CLS token; flatten + MLP head)."""
    h, w, _ = input_shape
    num_patches = (h // patch_size) * (w // patch_size)

    inputs = keras.layers.Input(shape=input_shape)
    patches = Patches(patch_size)(inputs)
    encoded = PatchEncoder(num_patches, projection_dim)(patches)

    for _ in range(transformer_layers):
        x1 = keras.layers.LayerNormalization(epsilon=1e-6)(encoded)
        attn = keras.layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=projection_dim, dropout=0.1
        )(x1, x1)
        x2 = keras.layers.Add()([attn, encoded])
        x3 = keras.layers.LayerNormalization(epsilon=1e-6)(x2)
        x3 = mlp(x3, hidden_units=transformer_units, dropout_rate=0.1)
        encoded = keras.layers.Add()([x3, x2])
        
    # TODO: play with the number of dropout layers in the ViT and their placement
    # TODO: see if a recurrent connection is appropriate (are there enough unshuffled frames in each 100-frame
    #  labelled sequence in the dataset to use the time-dependent information, or do I need a different dataset
    #  for that?)
    # TODO: see if a contour-based method, rather than the uniform 16x16 grid, would work for the initial selection
    #  of attention regions for this ViT. These images may be good candidates – the shapes we're looking at don't have
    #  foreground and background regions nor internal/hierarchical objects, like a dog's face. They're basic shapes
    #  like oblong/oval shapes, blobs or rings of light on dark.

    representation = keras.layers.LayerNormalization(epsilon=1e-6)(encoded)
    representation = keras.layers.Flatten()(representation)
    representation = keras.layers.Dropout(0.5)(representation)
    features = mlp(representation, hidden_units=mlp_head_units, dropout_rate=0.5)
    logits = keras.layers.Dense(num_classes)(features)

    model = keras.Model(inputs=inputs, outputs=logits)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=[keras.metrics.SparseCategoricalAccuracy(name="accuracy")],
    )
    return model

# ============================
# Data loading
# ============================
def load_subject_arrays(root, mode, subject, image_size):
    """
    Load four arrays for a subject:
      X_m_train.npy, X_m_test.npy, y_m_train.npy, y_m_test.npy
    Then: add channel dim if needed, resize to (image_size, image_size), normalize to [0,1].
    """
    d = os.path.join(root, mode, subject)
    x_train = np.load(os.path.join(d, "X_m_train.npy"))
    x_test  = np.load(os.path.join(d, "X_m_test.npy"))
    y_train = np.load(os.path.join(d, "y_m_train.npy"))
    y_test  = np.load(os.path.join(d, "y_m_test.npy"))

    y_train = y_train.astype(np.int64).ravel()
    y_test  = y_test.astype(np.int64).ravel()

    # Add channel dim if needed: (N,H,W) -> (N,H,W,1)
    if x_train.ndim == 3:
        x_train = x_train[..., np.newaxis]
    if x_test.ndim == 3:
        x_test = x_test[..., np.newaxis]

    # Resize to ViT input size
    x_train = tf.image.resize(tf.convert_to_tensor(x_train), (image_size, image_size)).numpy()
    x_test  = tf.image.resize(tf.convert_to_tensor(x_test ), (image_size, image_size)).numpy()

    # Normalize to [0,1]
    if x_train.dtype != np.float32:
        x_train = x_train.astype("float32")
        x_test  = x_test.astype("float32")
    maxv = max(float(x_train.max()), 1.0)
    x_train /= maxv
    x_test  /= maxv

    num_classes = int(max(y_train.max(), y_test.max()) + 1)
    return (x_train, y_train), (x_test, y_test), num_classes

def plot_history_together(training_history: History):
    # Convert history dictionary to DataFrame
    history_df = pd.DataFrame(training_history.history)
    
    # Plot all metrics at once
    history_df.plot(figsize=(10, 6))
    plt.grid(True)
    # plt.gca().set_ylim(0, 1) # Optional: clamp y-axis between 0 and 1 for accuracy
    plt.xlabel("Epochs")
    plt.show()
    
def plot_history_separately(training_history: History):
    # Create a figure with two subplots side-by-side
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot Training & Validation Loss
    ax1.plot(training_history.history['loss'], label='Train Loss', color='blue', linewidth=2)
    if 'val_loss' in training_history.history:
        ax1.plot(training_history.history['val_loss'], label='Val Loss', color='orange', linestyle='--', linewidth=2)
    ax1.set_title('Model Loss Over Epochs')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True)
    
    # Plot Training & Validation Accuracy
    # Note: Use 'acc' instead of 'accuracy' if you are using an older Keras version
    acc_key = 'accuracy' if 'accuracy' in training_history.history else 'acc'
    ax2.plot(training_history.history[acc_key], label='Train Accuracy', color='blue', linewidth=2)
    
    val_acc_key = 'val_' + acc_key
    if val_acc_key in training_history.history:
        ax2.plot(training_history.history[val_acc_key], label='Val Accuracy', color='orange', linestyle='--', linewidth=2)
    ax2.set_title('Model Accuracy Over Epochs')
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('Accuracy')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.show()


# ============================
# Main
# ============================
def main():
    # TODO: ** find additional ViT examples (with attention biasing)
    # TODO: ** find any ultrasound ViT examples (with or without attention biasing)

    # MPS for pytorch
    # confirm that an accelerator device is available (we expect
    # mps.device_count >= 1)
    # print(f"checking mps.device_count using pytorch...")
    # print(f"mps.device_count: {torch.mps.device_count()}")
    # use the MPS Pytorch accelerator (see https://docs.pytorch.org/docs/stable/mps.html#module-torch.mps)
    # mps_device = torch.device("mps:0" if torch.mps.is_available() else "cpu")
    
    # MPS for tensorflow
    print(f"\nchecking device count using tensorflow-metal...")
    # Check for available physical devices
    physical_devices = tf.config.list_physical_devices('GPU')
    
    if len(physical_devices) > 0:
        print(f"✅ Metal GPU Acceleration is active! Found: {physical_devices}")
    else:
        print("❌ GPU not found. TensorFlow is falling back to the CPU.")
    
    # exit(0)
    
    parser = argparse.ArgumentParser(description="Run ViT on Subject_1 ultrasound data with progress bars.")
    # Paths / data
    parser.add_argument("--root", type=str,
        # default=r"C:\Users\bimbr\Documents\Mirror_Paper\Data_Upload",
        # default=r"/Users/rickgladwin/Code/u_of_hull/dissertation/bimbraw_2025_dataset/data/",
        default=config.default_dataset_path,
        help="Root folder containing 'mirror' and 'perp'.")
    parser.add_argument("--mode", type=str, choices=["mirror", "perp"], default="mirror",
        help="Dataset mode: mirror or perp.")
    parser.add_argument("--subject", type=str, default="Subject_1",
        help="Subject folder name.")
    parser.add_argument("--image-size", type=int, default=640,
        help="Model input size (pixels).")
    # Training
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs.")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--val-split", type=float, default=0.1, help="Validation split from training set.")
    parser.add_argument("--progress", type=str, choices=["tqdm", "none"], default="tqdm",
        help="Use tqdm progress bars (tqdm) or Keras logging only (none).")
    # Save / load
    parser.add_argument("--load-model", type=str, default="", help="Path to an existing .keras model to load (skip training if provided).")
    parser.add_argument("--save-model", type=str, default="", help="Path to save trained model, e.g., results/vit_mirror_subject1.keras")
    parser.add_argument("--out", type=str, default="", help="Path to save metrics JSON, e.g., results/subject1_vit.json")
    parser.add_argument("--cm", type=str, default="", help="Path to save confusion matrix PNG, e.g., results/figs/subject1_vit_cm.png")

    args = parser.parse_args()
    set_seed(args.seed)

    # Load data
    print(f"-- Loading data from: {args.root}...")
    (x_train, y_train), (x_test, y_test), num_classes = load_subject_arrays(
        args.root, args.mode, args.subject, args.image_size
    )
    print(f"-- loaded {len(x_train)} training samples and {len(x_test)} test samples for {num_classes} classes")
    input_shape = (args.image_size, args.image_size, 1)
    print(f"-- input shape: {input_shape}")

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

    # Build or load model
    if args.load_model and os.path.isfile(args.load_model):
        print(f"Loading model from: {args.load_model}")
        model = keras.models.load_model(args.load_model, compile=False)
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=1e-3),
            loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=[keras.metrics.SparseCategoricalAccuracy(name='accuracy')],
        )
        trained = True
    else:
        model = build_vit(
            input_shape=input_shape,
            num_classes=num_classes,
            patch_size=32,
            projection_dim=64,
            num_heads=8,
            transformer_layers=6,
            transformer_units=(128, 64),
            mlp_head_units=(512, 256),
        )
        trained = False

    # Choose callbacks / logging
    callbacks = []
    verbose = 0 if args.progress == "tqdm" else 2  # let tqdm handle printing

    if args.progress == "tqdm":
        callbacks.append(TqdmProgress(enable=True))

    # Train (unless we loaded a pre-trained model)
    if not trained:
        print(f"-- training model for '{args.mode}/{args.subject}'...")
        # Note: with verbose=0, Keras won’t print per-batch/epoch lines; tqdm shows the progress instead.
        history = model.fit(
            x_train_fit, y_train_fit,
            batch_size=args.batch_size,
            epochs=args.epochs,
            validation_data=(x_val, y_val),
            callbacks=callbacks,
            verbose=verbose
        )
        print(f"-- training complete.")
        
        print(f"plotting training history...")

        plot_history_separately(history)
        # plot_history_together(history)
    
    # Evaluate on test
    logits = model.predict(x_test, verbose=0)
    y_pred = np.argmax(logits, axis=1)

    acc = accuracy_score(y_test, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(
        y_test, y_pred, average="macro", zero_division=0
    )

    print(f"\n[{args.mode}/{args.subject}]   Test Accuracy: {acc:.4f}")
    print(f"[{args.mode}/{args.subject}] Random Accuracy: {1/num_classes:.4f}")
    print(f"[{args.mode}/{args.subject}] Macro Precision: {prec:.4f}  Macro Recall: {rec:.4f}  Macro F1: {f1:.4f}")

    # Save CM and model/metrics if requested
    if args.cm:
        save_confusion_matrix_png(y_test, y_pred, args.cm)
        print(f"Saved confusion matrix to: {args.cm}")

    if args.save_model:
        ensure_dir(args.save_model)
        model.save(args.save_model)
        print(f"Saved model to: {args.save_model}")

    if args.out:
        result = {
            "model": "vit",
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
            },
        }
        dump_json(result, args.out)
        print(f"Saved metrics JSON to: {args.out}")

if __name__ == "__main__":
    # example command:
    # python3.10 ultrasound_gesture_vit_classification.py --mode perp --subject Subject_1 --epochs 2 --batch-size 64 --save-model results/vit_mirror_subject1.keras --out results/subject1_vit_mirror.json --cm results/figs/subject1_vit_mirror_cm.png
    main()
