#Author - Keshav Bimbraw

import os
import json
import argparse
from datetime import datetime
from symtable import Function

import numpy as np
import pandas as pd
import tensorflow as tf
from keras.src.callbacks import History
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
# import torch

# ViT optimizations
# see https://share.google/aimode/UwSqy8Wxk8WaXGWHX
from tensorflow.keras import mixed_precision

from ultrasound_gesture_cnn_classification import train_test_duration_display, is_apple_silicon, get_mac_system_info
from utilities import ensure_dir, set_seed, dump_json, process_pool_size
from visualizations import create_caption_from_details, plot_history_separately, save_confusion_matrix_png

# Enable mixed float16 precision (mat default to float32 for all operations otherwise)
# this changed the time per epoch from apx 1m14s to apx 43s
# early estimate 1h18s down from 2h11m
# ** check and see what the impact is on accuracy
policy = mixed_precision.Policy('mixed_float16')
mixed_precision.set_global_policy(policy)

# Progress bar (console "trackbar")
from tqdm.auto import tqdm

import config

# TODO: explore using an attention mask on a CNN
# TODO: explore using a ViT to generate an attention mask (using explainability techniques like segmentation or
#  post-training attention maps) and applying that to the CNN
# TODO: explore how a time-dependent attention mask, or an attention mask that was a function of
#  joint position, would work. Knowing the (theoretical/idealized) synergistic functions, how could we make a model
#  that would tweak or generate these functions for the purposes of making an attention mask function? One issue with
#  the synergistic function sets is that they're individualized. A ViT (etc.) might be able to modify or produce a
#  synergistic function set that would apply to the individual and/or dataset at hand, allowing us to make best use of
#  any given dataset, and/or help us build a transfer learning system or mapping for adapting to new subjects and/or
#  new positioning of the ultrasound probes.

# TODO: see if there are any other places where multiprocessing could be used.
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
# register this custom layer class with a decorator so that it can be deserialized if a saved .keras copy of the model is loaded
@keras.saving.register_keras_serializable()
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

@keras.saving.register_keras_serializable()
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

def build_vit_with_attention_output(input_shape, num_classes,
              patch_size=32, projection_dim=64, num_heads=8,
              transformer_layers=6, transformer_units=(128, 64),
              mlp_head_units=(512, 256), learning_rate: float=1e-3) -> keras.Model:
    """
    Build a compact ViT classifier (no CLS token; flatten + MLP head).
    Provide attention information for visualizing and using attention maps.
    """
    h, w, _ = input_shape
    num_patches = (h // patch_size) * (w // patch_size)

    inputs = keras.layers.Input(shape=input_shape)
    patches = Patches(patch_size)(inputs)
    encoded = PatchEncoder(num_patches, projection_dim)(patches)

    # Initialize a variable to store the final attention map
    final_attention_scores = None

    for i in range(transformer_layers):
        x1 = keras.layers.LayerNormalization(epsilon=1e-6)(encoded)
        # TODO: pass an argument for attention_mask to MultiHeadAttention
        # Instantiate the MHA layer
        attn_layer = keras.layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=projection_dim, dropout=0.1
        )

        # Determine if we are on the final transformer block
        is_final_layer = (i == transformer_layers - 1)

        # Call MHA and conditionally fetch attention scores
        if is_final_layer:
            attn_output, final_attention_scores = attn_layer(
                query=x1, value=x1, return_attention_scores=True
            )
        else:
            attn_output = attn_layer(query=x1, value=x1, return_attention_scores=False)

        x2 = keras.layers.Add()([attn_output, encoded])
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

    model = keras.Model(inputs=inputs, outputs=[logits, final_attention_scores])
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        # loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        # metrics=[keras.metrics.SparseCategoricalAccuracy(name="accuracy")],

        # Use a list of losses matching the order of outputs.
        # final_attention_scores does not need a loss, so we pass None.
        loss=[keras.losses.SparseCategoricalCrossentropy(from_logits=True), None],
        metrics={model.output_names[0]: keras.metrics.SparseCategoricalAccuracy(name="accuracy")},
    )
    return model


def get_vit_attention_model(existing_model):
    """
    Extract attention scores from a ViT model.
    Modified version of code from Google Gemini 3 (2026)
    'how to get the attention maps for all layers from an existing vision transformer in tensorflow'
    """
    # Find all attention score output tensors across the ViT layers
    attention_outputs = []

    for layer in existing_model.layers:
        # Check for ViT block structures or direct attention layers
        if 'attention' in layer.name.lower():
            # Ensure the layer outputs weights, or target the internal softmax tensor
            attention_outputs.append(layer.output[1]) # Index 1 usually holds scores if return_scores=True

    # Create a multi-output model
    return tf.keras.Model(
        inputs=existing_model.input,
        outputs={'preds': existing_model.output, 'attentions': attention_outputs}
    )


def build_vit(input_shape, num_classes,
              patch_size=32, projection_dim=64, num_heads=8,
              transformer_layers=6, transformer_units=(128, 64),
              mlp_head_units=(512, 256), learning_rate: float=1e-3) -> keras.Model:
    """Build a compact ViT classifier (no CLS token; flatten + MLP head)."""
    h, w, _ = input_shape
    num_patches = (h // patch_size) * (w // patch_size)

    inputs = keras.layers.Input(shape=input_shape)
    patches = Patches(patch_size)(inputs)
    encoded = PatchEncoder(num_patches, projection_dim)(patches)

    for _ in range(transformer_layers):
        x1 = keras.layers.LayerNormalization(epsilon=1e-6)(encoded)
        attn = keras.layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=projection_dim, dropout=0.1,
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
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
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


import numpy as np
import matplotlib.pyplot as plt
import cv2


def plot_attention_map(model, image, patch_size=32):
    # code extracted from Google Gemini 3 (2026)
    """
    Extracts attention scores, averages heads, and overlays a heatmap on the image.

    Args:
        model: The trained multi-output ViT model.
        image: A single numpy image array of shape (H, W, C).
        patch_size: The patch size used during model compilation.
    """
    print(f"image.shape: {image.shape}")
    # 1. Prepare image for prediction (add batch dimension: 1, H, W, C)
    input_tensor = np.expand_dims(image, axis=0)
    print(f"input_tensor.shape: {input_tensor.shape}")

    # 2. Extract outputs (logits and final attention scores)
    _, attention_scores = model.predict(input_tensor)

    # Shape of attention_scores is (batch, heads, target_seq_len, source_seq_len)
    # For self-attention, seq_len = num_patches
    attention_scores = attention_scores[0]  # Drop batch dimension -> (heads, num_patches, num_patches)

    # 3. Average across all attention heads
    avg_attention = np.mean(attention_scores, axis=0)  # Shape: (num_patches, num_patches)

    # 4. Collapse to a 1D attention weight per patch (sum or mean over source patches)
    # This represents how much each patch is attended to overall
    patch_weights = np.mean(avg_attention, axis=0)  # Shape: (num_patches,)
    print(f"patch_weights.shape: {patch_weights.shape}")

    # 5. Reshape the 1D patch weights back into the 2D grid spatial layout
    h, w, _ = image.shape
    grid_size = h // patch_size
    heatmap = patch_weights.reshape((grid_size, grid_size))
    print(f"heatmap.shape: {heatmap.shape}")
    print(f"heatmap.dtype: {heatmap.dtype}")
    print(f"(w, h): ({w, h})")
    # plt.imshow(heatmap, cmap="viridis")
    # plt.show()

    # convert data type for heatmap for compatibility with cv2.resize
    heatmap = heatmap.astype("float32")

    # 6. Resize heatmap to match original image dimensions using cubic interpolation
    heatmap = cv2.resize(heatmap, (w, h), interpolation=cv2.INTER_CUBIC)

    # 7. Normalize heatmap values strictly between 0 and 1 for clean rendering
    heatmap = (heatmap - np.min(heatmap)) / (np.max(heatmap) - np.min(heatmap) + 1e-8)

    # 8. Render the plot side-by-side: Original vs. Overlayed Heatmap
    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    plt.imshow(image.astype("uint8") if image.max() > 1 else image, cmap="gray")
    plt.title("Original Image")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(image.astype("uint8") if image.max() > 1 else image, vmin=0, vmax=1)
    # Overlay the heatmap using a semi-transparent jet colormap
    plt.imshow(heatmap, cmap="jet", alpha=0.4)
    plt.title("Attention Heatmap")
    plt.axis("off")

    plt.tight_layout()
    plt.show()


# ============================
# Main
# ============================
def main():
    # TODO: ** find additional ViT examples (with attention biasing)
    # TODO: ** find any ultrasound ViT examples (with or without attention biasing)

    # TODO: extract attention maps for comparison with CNN attribution maps

    # TODO: research TOAST (TOp-down Attention STeering), a transfer learning method
    #  that uses attention steering to improve accuracy, from Shi et al. (2023)

    # TODO: read Vision Transformers Need Registers from Darcet et al. (2023), regarding
    #  a technique to focus and improve accuracy of attention maps in ViTs.
    
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
    
    # settings from the paper:
    # Vision Transformers: The ViT model is trained with
    # a learning rate of 0.0005 and a weight decay of 0.0001,
    # using a batch size of 256, and for 200 epochs. The images
    # were downsized to 320 × 320 pixels and divided into
    # 32 ×32 patches, resulting in 100 patches per image. The
    # model uses 16 attention heads across eight transformer layers,
    # with each layer having projection dimensions of 64. The
    # final classification layers consist of two dense layers with
    # 2048 and 1024 units. Data augmentation techniques, including
    # normalization, resizing, random horizontal flipping, rotation,
    # and zooming, are applied before training.
    
    # TODO: get MPS working with ViT model. It's working for CNN and SVC.
    #  NOTE: it seems to be working but the CPU is lagging behind the GPU.
    #  TODO: make the CPU process run in parallel in order to keep ahead of the GPU.
    
    # TODO: ensure the train and test datasets are representative the way we want?
    #  Keshav's team has pre-split the data into train and test, rather than dividing
    #  them up at training time. Check and see what reasoning was used. Validation accuracy
    #  gets up around 97% while test accuracy is around 84%. We expect a difference
    #  in this direction, but one so large? Maybe overfitting on the training set? Or
    #  is there a difference between the test and training sets? We are shuffling the
    #  training set but not the test set. Maybe start there?

    file_datetime = datetime.now().strftime("%Y%m%d_%H%M%S")

    default_mode: str = "perp" # ["perp", "mirror"]
    default_subject_id: str = "4"
    default_epochs: int = 1 # paper used 500 (?)
    default_image_size: int = 320 # was 320, raw image is 640
    default_progress: str = "none" # ["tqdm", "none"]
    default_learning_rate: float = 0.0005
    default_weight_decay: float = 0.0001
    default_batch_size: int = 256 # was 256 (larger batch sizes are required for ViTs in order to saturate the GPU)
    default_patch_size: int = 32 # was 32, 320/16 = 20 (took several minutes and never finished the first iteration)
    default_num_heads: int = 16
    default_num_layers: int = 8
    default_projection_dim: int = 64
    default_dense_units: int = 2048

    # selects a model builder function that outputs attention from the MultiheadAttention layer
    # or uses the build_vit function without attention output
    default_output_attention_maps: bool = True

    # empty string for save or load model will skip save or load
    # default_save_model: str = f"results/models/vit/vit_{default_mode}_subject_{default_subject_id}_{default_epochs}_epochs_{default_image_size}px_{file_datetime}.keras"
    default_save_model: str = ""
    # default_load_model: str = f"results/models/vit/vit_perp_subject_2_1_epochs_20260717_191619.keras"
    # default_load_model: str = f"results/models/vit/vit_perp_subject_4_1_epochs_20260802_132156.keras"
    # default_load_model = f"results/models/vit/vit_perp_subject_4_1_epochs_320px_20260802_133434.keras"
    # default_load_model: str = f"results/models/vit/vit_perp_subject_4_1_epochs_320px_20260802_134208.keras"
    default_load_model: str = f"results/models/vit/vit_perp_subject_4_1_epochs_320px_20260802_181936.keras"
    # default_load_model: str = ""

    default_metrics_filepath: str = f"results/metrics/vit/metrics_vit_subject_{default_subject_id}_{default_epochs}_epochs_{file_datetime}.json"
    default_confusion_matrix_filepath: str = f"results/figs/vit/cm_vit_subject_{default_subject_id}_{default_epochs}_epochs_{file_datetime}.png"

    
    # TODO: include global tensorflow precision setting in training details
    # TODO: include model type in title for loss + acc plots
    
    parser = argparse.ArgumentParser(description=f"Run ViT on Subject_{default_subject_id} ultrasound data with progress bars.")
    # Paths / data
    parser.add_argument("--root", type=str,
        # default=r"C:\Users\bimbr\Documents\Mirror_Paper\Data_Upload",
        # default=r"/Users/rickgladwin/Code/u_of_hull/dissertation/bimbraw_2025_dataset/data/",
        default=config.default_dataset_path,
        help="Root folder containing 'mirror' and 'perp'.")
    # parser.add_argument("--mode", type=str, choices=["mirror", "perp"], default="mirror",
    parser.add_argument("--mode", type=str, choices=["mirror", "perp"], default=default_mode,
        help="Dataset mode: mirror or perp.")
    # parser.add_argument("--subject", type=str, default="Subject_1",
    parser.add_argument("--subject", type=str, default=f"Subject_{default_subject_id}",
        help="Subject folder name.")
    parser.add_argument("--image-size", type=int, default=default_image_size,
        help="Model input size (pixels).")
    # Training
    # parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs.")
    parser.add_argument("--epochs", type=int, default=default_epochs, help="Number of training epochs.")
    # parser.add_argument("--batch-size", type=int, default=64, help="Batch size.")
    parser.add_argument("--batch-size", type=int, default=default_batch_size, help="Batch size.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--val-split", type=float, default=0.1, help="Validation split from training set.")
    parser.add_argument("--lr", type=float, default=default_learning_rate, help="Adam learning rate.")
    parser.add_argument("--progress", type=str, choices=["tqdm", "none"], default=default_progress,
        help="Use tqdm progress bars (tqdm) or Keras logging only (none).")
    # Save / load
    parser.add_argument("--load-model", type=str, default=default_load_model, help="Path to an existing .keras model to load (skip training if provided).")
    parser.add_argument("--save-model", type=str, default=default_save_model, help="Path to save trained model, e.g., results/vit_mirror_subject1.keras")
    # parser.add_argument("--out", type=str, default="", help="Path to save metrics JSON, e.g., results/subject1_vit.json")
    file_datetime = datetime.now().strftime("%Y%m%d_%H%M%S")
    parser.add_argument("--out", type=str, default=default_metrics_filepath, help="Path to save metrics JSON, e.g., results/subject1_vit.json")
    parser.add_argument("--cm", type=str, default=default_confusion_matrix_filepath, help="Path to save confusion matrix PNG, e.g., results/figs/subject1_vit_cm.png")
    parser.add_argument("--output-attention-maps", type=bool, default=default_output_attention_maps, help="Whether to output attention maps when the model is built")

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

    # load a single test image tensor for attention mapping
    # gesture class index (0-11) and class shortnames
    # IndPinch is used as the first test in the CNN attribution mapping,
    # class index 4, sample index 410
    # from sample images image_source_file = "/Users/rickgladwin/Code/u_of_hull/dissertation/bimbraw_2025_dataset/data/perp/Subject_4/X_m_test.npy"
    gesture_class_index_and_shortnames: list[tuple[int, str]] = [
        (0, "IndFlex"),
        (1, "MidFlex"),
        (2, "RinFlex"),
        (3, "PinFlex"),
        (4, "IndPinch"),
        (5, "IndMidPinch"),
        (6, "IndMidRinPinch"),
        (7, "AllPinch"),
        (8, "MidRinPinch"),
        (9, "Fist"),
        (10, "Hook"),
        (11, "Open"),
    ]

    print(f"x_test shape: {x_test.shape}")

    # get single test sample image
    test_image_tensor = x_test[410]
    # plt.figure(figsize=(10, 10))
    # plt.imshow(test_image_tensor, cmap="gray")
    # plt.tight_layout()
    # plt.show()

    # exit(0)

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

    model_builder_function = build_vit_with_attention_output if args.output_attention_maps else build_vit

    # Build or load model
    if args.load_model and os.path.isfile(args.load_model):
        print(f"Loading model from: {args.load_model}")
        # model = keras.models.load_model(args.load_model, compile=False)
        model = keras.models.load_model(
            args.load_model,
            compile=False,
            # fix error: TypeError: Cannot deserialize object of type `Patches`. If `Patches` is a custom class, please register it using the `@keras.saving.register_keras_serializable()` decorator.
            # because Patches is a custom class
            custom_objects={'Patches': Patches, 'PatchEncoder': PatchEncoder}
        )
        model.compile(
            # optimizer=keras.optimizers.Adam(learning_rate=1e-3),
            optimizer=keras.optimizers.legacy.Adam(learning_rate=args.lr),
            loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=[keras.metrics.SparseCategoricalAccuracy(name='accuracy')],

        )
        trained = True
    else:
        # TODO: ** use build_vit_with_attention_model based on a config argument
        model = model_builder_function(
            input_shape=input_shape,
            num_classes=num_classes,
            patch_size=default_patch_size, # was 32
            projection_dim=default_projection_dim, # was 64
            num_heads=default_num_heads, # was 8
            transformer_layers=default_num_layers, # was 6
            transformer_units=(128, 64),
            mlp_head_units=(512, 256),
            learning_rate=default_learning_rate,
        )
        trained = False

    # Choose callbacks / logging
    callbacks = []
    verbose = 0 if args.progress == "tqdm" else 2  # let tqdm handle printing

    if args.progress == "tqdm":
        callbacks.append(TqdmProgress(enable=True))

    # training_details is used to label result plots
    training_details: dict = {
        "mode": args.mode,
        "subject": args.subject,
        "image_dimensions": f"{args.image_size}x{args.image_size}",
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.lr,
        # "dropout": args.dropout,
        "val_split": args.val_split,
    }
    
    history: keras.callbacks.History|None = None
    
    # Train (unless we loaded a pre-trained model)
    if not trained:
        print(f"-- training model for '{args.mode}/{args.subject}'...")
        # choose how many processors to use for multiprocessing
        worker_count = process_pool_size(reserve_cores_count=1, verbose=True)
        # Note: with verbose=0, Keras won’t print per-batch/epoch lines; tqdm shows the progress instead.
        train_start_datetime = datetime.now()
        history = model.fit(
            x_train_fit, y_train_fit,
            batch_size=args.batch_size,
            epochs=args.epochs,
            validation_data=(x_val, y_val),
            callbacks=callbacks,
            verbose=verbose,
            use_multiprocessing=True,
            workers=worker_count,
        )
        training_details['training_set_size'] = len(x_train_fit)
        train_end_datetime = datetime.now()
        training_duration = train_test_duration_display(train_end_datetime - train_start_datetime)
        training_details['training_duration'] = training_duration
        print(f"-- training complete.")
        
        # TODO: add title and run details to these functions as arguments
        # plot_history_separately(history, details=training_details)
        # plot_history_together(history)

    # extract attention maps
    # if args.output_attention_maps:
        ## Instantiate the extraction model
        # attn_extraction_model = get_vit_attention_model(model)

        # Inference
        # FIXME: error when calling attn_extraction_model():
        # {{function_node __wrapped__AddV2_device_/job:localhost/replica:0/task:0/device:GPU:0}} Incompatible shapes: [320,0,64] vs. [100,64] [Op:AddV2] name:
        #
        # Call arguments received by layer 'patch_encoder' (type PatchEncoder):
        #   • patch=tf.Tensor(shape=(320, 0, 1024), dtype=float16)

        # DISABLED
        # outputs = attn_extraction_model(test_image_tensor, training=False)
        # attention_maps = outputs['attentions']  # List of tensors: [batch, heads, tokens, tokens]
        #
        # print(f"attention_maps.shape: {attention_maps.shape}")
        #
        # Average across attention heads for layer 0
        # layer_0_attn = tf.reduce_mean(attention_maps[0], axis=1)
        #
        # print(f"layer_0_attn.shape: {layer_0_attn.shape}")
        # END DISABLED

    # Evaluate on test
    # logits = model.predict(x_test, verbose=0)
    # y_pred = np.argmax(logits, axis=1)

    test_start = datetime.now()
    [logits, attention_scores] = model.predict(x_test, verbose=0)
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

        loss_title: str = "ViT Model Loss Over Epochs"
        accuracy_title: str = "ViT Model Accuracy Over Epochs"

        # use scientific notation format for learning rate in filepaths
        # (don't use decimal point)
        learning_rate_string: str = f"{args.lr:.2e}"
        learning_rate_string = learning_rate_string.replace(".", "p")
        history_plot_filename: str = f"results/figs/vit/history_vit_{args.epochs}_epochs_{learning_rate_string}_lr_{file_datetime}"

        print(f"plotting training history...")

        print(f"plotting history: {history}")

        # plot_history_separately(training_history=history, loss_plot_title=loss_title, acc_plot_title=accuracy_title, details=training_details, save_plots=True, plot_filename=history_plot_filename)
        # plot_history_together(history)
        print(f"Saved training history plots to: {history_plot_filename}")

    print(f"plotting attention map...")
    plot_attention_map(model, test_image_tensor, patch_size=32)
    print(f"done plotting attention map")

    # Save CM and model/metrics if requested
    if args.cm:
        confusion_matrix_title: str = f"ViT Confusion Matrix"
        save_confusion_matrix_png(y_test, y_pred, args.cm, cm_title=confusion_matrix_title, details=training_details)
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
    
    # Get attention maps from ViT model
    ## Instantiate the extraction model
    attn_extraction_model = get_vit_attention_model(model)

    # TODO: research and implement GRAD-CAM attention visualization for this ViT
    #  GRAD-CAM visualizes one class at a time, which is exactly what we need for
    #  attention maps that we expect to correspond to muscle synergy groups.
    #  NOTE: if these attention maps turn out to not overlap with expected regions
    #  of the images, consider using:
    #  - a different visualization technique
    #  - a different way of generating attention maps (look for a kind of "feature significance"
    #    calculation, based on small increments to the relevant regions and their effect on
    #    the class prediction accuracy. Could also try masking techniques, though these
    #    don't always result in useful visualizations, especially for small datasets (?)
    
    # Inference
    # TODO: follow the rest of the Gemini-generated suggestions for visualization and rollout
    # TODO: consider following the Medium article, though it's a paid article.
    # outputs = attn_extraction_model(image_tensor, training=False)
    # attention_maps = outputs['attentions'] # List of tensors: [batch, heads, tokens, tokens]
    
    # Average across attention heads for layer 0
    # layer_0_attn = tf.reduce_mean(attention_maps[0], axis=1)

if __name__ == "__main__":
    # example command:
    # python3.10 ultrasound_gesture_vit_classification.py --mode perp --subject Subject_1 --epochs 2 --batch-size 64 --save-model results/vit_mirror_subject1.keras --out results/subject1_vit_mirror.json --cm results/figs/subject1_vit_mirror_cm.png
    main()

# References
# Shi, B., Gai, S., Darrell, T. & Wang, X. (2023) Toast: Transfer learning via attention steering. arXiv:2305.15542. https://ui.adsabs.harvard.edu/abs/2023arXiv230515542S [Accessed May 01, 2023].
#
# Darcet, T., Oquab, M., Mairal, J. & Bojanowski, P. (2023) Vision transformers need registers. arXiv e-prints, arXiv:2309.16588. https://doi.org/10.48550/arXiv.2309.16588
#
# Google Gemini 3 (2026) "Yes please provide an example snippet demonstrating how to format and average the multi-head attention scores to plot a clean heatmap over the input image." (mid-chat prompt) [LLM chat]. 2026–08–02 5:32 PM EDT.

