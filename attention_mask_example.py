import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from attention_mapping import import_attention_mask

# 1. Define ViT Sequence Dimensions
# batch_size = 4
batch_size = 256
# num_patches = 16 # e.g., a 4x4 grid of patches
num_patches = 16 * 16 # 256
projection_dim = 64
# num_heads = 4
num_heads = 16

# Simulate random patch embeddings (Batch, Num_Patches, Projection_Dim)
patch_embeddings = tf.random.normal((batch_size, num_patches, projection_dim))

# 2. Construct the Attention Mask
# Scenario: Suppose the last 4 patches are padding/empty space and should be ignored.
valid_patches_count = 12

print(f"sequence lengths: {[valid_patches_count]}")

# Create a 1D mask marking which patches are valid (1 = valid, 0 = masked)
# Shape: (num_patches,) -> [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0]
sample_mask_1d = tf.sequence_mask([valid_patches_count], maxlen=num_patches, dtype=tf.int32)
sample_mask_1d = tf.squeeze(sample_mask_1d, axis=0)

print(f"sample_mask_1d shape: {sample_mask_1d.shape}")
print(f"sample_mask_1d:\n{sample_mask_1d}")

# create 1d mask from imported 16x16 boolean attention mask
attention_mask_filepath: str = "/Users/rickgladwin/Code/u_of_hull/dissertation/Integrated-Decision-Gradients/results/attention_maps/attn_boolean_map_threshold=0p45_20260817_005014.npy"
attention_mask_array: np.ndarray = import_attention_mask(path=attention_mask_filepath, show_plot=True)
print(f"imported attention mask shape: {attention_mask_array.shape}")

# convert the imported attention mask into a 1d mask
attention_mask_array_1d: np.ndarray = attention_mask_array.flatten()
print(f"attention_mask_array_1d shape: {attention_mask_array_1d.shape}")

# Convert the 1D patch mask into a 2D matrix (num_patches, num_patches)
# A query patch can only attend to a key patch if BOTH are valid.
mask_2d = tf.logical_and(
    # sample_mask_1d[:, tf.newaxis] == 1,
    attention_mask_array_1d[:, tf.newaxis] == 1,
    # sample_mask_1d[tf.newaxis, :] == 1
    attention_mask_array_1d[tf.newaxis, :] == 1
)

print(f"mask_2d:\n{mask_2d}")

print(f"mask_2d shape: {mask_2d.shape}")

# 3. Initialize and Run the MultiHeadAttention Layer
attention_layer = layers.MultiHeadAttention(num_heads=num_heads, key_dim=projection_dim)

# Pass the boolean mask to the attention_mask argument during execution
attention_output = attention_layer(
    query=patch_embeddings,
    value=patch_embeddings,
    key=patch_embeddings,  # Optional: defaults to value if omitted
    attention_mask=mask_2d  # Shape: (16, 16) - automatically broadcasted to batches
)

print("Input shape:", patch_embeddings.shape)
print("Mask shape:", mask_2d.shape)
print("Output shape:", attention_output.shape)

# error when using a 16 x 16 attention mask:
# ValueError: Exception encountered when calling layer 'softmax' (type Softmax).
# 
# Dimensions must be equal, but are 256 and 16 for '{{node multi_head_attention/softmax/add}} = AddV2[T=DT_HALF](multi_head_attention/einsum/Einsum, multi_head_attention/softmax/mul)' with input shapes: [?,16,256,256], [1,1,16,16].
# 
# Call arguments received by layer 'softmax' (type Softmax):
#   • inputs=tf.Tensor(shape=(None, 16, 256, 256), dtype=float16)
#   • mask=tf.Tensor(shape=(1, 1, 16, 16), dtype=bool)

# example code gives:
# sequence lengths: [12]
# sample_mask_1d shape: (256,)
# sample_mask_1d:
# [1 1 1 1 1 1 1 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0...0 0 0 0] # length 256
# mask_2d:
# [[ True  True  True ... False False False]
#  [ True  True  True ... False False False]
#  [ True  True  True ... False False False]
#  ...
#  [False False False ... False False False]
#  [False False False ... False False False]
#  [False False False ... False False False]]
# mask_2d shape: (256, 256)
# Input shape: (256, 256, 64)
# Mask shape: (256, 256)
# Output shape: (256, 256, 64)
