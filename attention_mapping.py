# imports and processes attribution and attention maps and attention masks
import enum
import numpy as np
from matplotlib import pyplot as plt
import tensorflow as tf


class AttentionArrayType(enum.Enum):
    """
    Enum for attention array types.
    MASK type has boolean values.
    MAP type has float values in [0, 1].
    """
    MAP = 'map'
    MASK = 'mask'


def import_attention_mask(path: str, show_plot: bool=False) -> np.ndarray:
    attention_mask: np.ndarray = np.load(path)
    # confirm attention mask contains only boolean values
    assert np.all(attention_mask == attention_mask.astype(bool)), "Attention mask contains non-boolean values"
    
    if show_plot:
        display_plot(attention_mask, AttentionArrayType.MASK, path)
        
    return attention_mask


def import_normalized_attention_map(path: str, show_plot: bool=False) -> np.ndarray:
    attention_map: np.ndarray = np.load(path)
    # confirm attention map contains only float values in [0, 1]
    assert np.all(attention_map == attention_map.astype(float)), "Attention map contains non-float values"
    assert np.all(attention_map >= 0) and np.all(attention_map <= 1), "Attention map contains values outside of [0, 1]"
    
    if show_plot:
        display_plot(attention_map, AttentionArrayType.MAP, path)
    
    return attention_map


def display_plot(attention_object: np.ndarray, attention_object_type: AttentionArrayType, source_filepath: str) -> None:
    title_prefix: str
    plot_cmap: str
    
    match attention_object_type:
        case AttentionArrayType.MAP:
            title_prefix = 'Attention Map'
            plot_cmap = 'inferno'
        case AttentionArrayType.MASK:
            title_prefix = 'Attention Mask'
            plot_cmap = 'gray'
        case _:
            raise ValueError(f"Invalid attention object type: {attention_object_type}")
            
    title: str = f"{title_prefix}: {source_filepath.split('/')[-1]}"
    plt.title(title, wrap=True)
    plt.xticks([x for x in range(attention_object.shape[1])])
    plt.xlabel('patch column')
    plt.yticks([x for x in range(attention_object.shape[0])])
    plt.ylabel('patch row')
    plt.imshow(attention_object, cmap=plot_cmap)
    plt.show()
    

def attention_tensor_from_2d_mask(attention_mask_array: np.ndarray) -> tf.Tensor:
    """
    Converts a 2D attention mask into a 3D attention tensor for use in a MultiHeadAttention layer.
    NOTE: Uses broadcasting with the assumption that the attention mask for every image in the batch uses the same attention mask.
    """
    # convert the NxN attention mask array into a length N**2 1D array 
    attention_mask_array_1d: np.ndarray = attention_mask_array.flatten()
    print(f"attention_mask_array_1d shape: {attention_mask_array_1d.shape}")
    print(f"attention_mask_array_1d:\n{attention_mask_array_1d}")
    
    
    # assume the 1D attention mask applies to every image in the batch, therefore we can use broadcasting
    # and repeat the 1D attention mask array B times, where B is the batch size.
    # The `logical_and` call creates a self-attention tensor.
    attention_tensor: tf.Tensor = tf.logical_and(
        attention_mask_array_1d[:, tf.newaxis] == 1,
        attention_mask_array_1d[tf.newaxis, :] == 1
    )
    plt.imshow(attention_tensor, cmap='gray', vmin=0, vmax=1)
    plt.title("(patches_per_dim\u00B2) x (patches_per_dim\u00B2) attention tensor", wrap=True)
    plt.xticks([16*x for x in range(17)], rotation=90)
    plt.yticks([16*x for x in range(17)])
    plt.show()
    return attention_tensor
    

if __name__ == "__main__":
    # in-place tests
    # There are 4 types of attention maps and masks made from attribution maps:
    # - normalized attention map from combined attribution maps for all gestures
    # - normalized sigmoid-filtered attention map from combined attribution maps for all gestures
    # - normalized boolean-filtered attention mask from combined attribution maps for all gestures
    # - normalized high-pass boolean filtered attention mask from combined attribution maps for all gestures
    
    ## thresholded attention mask from attribution map
    test_attention_mask_filepath: str = '/Users/rickgladwin/Code/u_of_hull/dissertation/Integrated-Decision-Gradients/results/attention_maps/attn_boolean_map_threshold=0p45_20260817_005014.npy'
    print(f"importing {test_attention_mask_filepath}")
    imported_attn_mask: np.ndarray = import_attention_mask(test_attention_mask_filepath, show_plot=True)
    attn_tensor: tf.Tensor = attention_tensor_from_2d_mask(imported_attn_mask)
    
    # example "all patches" attention mask
    all_patches_mask: np.ndarray = np.ones((16, 16), dtype=bool)
    all_patches_attn_tensor: tf.Tensor = attention_tensor_from_2d_mask(all_patches_mask)
    
    ## high pass attention mask from attribution map
    # NOTE: this npy file will need to be converted to boolean values before use.
    # This example file will raise and error.
    # test_attention_mask_filepath: str = '/Users/rickgladwin/Code/u_of_hull/dissertation/Integrated-Decision-Gradients/results/attention_maps/attn_high_pass_map_threshold=0p45_20260817_005014.npy'
    # print(f"importing {test_attention_mask_filepath}")
    # import_attention_mask(test_attention_mask_filepath, show_plot=True)
    
    ## sigmoid filtered attention map from attribution map
    test_attention_map_filepath: str = '/Users/rickgladwin/Code/u_of_hull/dissertation/Integrated-Decision-Gradients/results/attribution_maps/attrib_map_combo_sigmoid_x0=0p65_w=0p15_20260817_005014.npy'
    print(f"importing {test_attention_map_filepath}")
    import_normalized_attention_map(test_attention_map_filepath, show_plot=True)
    
    ## normalized combined attention map from attribution map
    test_attention_map_filepath: str = '/Users/rickgladwin/Code/u_of_hull/dissertation/Integrated-Decision-Gradients/results/attribution_maps/attrib_map_combo_norm_20260817_005014.npy'
    print(f"importing {test_attention_map_filepath}")
    import_normalized_attention_map(test_attention_map_filepath, show_plot=True)
