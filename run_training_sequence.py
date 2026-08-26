# configure and run a sequence of training runs
from datetime import datetime

import ultrasound_gesture_vit_classification

file_datetime = datetime.now().strftime("%Y%m%d_%H%M%S")

epochs_count: int = 1
attn_bias_strength: float = 0.75
learning_rate: float = 0.0001
subject_id: int = 4
mode: str = 'perp'

selected_args: list = [
    '--root', '/Users/rickgladwin/Code/u_of_hull/dissertation/bimbraw_2025_dataset/data/',
    '--mode', mode,
    '--subject', f'Subject_{subject_id}',
    '--image-size', '224',
    '--patch-size', '14',
    '--projection-dim', '64',
    '--num-heads', '16',
    '--num-layers', '8',
    '--epochs', str(epochs_count), # variable
    '--batch-size', '256',
    '--seed', '42',
    '--val-split', '0.1',
    '--lr', str(learning_rate),
    '--weight-decay', '0.001',
    '--progress', 'none',
    '--load-model', '',
    '--save-model', f'results/models/vit/vit_{mode}_subject_{subject_id}_{epochs_count}_epochs_224px_14_patch_size_attn_True_{file_datetime}.keras',
    '--out', f'results/metrics/vit/metrics_vit_subject_{subject_id}_{epochs_count}_epochs_{file_datetime}.json',
    '--cm', f'results/figs/vit/cm_vit_subject_{subject_id}_{epochs_count}_epochs_{file_datetime}.png',
    '--output-attention-maps', 'True',
    '--output-attention-map-series', 'False',
    '--attn-map-series-folder', f'results/attention/vit/series/{file_datetime}/',
    '--attn-map', f'results/figs/vit/attn_vit_subject_{subject_id}_{epochs_count}_epochs_{file_datetime}.png',
    '--apply-attention-mask', 'False',
    '--applied-attention-mask-path', '/Users/rickgladwin/Code/u_of_hull/dissertation/Integrated-Decision-Gradients/results/attention_maps/attn_boolean_map_threshold=0p45_20260817_005014.npy',
    '--apply-patch-bias-attention-map', 'False',
    '--applied-patch-bias-attention-map-path', '',
    '--apply-attention-map', 'True',
    '--applied-attention-map-path', '/Users/rickgladwin/Code/u_of_hull/dissertation/Integrated-Decision-Gradients/results/attribution_maps/attrib_map_combo_norm_20260817_005014.npy',
    '--attention-mask-layer-application', '1,0,0,0,0,0,0,0',
    '--attention-bias-type', 'attn_bias_combo',
    '--attention-bias-strength', str(attn_bias_strength),
]

# run main training loop from the other module
ultrasound_gesture_vit_classification.main(selected_args)


