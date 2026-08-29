# configure and run a sequence of training runs
import json
from datetime import datetime, timedelta

import ultrasound_gesture_vit_classification


# attn_bias_strengths: list[float] = [ 0.00, 0.10, 0.25, 0.50, 0.75, 1.00, 2.00, 4.00, 6.00, 10.00, 15.00, 20.00 ]
# attn_bias_strengths: list[float] = [ 0.00, 0.10, 0.25 ]
# attn_bias_strengths: list[float] = [ 0.25, 0.50, 0.75, 1.00, 2.00, 4.00, 6.00, 10.00, 15.00, 20.00 ]

# set training parameter value sets
# attn_bias_types: list[str] = [ 'attn_bias_sigmoid', 'attn_bias_combo' ]
# attn_bias_types: list[str] = [ 'attn_bias_combo' ]
attn_bias_types: list[str] = [ 'attn_bias_sigmoid' ]

attn_bias_types_and_strengths: dict[str, list[float]] = {
    # 'attn_bias_sigmoid': [ 0.75, 1.0, 6.0, 10.0, 15.0, 20.0 ],
    # 'attn_bias_combo':   [ 0.10, 0.75, 1.0, 6.0, 10.0, 20.0 ],
    # 'attn_bias_sigmoid': [ 0.75, 1.0 ],
    # 'attn_bias_combo':   [ 0.10 ],
    # 'attn_bias_sigmoid': [ 0.75 ],
    # 'attn_bias_combo':   [ 0.10, 0.75 ],
    # 'attn_bias_combo': [ x / 100 for x in range(200, 610, 25) ],
    # 'attn_bias_sigmoid': [ 0.25, 0.50, 0.75, 1.00, 2.00, 4.00, 6.00, 10.00, 15.00, 20.00 ],
    # 'attn_bias_sigmoid': [ 0.75, 1.00, 6.00, 10.00, 15.00, 20.00 ],
    'attn_bias_sigmoid': [
        0.00, 0.00, 0.00,
        0.01, 0.01, 0.01,
        0.10, 0.10, 0.10,
        0.25, 0.25, 0.25,
        # 0.50, 0.50, 0.50,
        # 0.75, 0.75, 0.75,
        # 1.00, 1.00, 1.00,
        2.00, 2.00, 2.00,
        4.00, 4.00, 4.00,
        6.00, 6.00, 6.00,
        8.00, 8.00, 8.00,
    ],
    # 'attn_bias_sigmoid': [ x / 100 for x in range(200, 610, 25) ],
    # 'attn_bias_sigmoid': [ 0.00, 0.10, 0.25, 0.50, 0.75, 1.00, 2.00, 4.00, 6.00, 10.00, 15.00, 20.00 ],
    # 'attn_bias_combo':   [ 0.00, 0.10, 0.25, 0.50, 0.75, 1.00, 2.00, 4.00, 6.00, 10.00, 15.00, 20.00 ],
}

# epoch_counts: list[int] = [ 1, 5, 20 ]
# epoch_counts: list[int] = [ 1 ]
epoch_counts: list[int] = [ 150 ]

# 200 epochs * 2 bias types * 6 attention bias strengths * 8s/epoch == 9600s == 160m == 2h40m

# epochs_count: int = 200
# attn_bias_type: str = 'attn_bias_combo'
# attn_bias_type: str = 'attn_bias_sigmoid'
# learning_rate: float = 0.0001
learning_rate: float = 0.00005
subject_id: int = 4
mode: str = 'perp'
item_size: int = 224

# example results schema:
# {
#   'attn_bias_sigmoid': [
#    {
#       'epochs': 200,
#       'attn_bias_strength': 0.75,
#       'test_accuracy': 0.0875
#    },
#    {
#       'epochs': 200,
#       'attn_bias_strength': 1.0,
#       'test_accuracy': 0.0775
#    },
# ]

test_batch_datetime: str = datetime.now().strftime("%Y%m%d_%H%M%S")

results: dict[str, list[dict[str, float|int|str]]] = {}

for epochs_count in epoch_counts:
    for attn_bias_type, attn_bias_strengths in attn_bias_types_and_strengths.items():
        
        if attn_bias_type not in results:
            results[attn_bias_type] = []
        
        for attn_bias_strength in attn_bias_strengths:
            print(f'------------- running training with attn_bias_strength: {attn_bias_strength} for {attn_bias_type} ------------')
            
            training_run_start_datetime: datetime = datetime.now()
            
            file_datetime = datetime.now().strftime("%Y%m%d_%H%M%S")
        
            if attn_bias_type == "attn_bias_sigmoid":
                apply_attention_map: bool = True
                applied_attention_map_filepath: str = f"/Users/rickgladwin/Code/u_of_hull/dissertation/Integrated-Decision-Gradients/results/attribution_maps/attrib_map_combo_sigmoid_x0=0p65_w=0p15_20260817_005014.npy"
            elif attn_bias_type == "attn_bias_combo":
                apply_attention_map: bool = True
                applied_attention_map_filepath: str = f"/Users/rickgladwin/Code/u_of_hull/dissertation/Integrated-Decision-Gradients/results/attribution_maps/attrib_map_combo_norm_20260817_005014.npy"
            else:
                raise ValueError(f'Invalid or missing attn_bias_type: {attn_bias_type}')
        
            selected_args: list = [
                '--root', '/Users/rickgladwin/Code/u_of_hull/dissertation/bimbraw_2025_dataset/data/',
                '--mode', mode,
                '--subject', f'Subject_{subject_id}',
                '--image-size', str(item_size),
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
                '--save-model', f'results/models/vit/vit_{mode}_subject_{subject_id}_{epochs_count}_epochs_{item_size}px_14_patch_size_attn_True_{file_datetime}.keras',
                '--out', f'results/metrics/vit/metrics_vit_subject_{subject_id}_{epochs_count}_epochs_{file_datetime}.json',
                '--cm', f'results/figs/vit/cm_vit_subject_{subject_id}_{epochs_count}_epochs_{file_datetime}.png',
                # '--output-attention-maps', 'True',
                # '--output-attention-map-series', 'True',
                '--attn-map-series-folder', f'results/attention/vit/series/{file_datetime}/',
                '--attn-map', f'results/figs/vit/attn_vit_subject_{subject_id}_{epochs_count}_epochs_{file_datetime}.png',
                # '--apply-attention-mask', 'True',
                # '--applied-attention-mask-path', '/Users/rickgladwin/Code/u_of_hull/dissertation/Integrated-Decision-Gradients/results/attention_maps/attn_boolean_map_threshold=0p45_{file_datetime}.npy',
                # '--apply-patch-bias-attention-map', 'True',
                # '--applied-patch-bias-attention-map-path', '',
                '--apply-attention-map', 'True',
                '--applied-attention-map-path', f'{applied_attention_map_filepath}',
                '--attention-mask-layer-application', '1,0,0,0,0,0,0,0',
                '--attention-bias-type', attn_bias_type,
                '--attention-bias-strength', str(attn_bias_strength),
            ]
        
            # run main training loop from the other module
            test_accuracy_result: float = ultrasound_gesture_vit_classification.main(selected_args)
            
            training_run_end_datetime: datetime = datetime.now()
            training_run_duration: timedelta = training_run_end_datetime - training_run_start_datetime
            
            # example results schema:
            # {
            #   'attn_bias_sigmoid': [
            #    {
            #       'epochs': 200,
            #       'attn_bias_strength': 0.75,
            #       'test_accuracy': 0.0875
            #    },
            #    {
            #       'epochs': 200,
            #       'attn_bias_strength': 1.0,
            #       'test_accuracy': 0.0775
            #    },
            # ]
            training_run_results: dict[str, float|int|str] = {
                'epochs': epochs_count,
                'attn_bias_strength': attn_bias_strength,
                'training_run_duration': str(training_run_duration),
                'test_accuracy': test_accuracy_result,
            }
            
            results[attn_bias_type].append(training_run_results)
            
            print(f'------------- completed training with {training_run_results} ------------')

results_batch_filepath: str = f'results/batches/batch_{test_batch_datetime}.json'
print(f'saving results to {results_batch_filepath}')

with open(results_batch_filepath, 'w') as f:
    json.dump(results, f)



