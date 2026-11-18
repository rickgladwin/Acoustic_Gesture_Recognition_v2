# parse and compile result data
import glob
import json
import pprint as pp


def load_training_history(path: str) -> dict:
    """
    Loads a training history from a JSON file at the given path.

    Args:
        path (str): The path to the JSON file containing the training history.

    Returns:
        dict: The loaded training history as a dictionary.
    """
    with open(path, 'r') as f:
        return json.load(f)


def load_metrics(path: str) -> dict:
    """
    Loads metrics from a JSON file at the given path.

    Args:
        path (str): The path to the JSON file containing the metrics.

    Returns:
        dict: The loaded metrics as a dictionary.
    """
    
    with open(path, 'r') as f:
        return json.load(f)
    
    
def datetime_id_from_filepath(filepath: str) -> str:
    # 
    # filepath examples:
    # history_vit_subject_4_150_epochs_bias_attn_bias_sigmoid_strength_4.0_20260907_102253.json
    # metrics_vit_subject_4_150_epochs_20260907_100300.json
    file_extension: str = filepath.split('.')[-1]
    filepath_without_extension: str = filepath.replace(f'.{file_extension}', '')
    datetime_id: str = '_'.join(filepath_without_extension.split('_')[-2:])
    return datetime_id


def test_accuracy_for_run(datetime_id: str) -> float:
    metrics_folder_path: str = 'results/metrics/vit/'
    metrics_filename_glob: str = f'*_{datetime_id}.json'
    
    metrics_filepath: str = glob.glob(f'{metrics_folder_path}{metrics_filename_glob}')[0]
    metrics: dict = load_metrics(metrics_filepath)
    return metrics['training_details']['test_accuracy']


if __name__ == '__main__':
    training_history_folder: str = 'results/training_history/vit/'
    # example filepath
    # history_vit_subject_4_150_epochs_bias_attn_bias_sigmoid_strength_0.0_20260907_084620.json
    # training_history_filename_prefix: str = 'history_vit_subject_4_150_epochs_bias_attn_bias_sigmoid_strength_0.0_20260907_'
    # training_history_filename_prefix: str = 'history_vit_subject_4_150_epochs_bias_attn_bias_sigmoid_strength_4.0_20260907_'
    # training_history_filename_prefix: str = 'history_vit_subject_4_150_epochs_bias_attn_bias_combo_strength_4.0_20260907_'
    # training_history_filename_prefix: str = 'history_vit_subject_4_150_epochs_bias_attn_bias_combo_strength_4.0_20260908_'
    # training_history_filename_prefix: str = 'history_vit_subject_4_150_epochs_bias_attn_bias_combo_strength_0.25_20260912_'
    # training_history_filename_prefix: str = 'history_vit_subject_4_50_epochs_bias_attn_bias_combo_strength_0.25_20260911_'
    # training_history_filename_prefix: str = 'history_vit_subject_4_50_epochs_bias_attn_bias_combo_strength_4.0_20260911_'
    # training_history_filename_prefix: str = 'history_vit_subject_4_50_epochs_bias_attn_bias_sigmoid_strength_4.0_20260911_'
    # training_history_filename_prefix: str = 'history_vit_subject_4_50_epochs_bias_attn_bias_sigmoid_strength_0.0_20260910_'
    # training_history_filename_prefix: str = 'history_vit_subject_4_50_epochs_bias_attn_bias_sigmoid_strength_0.25_20260911_'
    # training_history_filename_prefix: str = 'history_vit_subject_4_50_epochs_bias_attn_bias_sigmoid_strength_0.25_20260910_23'
    # training_history_filename_prefix: str = 'history_vit_subject_4_150_epochs_bias_attn_bias_sigmoid_strength_0.25_20260912'
    # training_history_filename_prefix: str = 'history_vit_subject_4_250_epochs_bias_attn_bias_sigmoid_strength_4.0_20260910'
    # training_history_filename_prefix: str = 'history_vit_subject_4_250_epochs_bias_attn_bias_combo_strength_4.0_20260910'
    # training_history_filename_prefix: str = 'history_vit_subject_4_250_epochs_bias_attn_bias_sigmoid_strength_0.0_20260909'
    # training_history_filename_prefix: str = 'history_vit_subject_4_250_epochs_bias_attn_bias_sigmoid_strength_0.0_20260910'
    training_history_filename_prefix: str = 'history_vit_subject_4_250_epochs_bias_attn_bias_sigmoid_strength_0.0_20260913'
    # training_history_filename_prefix: str = 'history_vit_subject_4_250_epochs_bias_attn_bias_sigmoid_strength_4.0_20260910'
    training_history_filepath_glob: str = f'{training_history_folder}{training_history_filename_prefix}*.json'
    print(f'training_history_filepath_glob:\n{training_history_filepath_glob}')
    training_history_filepaths: list[str] = glob.glob(training_history_filepath_glob)
    training_history_filepaths.sort()
    
    print(f'filepaths:')
    print(f'n={len(training_history_filepaths)}\n')
    pp.pprint(training_history_filepaths)
    
    first_history_filepath: str = training_history_filepaths[0]
    first_history: dict = load_training_history(first_history_filepath)
    print(f'first_history:')
    pp.pprint(first_history)
    
    first_datetime_id: str = datetime_id_from_filepath(first_history_filepath)
    print(f'first_datetime_id:\n{first_datetime_id}')
    
    first_test_accuracy: float = test_accuracy_for_run(first_datetime_id)
    print(f'first_test_accuracy:\n{first_test_accuracy}')
    
    test_accuracies: list[float] = []
    
    for history_filepath in training_history_filepaths:
        datetime_id: str = datetime_id_from_filepath(history_filepath)
        test_accuracy: float = test_accuracy_for_run(datetime_id)
        test_accuracies.append(test_accuracy)
    
    print(f'test accuracies for {training_history_filepath_glob}\nn={len(test_accuracies)}\n----\n')
    for test_accuracy in test_accuracies:
        print(test_accuracy)
