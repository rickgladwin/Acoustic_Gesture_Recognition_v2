import json
import os
import platform
import random
import subprocess

import numpy as np
import tensorflow as tf


def get_mac_system_info() -> dict:
    system_details: dict[str, str] = {}

    # Query the exact Mac processor name using sysctl
    try:
        cmd = ["sysctl", "-n", "machdep.cpu.brand_string"]
        processor = subprocess.check_output(cmd).decode().strip()
    except Exception:
        processor = platform.processor()

    system_details['processor_type'] = processor

    logical_cpu_cores_count = subprocess.check_output(["sysctl", "-n", "hw.logicalcpu"]).decode().strip()

    system_details['cpu_cores'] = logical_cpu_cores_count

    # Queries the Mac I/O Registry for core-count allocations
    gpu_cores_raw = subprocess.check_output("ioreg -l | grep gpu-core-count", shell=True).decode()
    # Extracts the digit assignment from the property string
    gpu_cores = [s for s in gpu_cores_raw.split() if s.isdigit()][0]

    system_details['gpu_cores'] = gpu_cores

    return system_details


def is_apple_silicon():
    # Direct check (Returns 'arm64' if running natively)
    if platform.system() == "Darwin" and platform.machine() == "arm64":
        return True

    # Deep check (Catches Apple Silicon even when emulated via Rosetta)
    if platform.system() == "Darwin":
        try:
            brand = subprocess.check_output(["sysctl", "-n", "machdep.cpu.brand_string"]).decode().strip()
            if "Apple" in brand:
                return True
        except Exception:
            pass

    return False


def set_seed(seed: int):
    """Make runs reproducible across numpy and TF."""
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def ensure_dir(path_or_file):
    """Create parent directory for a file path if it does not exist."""    
    if not path_or_file:
        return
    d = path_or_file if os.path.isdir(path_or_file) else os.path.dirname(path_or_file)
    if d:
        os.makedirs(d, exist_ok=True)


def dump_json(obj, path, indent_spaces: int=2, char_encoding: str="utf-8"):
    """Dump a JSON file with custom indentation."""
    print(f"-- saving JSON results to '{path}'...")    
    if not path:
        return
    ensure_dir(path)
    with open(path, "w", encoding=char_encoding) as f:
        json.dump(obj, f, indent=indent_spaces)


def process_pool_size(reserve_cores_count: int=2, verbose: bool=False) -> int:
    """
    Returns the number of processes to use for a ProcessPoolExecutor.
    Defaults to the number of CPU cores available minus 2.
    Returns 1 if there are fewer than <reserve_cores_count> cores or the number of cores cannot be determined.
    :param reserve_cores_count: number of cores to reserve for other processes (exclude this many cores from the process pool)
    :param verbose: whether to print debug messages to the console
    :return: number of processes to use for a ProcessPoolExecutor
    """
    cpu_count: int|None = os.cpu_count()
    print(f"os.cpu_count(): {cpu_count}") if verbose else None
    process_pool_size = (cpu_count - reserve_cores_count) if cpu_count and cpu_count > reserve_cores_count else 1
    print(f"using process pool size {process_pool_size}") if verbose else None
    return process_pool_size


def convert_keras_model_for_tensorflow(model: tf.keras.Model) -> tf.keras.Model:
    """
    Converts a Keras model to a TensorFlow model.
    :param model: Keras model to convert
    :return: TensorFlow model
    """
    return tf.keras.models.clone_model(model)
