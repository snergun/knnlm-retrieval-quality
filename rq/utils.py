import time
import os
import shutil
import numpy as np

def log_progress(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

def copy_to_tmp(source_path, shape, dtype):
    """Copy a memmap file to /tmp for faster access"""
    filename = os.path.basename(source_path)
    dir_path = os.path.dirname(source_path)
    tmp_dir = os.path.join('/tmp', os.path.basename(dir_path))
    os.makedirs(tmp_dir, exist_ok=True)
    tmp_path = os.path.join(tmp_dir, filename)
    
    if not os.path.exists(tmp_path):
        log_progress(f"Copying {source_path} to {tmp_path}")
        shutil.copy2(source_path, tmp_path)
    else:
        log_progress(f"File already exists: {tmp_path}")
    
    return np.memmap(tmp_path, dtype=dtype, mode='r', shape=shape)