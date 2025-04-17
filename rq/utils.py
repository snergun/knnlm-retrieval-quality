import time
import os
import shutil
import numpy as np
import wandb

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

def set_presets(args):
    if args.preset is None:
        args.preset = 'wiki_valid'

    if args.preset == 'wiki_valid':
        args.vocab = 'data-bin/wikitext-103/dict.txt'
        args.dstore = 'datastore/wikitext-103/train'
        args.external_lm_prob = '/u/eergun/pos_lm/misc/ADT_validation_results.pt'
        args.pos_prob = "/u/eergun/pos_lm/misc/ADTPOS_validation_results.pt"
        args.dstore_size = 103225485
        args.eval_dstore = 'datastore/wikitext-103/valid'
        args.eval_dstore_cache = 'datastore/wikitext-103/valid.cache'
        args.eval_dstore_size = 217646
        args.eval_external_knns = False

    if args.preset == 'wiki_test':
        args.vocab = 'data-bin/wikitext-103/dict.txt'
        args.dstore = 'datastore/wikitext-103/train'
        args.external_lm_prob = '/u/eergun/pos_lm/misc/ADT_test_results.pt'
        args.pos_prob = "/u/eergun/pos_lm/misc/ADTPOS_test_results.pt"
        args.dstore_size = 103225485
        args.eval_dstore = 'datastore/wikitext-103/test'
        args.eval_dstore_cache = 'datastore/wikitext-103/test.cache'
        args.eval_dstore_size = 245569
        args.eval_external_knns = False

    if args.preset == 'ptb_valid':
        args.vocab = 'data-bin/ptb/dict.txt'
        args.dstore = './work_data/ptb.train'
        args.dstore_size = 1003610
        args.eval_dstore = './work_data/ptb.valid'
        args.eval_dstore_cache = './work_data/ptb.valid.cache'
        args.eval_dstore_size = 42355
        args.eval_external_knns = False

    args.dstore_knn_index = f'{args.dstore}/knn.index'

def setup_wandb(args):
    if args.use_wandb:
        wandb_config = {
            'args': vars(args),
        }
        wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=args.wandb_name,
            config=wandb_config
        )
        log_progress("Initialized wandb logging")