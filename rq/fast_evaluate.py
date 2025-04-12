"""
Basic Instructions:

  1. Cache the approximate retrieval. `python rq/fast_evaluate.py --preset wiki_valid --save-knns`
  2. Cache the exact vector distance. `python rq/fast_evaluate.py --preset wiki_valid --save-exact`
  3. Compute perplexity with exact distance. `python rq/fast_evaluate.py --preset wiki_valid --exact`

"""
import collections
import json
import os
import math
import sys
import time
import argparse
import shutil

from utils import log_progress, copy_to_tmp

log_progress("Starting imports")
# Time individual imports
import numpy as np
import torch
from tqdm import tqdm
from data_structures import Dataset, Dstore
from vocab import Dictionary
import knnlm_func
log_progress("Imports completed")


def argument_parser():
    parser = argparse.ArgumentParser()

    # Filepaths.
    parser.add_argument('--vocab', default=None, type=str,
                        help='Path to vocab file.')
    parser.add_argument('--dstore', default=None, type=str,
                        help='Path to dstore.')
    parser.add_argument('--dstore-size', default=None, type=int)
    parser.add_argument('--eval-dstore', default=None, type=str,
                        help='Path to precomputed evaluation information. Similar to dstore, but for evaluation.')
    parser.add_argument('--eval-dstore-size', default=None, type=int)
    parser.add_argument('--eval-dstore-cache', default=None, type=str,
                        help='Path to additional evaluation information.')
    parser.add_argument('--eval-external-knns', default=None, type=str,
                        help='If set, then override the kNNs that would have been returned from faiss.')

    # Algorithm configuration.
    parser.add_argument('--k', default=1024)
    parser.add_argument('--exact', action='store_true',
                        help='If set, then use the exact distances (these should be cached with --save-exact).')
    parser.add_argument('--from_cache', action='store_true',
                        help='Set if evaluating from cached neighbors, values and probs.')
    parser.add_argument('--validation-split', action='store_true',
                    help='If set, split validation data for two-stage parameter tuning.')
    # Commands.
    parser.add_argument('--save-knns', action='store_true')
    parser.add_argument('--save-exact', action='store_true')

    # Preset configuration.
    parser.add_argument('--preset', default=None, type=str,
                        help='Use a preset configuration for different datasets.')

    # Hardware specific.
    parser.add_argument('--load-pct', default=10, type=int,
                        help='Should be [0,100] corresponding to percent of keys to load in mem.')
    parser.add_argument('--cuda', action='store_true')

    return parser


def set_presets(args):
    if args.preset is None:
        args.preset = 'wiki_valid'

    if args.preset == 'wiki_valid':
        args.vocab = 'data-bin/wikitext-103/dict.txt'
        args.dstore = 'datastore/wikitext-103/train'
        args.dstore_size = 103225485
        args.eval_dstore = 'datastore/wikitext-103/valid'
        args.eval_dstore_cache = 'datastore/wikitext-103/valid.cache'
        args.eval_dstore_size = 217646

    if args.preset == 'wiki_test':
        args.vocab = 'data-bin/wikitext-103/dict.txt'
        args.dstore = 'datastore/wikitext-103/train'
        args.dstore_size = 103225485
        args.eval_dstore = 'datastore/wikitext-103/test'
        args.eval_dstore_cache = 'datastore/wikitext-103/test.cache'
        args.eval_dstore_size = 245569

    if args.preset == 'ptb_valid':
        args.vocab = 'data-bin/ptb/dict.txt'
        args.dstore = './work_data/ptb.train'
        args.dstore_size = 1003610
        args.eval_dstore = './work_data/ptb.valid'
        args.eval_dstore_cache = './work_data/ptb.valid.cache'
        args.eval_dstore_size = 42355

    args.dstore_knn_index = f'{args.dstore}/knn.index'


#
# Serialization Methods
#

def save_knns(args, dataset, dstore):
    cache = collections.defaultdict(list)

    batch_size = 128

    print('Precomputing neighbors...')
    for start in tqdm(range(0, dataset.query.shape[0], batch_size)):
        end = min(start + batch_size, dataset.query.shape[0])

        query, target = dataset.query[start:end], dataset.target[start:end]
        dists, knns = dstore.get_knns(query)
        cache['dists'].append(dists)
        cache['knns'].append(knns)

    os.system(f'mkdir -p {args.eval_dstore_cache}')

    dists = np.concatenate(cache['dists'], 0)
    knns = np.concatenate(cache['knns'], 0)

    # Save to /tmp first
    tmp_cache_dir = os.path.join('/tmp', os.path.basename(args.eval_dstore_cache))
    tmp_dists_path = os.path.join(tmp_cache_dir, 'dstore_cache_dists.npy')
    tmp_knns_path = os.path.join(tmp_cache_dir, 'dstore_cache_knns.npy')
    
    dstore_dists = np.memmap(tmp_dists_path, dtype=np.float32, mode='w+', shape=dists.shape)
    dstore_dists[:] = dists
    dstore_knns = np.memmap(tmp_knns_path, dtype=np.int32, mode='w+', shape=knns.shape)
    dstore_knns[:] = knns
    
    # Then copy back to original location
    orig_cache_dir = args.eval_dstore_cache
    os.makedirs(orig_cache_dir, exist_ok=True)
    
    orig_dists_path = os.path.join(orig_cache_dir, 'dstore_cache_dists.npy')
    orig_knns_path = os.path.join(orig_cache_dir, 'dstore_cache_knns.npy')
    
    log_progress(f"Copying {tmp_dists_path} to {orig_dists_path}")
    shutil.copy2(tmp_dists_path, orig_dists_path)
    
    log_progress(f"Copying {tmp_knns_path} to {orig_knns_path}")
    shutil.copy2(tmp_knns_path, orig_knns_path)

def save_exact(args, dataset, dstore):

    keys = dstore.keys
    vals = dstore.vals
    query = dataset.query
    target = dataset.target
    knns = dataset.knns
    scores = dataset.dists

    new_dist = np.ones(scores.shape, dtype=scores.dtype)

    def run_block(start_k, end_k):
        batch_size = 1024
        in_mem_keys = np.empty(shape=(end_k - start_k, keys.shape[1]), dtype=keys.dtype)
        for start in tqdm(range(0, in_mem_keys.shape[0], batch_size), desc='load-pct'):
            end = min(start + batch_size, in_mem_keys.shape[0])
            in_mem_keys[start:end] = keys[start_k + start: start_k + end]

        batch_size = 128

        # TODO: GPU usage is low. Try using dataloader?
        for start in tqdm(range(0, query.shape[0], batch_size), desc='exact'):
            end = min(start + batch_size, query.shape[0])

            q_vecs = torch.from_numpy(query[start:end]).float().cuda()
            k_idx = knns[start:end]

            batch_keys = np.zeros(shape=(k_idx.shape[0], k_idx.shape[1], keys.shape[1]), dtype=keys.dtype)
            batch_mask = np.logical_and(k_idx >= start_k, k_idx < end_k)
            batch_keys[batch_mask] = in_mem_keys[k_idx[batch_mask] - start_k]

            # TODO: This is doing a lot of extra work, since many keys are blank.
            k_vecs = torch.from_numpy(batch_keys).float().cuda()
            d = -torch.sum((q_vecs[:, None, :] - k_vecs)**2, 2)

            d = d.cpu().numpy()
            batch_dist = new_dist[start:end].copy()
            batch_dist[batch_mask] = d[batch_mask]

            new_dist[start:end] = batch_dist

    block_size = int(args.load_pct / 100 * keys.shape[0])
    num_blocks = math.ceil(100 / args.load_pct)
    print(f'num_blocks = {num_blocks}')
    for i in range(num_blocks):
        start_k = i * block_size
        end_k = start_k + block_size
        if i == num_blocks - 1:
            end_k = keys.shape[0]
        assert start_k < end_k
        header = '\n\n' + '*' * 20 + f'{i * args.load_pct}/100 ({i}/{num_blocks})' + '*'*20 + '\n'
        header += f'slice = {start_k}:{end_k} of {keys.shape[0]}\n'
        header += '\n\n'
        print(header)
        run_block(start_k, end_k)

    # Save to /tmp first
    tmp_cache_dir = os.path.join('/tmp', os.path.basename(args.eval_dstore_cache))
    os.makedirs(tmp_cache_dir, exist_ok=True)
    tmp_exact_dists_path = os.path.join(tmp_cache_dir, 'dstore_cache_exact_dists.npy')
    
    dstore_exact_dists = np.memmap(tmp_exact_dists_path, dtype=np.float32, mode='w+', shape=scores.shape)
    dstore_exact_dists[:] = new_dist
    
    # Then copy back to original location
    orig_cache_dir = args.eval_dstore_cache
    os.makedirs(orig_cache_dir, exist_ok=True)
    orig_exact_dists_path = os.path.join(orig_cache_dir, 'dstore_cache_exact_dists.npy')
    
    log_progress(f"Copying {tmp_exact_dists_path} to {orig_exact_dists_path}")
    shutil.copy2(tmp_exact_dists_path, orig_exact_dists_path)

    time.sleep(1)


#
# Main
#

def main(args):
    log_progress("Starting dataset load")
    t0 = time.time()
    dataset = Dataset(args)
    log_progress(f"Dataset loaded in {time.time() - t0:.2f}s")

    log_progress("Starting dstore load")
    t0 = time.time()
    dstore = Dstore(args)
    log_progress(f"Dstore loaded in {time.time() - t0:.2f}s")

    if args.save_knns:
        log_progress("Finding and Saving Neighbors")
        save_knns(args, dataset, dstore)
        log_progress("Done")
        sys.exit()

    log_progress("Starting cache load")
    t0 = time.time()
    dataset.load_cache()
    log_progress(f"Cache loaded in {time.time() - t0:.2f}s")

    if args.save_exact:
        log_progress("Calculating and Saving Exact Distances")
        save_exact(args, dataset, dstore)
        log_progress("Done")
        sys.exit()

    if args.exact:
        log_progress("Loading exact distances")
        t0 = time.time()
        dataset.load_exact_dists()
        dists = dataset.exact_dists
        log_progress(f"Exact distances loaded in {time.time() - t0:.2f}s")
    else:
        dists = -1 * dataset.dists

    # Vocab
    log_progress("Loading dictionary")
    t0 = time.time()
    vocab = Dictionary()
    vocab.add_from_file(args.vocab)
    vocab.finalize()
    log_progress(f"Dictionary loaded in {time.time() - t0:.2f}s")
    log_progress(f'found {len(vocab)} tokens in vocab {args.vocab}')
    
    # Check dtype of probabilities
    log_progress(f"LM prob dtype: {dataset.prob.dtype}")
    log_progress(f"LM prob min: {dataset.prob.min()}, max: {dataset.prob.max()}")
    
    # Context
    context = {}
    context['args'] = args
    context['vocab'] = vocab
    context['dstore'] = dstore
    context['dataset'] = dataset
    context['dists'] = dists

    log_progress("Running knnlm_func.run_eval_ppl")
    knnlm_func.run_eval_ppl(context,args.validation_split)

if __name__ == '__main__':
    args = argument_parser().parse_args()

    set_presets(args)

    with torch.no_grad():
        main(args)




