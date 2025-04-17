import os

# import faiss
import numpy as np
import torch
from torch.utils.data import Dataset as TorchDataset
import time

import shutil
from utils import log_progress, copy_to_tmp

class Dataset(object):
    def __init__(self, args):
        self.args = args
        path = args.eval_dstore
        dstore_size = args.eval_dstore_size
        # TODO: We should allow for more (or less) neighbors to be included.
        # Copy data to /tmp for faster access
        self.query = copy_to_tmp(f'{path}/dstore_keys.npy', dtype=np.float32, shape=(dstore_size, 1024))
        self.target = copy_to_tmp(f'{path}/dstore_vals.npy', dtype=np.int64, shape=(dstore_size, 1))
        #This needs to be loaded float32 if dstore is float32
        self.prob = copy_to_tmp(f'{path}/dstore_prob.npy', dtype=np.float32, shape=(dstore_size, 1))
        log_progress("Dataset loaded with data from /tmp")
        for k in ['query', 'target', 'prob']:
            v = getattr(self, k)
            new_v = np.ones(v.shape, dtype=v.dtype)
            new_v[:] = v
            setattr(self, k, new_v)

    def load_cache(self):
        args = self.args
        path = args.eval_dstore_cache
        dstore_size = args.eval_dstore_size

        if not args.eval_external_knns:
            self.dists = copy_to_tmp(f'{path}/dstore_cache_dists.npy', dtype=np.float32, shape=(dstore_size, 1024))
            self.knns = copy_to_tmp(f'{path}/dstore_cache_knns.npy', dtype=np.int32, shape=(dstore_size, 1024))
        else:
            # TODO: We don't load approx. distances since we assume the neighbors were set without faiss.
            #self.dists = np.memmap(f'{path}/dstore_cache_dists.npy', dtype=np.float32, mode='r', shape=(dstore_size, 1024))
            self.dists = np.ones(dtype=np.float32, shape=(dstore_size, 1024))
            self.knns = copy_to_tmp(args.eval_external_knns, dtype=np.int32, shape=(dstore_size, 1024))
        log_progress("Cache loaded from /tmp")

    def load_exact_dists(self):
        args = self.args
        path = args.eval_dstore_cache
        dstore_size = args.eval_dstore_size
        if not args.eval_external_knns:
            filename = f'{path}/dstore_cache_exact_dists.npy'
        else:
            filename = f'{args.eval_external_knns}.exact_dists.npy'
        assert os.path.exists(filename)
        self.exact_dists = copy_to_tmp(filename, dtype=np.float32, shape=(dstore_size, 1024))
        log_progress("Exact distances loaded from /tmp")


class Dstore(object):
    def __init__(self, args):
        path = args.dstore
        dstore_size = args.dstore_size

        self.sim_func = 'do_not_recomp_l2'
        self.k = 1024

        # Copy data to /tmp for faster access
        tmp_path = os.path.join('/tmp', os.path.basename(path))
        os.makedirs(tmp_path, exist_ok=True)
        
        keys_path = f'{path}/dstore_keys.npy'
        vals_path = f'{path}/dstore_vals.npy'
        tmp_keys_path = os.path.join(tmp_path, 'dstore_keys.npy')
        tmp_vals_path = os.path.join(tmp_path, 'dstore_vals.npy')
        if not args.from_cache:
            #Don't need to load keys if evaluating from cache
            if not os.path.exists(tmp_keys_path):
                log_progress(f"Copying {keys_path} to {tmp_keys_path}")
                shutil.copy2(keys_path, tmp_keys_path)
            else:
                log_progress(f"File already exists: {tmp_keys_path}")
            
        if not os.path.exists(tmp_vals_path):
            log_progress(f"Copying {vals_path} to {tmp_vals_path}")
            shutil.copy2(vals_path, tmp_vals_path)
        else:
            log_progress(f"File already exists: {tmp_vals_path}")
        #Don't use tmp for datastore keys if not needed.
        if args.from_cache:
            self.keys = np.memmap(keys_path, dtype=np.float32, mode='r', shape=(dstore_size, 1024))
        else:
            #Keys will be loaded from tmp if we need them
            self.keys = np.memmap(tmp_keys_path, dtype=np.float32, mode='r', shape=(dstore_size, 1024))
        self.vals = np.memmap(tmp_vals_path, dtype=np.int64, mode='r', shape=(dstore_size, 1))

        print('load index')
        index_path = args.dstore_knn_index
        tmp_index_path = os.path.join(tmp_path, os.path.basename(index_path))
        
        if not os.path.exists(tmp_index_path):
            log_progress(f"Copying {index_path} to {tmp_index_path}")
            shutil.copy2(index_path, tmp_index_path)
        else:
            log_progress(f"File already exists: {tmp_index_path}")
        #Comment this out to run with anaconda3_cpu, if index is not needed
        if not args.from_cache:
            #Load index if not evaluating from cache
            import faiss
            self.index = faiss.read_index(tmp_index_path, faiss.IO_FLAG_ONDISK_SAME_DIR)
        else:
            self.index = None
        self.half = True
        self.metric_type = 'l2'

    def combine_knn_and_vocab_probs(self, knn_p, vocab_p, coeff):
        combine_probs = torch.stack([vocab_p, knn_p], dim=0)
        coeffs = torch.ones_like(combine_probs)
        coeffs[0] = np.log(1 - coeff) if coeff < 1 else -1e13
        coeffs[1] = np.log(coeff) if coeff > 0 else -1e13
        curr_prob = torch.logsumexp(combine_probs + coeffs, dim=0)

        return curr_prob

    def get_knns(self, query, k=None):
        if k is None:
            k = self.k
        if query.dtype == np.float16:
            query = query.astype(np.float32)
        dists, knns = self.index.search(query, k)
        return dists, knns

# Dataset class for training the combiner
class ProbDataset(TorchDataset):
    def __init__(self, features, probs, targets):
        """
        features: Input features for the model (query embeddings or other features)
        probs: List of probability distributions [lm_probs, knn_probs, pos_modified_probs]
        targets: Target tokens for calculating loss
        """
        self.features = features
        self.probs = probs
        self.targets = targets

    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        return self.features[idx], [p[idx] for p in self.probs], self.targets[idx]