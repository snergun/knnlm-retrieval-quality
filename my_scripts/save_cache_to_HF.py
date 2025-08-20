import numpy as np
from datasets import Dataset, DatasetDict, Features, Sequence, Value
import os

dstore_dir = "datastore/wikitext-103/"
split_sizes = {"valid": 217646, "test": 245569}

# (Optional) if you actually use this elsewhere

# Define schema explicitly
features = Features({
    "exact_dists": Sequence(Value("float32"), length=1024),
    "dists":       Sequence(Value("float32"), length=1024),
    "knns":        Sequence(Value("int32"), length=1024),
})

hf_dataset = {}
for split in ["valid", "test"]:
    split_size = split_sizes[split]
    base = os.path.join(dstore_dir, f"{split}.cache")
    exact_dists = np.copy(np.memmap(os.path.join(base, "dstore_cache_exact_dists.npy"),
                            dtype=np.float32, mode='r', shape=(split_size, 1024)))
    dists       = np.copy(np.memmap(os.path.join(base, "dstore_cache_dists.npy"),
                            dtype=np.float32, mode='r', shape=(split_size, 1024)))
    knns        = np.copy(np.memmap(os.path.join(base, "dstore_cache_knns.npy"),
                            dtype=np.int32,   mode='r', shape=(split_size, 1024)))

    split_dataset = {"exact_dists": exact_dists, "dists": dists, "knns": knns}
    split_name = "validation" if split == "valid" else split

    # IMPORTANT: pass features so Arrow stores float32/int32
    hf_dataset[split_name] = Dataset.from_dict(split_dataset, features=features)

hf_dataset = DatasetDict(hf_dataset)
hf_dataset.push_to_hub("knn-wiki-cache-v3")
