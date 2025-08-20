import numpy as np
from datasets import Dataset, DatasetDict, load_from_disk
import os

dstore_dir = "datastore/wikitext-103/"
split_sizes = {"valid": 217646, "test": 245569}

features = {}
features["exact_dists"] = {"mmap_name": "dstore_cache_exact_dists.npy", 
                        "shape" : (1024,),
                            "dtype" : np.float32}
features["dists"] = {"mmap_name": "dstore_cache_dists.npy",
                        "shape" : (1024,),
                        "dtype" : np.float32}
features["knns"] = {"mmap_name": "dstore_cache_knns.npy",
                    "shape" : (1024,),
                    "dtype" : np.int32}

hf_dataset = DatasetDict({
    split : Dataset.from_dict(
        {"idx": np.arange(split_sizes[split])}
        ) for split in ["valid", "test"]
})
hf_dataset.save_to_disk("dataset_cache")
hf_dataset = load_from_disk("dataset_cache")

def add_new_column(df, col_name, col_values):
    # Define a function to add the new column
    def create_column(examples):
        examples[col_name] = col_values[examples["idx"]]  # Assign specific values from memmap
        return examples
    # Apply the function to each item in the dataset
    df = df.map(create_column)
    return df

for split in ["valid", "test"]:
    split_size = split_sizes[split]
    base = os.path.join(dstore_dir, f"{split}.cache")
    mmaps = {
        k: np.memmap(os.path.join(base, feat["mmap_name"]), dtype=feat["dtype"], mode='r', shape=(split_size,) + feat["shape"])
        for k, feat in features.items()
    }
    for k, v in mmaps.items():
        hf_dataset[split] = add_new_column(hf_dataset[split], k, v)
    hf_dataset[split] = hf_dataset[split].remove_columns(["idx"])
os.system("rm -rf dataset_cache")
hf_dataset.push_to_hub("knn-wiki-cache-v3")
