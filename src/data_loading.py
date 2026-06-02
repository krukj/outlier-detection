import os
from dataclasses import dataclass

import clustbench
import numpy as np
from scipy.io import loadmat

from config import (BENCGMARK_BATTERIES, BENCHMARK_DATA_PATH,
                    BENCHMARK_DATASETS, DATA_PATH)


@dataclass
class Dataset:
    name: str
    data: np.ndarray
    labels: np.ndarray


def load_data() -> list[Dataset]:
    datasets = []

    files = os.listdir(DATA_PATH)
    for f in files:
        file_path = os.path.join(DATA_PATH, f)
        loaded = loadmat(file_path)
        data = loaded.get("X")
        labels = loaded.get("y")
        name = f.removesuffix(".mat")
        datasets.append(Dataset(name=name, data=data, labels=labels))

    
    for battery in BENCGMARK_BATTERIES:
        datasets_battery = clustbench.get_dataset_names(battery, path=BENCHMARK_DATA_PATH)
        for ds in datasets_battery:
            if ds in BENCHMARK_DATASETS:
                loaded_ds = clustbench.load_dataset(battery, ds, BENCHMARK_DATA_PATH)
                labels = loaded_ds.labels[0]
                binary_labels = (loaded_ds.labels[0] == 0).astype(int)

                datasets.append(
                    Dataset(
                        name=ds,
                        data=loaded_ds.data,
                        labels=binary_labels,
                    )
                )

    print(f"Successfully loaded {len(datasets)} datasets")
    return datasets
