from deadwood import Deadwood
from sklearn.cluster import DBSCAN
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM

DATA_PATH = "data/"
BENCHMARK_DATA_PATH = "clustering-data-v1"
SEEDS = [42, 100, 210, 410, 500]
ALGORITHMS = [OneClassSVM, IsolationForest, LocalOutlierFactor, DBSCAN, Deadwood]
BENCGMARK_BATTERIES = ["graves", "other"]
BENCHMARK_DATASETS = ["ring_noisy", "zigzag_noisy", "hdbscan"]

PARAMS_GRID = {
    "OneClassSVM": {
        "kernel": ["rbf", "linear", "poly", "sigmoid"],
        "gamma": ["scale", "auto", 0.001, 0.01, 0.1, 1.0],
        "nu": [0.01, 0.05, 0.1, 0.2, 0.3, 0.5],
    },
    "IsolationForest": {
        "n_estimators": [50, 100, 200, 500],
    },
    "DBSCAN": {
        "eps": [0.1, 0.3, 0.5, 1.0, 2.0],
        "min_samples": [3, 5, 10, 15, 20],
        "metric": ["euclidean", "manhattan", "chebyshev"],
    },
    "LocalOutlierFactor": {
        "n_neighbors": [5, 10, 20, 35, 50, 100],
        "algorithm": ["auto", "ball_tree", "kd_tree", "brute"],
        "leaf_size": [10, 20, 30, 50],
    },
    "Deadwood": {
        "max_debris_size": ["auto", 5, 10, 20, 50],
    },
}
