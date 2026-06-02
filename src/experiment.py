import json
from pathlib import Path

import numpy as np
from deadwood import Deadwood
from sklearn.base import BaseEstimator
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score, roc_auc_score)
from tqdm import tqdm

from config import ALGORITHMS, DATA_PATH, PARAMS_GRID, SEEDS
from data_loading import load_data
from utils import find_epsilon, unified_scoring


def build_model(
    class_name: str,
    alg: BaseEstimator,
    data: np.ndarray,
    contamination: float,
    seed=42,
    extra_params: dict = None,
) -> BaseEstimator | Deadwood:
    params = dict(extra_params) if extra_params else {}

    if class_name == "DBSCAN":
        n_samples, n_features = data.shape
        min_samples_auto = min(max(2 * n_features, 5), n_samples - 1)
        eps_auto = find_epsilon(data, k=min_samples_auto)

        eps = params.pop("eps", eps_auto)
        min_samples = params.pop("min_samples", min_samples_auto)

        return alg(
            eps=eps,
            min_samples=min_samples,
            **params,
        )
    elif class_name in ["LocalOutlierFactor", "OneClassSVM"]:
        return alg(**params)
    elif class_name == "Deadwood":
        return alg(contamination=contamination, **params)
    elif class_name == "IsolationForest":
        return alg(random_state=seed, **params)


def update_results(
    results: dict,
    dataset,
    data_shape,
    class_name: str,
    labels_true,
    labels_pred,
    u_scores,
    roc_auc,
    extra_params=None,
) -> None:
    results["dataset"].append(dataset)
    results["data_shape"].append(data_shape)
    results["algorithm"].append(class_name)
    results["accuracy"].append(accuracy_score(labels_true, labels_pred))
    results["precision"].append(
        precision_score(labels_true, labels_pred, zero_division=0, average="macro")
    )
    results["recall"].append(
        recall_score(labels_true, labels_pred, zero_division=0, average="macro")
    )
    results["f1"].append(
        f1_score(labels_true, labels_pred, zero_division=0, average="macro")
    )
    results["unified_score"].append(u_scores if u_scores is not None else [])
    results["roc_auc"].append(roc_auc)
    if extra_params is not None:
        results["params"].append(extra_params)


def run_experiment(
    exp_name: str, contamination: float = 0.05, param_grid: dict | None = None
):
    datasets = load_data()
    results = {
        "dataset": [],
        "data_shape": [],
        "algorithm": [],
        "accuracy": [],
        "precision": [],
        "recall": [],
        "roc_auc": [],
        "f1": [],
        "unified_score": [],
    }
    if param_grid is not None:
        results["params"] = []

    for ds in tqdm(datasets):
        print("=" * 30 + f" DATASET {ds.name} " + "=" * 30)
        print()
        data, unique_indices = np.unique(ds.data, axis=0, return_index=True)
        labels_true = ds.labels[unique_indices]

        data_shape = data.shape
        for alg in ALGORITHMS:
            class_name = alg.__name__

            is_nondeterministic = class_name == "IsolationForest"
            current_seeds = SEEDS if is_nondeterministic else [42]

            if param_grid is not None:
                alg_params_dict = param_grid.get(class_name)

                for param_name in alg_params_dict.keys():
                    for param_value in alg_params_dict.get(param_name):
                        extra_params = {param_name: param_value}
                        for seed in current_seeds:

                            model = build_model(
                                class_name,
                                alg,
                                data,
                                contamination,
                                seed,
                                extra_params,
                            )

                            if class_name == "Deadwood":
                                labels_pred = model.fit_predict(data)
                                labels_pred = (labels_pred == -1).astype(int)  # -1 to 0
                                roc_auc = None
                                u_scores = None
                            elif class_name == "DBSCAN":
                                model.fit(data)
                                labels_pred = np.where(model.labels_ == -1, 1, 0)
                                roc_auc = None
                                u_scores = None
                            else:
                                model.fit(data)
                                u_scores = unified_scoring(
                                    method=class_name, fitted_model=model, X=data
                                )
                                roc_auc = roc_auc_score(labels_true, u_scores)
                                threshold = np.percentile(
                                    u_scores, 100 * (1 - contamination)
                                )
                                labels_pred = (u_scores >= threshold).astype(int)
                            update_results(
                                results,
                                ds.name,
                                data_shape,
                                class_name,
                                labels_true,
                                labels_pred,
                                u_scores,
                                roc_auc,
                                extra_params,
                            )
                            print(
                                f"Dataset: {ds.name}, Algorithm: {class_name}, Seed: {seed}, Params: {param_name}:{param_value}"
                            )
            else:

                for seed in current_seeds:
                    model = build_model(class_name, alg, data, contamination, seed)
                    if class_name == "Deadwood":
                        labels_pred = model.fit_predict(data)
                        labels_pred = (labels_pred == -1).astype(int)  # -1 to 0
                        roc_auc = None
                        u_scores = None
                    elif class_name == "DBSCAN":
                        model.fit(data)
                        labels_pred = np.where(model.labels_ == -1, 1, 0)
                        roc_auc = None
                        u_scores = None
                    else:
                        model.fit(data)
                        u_scores = unified_scoring(
                            method=class_name, fitted_model=model, X=data
                        )
                        roc_auc = roc_auc_score(labels_true, u_scores)
                        threshold = np.percentile(u_scores, 100 * (1 - contamination))
                        labels_pred = (u_scores >= threshold).astype(int)

                    update_results(
                        results,
                        ds.name,
                        data_shape,
                        class_name,
                        labels_true,
                        labels_pred,
                        u_scores,
                        roc_auc,
                        extra_params=None,
                    )
                    print(f"Dataset: {ds.name}, Algorithm: {class_name}, Seed: {seed}")

    results_path = Path("results/").resolve()
    results_path.mkdir(parents=True, exist_ok=True)
    with open(results_path / f"{exp_name}.json", "w") as f:
        json.dump(results, f, indent=4)


if __name__ == "__main__":
    # for cont in [0.05, 0.1, 0.15, 0.2]:
    #     run_experiment(f"exp_contamination_{cont}", contamination=cont)

    run_experiment(exp_name="experiment_params_cont_05", param_grid=PARAMS_GRID, contamination=0.05)
