import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.io import loadmat
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import warnings

import clustbench
import numpy as np
from scipy.io import loadmat

from src.config import (BENCGMARK_BATTERIES, BENCHMARK_DATA_PATH,
                    BENCHMARK_DATASETS, DATA_PATH)


warnings.filterwarnings("ignore")
from src.config import PARAMS_GRID

ALGORITHMS = [
    "OneClassSVM",
    "IsolationForest",
    "LocalOutlierFactor",
    "DBSCAN",
    "Deadwood",
]
plt.rcParams.update(
    {
        "figure.dpi": 130,
        # "font.family": "DejaVu Sans",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.3,
    }
)
COLORS = {
    "OneClassSVM": "#4C72B0",
    "IsolationForest": "#DD8452",
    "LocalOutlierFactor": "#55A868",
    "DBSCAN": "#C44E52",
    "Deadwood": "#8172B2",
}


def visualize_dataset(name: str, benchmark = False, battery = None):
    if benchmark:
        datasets_battery = clustbench.get_dataset_names(battery, path=BENCHMARK_DATA_PATH)
        for ds in datasets_battery:
            if ds == name:
                loaded_ds = clustbench.load_dataset(battery, ds, BENCHMARK_DATA_PATH)
                X = loaded_ds.data 
                y = (loaded_ds.labels[0] == 0).astype(int)
    else:

        data = loadmat(f"data/{name}.mat")
        X = data["X"]
        y = data["y"].ravel()

    print(f"X shape : {X.shape}")
    print(f"Number of anomalies: {sum(y)} ({sum(y)/len(y)*100:.1f}%)")
    
    palette = {0: "#4C72B0", 1: "#C44E52"}
    labels = {0: "Inlier", 1: "Outlier"}

    if X.shape[1] > 2:

        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X)

        tsne = TSNE(n_components=2, perplexity=30, random_state=42)
        X_tsne = tsne.fit_transform(X)

        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        fig.suptitle(f"{name} dataset", fontsize=14)

    
        sns.scatterplot(
            x=X_pca[:, 0],
            y=X_pca[:, 1],
            hue=y,
            palette=palette,
            alpha=0.7,
            edgecolor="k",
            ax=axes[0],
        )
        axes[0].set_title("PCA")
        axes[0].set_xlabel("PC1")
        axes[0].set_ylabel("PC2")
        handles, _ = axes[0].get_legend_handles_labels()
        axes[0].legend(handles, [labels[0], labels[1]])

        sns.scatterplot(
            x=X_tsne[:, 0],
            y=X_tsne[:, 1],
            hue=y,
            palette=palette,
            alpha=0.7,
            edgecolor="k",
            ax=axes[1],
        )
        axes[1].set_title("t-SNE")
        handles, _ = axes[1].get_legend_handles_labels()
        axes[1].legend(handles, [labels[0], labels[1]])

        plt.tight_layout()
        plt.show()
    
    else:
        fig, axes = plt.subplots(1, 1, figsize=(8, 6))
        fig.suptitle(f"{name} dataset", fontsize=14)
        sns.scatterplot(
            x=X[:, 0],
            y=X[:, 1],
            hue=y,
            palette=palette,
            alpha=0.7,
            edgecolor="k",
            ax=axes,
        )


def plot_metric_boxplot(df_agg: pd.DataFrame, contamination_vals: list, metric: str):
    fig, axes = plt.subplots(
        1,
        len(contamination_vals),
        figsize=(5 * len(contamination_vals), 5),
        sharey=True,
    )
    fig.suptitle(
        f"{metric.upper().replace('_', ' ')} distribution per algorithm",
        fontsize=14,
    )

    for ax, cont in zip(axes, contamination_vals):
        df_sub = df_agg[df_agg["contamination"] == cont]
        data_to_plot = [
            df_sub[df_sub["algorithm"] == alg][metric].dropna().values
            for alg in ALGORITHMS
        ]
        bp = ax.boxplot(
            data_to_plot,
            patch_artist=True,
            medianprops=dict(color="black", linewidth=2),
        )
        for patch, alg in zip(bp["boxes"], ALGORITHMS):
            patch.set_facecolor(COLORS[alg])
            patch.set_alpha(0.8)
        ax.set_title(f"contamination = {cont}")
        ax.set_xticks(range(1, len(ALGORITHMS) + 1))
        ax.set_xticklabels(ALGORITHMS, rotation=30, ha="right", fontsize=9)
        ax.set_ylim(0, 1.05)

    axes[0].set_ylabel(metric.upper().replace("_", " "))
    plt.tight_layout()
    plt.show()


def plot_contamination_influence(df_agg: pd.DataFrame, metrics: list):
    fig, axes = plt.subplots(
        1, len(metrics), figsize=(6 * len(metrics), 5), sharey=True
    )
    fig.suptitle(
        "Contamination influence on metrics (median over datasets)", fontsize=14
    )

    for ax, metric in zip(axes, metrics):
        for alg in ALGORITHMS:
            sub = (
                df_agg[df_agg["algorithm"] == alg]
                .groupby("contamination")[metric]
                .median()
            )
            if sub.dropna().empty:
                continue
            ax.plot(
                sub.index,
                sub.values,
                marker="o",
                label=alg,
                color=COLORS[alg],
                linewidth=2,
                markersize=7,
            )
        ax.set_title(metric.upper().replace("_", " "))
        ax.set_xlabel("contamination")
        ax.set_xticks([0.05, 0.1, 0.15, 0.2])
        ax.set_ylim(0, 1.05)

    axes[0].set_ylabel("median score")
    axes[-1].legend(loc="lower left", fontsize=8, framealpha=0.7)
    plt.tight_layout()
    plt.show()


def plot_heatmap_dflt(
    df_agg: pd.DataFrame, contamination_levels: list, algorithms: list, metric: str
):
    for contamination_val in contamination_levels:
        pivot = (
            df_agg[df_agg["contamination"] == contamination_val]
            .pivot_table(index="dataset", columns="algorithm", values=metric)
            .reindex(columns=algorithms)
        )

        fig, ax = plt.subplots(figsize=(10, max(5, len(pivot) * 0.3)))
        im = ax.imshow(pivot.values, aspect="auto", cmap="RdYlGn", vmin=0, vmax=1)

        ax.set_xticks(range(len(algorithms)))
        ax.set_xticklabels(algorithms, rotation=30, ha="right")
        ax.set_yticks(range(len(pivot)))
        ax.set_yticklabels(pivot.index, fontsize=8)

        for i in range(pivot.shape[0]):
            for j in range(pivot.shape[1]):
                val = pivot.values[i, j]
                if not np.isnan(val):
                    ax.text(
                        j,
                        i,
                        f"{val:.2f}",
                        ha="center",
                        va="center",
                        fontsize=7,
                        color="black" if 0.3 < val < 0.8 else "white",
                    )
                else:
                    ax.text(
                        j, i, "N/A", ha="center", va="center", fontsize=7, color="gray"
                    )

        plt.colorbar(im, ax=ax, label=metric.upper().replace("_", " "))
        ax.set_title(
            f"{metric.upper().replace('_', ' ')} per dataset vs. algorithm (contamination = {contamination_val})"
        )
        plt.tight_layout()
        plt.show()


def plot_metric_comparison_per_contamination(
    df_agg: pd.DataFrame, metric: str, contamination_levels: list
):
    df_metric = df_agg[df_agg[metric].notna()]

    fig, axes = plt.subplots(1, len(contamination_levels), figsize=(15, 5), sharey=True)
    fig.suptitle(
        f"{metric.upper().replace('_', ' ')} per algorithm for different contamination levels",
        fontsize=14,
    )

    for ax, contamination_val in zip(axes, contamination_levels):
        sub = df_metric[df_metric["contamination"] == contamination_val]
        data_to_plot = [
            sub[sub["algorithm"] == alg][metric].dropna().values for alg in ALGORITHMS
        ]
        bp = ax.boxplot(
            data_to_plot,
            patch_artist=True,
            medianprops=dict(color="black", linewidth=2),
        )
        for patch, alg in zip(bp["boxes"], ALGORITHMS):
            patch.set_facecolor(COLORS[alg])
            patch.set_alpha(0.8)

        ax.set_title(f"contamination = {contamination_val}")
        ax.set_xticks(range(1, len(ALGORITHMS) + 1))
        ax.set_xticklabels(ALGORITHMS, rotation=20, ha="right")
        ax.set_ylim(0, 1.05)

    axes[0].set_ylabel(metric.upper().replace("_", " "))
    if metric == "roc_auc":
        axes[0].legend(fontsize=8)

    plt.tight_layout()
    plt.show()


def plot_algorithm_ranking(
    df_agg: pd.DataFrame, metric: str, contamination_levels: list
):
    fig, axes = plt.subplots(1, len(contamination_levels), figsize=(15, 5), sharey=True)
    fig.suptitle(
        f"Mean {metric.upper().replace('_', ' ')} rank per algorithm", fontsize=14
    )

    for ax, contamination_val in zip(axes, contamination_levels):
        sub = df_agg[df_agg["contamination"] == contamination_val].copy()
        sub["rank"] = sub.groupby("dataset")[metric].rank(
            ascending=False, method="average"
        )
        mean_rank = sub.groupby("algorithm")["rank"].mean().reindex(ALGORITHMS)

        bars = ax.barh(
            ALGORITHMS[::-1],
            mean_rank.reindex(ALGORITHMS[::-1]),
            color=[COLORS[a] for a in ALGORITHMS[::-1]],
            alpha=0.85,
            edgecolor="white",
        )
        for bar, val in zip(bars, mean_rank.reindex(ALGORITHMS[::-1])):
            if not np.isnan(val):
                ax.text(
                    val + 0.05,
                    bar.get_y() + bar.get_height() / 2,
                    f"{val:.2f}",
                    va="center",
                    fontsize=9,
                )
        ax.set_title(f"contamination = {contamination_val}")
        ax.set_xlabel("Mean rank")
        ax.set_xlim(0, len(ALGORITHMS) + 0.5)
        ax.set_yticklabels(ALGORITHMS[::-1])

    plt.tight_layout()
    plt.show()


def plot_if_stability(
    df: pd.DataFrame, metric: str, contamination_levels: list, palette: list = None
):
    if palette is None:
        palette = ["#4C72B0", "#DD8452", "#55A868", "#C44E52"]

    df_if = df[df["algorithm"] == "IsolationForest"]
    std_col_name = f"{metric}_std"
    std_per_ds = df_if.groupby(["dataset", "contamination"])[metric].std().reset_index()
    std_per_ds.columns = ["dataset", "contamination", std_col_name]

    fig, ax = plt.subplots(figsize=(12, 5))
    datasets_sorted = (
        std_per_ds.groupby("dataset")[std_col_name]
        .mean()
        .sort_values(ascending=False)
        .index
    )

    x = np.arange(len(datasets_sorted))
    width = 0.8 / len(contamination_levels)

    for i, contamination_val in enumerate(contamination_levels):
        color = palette[i % len(palette)]
        vals = (
            std_per_ds[std_per_ds["contamination"] == contamination_val]
            .set_index("dataset")
            .reindex(datasets_sorted)[std_col_name]
        )
        ax.bar(
            x + i * width,
            vals,
            width,
            label=f"contamination={contamination_val}",
            color=color,
            alpha=0.8,
        )

    ax.set_title(
        f"IsolationForest stability - std({metric.upper()}) over seeds per dataset"
    )
    ax.set_xticks(x + (width * (len(contamination_levels) - 1)) / 2)
    ax.set_xticklabels(datasets_sorted, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel(f"std({metric.upper()})")
    ax.legend()

    plt.tight_layout()
    plt.show()

def format_param_value(val):
    if isinstance(val, (int, float)) and not isinstance(val, bool):
        return str(round(val, 5))
    return str(val)

def agg_by_param(df, alg, param, metric="f1"):
    sub = df[(df["algorithm"] == alg) & (df["param_name"] == param)].copy()
    if alg == "IsolationForest":
        sub = sub.groupby(["dataset", "param_value_str"], as_index=False)[metric].mean()
    return sub.groupby("param_value_str")[metric].agg(["median", "std"]).reset_index()


def is_numeric_param(values):
    try:
        [float(v) for v in values if v != "auto"]
        return True
    except (ValueError, TypeError):
        return False


def plot_param(ax, agg, param_values_order, metric, color, title, param=None):
    order_str = [format_param_value(v) for v in param_values_order]

    agg = agg.set_index("param_value_str").reindex(order_str).reset_index()
    x = np.arange(len(agg))
    medians = agg["median"].values
    stds = agg["std"].fillna(0).values

    if is_numeric_param(agg["param_value_str"].tolist()):
        ax.plot(x, medians, marker="o", color=color, linewidth=2, markersize=7)
        ax.fill_between(x, medians - stds, medians + stds, alpha=0.15, color=color)
    else:
        bars = ax.bar(
            x, medians, color=color, alpha=0.8, edgecolor="white", linewidth=0.8
        )
        ax.errorbar(
            x, medians, yerr=stds, fmt="none", color="black", capsize=4, linewidth=1.2
        )
        
    ax.set_xlabel(param)
    ax.set_xticks(x)
    ax.set_xticklabels(agg["param_value_str"], rotation=30, ha="right", fontsize=8)
    ax.set_title(title)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel(metric.upper())


def plot_alg_params(df: pd.DataFrame, alg: str, metric: str, params_dict: dict):
    color = COLORS[alg]

    fig, axes = plt.subplots(
        1, len(params_dict), figsize=(16 if len(params_dict) > 1 else 8, 5)
    )
    fig.suptitle(
        f"{alg} - {metric} median over datasets for different parameters", fontsize=13
    )

    if isinstance(axes, np.ndarray):
        for ax, (param, order) in zip(axes, params_dict.items()):
            agg = agg_by_param(df, alg, param, metric)
            plot_param(ax, agg, order, metric, color, f"Parameter: {param}", param)
    else:
        for param, order in params_dict.items():
            agg = agg_by_param(df, alg, param, metric)
            plot_param(axes, agg, order, metric, color, f"Parameter: {param}", param)

    plt.tight_layout()
    plt.show()


def plot_sd_if(df: pd.DataFrame, metric: str):

    alg = "IsolationForest"
    color = COLORS[alg]
    order_if = PARAMS_GRID.get("IsolationForest").get("n_estimators")
    order_if = [str(v) for v in order_if]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("IsolationForest", fontsize=13)

    agg = agg_by_param(df, alg, "n_estimators", metric)
    plot_param(
        axes[0],
        agg,
        order_if,
        metric,
        color,
        f"{metric} median over datatsets",
        param="n_estimators",
    )

    sub_if = df[(df["algorithm"] == alg) & (df["param_name"] == "n_estimators")]
    sd_seed = sub_if.groupby(["dataset", "param_value_str"])[metric].std().reset_index()
    sd_seed.columns = ["dataset", "param_value_str", f"{metric}_sd"]
    mean_std = (
        sd_seed.groupby("param_value_str")[f"{metric}_sd"].mean().reindex(order_if)
    )

    axes[1].bar(order_if, mean_std.values, color=color, alpha=0.8, edgecolor="white")
    axes[1].set_title(f"Mean sd({metric}) over seeds")
    axes[1].set_xlabel("n_estimators")
    axes[1].set_ylabel(f"sd({metric})")

    plt.tight_layout()
    plt.show()


def plot_heatmap(df: pd.DataFrame, alg: str, param_name: str, metric: str):

    sub = df[(df["algorithm"] == alg) & (df["param_name"] == param_name)]
    pivot = sub.pivot_table(index="dataset", columns="param_value_str", values=metric)
    values = PARAMS_GRID.get(alg).get(param_name)
    values = [str(v) for v in values]
    pivot = pivot.reindex(columns=values)

    fig, ax = plt.subplots(figsize=(10, max(4, len(pivot) * 0.45)))
    im = ax.imshow(pivot.values, aspect="auto", cmap="RdYlGn", vmin=0, vmax=1)
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns)
    ax.set_xlabel(param_name)
    ax.set_yticks(range(len(pivot)))
    ax.set_yticklabels(pivot.index, fontsize=8)
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            val = pivot.values[i, j]
            if not np.isnan(val):
                ax.text(
                    j,
                    i,
                    f"{val:.2f}",
                    ha="center",
                    va="center",
                    fontsize=7,
                    color="black" if 0.3 < val < 0.8 else "white",
                )
    plt.colorbar(im, ax=ax, label=metric)
    ax.set_title(f"{alg} - {metric} per dataset vs. {param_name}")
    plt.tight_layout()
    plt.show()


def plot_one_param_and_heatmap(
    df: pd.DataFrame, alg: str, param_name: str, metric: str
):

    color = COLORS[alg]
    values = PARAMS_GRID.get(alg).get(param_name)
    values = [str(v) for v in values]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(f"{alg} - {param_name}", fontsize=13)

    agg = agg_by_param(df, alg, param_name, metric)
    plot_param(
        axes[0],
        agg,
        values,
        metric,
        color,
        f"{metric} median over datasets",
        param=param_name,
    )

    # per dataset
    sub_dw = df[(df["algorithm"] == alg) & (df["param_name"] == param_name)]
    pivot_dw = sub_dw.pivot_table(
        index="dataset", columns="param_value_str", values=metric
    )
    pivot_dw = pivot_dw.reindex(columns=values)

    im = axes[1].imshow(pivot_dw.values, aspect="auto", cmap="RdYlGn", vmin=0, vmax=1)
    axes[1].set_xticks(range(len(pivot_dw.columns)))
    axes[1].set_xticklabels(pivot_dw.columns)
    axes[1].set_xlabel(param_name)
    axes[1].set_yticks(range(len(pivot_dw)))
    axes[1].set_yticklabels(pivot_dw.index, fontsize=8)
    for i in range(pivot_dw.shape[0]):
        for j in range(pivot_dw.shape[1]):
            val = pivot_dw.values[i, j]
            if not np.isnan(val):
                axes[1].text(
                    j,
                    i,
                    f"{val:.2f}",
                    ha="center",
                    va="center",
                    fontsize=7,
                    color="black" if 0.3 < val < 0.8 else "white",
                )
    plt.colorbar(im, ax=axes[1], label=metric)
    axes[1].set_title(f"{metric} per dataset vs. {param_name}")

    plt.tight_layout()
    plt.show()


def plot_params_sensitivity(df: pd.DataFrame, metric: str):

    cv_data = {}
    for alg in ALGORITHMS:
        sub = df[df["algorithm"] == alg]
        if alg == "IsolationForest":
            sub = sub.groupby(
                ["dataset", "param_name", "param_value_str"], as_index=False
            )[metric].mean()
        if sub.empty:
            continue
        metric_mean = sub.groupby("param_name")[metric].mean()
        metric_sd = sub.groupby("param_name")[metric].std()
        cv = (metric_sd / metric_mean).fillna(0)
        cv_data[alg] = cv.mean()

    fig, ax = plt.subplots(figsize=(9, 5))
    algs = list(cv_data.keys())
    cvs = [cv_data[a] for a in algs]
    bars = ax.bar(
        algs, cvs, color=[COLORS[a] for a in algs], alpha=0.85, edgecolor="white"
    )
    for bar, val in zip(bars, cvs):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            val + 0.001,
            f"{val:.3f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    ax.set_ylabel(f"Mean coefficient of variance of {metric}")
    ax.set_title("Algorithms sensitivity to hyperparameters")
    ax.set_xticklabels(ALGORITHMS)
    plt.tight_layout()
    plt.show()
